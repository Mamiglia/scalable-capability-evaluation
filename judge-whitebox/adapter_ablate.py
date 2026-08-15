"""Unmerged-adapter loader: attaches a LoRA adapter without merging it into the base weights, so
individual LoRA modules stay addressable for causal ablation."""
import contextlib
import re

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import verdicts
import wbconfig
import wbprompts


def load_unmerged(model_name, adapter, c):
    """Mirrors verdicts.load_model, but keeps the adapter attached as an unmerged PeftModel
    (no merge_and_unload) and freezes all parameters."""
    model_path = wbconfig.resolve_model(model_name, c)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)

    device = verdicts.pick_device()
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, device


def lora_layers(model):
    """decoder layer index -> its LoRA-wrapped linears: every module carrying lora_A, addressed
    under a `.layers.N.` path segment. Must be exhaustive over the adapter's LoRA modules, so
    raises rather than skipping a module whose name-tail isn't one of adapter_ablate.target_modules
    or that has no `.layers.N.` segment — either means the adapter reaches somewhere unexpected."""
    kinds = wbconfig.cfg()["adapter_ablate"]["target_modules"]
    layers = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        tail = name.rsplit(".", 1)[-1]
        assert tail in kinds, f"lora_A module {name!r} not in adapter_ablate.target_modules {kinds}"
        m = re.search(r"\.layers\.(\d+)\.", name)
        assert m, f"lora_A module {name!r} has no `.layers.N.` path segment"
        layers.setdefault(int(m.group(1)), []).append(module)
    return layers


@contextlib.contextmanager
def toggle(model, active_layers: set[int]):
    """Context manager: for the duration of the block, layers outside `active_layers` are forced
    base-only by zeroing their LoRA modules' `.scaling` (the trained alpha/r multiplier on the
    B(Ax) branch); layers in `active_layers` keep their trained scaling. Restores the original
    scaling on exit, regardless of exceptions inside the block. Not nesting-safe: an outer
    toggle's zeroing can't be undone by an inner toggle's active set."""
    layers = lora_layers(model)
    unknown = active_layers - layers.keys()
    assert not unknown, f"active_layers {unknown} not in model's layers {set(layers)}"
    saved = []  # (module, adapter_name, original_scale)
    for idx, modules in layers.items():
        if idx in active_layers:
            continue
        for module in modules:
            for name in module.scaling:
                saved.append((module, name, module.scaling[name]))
                module.scaling[name] = 0.0
    try:
        yield model
    finally:
        for module, name, scale in saved:
            module.scaling[name] = scale


def verdicts_under(model, tokenizer, df, active_layers, c, invert=False):
    """Verdict readout with only `active_layers`' LoRA active (single `toggle` for the whole
    batched forward). Mirrors verdicts.run's rendering/padding/readout, returns a DataFrame
    instead of writing jsonl. Columns: pair_uid, order, l1, l2, gap, pick, label."""
    device = next(model.parameters()).device
    max_tokens = c["verdicts"]["max_input_tokens"]
    batch_size = c["verdicts"]["batch_size"]

    rows, encs = [], []
    n_skipped = 0
    for _, row in df.iterrows():
        prompt = wbprompts.render_verdict_prompt(row, tokenizer, invert=invert)
        enc = tokenizer(prompt)
        if len(enc["input_ids"]) > max_tokens:
            n_skipped += 1
            continue
        rows.append(row)
        encs.append(enc)
    if n_skipped:
        print(f"skipped {n_skipped} rows over {max_tokens} tokens")

    out = []
    with toggle(model, active_layers):
        for i in range(0, len(encs), batch_size):
            batch_rows = rows[i:i + batch_size]
            batch = tokenizer.pad(encs[i:i + batch_size], return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**batch).logits[:, -1, :]
            for row, row_logits in zip(batch_rows, logits):
                l1, l2 = verdicts.readout(row_logits, tokenizer)
                pick = 1 if l1 > l2 else 2
                out.append(dict(pair_uid=row["pair_uid"], order=int(row["order"]),
                                 l1=l1, l2=l2, gap=l1 - l2, pick=pick, label=int(row["label"])))
    return pd.DataFrame(out, columns=["pair_uid", "order", "l1", "l2", "gap", "pick", "label"])


def metrics(df):
    """Verdict metrics from a verdicts_under-shaped df. order_even/order_odd average and
    anti-average the two orders' gap per pair_uid; pairs missing an order are excluded, with
    the excluded count printed. pick_weaker_acc is the meaningful number when df came from an
    invert=True run, but is always returned."""
    acc = (df["pick"] == df["label"]).mean()
    pos_term = 2 * (df["pick"] == 1).mean() - 1
    pick_weaker_acc = (df["pick"] == 3 - df["label"]).mean()

    by_order = df.groupby(["pair_uid", "order"])["gap"].mean().unstack("order").reindex(columns=[0, 1])
    both = by_order.dropna()
    n_excluded = len(by_order) - len(both)
    if n_excluded:
        print(f"excluded {n_excluded} pair_uid(s) missing an order from order_even/odd")
    order_even = ((both[0] + both[1]) / 2).mean()
    order_odd = ((both[0] - both[1]) / 2).mean()

    return dict(acc=acc, pos_term=pos_term, order_even=order_even, order_odd=order_odd,
                pick_weaker_acc=pick_weaker_acc)


def run_ablation(model, tokenizer, df, groups: dict[str, set[int]], c):
    """Four-condition LoRA ablation per named group (base/group_only/full/full_minus), each a
    sequential verdicts_under + metrics call (never nested), plus derived sufficiency
    (group_only - base), necessity (full - full_minus) and interaction (necessity - sufficiency).
    One row per (group, metric)."""
    all_layers = set(lora_layers(model))
    rows = []
    for group_name, group in groups.items():
        conditions = {
            "base": set(),
            "group_only": group,
            "full": all_layers,
            "full_minus": all_layers - group,
        }
        results = {cond: metrics(verdicts_under(model, tokenizer, df, active_layers, c))
                   for cond, active_layers in conditions.items()}
        for metric_name in results["base"]:
            base, group_only, full, full_minus = (results[cond][metric_name] for cond in
                                                    ("base", "group_only", "full", "full_minus"))
            sufficiency = group_only - base
            necessity = full - full_minus
            rows.append(dict(group=group_name, metric=metric_name, base=base, group_only=group_only,
                              full=full, full_minus=full_minus, sufficiency=sufficiency,
                              necessity=necessity, interaction=necessity - sufficiency))
    return pd.DataFrame(rows, columns=["group", "metric", "base", "group_only", "full", "full_minus",
                                        "sufficiency", "necessity", "interaction"])
