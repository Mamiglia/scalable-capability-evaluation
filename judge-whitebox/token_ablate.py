"""Token-resolved causal ablation of an LLM judge's verdict.

token_groups maps a rendered verdict prompt's tokens to interpretable groups (structural spans
+ marker-word spans). The two probes then intervene BY TOKEN POSITION, batch-1 so positions map
straight from token_groups (no left-pad offset):
- FT probe (lora_at_positions): keep the LoRA dW.x only at chosen positions, zero it elsewhere
  -> which tokens' LoRA-write reproduces the base->ft change.
- BASE probe (mean_ablate): replace the residual at chosen positions with the mean over the other
  positions, at every layer -> which tokens the base verdict causally depends on.
Metrics reuse adapter_ablate.metrics (pos_term, content_gap, acc, ...).
"""
import argparse
import contextlib
import pathlib
import re

import numpy as np
import pandas as pd
import torch

import adapter_ablate
import verdicts
import wbconfig
import wbpairs
import wbprompts


def _find_span(haystack, needle, start=0, end=None):
    """Char span (start, end) of the first `needle` in haystack[start:end]. None if absent
    or needle is empty."""
    if not needle:
        return None
    idx = haystack.find(needle, start, end)
    if idx == -1:
        return None
    return idx, idx + len(needle)


def _overlapping_tokens(offsets, span):
    """Token indices whose (start, end) offset overlaps span."""
    start, end = span
    return {i for i, (s, e) in enumerate(offsets) if s < end and e > start}


def _marker_tokens(text, base, lexicon, offsets):
    """Token indices of lexicon words found in `text` (a span starting at `base` in the
    prompt), matched whole-word, case-insensitive."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in lexicon) + r")\b", re.IGNORECASE)
    idx = set()
    for m in pattern.finditer(text):
        idx |= _overlapping_tokens(offsets, (base + m.start(), base + m.end()))
    return idx


def token_groups(row, tokenizer, c, invert=False, fine=False):
    """Map a rendered verdict prompt's tokens to interpretable groups.

    Structural groups (instr_format, slot_labels, question, transcript_1, transcript_2,
    answer_readout) partition every token index exactly once. Marker groups (one per
    token_ablate.markers class in config) are word hits inside transcript_1/transcript_2 and
    are not required disjoint from the structural groups.
    """
    prompt = wbprompts.render_verdict_prompt(row, tokenizer, invert=invert)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)

    text_1_span = _find_span(prompt, row["text_1"])
    assert text_1_span is not None, "text_1 not found in rendered prompt"
    start1, end1 = text_1_span
    text_2_span = _find_span(prompt, row["text_2"], end1)
    assert text_2_span is not None, "text_2 not found after text_1 in rendered prompt"
    start2, end2 = text_2_span
    question_span = _find_span(prompt, row["question"])
    assert question_span is not None, "question not found in rendered prompt"

    label1 = prompt.rfind("Transcript 1:", 0, start1)
    assert label1 != -1, "'Transcript 1:' label not found before text_1"
    digit1 = label1 + len("Transcript ")
    label2 = prompt.find("Transcript 2:", end1, start2)
    assert label2 != -1, "'Transcript 2:' label not found between text_1 and text_2"
    digit2 = label2 + len("Transcript ")

    slot_spans = [(digit1, digit1 + 1), (digit2, digit2 + 1)]
    for span in (_find_span(prompt, row["options_1"], 0, label1),
                 _find_span(prompt, row["options_2"], end1, label2)):
        if span is not None:
            slot_spans.append(span)

    # priority-claimed so structural groups partition the sequence even if a token straddles
    # a span boundary (e.g. merges trailing/leading whitespace into an adjacent span)
    claimed = set()

    def claim(spans):
        idx = set()
        for span in spans:
            idx |= _overlapping_tokens(offsets, span)
        idx -= claimed
        claimed.update(idx)
        return idx

    transcript_1_idx = claim([(start1, end1)])
    transcript_2_idx = claim([(start2, end2)])
    question_idx = claim([question_span])
    slot_labels_idx = claim(slot_spans)
    answer_idx = {n - 1}
    claimed.update(answer_idx)
    instr_format_idx = set(range(n)) - claimed

    groups = {
        "instr_format": sorted(instr_format_idx),
        "slot_labels": sorted(slot_labels_idx),
        "question": sorted(question_idx),
        "transcript_1": sorted(transcript_1_idx),
        "transcript_2": sorted(transcript_2_idx),
        "answer_readout": sorted(answer_idx),
    }

    for name, lexicon in c["token_ablate"]["markers"].items():
        idx = _marker_tokens(row["text_1"], start1, lexicon, offsets) \
            | _marker_tokens(row["text_2"], start2, lexicon, offsets)
        groups[name] = sorted(idx)

    if fine:  # split each transcript into a reasoning-setup head and a conclusion/answer tail
        for tname in ("transcript_1", "transcript_2"):
            idx = groups[tname]
            half = len(idx) // 2
            groups[f"{tname}_head"] = idx[:half]
            groups[f"{tname}_tail"] = idx[half:]

    return groups


class _Holder:
    """Per-forward intervention target: the set of token positions the active probe acts on.
    positions=None disables the intervention (a plain forward)."""
    positions = None


@contextlib.contextmanager
def lora_at_positions(model, holder):
    """FT probe. Forward-hook on every LoRA module recomputes its dW.x increment and subtracts it
    at every position NOT in holder.positions, so the LoRA is active only at holder.positions and
    the base weight path is untouched everywhere. holder.positions is set per row by the caller;
    None = full LoRA (no masking). Batch-1 assumed (positions index the unpadded sequence)."""
    handles = []
    for _, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue

        def hook(mod, inputs, output, mod_ref=None):
            if holder.positions is None:
                return output
            x = inputs[0]
            inc = None
            for name in mod.active_adapters:
                lora_a = mod.lora_A[name]
                # peft keeps LoRA in fp32 on CUDA even under a bf16 base; cast x to the LoRA
                # weight dtype for the matmul, then back to the output dtype before combining.
                term = mod.lora_B[name](lora_a(x.to(lora_a.weight.dtype))) * mod.scaling[name]
                inc = term if inc is None else inc + term
            if inc is None:
                return output
            inc = inc.to(output.dtype)
            keep = torch.zeros(output.shape[1], device=output.device, dtype=output.dtype)
            keep[list(holder.positions)] = 1.0
            return output - inc * (1.0 - keep)[None, :, None]

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _decoder_layers(model):
    """The list of decoder blocks, for either a plain CausalLM or an unmerged PeftModel."""
    for attr in ("model.layers", "base_model.model.model.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers")


def _causal_mean(hs, pos):
    """Replace the residual at each ablated position `pos` with the mean over KEPT positions
    strictly EARLIER than it, so no future-token content leaks into an earlier position (a plain
    mean over ALL other positions would, artefactually inflating later-span responsibility).
    Positions with no earlier kept position are left unchanged. hs: [1, T, H]; returns a new tensor."""
    hs = hs.clone()
    keep = torch.ones(hs.shape[1], dtype=torch.bool, device=hs.device)
    keep[pos] = False
    kept_idx = keep.nonzero(as_tuple=True)[0]
    for p in pos:  # kept positions are never in pos, so reads stay on original values
        earlier = kept_idx[kept_idx < p]
        if len(earlier):
            hs[:, p, :] = hs[:, earlier, :].mean(dim=1).to(hs.dtype)
    return hs


@contextlib.contextmanager
def mean_ablate(model, holder, baseline="mean"):
    """BASE probe. Forward-hook on every decoder block replaces the residual at holder.positions,
    at every layer, with either the causal mean over the OTHER (earlier, kept) positions of that
    sequence (baseline='mean', on-manifold) or zeros (baseline='zero', off-manifold).
    holder.positions set per row by the caller; None or empty = no-op. Batch-1."""
    handles = []

    def hook(mod, inputs, output):
        if not holder.positions:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        pos = list(holder.positions)
        if baseline == "zero":
            hs = hs.clone()
            hs[:, pos, :] = 0.0
        else:
            hs = _causal_mean(hs, pos)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    for layer in _decoder_layers(model):
        handles.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _kept_rows(df, tokenizer, c, fine=False):
    """Rows whose rendered prompt is within max_input_tokens, with their token_groups and length."""
    max_tokens = c["verdicts"]["max_input_tokens"]
    kept, n_skipped = [], 0
    for _, row in df.iterrows():
        prompt = wbprompts.render_verdict_prompt(row, tokenizer)
        ids = tokenizer(prompt)["input_ids"]
        if len(ids) > max_tokens:
            n_skipped += 1
            continue
        kept.append((row, token_groups(row, tokenizer, c, fine=fine), len(ids)))
    if n_skipped:
        print(f"skipped {n_skipped} rows over {max_tokens} tokens")
    return kept


def _verdicts(model, tokenizer, kept, holder, positions_fn):
    """Batch-1 verdict pass; per row set holder.positions=positions_fn(groups, seq) and read
    l1/l2 at the last token. The intervention context manager must already be entered."""
    device = next(model.parameters()).device
    out = []
    for row, grp, seq in kept:
        holder.positions = positions_fn(grp, seq)
        enc = tokenizer(wbprompts.render_verdict_prompt(row, tokenizer), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        l1, l2 = verdicts.readout(logits[-1], tokenizer)
        out.append(dict(pair_uid=row["pair_uid"], order=int(row["order"]), l1=l1, l2=l2,
                         gap=l1 - l2, pick=1 if l1 > l2 else 2, label=int(row["label"])))
    return pd.DataFrame(out, columns=["pair_uid", "order", "l1", "l2", "gap", "pick", "label"])


def _metrics(df, single_order):
    """Verdict metrics. Default reuses adapter_ablate.metrics (drops pairs missing an order);
    single_order computes per-row metrics directly (for the --order recency split, where every
    pair carries only one order)."""
    if not single_order:
        return adapter_ablate.metrics(df)
    sgap = df["gap"].to_numpy() * np.where(df["label"].to_numpy() == 1, 1.0, -1.0)
    return {"order_even": float(df["gap"].mean()),  # single-order: the raw signed slot gap
            "acc": (df["pick"] == df["label"]).mean(),
            "pos_term": 2 * (df["pick"] == 1).mean() - 1,
            "content_gap": float(sgap.mean()),
            "pick_weaker_acc": (df["pick"] == 3 - df["label"]).mean()}


def run_token_ablation(model, tokenizer, df, c, mode, baseline="mean", fine=False, single_order=False):
    """mode='ft': keep the LoRA only at each token-group (vs LoRA-nowhere=base and LoRA-everywhere
    =full) -> sufficiency/necessity of each group's LoRA-write. mode='base': ablate each
    token-group (vs intact) -> the base verdict's dependence on each group; `baseline` picks the
    ablation value (mean|zero). `fine` adds transcript head/tail groups. `single_order` uses
    per-order metrics (for the --order recency split). Returns a tidy df, one row per (group, metric)."""
    kept = _kept_rows(df, tokenizer, c, fine=fine)
    assert kept, "no rows within max_input_tokens"
    holder = _Holder()
    mm = lambda d: _metrics(d, single_order)  # noqa: E731 - metric closure reused per condition
    group_names = list(kept[0][1])
    metric_keys = ["order_even", "content_gap", "pos_term", "acc", "pick_weaker_acc"]

    rows = []
    if mode == "ft":
        with lora_at_positions(model, holder):
            none_m = mm(_verdicts(model, tokenizer, kept, holder, lambda g, s: set()))
            full_m = mm(_verdicts(model, tokenizer, kept, holder, lambda g, s: set(range(s))))
            for gn in group_names:
                only_m = mm(_verdicts(model, tokenizer, kept, holder,
                                                          lambda g, s, gn=gn: set(g[gn])))
                minus_m = mm(_verdicts(model, tokenizer, kept, holder,
                                                           lambda g, s, gn=gn: set(range(s)) - set(g[gn])))
                for mk in metric_keys:
                    rows.append(dict(group=gn, metric=mk, none=none_m[mk], full=full_m[mk],
                                     only=only_m[mk], minus=minus_m[mk],
                                     sufficiency=only_m[mk] - none_m[mk],
                                     necessity=full_m[mk] - minus_m[mk]))
    else:
        with mean_ablate(model, holder, baseline=baseline):
            intact_m = mm(_verdicts(model, tokenizer, kept, holder, lambda g, s: set()))
            for gn in group_names:
                ablate_m = mm(_verdicts(model, tokenizer, kept, holder,
                                                            lambda g, s, gn=gn: set(g[gn])))
                for mk in metric_keys:
                    rows.append(dict(group=gn, metric=mk, intact=intact_m[mk], ablate=ablate_m[mk],
                                     effect=ablate_m[mk] - intact_m[mk]))
    return pd.DataFrame(rows)


def main():
    c = wbconfig.cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="required for --mode ft")
    p.add_argument("--mode", required=True, choices=["base", "ft"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ablate-baseline", choices=["mean", "zero"], default="mean",
                   help="base probe: mean (on-manifold) or zero (off-manifold) ablation value")
    p.add_argument("--fine", action="store_true", help="add transcript head/tail groups")
    p.add_argument("--order", type=int, default=None, choices=[0, 1],
                   help="restrict to one slot order (recency check); metrics computed per-order")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # float32 so the causal interventions are exact: bf16 rounding otherwise leaves 'LoRA off'
    # ~0.4 logit away from the true base, biasing every sufficiency/necessity by that offset.
    if args.mode == "ft":
        assert args.adapter, "--mode ft needs --adapter"
        model, tokenizer, _ = adapter_ablate.load_unmerged(args.model, args.adapter, c, dtype=torch.float32)
    else:
        model, tokenizer, _ = verdicts.load_model(args.model, None, c, dtype=torch.float32)

    df = wbpairs.load_pairs(c, split=c["verdicts"]["eval_split"], limit=args.limit)
    if args.order is not None:
        df = df[df["order"] == args.order]
    out = run_token_ablation(model, tokenizer, df, c, args.mode,
                             baseline=args.ablate_baseline, fine=args.fine,
                             single_order=args.order is not None)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{args.mode} token-ablation -> {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
