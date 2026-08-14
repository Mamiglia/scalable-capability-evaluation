"""Ablation harness: zero (or scale) top-ranked SVD components in place and re-run the verdict +
control battery. single mode ablates one component at a time then restores; cumulative ablates
greedily in rank order without restoring, so effects compound.
"""
import argparse
import json
import pathlib

import pandas as pd
import torch

import corpus
import importance_report
import screen
import verdicts
import wbconfig
import wbfig
import wbpairs
import wbprompts


def apply_ablation(model, decomp, comps, scale):
    """comps: list of (key, comp_idx). For each, W_slice -= (1-scale) * S[c] * U[:,c] @ V[:,c].T,
    written into the hosting module's weight at decomp's row/col slice, in the weight's dtype.
    Returns {key: (module, lo, hi, by_columns, original_slice)} — the slice as it was BEFORE any
    of these comps touched it (saved once per key even if comps hits the same key twice), for
    restore(). Cheaper than reloading the model per component.
    """
    entry_map = screen.qwen_entry_map(model, decomp)
    saved = {}
    for key, comp in comps:
        module, by_columns = entry_map[key]
        U, S, V = decomp.entries[key]
        head = decomp.meta[key].get("head")
        lo, hi = screen._slice_bounds(by_columns, head, U, V)
        W = module.weight.data
        sl = W[:, lo:hi] if by_columns else W[lo:hi, :]
        if key not in saved:
            saved[key] = (module, lo, hi, by_columns, sl.clone())
        delta = (S[comp] * torch.outer(U[:, comp], V[:, comp])).to(dtype=W.dtype, device=W.device)
        sl -= (1 - scale) * delta
    return saved


def restore(model, saved):
    """Undo apply_ablation: write each key's saved original slice back into its module weight."""
    for module, lo, hi, by_columns, original in saved.values():
        W = module.weight.data
        if by_columns:
            W[:, lo:hi] = original
        else:
            W[lo:hi, :] = original


def _verdict_rows(model, tokenizer, pairs_df, c):
    """Same batching/readout as verdicts.run(), returning rows instead of writing them to a file."""
    max_tokens = c["verdicts"]["max_input_tokens"]
    batch_size = c["verdicts"]["batch_size"]
    device = next(model.parameters()).device

    rows, encs = [], []
    for _, row in pairs_df.iterrows():
        prompt = wbprompts.render_verdict_prompt(row, tokenizer)
        enc = tokenizer(prompt)
        if len(enc["input_ids"]) > max_tokens:
            continue
        rows.append(row)
        encs.append(enc)

    out = []
    with torch.no_grad():
        for i in range(0, len(encs), batch_size):
            batch_rows = rows[i:i + batch_size]
            batch = tokenizer.pad(encs[i:i + batch_size], return_tensors="pt").to(device)
            logits = model(**batch).logits[:, -1, :]
            for row, row_logits in zip(batch_rows, logits):
                l1, l2 = verdicts.readout(row_logits, tokenizer)
                pick = 1 if l1 > l2 else 2
                label = int(row["label"])
                out.append({"pair_uid": row["pair_uid"], "order": int(row["order"]),
                            "l1": l1, "l2": l2, "pick": pick, "label": label, "correct": pick == label})
    return out


def _control_nll(model, tokenizer, control_texts, c):
    """Mean per-token NLL over control_texts (token-count-weighted across texts, not text-averaged)."""
    device = next(model.parameters()).device
    max_tokens = c["screen"]["batch_tokens"]
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for entry in control_texts:
            text = screen._resolve_text(entry, "nll", tokenizer)
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            if ids.shape[1] > max_tokens:
                continue
            ids = ids.to(device)
            logits = model(input_ids=ids).logits[0].float()
            loss = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:])
            n_tok = ids.shape[1] - 1
            total_nll += loss.item() * n_tok
            total_tokens += n_tok
    return total_nll / total_tokens if total_tokens else float("nan")


def _load_reference(reference):
    """reference: None, a {(pair_uid, order): pick} dict, or a verdicts-style jsonl path."""
    if reference is None or isinstance(reference, dict):
        return reference
    picks = {}
    with open(reference) as f:
        for line in f:
            row = json.loads(line)
            picks[(row["pair_uid"], row["order"])] = row["pick"]
    return picks


def battery(model, tokenizer, pairs_df, control_texts, c, reference=None):
    """Runs on the model's CURRENT weights (caller applies/restores ablation around this call).
    reference: picks to diff against for flip_rate (None skips it, e.g. for the baseline call).
    Returns acc, position_term (2*P(pick==1)-1 over rows), flip_rate, control_nll, n_pairs, and
    picks (this call's own {(pair_uid, order): pick}, for use as the next call's reference).
    """
    rows = _verdict_rows(model, tokenizer, pairs_df, c)
    picks = {(r["pair_uid"], r["order"]): r["pick"] for r in rows}
    acc = sum(r["correct"] for r in rows) / len(rows)
    position_term = 2 * (sum(r["pick"] == 1 for r in rows) / len(rows)) - 1

    ref = _load_reference(reference)
    flip_rate = None
    if ref is not None:
        common = [k for k in picks if k in ref]
        flip_rate = sum(picks[k] != ref[k] for k in common) / len(common) if common else float("nan")

    return {"acc": acc, "position_term": position_term, "flip_rate": flip_rate,
            "control_nll": _control_nll(model, tokenizer, control_texts, c),
            "n_pairs": len({r["pair_uid"] for r in rows}), "picks": picks}


def _row(mode, comps, key, comp, result):
    return {"mode": mode, "comps": comps, "key": key, "comp": comp,
            "acc": result["acc"], "position_term": result["position_term"],
            "flip_rate": result["flip_rate"], "control_nll": result["control_nll"],
            "n_pairs": result["n_pairs"]}


def run_single(model, tokenizer, decomp, pairs_df, control_texts, comps, scale, c, out_f):
    baseline = battery(model, tokenizer, pairs_df, control_texts, c)
    out_f.write(json.dumps(_row("single", [], None, None, baseline)) + "\n")
    for key, comp in comps:
        saved = apply_ablation(model, decomp, [(key, comp)], scale)
        result = battery(model, tokenizer, pairs_df, control_texts, c, reference=baseline["picks"])
        restore(model, saved)
        out_f.write(json.dumps(_row("single", [[key, comp]], key, comp, result)) + "\n")
        out_f.flush()


def run_cumulative(model, tokenizer, decomp, pairs_df, control_texts, comps, scale, c, out_f):
    baseline = battery(model, tokenizer, pairs_df, control_texts, c)
    out_f.write(json.dumps(_row("cumulative", [], None, None, baseline)) + "\n")
    saved_all, ablated = {}, []
    for key, comp in comps:
        saved = apply_ablation(model, decomp, [(key, comp)], scale)
        for k, v in saved.items():
            saved_all.setdefault(k, v)
        ablated = ablated + [[key, comp]]
        result = battery(model, tokenizer, pairs_df, control_texts, c, reference=baseline["picks"])
        out_f.write(json.dumps(_row("cumulative", ablated, key, comp, result)) + "\n")
        out_f.flush()
    restore(model, saved_all)


def fig_deltas(df, tag):
    """Single- and cumulative-mode: per-ablated-row paired bars of Δacc/Δposition_term/ΔNLL vs
    baseline, labelled like importance_report's top-30 chart."""
    base = df[df["key"].isna()].iloc[0]
    rows = df[df["key"].notna()].reset_index(drop=True)
    labels = [importance_report._label(*importance_report._parse_key(r.key), r.comp)
              for r in rows.itertuples()]
    metrics = [("acc", "Δaccuracy"), ("position_term", "Δposition term"), ("control_nll", "ΔNLL")]

    fig = wbfig.new_fig(wbfig.WIDE, max(2.0, 0.18 * len(rows) + 0.4))
    y_pos = list(range(len(rows)))
    for i, (col, xlabel) in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i + 1)
        ax.barh(y_pos, rows[col] - base[col], color=wbfig.PALETTE[i])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels if i == 0 else [], fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color=wbfig.INK, linewidth=0.8)
        ax.set_xlabel(xlabel)
    wbfig.save(fig, f"ablate_{tag}_deltas")


def fig_dose(df, tag):
    """Cumulative mode: x = components ablated (0 = baseline), accuracy and position term on
    twin y-axes."""
    df = df.copy()
    df["n_ablated"] = df["comps"].apply(len)
    df = df.sort_values("n_ablated")

    fig = wbfig.new_fig(wbfig.WIDE, 3.0)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    l1, = ax.plot(df["n_ablated"], df["acc"], color=wbfig.PALETTE[0], marker="o", label="accuracy")
    l2, = ax2.plot(df["n_ablated"], df["position_term"], color=wbfig.PALETTE[1], marker="o",
                    label="position term")
    ax.set_xlabel("components ablated")
    ax.set_ylabel("accuracy", color=wbfig.PALETTE[0])
    ax2.set_ylabel("position term", color=wbfig.PALETTE[1])
    ax.legend(handles=[l1, l2], frameon=False)
    wbfig.save(fig, f"ablate_{tag}_dose")


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name key into models.hf_ids")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    parser.add_argument("--components", required=True, help="importance_x.csv path")
    parser.add_argument("--top-n", type=int, default=c["ablate"]["top_n"])
    parser.add_argument("--mode", required=True, choices=["single", "cumulative"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decomp", default=None, help="SVD cache path: load if present, else compute+save")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model, tokenizer, device = verdicts.load_model(args.model, args.adapter, c)
    decomp = screen._load_or_build_decomp(model, args.decomp, args.model, args.adapter)

    pairs_df = wbpairs.load_pairs(c, split=c["verdicts"]["eval_split"], limit=args.limit)
    control_texts = corpus.build_control(pairs_df)
    comps = importance_report.top_components(args.components, args.top_n)
    scale = c["ablate"]["scale"]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_fn = run_single if args.mode == "single" else run_cumulative
    with out_path.open("w") as f:
        run_fn(model, tokenizer, decomp, pairs_df, control_texts, comps, scale, c, f)

    with out_path.open() as f:
        df = pd.DataFrame(json.loads(line) for line in f)
    tag = out_path.stem
    fig_deltas(df, tag)
    if args.mode == "cumulative":
        fig_dose(df, tag)

    print(f"mode={args.mode} n_components={len(comps)} -> {out_path}")


if __name__ == "__main__":
    main()
