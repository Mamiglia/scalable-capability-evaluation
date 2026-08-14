"""Summary bars: base / plain-FT / combined-FT judge accuracy across the three
held-out sets (plain eval_mm, confidence spoof, style swap). Shows the whole
robustness story in one figure — plain-FT aces the natural sets but collapses on
the sandbag; combined-FT holds everywhere.

P(pick the more-capable author), counter-balanced over each pair's two orders.
Reads the verdict jsonls directly (label + judge slot logits are in the rows)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import _picked, read_jsonl

# (jsonl relative to --root, split within it). None = file not produced for this judge.
FILES = {
    # all three judges pair-matched: spoof on pairs_spoof_heldout, style on pairs_style_heldout
    "8b": {
        "base":     {"eval_mm": ("out/verdicts_base.jsonl", "eval_mm"),
                     "spoof":   ("out/spoof_heldout_base.jsonl", None),
                     "style":   ("out/style_heldout_base.jsonl", None)},
        "plain":    {"eval_mm": ("out/verdicts_ft.jsonl", "eval_mm"),
                     "spoof":   ("out/spoof_heldout_plain.jsonl", None),
                     "style":   ("out/style_heldout_plain.jsonl", None)},
        "combined": {"eval_mm": ("out_combined8b/eval2_plain.jsonl", "eval_mm"),
                     "spoof":   ("out_combined8b/eval2_spoof.jsonl", None),
                     "style":   ("out_combined8b/eval2_style.jsonl", None)},
    },
}

# (set, split-in-file) -> column label. spoof/style files carry two splits each.
CONDITIONS = [
    ("eval_mm", "eval_mm",         "plain\n(eval_mm)"),
    ("spoof",   "natural",         "spoof\nnatural"),
    ("spoof",   "spoof_uncertain", "spoof\nsandbag"),
    ("style",   "natural",         "style\nnatural"),
    ("style",   "style_swap",      "style\nswap"),
]
JUDGES = ["base", "plain", "combined"]


def pair_acc(path, split):
    """Counter-balanced P(pick stronger) + SE over pairs, for one split."""
    d = read_jsonl(path)
    if split is not None:
        d = d[d.split == split]
    d = d.assign(hit=(_picked(d) == d.label).astype(float))
    pair = d.groupby("pair_uid").hit.mean()
    return pair.mean(), pair.std(ddof=1) / np.sqrt(len(pair)), len(pair)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="8b", choices=list(FILES))
    ap.add_argument("--root", default="judge_results/judge_ft")
    ap.add_argument("--out-stub", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    fmap = FILES[args.size]
    out_stub = args.out_stub or f"{root}/combined_summary_{args.size}"

    rows, ns = {}, {}
    for j in JUDGES:
        for kset, ksplit, _ in CONDITIONS:
            fn, filesplit = fmap[j][kset]
            key = (j, kset, ksplit)
            if fn is None:
                rows[key] = (np.nan, np.nan); continue
            # spoof/style files hold both splits under `split`; eval_mm file needs its own split filter
            split = ksplit if kset in ("spoof", "style") else filesplit
            p, se, n = pair_acc(root / fn, split)
            rows[key] = (p, se); ns[key] = n

    sys.path.insert(0, str(Path(__file__).parents[1] / "analyses"))
    from plots import PALETTE, MUTED, chance, legend, save, style
    import matplotlib.pyplot as plt

    jcolor = {"base": MUTED, "plain": PALETTE[0], "combined": PALETTE[2]}
    jlabel = {"base": "base (no FT)", "plain": "plain-FT", "combined": "combined-FT"}
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    chance(ax)
    x = np.arange(len(CONDITIONS)); w = 0.26
    for i, j in enumerate(JUDGES):
        vals = [rows[(j, ks, kp)][0] for ks, kp, _ in CONDITIONS]
        ses = [rows[(j, ks, kp)][1] for ks, kp, _ in CONDITIONS]
        xpos = x + (i - 1) * w
        ax.bar(xpos, vals, w, color=jcolor[j], label=jlabel[j], zorder=2)
        ax.errorbar(xpos, vals, yerr=ses, fmt="none", ecolor=MUTED, elinewidth=0.8,
                    capsize=2, zorder=3)
        for xp, v in zip(xpos, vals):
            if np.isnan(v):
                ax.text(xp, 0.06, "n/a", ha="center", va="bottom", fontsize=7, color=MUTED, rotation=90)
            else:
                ax.text(xp, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in CONDITIONS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("P(judge picks the more-capable author)")
    ax.set_title(f"Qwen3-{args.size} judge — robustness across held-out sets", color="#0b0b0b", fontsize=11)
    style(ax)
    legend(ax, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.3))
    ax.text(0.5, -0.42,
            "bars: counter-balanced P over both-correct pairs ± SE  ·  dashed = chance  ·  "
            "spoof = strong author told to sound uncertain (accuracy held); style-swap = voices exchanged",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=6.5)
    save(fig, out_stub)
    tab = pd.DataFrame([{"judge": j, "condition": f"{ks}:{kp}", "p": round(rows[(j,ks,kp)][0],3),
                         "n_pairs": ns.get((j,ks,kp))} for j in JUDGES for ks,kp,_ in CONDITIONS])
    print(tab.to_string(index=False))
    print(f"\nwrote {out_stub}.png/.svg")


if __name__ == "__main__":
    main()
