"""Sandbag-robustness readout for the FT judge.

Does the judge keep picking the stronger author when that author is told to
sound uncertain? Compares P(pick stronger) on matched natural vs
spoof_uncertain pairs, base vs FT, binned by true capability gap. The
zero-shot reference: strong-told-uncertain crushed zero-shot judges to
0.31 at the largest gap (docs/2026-08-06-style-cap-findings.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import _picked, read_jsonl, gap_fit

SPLITS = ["natural", "spoof_uncertain"]


def add_gap(v, pairs):
    """Attach the true capability gap (|acc_1 - acc_2|) via the pairs parquet."""
    m = pairs[["pair_uid", "order", "cap_1", "cap_2"]]
    v = v.merge(m, on=["pair_uid", "order"], how="left")
    v["gap"] = (v.cap_1 - v.cap_2).abs()
    v["hit"] = (_picked(v) == v.label).astype(float)
    return v


def pair_scores(v):
    """Per-pair counter-balanced P(pick stronger): average over a pair's two orders."""
    return v.groupby(["split", "pair_uid"]).agg(
        hit=("hit", "mean"), gap=("gap", "first")).reset_index()


def summarise(scores, tag):
    rows = []
    for split in SPLITS:
        g = scores[scores.split == split]
        p, se = g.hit.mean(), g.hit.std(ddof=1) / np.sqrt(len(g))
        rows.append(dict(judge=tag, split=split, p_pick_stronger=round(p, 3),
                         se=round(se, 3), n_pairs=len(g)))
    return rows


def plot(all_rows, all_scores, out_stub, model_label):
    sys.path.insert(0, str(Path(__file__).parents[1] / "analyses"))
    from plots import PALETTE, MUTED, chance, legend, save, style
    import matplotlib.pyplot as plt

    series = {  # (judge, split) -> style
        ("base", "natural"): dict(color=PALETTE[0], marker="o", ls="--"),
        ("base", "spoof_uncertain"): dict(color=PALETTE[0], marker="x", ls=":"),
        ("ft", "natural"): dict(color=PALETTE[1], marker="o", ls="-"),
        ("ft", "spoof_uncertain"): dict(color=PALETTE[1], marker="x", ls="-."),
    }
    label = {"natural": "natural", "spoof_uncertain": "strong told uncertain"}
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    chance(ax)
    x_top = max(s.gap.max() for s in all_scores.values())
    xs = np.linspace(0, x_top, 200)
    for (tag, split), st in series.items():
        pts = all_scores[tag]
        pts = pts[pts.split == split]
        by = pts.groupby(pts.gap.round(3)).agg(x=("gap", "mean"), p=("hit", "mean"),
                                               n=("hit", "size"))
        se = np.sqrt(by.p * (1 - by.p) / by.n)
        ax.errorbar(by.x, by.p, yerr=se, fmt=st["marker"], color=st["color"],
                    ms=5, lw=0, elinewidth=0.9, capsize=0, zorder=3)
        rows = all_rows[tag]
        rows = rows[rows.split == split]  # fit on raw 0/1 verdict rows, not pair means
        a, d = gap_fit(rows.gap, rows.hit)
        ax.plot(xs, 1 / (1 + np.exp(-(a + d * xs))), color=st["color"],
                ls=st["ls"], lw=1.6, zorder=2,
                label=f"{model_label} {tag} · {label[split]} (n={len(pts)})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, x_top * 1.03)
    ax.set_xlabel("capability gap (GPQA accuracy difference)")
    ax.set_ylabel("P(judge picks the stronger author)")
    style(ax)
    legend(ax, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.text(0.98, 0.04,
            "points: per-model-pair means ± SE, both-correct pairs · lines: sigmoid fits\n"
            "strong = qwen3.5-122b / qwen3.6-35b told to sound uncertain (accuracy held)",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=6.5)
    save(fig, out_stub)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="judge_results/judge_ft/out_qwen06",
                    help="dir holding spoof_base.jsonl / spoof_ft.jsonl")
    ap.add_argument("--pairs", default="judge_results/judge_ft/pairs_spoof.parquet")
    ap.add_argument("--out-stub", default="judge_results/judge_ft/spoof_curve")
    ap.add_argument("--model-label", default="0.6b", help="legend prefix, e.g. 0.6b / 8b")
    args = ap.parse_args()
    pairs = pd.read_parquet(args.pairs)

    all_rows, all_scores, summary = {}, {}, []
    for tag in ["base", "ft"]:
        v = add_gap(read_jsonl(Path(args.dir) / f"spoof_{tag}.jsonl"), pairs)
        s = pair_scores(v)
        all_rows[tag], all_scores[tag] = v, s
        summary += summarise(s, tag)
    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    table.to_csv(Path(args.out_stub).with_name("spoof_summary.csv"), index=False)
    plot(all_rows, all_scores, args.out_stub, args.model_label)
    print(f"wrote {args.out_stub}.png")


if __name__ == "__main__":
    main()
