"""Story figures for the DeepSeek finding (gpqa, all judges pooled).

14: the same gap curve three ways — all pairs / without DeepSeek / DeepSeek
only. 15: visible answer length per member, DeepSeek highlighted.

Usage:
  uv run python scripts/judge/fig_deepseek_story.py --out-dir judge_results/multi-eval-pilot/figures
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("interactive_viewer/data/data-expansion-gpqa")
GAP_BINS = [0, 5, 10, 15, 25, 45]
DSK = "deepseek-v4-flash-0731"


def wilson(c, n, z=1.96):
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_runs():
    idx = json.load(open(DATA / "index.json"))
    runs = []
    for r in idx["runs"]:
        if r["is_self"] or r["score"] is None:
            continue
        M, N = idx["models"][r["major"]], idx["models"][r["minor"]]
        ea, eb = M["eci"], N["eci"]
        if not ea or not eb:
            continue
        score = r["score"] if ea >= eb else 1 - r["score"]
        runs.append({"gap": abs(ea - eb), "score": score, "n": 2 * r["n_items"],
                     "dsk": DSK in (r["major"], r["minor"])})
    return idx, runs


def panel(ax, runs, title, color):
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    ax.scatter([r["gap"] for r in runs], [r["score"] for r in runs],
               s=[6 + r["n"] / 40 for r in runs], color=color, alpha=0.4,
               edgecolors="white", linewidths=0.4)
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        sel = [r for r in runs if lo <= r["gap"] < hi]
        if not sel:
            continue
        c = sum(r["score"] * r["n"] for r in sel)
        n = sum(r["n"] for r in sel)
        wlo, whi = wilson(c, n)
        x = (lo + hi) / 2
        ax.errorbar(x, c / n, yerr=[[c / n - wlo], [whi - c / n]], color=color,
                    marker="o", ms=8, capsize=3, lw=2.2, zorder=5)
    xs, ys = [], []
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        sel = [r for r in runs if lo <= r["gap"] < hi]
        if sel:
            xs.append((lo + hi) / 2)
            ys.append(sum(r["score"] * r["n"] for r in sel) / sum(r["n"] for r in sel))
    ax.plot(xs, ys, color=color, lw=2.2, zorder=4)
    ax.set_title(title, fontsize=10, loc="left", color="#333")
    ax.set_xlabel("capability gap (ECI)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-2, 42)


def fig_three_way(runs, out):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    panel(axs[0], runs, "1. All GPQA pairs", "#555")
    panel(axs[1], [r for r in runs if not r["dsk"]], "2. Without DeepSeek", "#2a78d6")
    panel(axs[2], [r for r in runs if r["dsk"]], "3. DeepSeek pairs only", "#c23b22")
    axs[0].set_ylabel("P(pick stronger)")
    fig.tight_layout()
    fig.savefig(out / "14_gap_curve_three_way.png", dpi=150)
    fig.savefig(out / "14_gap_curve_three_way.svg")
    plt.close(fig)
    print("  14_gap_curve_three_way")


def fig_lengths(idx, out):
    rows = []
    for m, meta in idx["models"].items():
        f = DATA / f"transcripts/{m}.json"
        if meta["arm"] != "A" or not f.exists():
            continue
        tr = json.load(open(f))
        if tr:
            rows.append((meta["label"], [len(t) for t in tr.values()], m == DSK))
    rows.sort(key=lambda r: np.median(r[1]))
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 1.8))
    style(ax)
    bp = ax.boxplot([r[1] for r in rows], orientation="horizontal", showfliers=False,
                    patch_artist=True, medianprops={"color": "#222"}, widths=0.6)
    for patch, (_, _, dsk) in zip(bp["boxes"], rows):
        patch.set_facecolor("#c23b22" if dsk else "#9ec2ea")
        patch.set_alpha(0.9)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("answer length (chars)", fontsize=11)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out / "15_transcript_lengths.png", dpi=150)
    fig.savefig(out / "15_transcript_lengths.svg")
    plt.close(fig)
    print("  15_transcript_lengths")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    idx, runs = load_runs()
    fig_three_way(runs, args.out_dir)
    fig_lengths(idx, args.out_dir)


if __name__ == "__main__":
    main()
