"""Judge pick rate vs the stronger member's per-item answer length (gpqa),
all grid pairs pooled, with each member placed on the curve at its median.

Usage:
  uv run python scripts/judge/fig_shown_work_curve.py --out judge_results/multi-eval-pilot/figures/13_shown_work_curve
"""

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

DATA = Path("interactive_dashboards/transcript_viewer/data/data-expansion-gpqa")
EDGES = [0, 175, 300, 425, 575, 750, 950, 1200, 1600, 2400, 99999]
BINS = list(zip(EDGES[:-1], EDGES[1:]))
CENTERS = [120, 232, 358, 494, 656, 845, 1068, 1385, 1959, 3200]
DSK = "deepseek-v4-flash-0731"


def wilson(c, n, z=1.96):
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def load():
    idx = json.load(open(DATA / "index.json"))
    tr = {}
    for m in idx["models"]:
        f = DATA / f"transcripts/{m}.json"
        if f.exists():
            tr[m] = json.load(open(f))
    bins = [[0, 0] for _ in BINS]
    for pf in glob.glob(str(DATA / "pairs/*__full.json")):
        major, minor, _ = Path(pf).stem.split("__")
        if major == minor or major not in tr or minor not in tr:
            continue
        if idx["models"][major]["arm"] != "A" or idx["models"][minor]["arm"] != "A":
            continue
        for item, judges in json.load(open(pf)).items():
            L = len(tr[major].get(item, ""))
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= L < hi)
            for j, cell in judges.items():
                for o in cell["orders"]:
                    if o and o["picked_major"] is not None:
                        bins[b][0] += o["picked_major"]
                        bins[b][1] += 1
    members = {m: statistics.median(len(t) for t in tr[m].values())
               for m, meta in idx["models"].items() if meta["arm"] == "A" and m in tr and tr[m]}
    labels = {m: idx["models"][m]["label"] for m in members}
    return bins, members, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bins, members, labels = load()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    ax.axvspan(150, 1150, color="#eda100", alpha=0.10, zorder=0)

    keep = [i for i, (c, n) in enumerate(bins) if n]
    xs = np.array([CENTERS[i] for i in keep], float)
    bins = [bins[i] for i in keep]
    ys = np.array([c / n for c, n in bins])
    ws = np.array([n for _, n in bins], float)
    def scaled_sigmoid(lx, a, d, lo, hi):
        return lo + (hi - lo) / (1 + np.exp(-a * (lx - d)))
    (sa, sd, slo, shi), _ = curve_fit(scaled_sigmoid, np.log(xs), ys,
                                      p0=[3.0, math.log(700), 0.08, 0.67],
                                      sigma=1 / np.sqrt(ws), maxfev=10000)
    gx = np.geomspace(xs.min(), xs.max(), 200)
    fit_y = scaled_sigmoid(np.log(gx), sa, sd, slo, shi)
    ax.plot(gx, fit_y, color="#999", lw=1.6, ls="--", zorder=2)
    for x, (c, n) in zip(xs, bins):
        lo, hi = wilson(c, n)
        ax.plot([x, x], [lo, hi], color="#555", lw=1.5)
        ax.scatter([x], [c / n], color="#555", s=45, zorder=4)

    # members placed on the curve at their median answer length
    for i, (m, L) in enumerate(sorted(members.items(), key=lambda kv: kv[1])):
        y = float(slo + (shi - slo) / (1 + np.exp(-sa * (math.log(L) - sd))))
        dsk = m == DSK
        ax.scatter([L], [y], s=110, color="#c23b22" if dsk else "#2a78d6",
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(labels[m], (L, y), textcoords="offset points",
                    xytext=(0, (10, -16, 24, -30)[i % 4]), ha="center", fontsize=7.5,
                    color="#c23b22" if dsk else "#1f5da8",
                    fontweight="bold" if dsk else "normal")

    ax.text(400, 0.88, "sensitive zone", fontsize=9, color="#a97100", ha="center")
    ax.text(1900, 0.78, "plateau", fontsize=9, color="#555", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("stronger member's answer length (chars)")
    ax.set_ylabel("P(judge picks stronger)")
    ax.set_ylim(0, 0.95)
    h = [plt.Line2D([], [], marker="o", ls="", c="#555", ms=7,
                    label="verdicts binned by answer length (Wilson 95% CI)"),
         plt.Line2D([], [], ls="--", c="#999", label="sigmoid fit"),
         plt.Line2D([], [], marker="o", ls="", c="#2a78d6", ms=9,
                    label="member at its median answer length (on fit)"),
         plt.Line2D([], [], marker="o", ls="", c="#c23b22", ms=9, label="DeepSeek V4 Flash")]
    ax.legend(handles=h, frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Pick rate vs shown work — all gpqa grid pairs pooled", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    fig.savefig(f"{args.out}.svg")
    print(f"wrote {args.out}.png/.svg")


if __name__ == "__main__":
    main()
