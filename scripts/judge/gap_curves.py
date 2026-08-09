"""Gap-curve figures from judge logs: accuracy vs ECI gap, by judge and by eval.

Reads every success log under the given log roots, pools verdicts per
(bench, judge, pair), and writes a summary CSV plus two PNGs.

Usage:
  uv run python scripts/judge/gap_curves.py --out-dir judge_results/multi-eval-pilot
"""

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

MAIN_ROOT = "logs/data-expansion"
PROBE_ROOT = "logs/data-expansion-biggap"
COMPARISONS_CSV = "configs/comparisons/comparisons.csv"

# ECI gaps for the epoch-transcript probe pairs (ECIs in configs/judge_biggap_gpqa.yaml).
PROBE_GAPS = {
    "qwen3.6-plus-vs-llama-3.1-70b": 23.8,
    "gpt-5.5-low-vs-claude-3-5-sonnet-0620": 29.2,
    "gemini-3.5-flash-vs-llama-3.1-70b": 29.4,
    "gpt-5.5-low-vs-llama-3.1-70b": 34.0,
    "gpt-5.5-low-vs-gemma-2-27b": 37.1,
}

# display name (comparisons.csv) per shortname, for delta_eci lookup
DISPLAY = {
    "qwen3-8b": "Qwen3 8B",
    "qwen3-32b": "Qwen3 32B",
    "qwen3-vl-30b-a3b-instruct": "Qwen3 VL 30B A3B Instruct",
    "glm-4.6": "GLM-4.6",
    "ling-2.6-flash": "Ling-2.6-flash",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude-opus-4.7": "Claude Opus 4.7",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gpt-4.1": "GPT-4.1",
    "o4-mini": "o4-mini",
}

# fixed judge order (by ECI) -> fixed categorical slot; never re-mapped
JUDGE_ORDER = ["qwen3-8b", "qwen3-32b", "qwen3-vl-30b-a3b-instruct",
               "ling-2.6-flash", "glm-4.6", "deepseek-v4-flash-0731"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
EVAL_ORDER = ["gpqa", "aime2025", "aime2026", "gpqa (epoch transcripts)"]
GRID_BINS = [0, 5, 10, 15, 25]


def wilson(c, n, z=1.96):
    if n == 0:
        return 0, 0
    p = c / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def load_rows(main_root, probe_root, comparisons_csv):
    delta = {}
    for r in csv.DictReader(open(comparisons_csv)):
        delta[frozenset([r["model_a"], r["model_b"]])] = float(r["delta_eci"])
    rows = []  # (source_eval, judge, pair, gap, correct, n)
    counts = defaultdict(lambda: [0, 0])
    for p in glob.glob(f"{main_root}/*/*/*/phase-*/*.eval") + glob.glob(f"{probe_root}/*/*/*/phase-*/*.eval"):
        parts = Path(p).parts
        root, bench, judge, pair = parts[0] + "/" + parts[1], parts[2], parts[3], parts[4]
        a, b = pair.split("-vs-")
        if a == b:
            continue  # self-pairs have no ground truth
        if read_eval_log(p, header_only=True).status != "success":
            continue
        if probe_root.endswith(parts[1]):
            src, gap = "gpqa (epoch transcripts)", PROBE_GAPS[pair]
        else:
            src, gap = bench, delta[frozenset([DISPLAY[a], DISPLAY[b]])]
        for s in read_eval_log(p).samples:
            sc = list(s.scores.values())[0] if s.scores else None
            if sc and sc.value in ("C", "I"):
                key = (src, judge, pair, gap)
                counts[key][1] += 1
                if sc.value == "C":
                    counts[key][0] += 1
    for (src, judge, pair, gap), (c, n) in counts.items():
        rows.append({"eval": src, "judge": judge, "pair": pair, "eci_gap": gap,
                     "n_verdicts": n, "n_correct": c, "accuracy": c / n})
    return rows


def binned(rows, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["eci_gap"] < hi]
        c = sum(r["n_correct"] for r in sel)
        n = sum(r["n_verdicts"] for r in sel)
        if n:
            out.append(((lo + hi) / 2, c / n, n, *wilson(c, n)))
    return out


def style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    ax.set_ylim(0.1, 1.05)


def fig_by_judge(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, judge, color in zip(axes.flat, JUDGE_ORDER, PALETTE):
        mine = [r for r in rows if r["judge"] == judge]
        grid = [r for r in mine if r["eval"] != "gpqa (epoch transcripts)"]
        probe = [r for r in mine if r["eval"] == "gpqa (epoch transcripts)"]
        style(ax)
        ax.scatter([r["eci_gap"] for r in grid], [r["accuracy"] for r in grid],
                   s=[8 + r["n_verdicts"] / 6 for r in grid], color=color, alpha=0.45,
                   edgecolors="white", linewidths=0.5)
        for x, y, n, lo, hi in binned(grid, GRID_BINS):
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], color=color, marker="o",
                        ms=7, capsize=3, lw=2, zorder=5)
        bl = binned(grid, GRID_BINS)
        ax.plot([b[0] for b in bl], [b[1] for b in bl], color=color, lw=2, zorder=4)
        if probe:
            ax.scatter([r["eci_gap"] for r in probe], [r["accuracy"] for r in probe],
                       s=[8 + r["n_verdicts"] / 6 for r in probe], marker="D",
                       facecolors="none", edgecolors=color, linewidths=1.6, zorder=6)
        ax.set_title(judge, fontsize=10, color="#333")
    for ax in axes[1]:
        ax.set_xlabel("ECI gap between pair members")
    for ax in axes[:, 0]:
        ax.set_ylabel("judge accuracy")
    h = [plt.Line2D([], [], marker="o", ls="-", c="#666", ms=7, label="grid pairs, binned (95% Wilson CI)"),
         plt.Line2D([], [], marker="o", ls="", c="#666", alpha=0.45, label="single pair (dot area ∝ verdicts)"),
         plt.Line2D([], [], marker="D", ls="", mfc="none", mec="#666", label="epoch-transcript probe pair"),
         plt.Line2D([], [], ls="--", c="#8a8a86", label="chance (0.5)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Judge accuracy vs capability gap, by judge — gpqa + aime2025 + aime2026 pooled;"
                 " both orders per pair; self-pairs excluded", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_by_eval(rows, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    style(ax)
    n_offsets = [-14, 11, -24, -14]  # per-series stagger so n labels don't collide
    for si, (src, color) in enumerate(zip(EVAL_ORDER, PALETTE)):
        mine = [r for r in rows if r["eval"] == src]
        if not mine:
            continue
        if src == "gpqa (epoch transcripts)":
            pts = sorted({(r["eci_gap"],) for r in mine})
            data = []
            for (g,) in pts:
                sel = [r for r in mine if r["eci_gap"] == g]
                c, n = sum(r["n_correct"] for r in sel), sum(r["n_verdicts"] for r in sel)
                data.append((g, c / n, n, *wilson(c, n)))
        else:
            data = binned(mine, GRID_BINS)
        xs, ys = [d[0] for d in data], [d[1] for d in data]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7,
                ls="-" if src != "gpqa (epoch transcripts)" else ":")
        for x, y, n, lo, hi in data:
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], color=color, capsize=3, lw=1.5)
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, n_offsets[si]),
                        fontsize=7, color="#666", ha="center")
        ax.annotate(src, (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("ECI gap between pair members")
    ax.set_ylabel("judge accuracy")
    ax.set_xlim(0, 42)
    ax.set_title("Judge accuracy vs capability gap, by eval\nall 6 cheap judges pooled; both orders per pair; self-pairs excluded", fontsize=10, loc="left")
    ax.legend(handles=[plt.Line2D([], [], c=c, lw=2, ls=":" if s.endswith(")") else "-",
                                  marker="o", label=s) for s, c in zip(EVAL_ORDER, PALETTE)],
              frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--main-root", default=MAIN_ROOT)
    ap.add_argument("--probe-root", default=PROBE_ROOT)
    ap.add_argument("--comparisons", default=COMPARISONS_CSV)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.main_root, args.probe_root, args.comparisons)
    with open(args.out_dir / "gap_curve_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["eval"], r["judge"], r["eci_gap"])))
    fig_by_judge(rows, args.out_dir / "gap_curve_by_judge.png")
    fig_by_eval(rows, args.out_dir / "gap_curve_by_eval.png")
    print(f"{len(rows)} (eval, judge, pair) rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
