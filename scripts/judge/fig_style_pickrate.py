"""Pick rate of the stronger member vs its visible answer length (gpqa).

One point per model that appears as the (ECI-)stronger member of a judged
pair; y = P(judges pick it), x = median final-text transcript length.

Usage:
  uv run python scripts/judge/fig_style_pickrate.py --out judge_results/multi-eval-pilot/figures/12_style_vs_pickrate
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("interactive_viewer/data/data-expansion-gpqa")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    idx = json.load(open(DATA / "index.json"))
    picked, n = {}, {}
    for r in idx["runs"]:
        if r["is_self"] or r["score"] is None:
            continue
        w = 2 * r["n_items"]
        picked[r["major"]] = picked.get(r["major"], 0) + r["score"] * w
        n[r["major"]] = n.get(r["major"], 0) + w

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    ax.text(0.98, 0.505, "chance", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=7, color="#8a8a86")
    for m, c in picked.items():
        f = DATA / f"transcripts/{m}.json"
        if not f.exists() or n[m] < 300:
            continue
        tr = json.load(open(f))
        med = statistics.median(len(t) for t in tr.values())
        dsk = m == "deepseek-v4-flash-0731"
        ax.scatter([med], [c / n[m]], s=30 + n[m] / 60, color="#c23b22" if dsk else "#2a78d6",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(idx["models"][m]["label"], (med, c / n[m]), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.5, color="#c23b22" if dsk else "#444")
    ax.set_xscale("log")
    ax.set_xlabel("median visible answer length (chars, log) — final text only, reasoning stripped")
    ax.set_ylabel("P(judges pick it when it is the stronger member)")
    ax.set_ylim(0.1, 1.05)
    ax.set_title("Judges reward shown work: pick rate of the stronger member vs its answer length\n"
                 "gpqa, all judges + pairs pooled; dot area ∝ verdicts; red = DeepSeek V4 Flash",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    fig.savefig(f"{args.out}.svg")
    print(f"wrote {args.out}.png/.svg")


if __name__ == "__main__":
    main()
