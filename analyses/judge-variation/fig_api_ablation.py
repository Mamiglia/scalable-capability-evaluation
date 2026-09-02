"""Dumbbell figure: API reasoning ablation, baseline -> with-reasoning per judge.

Reads cheap_judges_aime_summary.csv; one row of panels per bench, one column per
metric. Usage: uv run python fig_cheap_judges_aime.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT_DIR = HERE.parent.parent / "judge_results" / "judging_variations"

BASELINE = "#2a78d6"
REASONING = "#eb6834"
METRICS = [("p_pick_stronger", "P(pick stronger)"),
           ("p_pick_longer", "P(pick longer)"),
           ("p_pick_pos1", "P(pick transcript 1)")]


def main():
    rows = list(csv.DictReader(open(OUT_DIR / "cheap_judges_aime_summary.csv")))
    benches = sorted({r["bench"] for r in rows})
    judges = sorted({r["judge"] for r in rows})

    fig, axes = plt.subplots(len(benches), len(METRICS),
                             figsize=(11, 2.1 * len(benches)),
                             sharex=True, sharey=True, squeeze=False)
    for bi, bench in enumerate(benches):
        for mi, (key, label) in enumerate(METRICS):
            ax = axes[bi][mi]
            ax.axvline(0.5, color="#b0b0b0", lw=1, ls="--", zorder=0)
            for ji, judge in enumerate(judges):
                vals = {r["condition"]: float(r[key]) for r in rows
                        if r["bench"] == bench and r["judge"] == judge and r[key]}
                b, w = vals.get("api_baseline"), vals.get("api_with_reasoning")
                if b is None or w is None:
                    continue
                ax.plot([b, w], [ji, ji], color="#999999", lw=1.2, zorder=1)
                ax.scatter([b], [ji], color=BASELINE, s=42, zorder=2)
                ax.scatter([w], [ji], color=REASONING, s=42, zorder=2)
            if bi == len(benches) - 1:
                ax.set_xlabel(label)
            if mi == 0:
                ax.set_yticks(range(len(judges)))
                ax.set_yticklabels(judges)
                ax.text(-0.42, 1.02, bench, transform=ax.transAxes,
                        fontsize=9, color="#52514e")
            ax.set_xlim(0.15, 0.85)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=-1)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BASELINE, label="final text only"),
        plt.Line2D([], [], marker="o", ls="", color=REASONING, label="reasoning shown"),
        plt.Line2D([], [], color="#b0b0b0", ls="--", label="chance"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3,
               bbox_to_anchor=(0.99, 1.04))
    fig.tight_layout()
    out = OUT_DIR / "cot_inclusion"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "cheap_judges_aime.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "cheap_judges_aime.svg", bbox_inches="tight")
    print(f"-> {out}/cheap_judges_aime.png")


if __name__ == "__main__":
    main()
