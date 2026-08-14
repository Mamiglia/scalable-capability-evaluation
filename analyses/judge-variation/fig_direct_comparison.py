"""Per-pair direct comparison scatters + reorganized outputs.

judge_results/judging_variations/
  cot_inclusion/       a vs e (matched pairs): dumbbells, pair scatter, summary
  ignore_instructions/ a vs b/c/d: conditions grid, pair scatters, summary

Each scatter point = one pair's P(pick stronger): x under baseline, y under the
comparison condition; diagonal = no effect. Pairs need >= MIN_PAIR_N parsed
verdicts on both sides. Small multiples: one column per judge (colour = judge,
kept exclusive to that encoding), one row per condition, so panels stay
readable as more judges land.

Usage: uv run python fig_direct_comparison.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5})

HERE = Path(__file__).parent
BASE = HERE.parent.parent / "judge_results" / "judging_variations"
COT_DIR = BASE / "cot_inclusion"
INSTR_DIR = BASE / "ignore_instructions"
MIN_PAIR_N = 20

JUDGES = {"qwen3-0.6b": "#2a78d6", "llama-3.2-3b": "#eb6834", "qwen3-8b": "#1baf7a",
          "qwen3-14b": "#e87ba4", "qwen3-32b": "#eda100", "gpt-oss-20b": "#9085e9", "gpt-oss-120b": "#4a3aa7"}
E_SERVING = {"qwen3-0.6b": "qwen3-0.6b-yarn", "llama-3.2-3b": "llama-3.2-3b",
             "qwen3-8b": "qwen3-8b-yarn", "qwen3-14b": "qwen3-14b-yarn",
             "qwen3-32b": "qwen3-32b-yarn"}
BENCH_MARK = {"gpqa": "o", "otis-aime": "D"}


def pair_rates(rows):
    """(judge-as-plotted, bench, major, minor) -> P(pick stronger), n."""
    grp = defaultdict(list)
    for r in rows:
        grp[(r["judge"], r["bench"], r["major"], r["minor"])].append(
            r["picked_major"] == "True")
    return {k: (sum(v) / len(v), len(v)) for k, v in grp.items() if len(v) >= MIN_PAIR_N}


def judge_cell(ax, judge, base_rates, cond_rates, judge_map):
    """One judge's pairs, condition vs baseline. Returns points drawn."""
    ax.plot([0, 1], [0, 1], color="#b0b0b0", lw=0.8, ls="--", zorder=0)
    n = 0
    for (j, bench, major, minor), (y, _) in cond_rates.items():
        base_judge = judge_map.get(j, j)
        if base_judge != judge:
            continue
        b = base_rates.get((base_judge, bench, major, minor))
        if b is None:
            continue
        ax.scatter([b[0]], [y], color=JUDGES[judge], s=8, alpha=0.55,
                   marker=BENCH_MARK[bench], zorder=2, linewidths=0)
        n += 1
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    return n


def bench_legend(fig):
    h = [plt.Line2D([], [], marker=m, ls="", ms=3, color="#52514e", label=b)
         for b, m in BENCH_MARK.items()]
    h.append(plt.Line2D([], [], color="#b0b0b0", ls="--", lw=0.8, label="no effect"))
    fig.legend(handles=h, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), fontsize=6, columnspacing=0.9,
               handletextpad=0.3)


def judges_with_data(rates, judge_map):
    present = {judge_map.get(k[0], k[0]) for k in rates}
    return [j for j in JUDGES if j in present]


def main():
    rows = [r for r in csv.DictReader(open(BASE / "verdicts.csv"))
            if r["parse_ok"] == "True"]
    for d in (COT_DIR, INSTR_DIR):
        d.mkdir(parents=True, exist_ok=True)

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    # baseline restricted to each bench's e-pairs, for the cot comparison
    e_pairs = {(r["bench"], r["major"], r["minor"]) for r in by_cond["e_with_reasoning"]}
    a_matched = [r for r in by_cond["a_baseline"]
                 if (r["bench"], r["major"], r["minor"]) in e_pairs]

    # ---- cot_inclusion scatter: a (matched) vs e, one panel per judge ----
    base_rates = pair_rates(a_matched)
    cond_rates = pair_rates(by_cond["e_with_reasoning"])
    judge_map = {v: k for k, v in E_SERVING.items()}  # serving name -> native judge
    cot_judges = judges_with_data(cond_rates, judge_map)
    fig, axes = plt.subplots(1, len(cot_judges),
                             figsize=(1.05 * len(cot_judges) + 0.4, 1.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, judge in zip(axes[0], cot_judges):
        n = judge_cell(ax, judge, base_rates, cond_rates, judge_map)
        ax.set_title(judge, fontsize=6, color=JUDGES[judge], loc="left")
        ax.set_xlabel("final text", fontsize=7)
        if n == 0:
            print(f"cot pair_scatter: {judge} empty")
    axes[0][0].set_ylabel("CoT shown", fontsize=7)
    bench_legend(fig)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(COT_DIR / f"pair_scatter.{ext}", dpi=300, bbox_inches="tight")
    print(f"-> {COT_DIR}/pair_scatter.png  (P(pick stronger) per pair)")

    # ---- ignore_instructions scatters: a vs b/c/d, judge x condition grid ----
    base_rates = pair_rates(by_cond["a_baseline"])
    conds = [("b_ignore_superficial", "ignore-style"),
             ("c_ignore_superficial_confidence", "+ confidence"),
             ("d_pick_weaker", "pick-weaker (inv)")]
    cond_rates = {c: pair_rates(by_cond[c]) for c, _ in conds}
    instr_judges = [j for j in JUDGES
                    if any(j in judges_with_data(cr, {}) for cr in cond_rates.values())]
    fig, axes = plt.subplots(len(conds), len(instr_judges),
                             figsize=(1.05 * len(instr_judges) + 0.6, 1.05 * len(conds) + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    for ri, (cond, label) in enumerate(conds):
        for ci, judge in enumerate(instr_judges):
            ax = axes[ri][ci]
            n = judge_cell(ax, judge, base_rates, cond_rates[cond], {})
            if ri == 0:
                ax.set_title(judge, fontsize=6, color=JUDGES[judge], loc="left")
            if ci == 0:
                ax.set_ylabel(label, fontsize=7)
            if n == 0:
                print(f"instr pair_scatter: {judge} x {label} empty")
    for ax in axes[-1]:
        ax.set_xlabel("baseline", fontsize=7)
    bench_legend(fig)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(INSTR_DIR / f"pair_scatter.{ext}", dpi=300, bbox_inches="tight")
    print(f"-> {INSTR_DIR}/pair_scatter.png  (P(pick stronger) per pair)")


if __name__ == "__main__":
    main()
