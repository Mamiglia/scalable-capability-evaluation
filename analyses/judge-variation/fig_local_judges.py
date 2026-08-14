"""Two figures from the local-judge grid.

conditions_q2: prompt conditions a-d (same full pair set) per judge x bench.
cot_q1: a -> e dumbbells on the matched CoT-pair subset (a restricted to
e's pairs; e from each judge's long-context serving).

Cells under MIN_N parsed verdicts are dropped (printed, not drawn).
Usage: uv run python fig_local_judges.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

HERE = Path(__file__).parent
OUT_DIR = HERE.parent.parent / "judge_results" / "judging_variations"
COT_DIR = OUT_DIR / "cot_inclusion"
INSTR_DIR = OUT_DIR / "ignore_instructions"
MIN_N = 300

JUDGES = {"qwen3-0.6b": "#2a78d6", "llama-3.2-3b": "#eb6834", "qwen3-8b": "#1baf7a",
          "qwen3-14b": "#e87ba4", "qwen3-32b": "#eda100", "gpt-oss-20b": "#9085e9", "gpt-oss-120b": "#4a3aa7"}
E_SERVING = {"qwen3-0.6b": "qwen3-0.6b-yarn", "llama-3.2-3b": "llama-3.2-3b",
             "qwen3-8b": "qwen3-8b-yarn", "qwen3-14b": "qwen3-14b-yarn",
             "qwen3-32b": "qwen3-32b-yarn"}
CONDITIONS = [("a_baseline", "baseline"),
              ("b_ignore_superficial", "ignore superficial"),
              ("c_ignore_superficial_confidence", "+ ignore confidence"),
              ("d_pick_weaker", "pick weaker (inv.)")]
METRICS = [("p_stronger", "P(pick stronger)"), ("p_longer", "P(pick longer)"),
           ("p_pos1", "P(pick transcript 1)")]


def rates(rs):
    n = len(rs)
    if n < MIN_N:
        return None
    out = {"n": n,
           "p_stronger": sum(r["picked_major"] == "True" for r in rs) / n,
           "p_pos1": sum(r["picked"] == "1" for r in rs) / n}
    pl = [r for r in rs if r["picked_longer"] != ""]
    out["n_longer"] = len(pl)
    out["p_longer"] = sum(r["picked_longer"] == "True" for r in pl) / max(len(pl), 1)
    return out


def ci95(p, n):
    return 1.96 * (p * (1 - p) / n) ** 0.5 if n else 0.0


def panel_axes(n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 1.8 * n_rows),
                             sharex=True, squeeze=False)
    return fig, axes


def style(ax, bench, mi, bi, n_rows, label):
    ax.axvline(0.5, color="#b0b0b0", lw=1, ls="--", zorder=0)
    ax.set_xlim(0.15, 0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=-1)
    if bi == n_rows - 1:
        ax.set_xlabel(label)
    if mi == 0:
        ax.text(-0.45, 1.02, bench, transform=ax.transAxes, fontsize=9,
                color="#52514e")


def main():
    rows = list(csv.DictReader(open(OUT_DIR / "verdicts.csv")))
    parsed = [r for r in rows if r["parse_ok"] == "True"]
    benches = sorted({r["bench"] for r in parsed})
    by = defaultdict(list)
    for r in parsed:
        by[(r["bench"], r["condition"], r["judge"])].append(r)

    # ---- Q2: conditions a-d ----
    fig, axes = panel_axes(len(benches), len(METRICS))
    for bi, bench in enumerate(benches):
        for mi, (key, label) in enumerate(METRICS):
            ax = axes[bi][mi]
            for judge, color in JUDGES.items():
                xs, errs = [], []
                for cond, cname in CONDITIONS:
                    rr = rates(by.get((bench, cond, judge), []))
                    if rr is None:
                        if mi == 0 and by.get((bench, cond, judge)):
                            print(f"drop {bench}/{cond}/{judge}: n={len(by[(bench, cond, judge)])}")
                        xs.append(float("nan"))
                        errs.append(0.0)
                        continue
                    n = rr["n_longer"] if key == "p_longer" else rr["n"]
                    xs.append(rr[key])
                    errs.append(ci95(rr[key], n))
                ys = range(len(CONDITIONS))
                ax.plot(xs, ys, color=color, lw=1, alpha=0.4, zorder=1)
                ax.errorbar(xs, ys, xerr=errs, fmt="o", color=color, ms=6.5,
                            elinewidth=1.2, capsize=2.5, zorder=2)
            ax.set_yticks(range(len(CONDITIONS)))
            ax.set_yticklabels([c[1] for c in CONDITIONS])
            ax.invert_yaxis()
            style(ax, bench, mi, bi, len(benches), label)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=j)
               for j, c in JUDGES.items()]
    handles.append(plt.Line2D([], [], color="#b0b0b0", ls="--", label="chance"))
    handles.append(plt.Line2D([], [], color="#52514e", lw=1, marker="|", ms=7,
                              label="95% CI"))
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=5,
               bbox_to_anchor=(0.99, 1.05))
    fig.tight_layout()
    COT_DIR.mkdir(exist_ok=True); INSTR_DIR.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(INSTR_DIR / f"conditions.{ext}", dpi=150,
                    bbox_inches="tight")
    print(f"-> {INSTR_DIR}/conditions.png")

    # ---- Q1: a -> e dumbbells on matched pairs ----
    fig, axes = panel_axes(len(benches), len(METRICS))
    judges = list(JUDGES)
    for bi, bench in enumerate(benches):
        e_pairs = {(r["major"], r["minor"]) for r in parsed
                   if r["bench"] == bench and r["condition"] == "e_with_reasoning"}
        for mi, (key, label) in enumerate(METRICS):
            ax = axes[bi][mi]
            for ji, judge in enumerate(judges):
                a = rates([r for r in by.get((bench, "a_baseline", judge), [])
                           if (r["major"], r["minor"]) in e_pairs])
                e = rates(by.get((bench, "e_with_reasoning", E_SERVING.get(judge, judge)), []))
                if a is None or e is None:
                    if mi == 0:
                        print(f"drop q1 {bench}/{judge}: a={'ok' if a else 'thin'} e={'ok' if e else 'thin'}")
                    continue
                ax.plot([a[key], e[key]], [ji, ji], color="#999999", lw=1.2, zorder=1)
                ax.scatter([a[key]], [ji], color=JUDGES[judge], s=42, zorder=2)
                ax.scatter([e[key]], [ji], color=JUDGES[judge], s=42, zorder=2,
                           marker="D")
            ax.set_yticks(range(len(judges)))
            ax.set_yticklabels(judges)
            ax.invert_yaxis()
            style(ax, bench, mi, bi, len(benches), label)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#52514e", label="final text only"),
               plt.Line2D([], [], marker="D", ls="", color="#52514e", label="reasoning shown"),
               plt.Line2D([], [], color="#b0b0b0", ls="--", label="chance")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3,
               bbox_to_anchor=(0.99, 1.05))
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(COT_DIR / f"cot_dumbbells.{ext}", dpi=150, bbox_inches="tight")
    print(f"-> {COT_DIR}/cot_dumbbells.png")


if __name__ == "__main__":
    main()
