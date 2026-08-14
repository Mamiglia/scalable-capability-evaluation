"""Capability curve: pointwise vs pairwise judging across judge capability.

Answers "is there a point to pairwise?" visually. Per judge (ordered by the
judge's own ECI), per bench:
  row 1  P(pick stronger) — pairwise (black) vs pointwise (grey), chance 0.5
  row 2  pointwise tie fraction (why the gap exists, and whether it collapses)

Only judges with BOTH pointwise picks (judge_results/direct_judge/picks.csv)
and pairwise a_baseline verdicts (judge_results/judging_variations/verdicts.csv)
on a bench appear in row 1. deepseek/ling (pointwise-only, no pairwise on these
benches) are shown as open reference markers on the tie row.

Unit of the trend = judge model (each x tick is one judge); no across-judge CI
because every unit is plotted. Usage: uv run python fig_capability_curve.py
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POOL = Path("/tmp/pool.json")  # from: git show judging-variations:model-sampler/model_pool.json
BENCHES = ["gpqa", "otis-aime"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLACK, GREY, REF = "#1a1a1a", "#8a8a8a", "#eb6834"


DROP = {"instruct", "chat", "hf"}


def norm(s):
    toks = re.sub(r"[^a-z0-9. ]", " ", s.lower()).split()
    return [t for t in toks if t not in DROP]


def judge_eci():
    pool = json.loads(POOL.read_text())
    keyed = {" ".join(norm(m["name"])): m["eci"] for m in pool if m["eci"] is not None}
    alias = {"deepseek-v4-flash": "deepseek v4 flash", "gpt-oss-20b": "gpt oss 20b",
             "gpt-oss-120b": "gpt oss 120b"}

    def lookup(short):
        s = alias.get(short.replace("api-", ""), short.replace("api-", ""))
        k = " ".join(norm(s))
        if k in keyed:
            return keyed[k]
        kd = " ".join(k.replace(".", " ").split())
        for pk, v in keyed.items():
            if " ".join(pk.replace(".", " ").split()) == kd:
                return v
        return None
    return lookup


def pairwise_rates():
    r = defaultdict(list)
    for row in csv.DictReader(open(ROOT / "judge_results/judging_variations/verdicts.csv")):
        if row["condition"] != "a_baseline" or row["parse_ok"] != "True" or row["picked_major"] == "":
            continue
        r[(row["judge"], row["bench"])].append(row["picked_major"] == "True")
    return {k: sum(v) / len(v) for k, v in r.items()}


def pointwise_rates():
    acc, tie = defaultdict(list), defaultdict(list)
    for row in csv.DictReader(open(ROOT / "judge_results/direct_judge/picks.csv")):
        k = (row["judge"].replace("api-", ""), row["bench"])
        acc[k].append(float(row["picked_major"]))
        tie[k].append(row["tie"] == "True")
    A = {k: sum(v) / len(v) for k, v in acc.items()}
    T = {k: sum(v) / len(v) for k, v in tie.items()}
    return A, T


def main():
    eci = judge_eci()
    pw = pairwise_rates()
    pt_acc, pt_tie = pointwise_rates()

    both = {j for (j, b) in pw} & {j for (j, b) in pt_acc}
    pt_only = {j for (j, b) in pt_acc} - both
    judges = sorted(both, key=lambda j: eci(j) if eci(j) is not None else -1e9)
    refs = sorted(pt_only, key=lambda j: eci(j) if eci(j) is not None else -1e9)
    print("head-to-head judges (ECI):", [(j, eci(j)) for j in judges])
    print("pointwise-only refs (ECI):", [(j, eci(j)) for j in refs])

    fig, axes = plt.subplots(2, len(BENCHES), figsize=(5.0, 3.6), sharex="col")
    x = list(range(len(judges)))
    for c, bench in enumerate(BENCHES):
        top, bot = axes[0][c], axes[1][c]
        pwy = [pw.get((j, bench)) for j in judges]
        pty = [pt_acc.get((j, bench)) for j in judges]
        top.axhline(0.5, ls="--", lw=0.8, color=GREY, zorder=0)
        top.plot(x, pwy, "-s", color=BLACK, lw=1, ms=3, label="pairwise")
        top.plot(x, pty, "-o", color=GREY, lw=1, ms=3, label="pointwise")
        top.set_ylim(0.45, 0.85); top.set_yticks([0.5, 0.7])
        top.text(0.02, 0.98, bench, transform=top.transAxes, va="top", ha="left",
                 fontsize=7, color=GREY)

        tiey = [pt_tie.get((j, bench)) for j in judges]
        bot.plot(x, tiey, "-o", color=GREY, lw=1, ms=3)
        bot.set_ylim(0, 0.65); bot.set_yticks([0, 0.3, 0.6])
        bot.set_xticks(x)
        bot.set_xticklabels(judges, rotation=90, ha="center")
        if c == 0:
            top.set_ylabel("P(pick stronger)")
            bot.set_ylabel("pointwise tie frac.")

    axes[0][0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=2,
                      frameon=False)
    fig.tight_layout()
    out = HERE / "fig_capability_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
