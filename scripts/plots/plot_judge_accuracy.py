#!/usr/bin/env python3
"""Judge accuracy per source quartile, one line per judge, for a domain's rank/ logs.

Left: accuracy (did the judge pick the more capable model). Right: consistent accuracy
(right in BOTH counterbalanced orders) -- a judge that reads position rather than capability
scores ~0.5 on the left and ~0 on the right. Error bars are 95% t-CIs across PAIRS, the unit
the claim replicates over (N = 56-89 pairs per point).

Run:  uv run python scripts/plots/plot_judge_accuracy.py [--domain coding]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log
from scipy import stats

from plot_difficulty import offset

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
COLOR = {"gpt-5.2": "#2a78d6", "deepseek-r1-0528": "#eb6834",
         "o4-mini": "#1baf7a", "gemma-4-31b-it": "#4a3aa7"}


def collect(domain: str) -> dict:
    """{(judge, q_src): {task: metrics}} from every successful log, latest run per task."""
    out = defaultdict(dict)
    for judge_dir in sorted((Path("logs/rank") / domain).iterdir()):
        for cond in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
            for f in cond.glob("*.eval"):
                h = read_eval_log(str(f), header_only=True)
                if h.status != "success" or not h.results:
                    continue
                q = (h.eval.metadata or {}).get("q_src")
                m = {r.name: r.value for s in h.results.scores for r in s.metrics.values()}
                if q is None or "accuracy" not in m:
                    continue
                prev = out[(judge_dir.name, q)].get(h.eval.task)
                # a re-run pair leaves two success logs; keep the newer verdicts
                if prev is None or prev["created"] < h.eval.created:
                    out[(judge_dir.name, q)][h.eval.task] = m | {"created": h.eval.created}
    return out


def mean_ci(values):
    v = np.asarray([x for x in values if x is not None], float)
    ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), ci


def main(domain: str):
    data = collect(domain)
    judges = sorted({j for j, _ in data}, key=lambda j: -np.mean(
        [m["accuracy"] for q in (1, 2, 3) for m in data[(j, q)].values()]))
    quartiles = sorted({q for _, q in data})

    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.9), sharex=True)
    for ax, key, label, chance in ((axs[0], "accuracy", "Accuracy", 0.5),
                                   (axs[1], "consistent_accuracy", "Consistent accuracy", 0.25)):
        ax.axhline(chance, color="#999", lw=0.8, ls="--", zorder=0)
        for i, judge in enumerate(judges):
            dodge = (i - (len(judges) - 1) / 2) * 0.06
            xs, ys, es = [], [], []
            for q in quartiles:
                m, ci = mean_ci([r.get(key) for r in data[(judge, q)].values()])
                xs.append(q + dodge); ys.append(m); es.append(ci)
            ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=3, lw=1, capsize=0,
                        color=COLOR.get(judge, f"C{i}"), label=judge)
        ax.set_xticks(quartiles, [f"Q{q}" for q in quartiles])
        ax.set_xlim(0.7, max(quartiles) + 0.3)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Items judged"); ax.set_ylabel(label)
        ax.tick_params(length=2)
        offset(ax)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.15, -0.35), ncol=2, frameon=False,
                  handlelength=1.2, columnspacing=1)
    fig.subplots_adjust(wspace=0.55)
    out = Path("analysis/judge")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{domain}_judge_accuracy.{ext}", dpi=300, bbox_inches="tight")
    for judge in judges:
        row = " ".join(f"Q{q} {mean_ci([r['accuracy'] for r in data[(judge,q)].values()])[0]:.3f}"
                       f"({len(data[(judge,q)])})" for q in quartiles)
        print(f"{judge:20s} {row}")
    print(f"wrote {out}/{domain}_judge_accuracy.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="coding")
    main(ap.parse_args().domain)
