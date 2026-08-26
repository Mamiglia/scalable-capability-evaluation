#!/usr/bin/env python3
"""Easy/hard split of the pooled GPQA + HLE-gold set: the easiest quartile by pooled-2PL
difficulty (analysis/irt/pooled_gold_difficulties.csv) vs the rest.

Writes analysis/irt/easy_hard_gold.{png,svg,csv}:
  top-left   item difficulties per benchmark, with the cut
  top-right  per-model accuracy easy vs hard (mean over items ±95% t-CI, N = items),
             Pearson r / Spearman rho across models; open markers = GPQA logs only
  bottom     performance curve: accuracy on each set per model, models ordered by pooled
             ability theta (rank numbers match the scatter labels)
Run:  uv run python scripts/plots/plot_easy_hard.py [--easy-frac 0.25]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_evals.hle.hle import get_verified_gold_uids
from scipy import stats

from plot_difficulty import BENCHES, offset
from plot_irt_pooled import COLOR, OUT, responses

SET = {"easy": dict(color="#999", ls="--"), "hard": dict(color="#222", ls="-")}


def ci95(m: pd.DataFrame) -> pd.Series:
    n = m.notna().sum()
    return stats.t.ppf(0.975, n - 1) * m.std() / np.sqrt(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--easy-frac", type=float, default=0.25)
    frac = ap.parse_args().easy_frac

    df = responses()
    df = df[(df.benchmark != "hle") | df.uid.isin(get_verified_gold_uids())]
    diffs = pd.read_csv(OUT / "pooled_gold_difficulties.csv").sort_values("b")
    n_easy = round(frac * len(diffs))
    easy_ids = set(map(tuple, diffs[["benchmark", "item"]].values[:n_easy]))
    cut = diffs.b.iloc[n_easy - 1]
    df["easy"] = [(b, i) in easy_ids for b, i in zip(df.benchmark, df.item)]

    M = df.groupby(["easy", "benchmark", "item", "model"]).correct.mean().unstack("model")
    easy, hard = M.loc[True], M.loc[False]
    theta = pd.read_csv(OUT / "pooled_gold_capabilities.csv").set_index("model").theta
    acc = pd.DataFrame({"easy": easy.mean(), "easy_ci": ci95(easy),
                        "hard": hard.mean(), "hard_ci": ci95(hard), "theta": theta,
                        "has_hle": M.columns.isin(df.model[df.benchmark == "hle"])})
    acc = acc.sort_values("theta", ascending=False)
    acc.round(3).to_csv(OUT / "easy_hard_gold.csv")
    r, rho = stats.pearsonr(acc.easy, acc.hard)[0], stats.spearmanr(acc.easy, acc.hard)[0]
    comp = diffs.iloc[:n_easy].benchmark.value_counts().to_dict()
    print(f"easy = {n_easy} items {comp}, b <= {cut:.2f}; hard = {len(diffs) - n_easy}")
    print(acc.round(3).to_string())
    print(f"easy vs hard accuracy across {len(acc)} models: r = {r:.3f}, rho = {rho:.3f}")

    fig = plt.figure(figsize=(3.4, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1], wspace=0.5, hspace=0.6)
    ax_b, ax_s, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    bins = np.linspace(-3, 3, 25)
    ax_b.hist([diffs.b[diffs.benchmark == b] for b in BENCHES], bins=bins, stacked=True,
              color=[COLOR[b] for b in BENCHES], alpha=0.7, lw=0, label=[b.upper() for b in BENCHES])
    ax_b.axvline(cut, color="#333", lw=0.8, ls="--")
    ax_b.text(cut - 0.1, ax_b.get_ylim()[1], f"easiest\n{frac:.0%}", fontsize=6, color="#333", ha="right", va="top")
    ax_b.set_xlabel("Difficulty b"); ax_b.set_ylabel("Items")
    ax_b.set_xlim(-3, 3); ax_b.set_xticks([-3, 0, 3])
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                handlelength=1, columnspacing=1)
    offset(ax_b)

    for full, sub in acc.groupby("has_hle"):
        ax_s.errorbar(sub.easy, sub.hard, xerr=sub.easy_ci, yerr=sub.hard_ci, fmt="o", ms=3, lw=0.6,
                      capsize=0, color="#333", mfc="#333" if full else "white", zorder=3)
    for i, row in enumerate(acc.itertuples(), 1):
        ax_s.annotate(i, (row.easy, row.hard), xytext=(2, 1), textcoords="offset points",
                      fontsize=5, color="#444")
    ax_s.text(0.03, 0.97, f"r = {r:.2f}\nρ = {rho:.2f}", transform=ax_s.transAxes, va="top",
              fontsize=7, color="grey")
    ax_s.set_xlabel("Accuracy, easy"); ax_s.set_ylabel("Accuracy, hard")
    ax_s.set_xlim(0.4, 1); ax_s.set_xticks([0.4, 0.7, 1], [".4", ".7", "1"])
    ax_s.set_ylim(0, 1); ax_s.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    offset(ax_s)

    x = np.arange(len(acc))
    offset(ax_c)  # before the tick labels: spine.set_position resets their rotation (mpl 3.11)
    for name, st in SET.items():
        ax_c.plot(x, acc[name], lw=1, marker="o", ms=2.5, label=f"{name} ({(n_easy if name == 'easy' else len(diffs) - n_easy)} items)", **st)
        ax_c.fill_between(x, acc[name] - acc[f"{name}_ci"], acc[name] + acc[f"{name}_ci"],
                          color=st["color"], alpha=0.2, lw=0)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{i} {m}" for i, m in enumerate(acc.index, 1)], rotation=90, fontsize=5.5)
    ax_c.set_xlim(-0.5, len(acc) - 0.5)
    ax_c.set_ylim(0, 1); ax_c.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    ax_c.set_ylabel("Accuracy"); ax_c.set_xlabel("Model, by pooled ability θ")
    ax_c.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, handlelength=1.5, columnspacing=1)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"easy_hard_gold.{ext}", dpi=300, bbox_inches="tight")
