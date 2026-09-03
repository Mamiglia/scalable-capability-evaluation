#!/usr/bin/env python3
"""Subject-model accuracy across a domain's difficulty quartiles (Q1 easiest -> Q4 hardest).

One thin line per model over the quartiles, plus the mean +- 95% t-CI across models (N =
models). Validates the IRT difficulty ordering: if the quartiles are real, every line slopes
down. Bars on the mean use models as the replication unit; per-model points are raw rates.

Run:  uv run python scripts/plots/plot_domain_accuracy.py [--domain coding]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs.domain import DomainConfig
from plot_difficulty import offset

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})


def main(domain_name: str):
    domain = DomainConfig.from_cli(f"configs/domain/{domain_name}.yml")
    df = domain.load_responses().merge(domain.load_items()[["bench", "item", "quartile"]])
    acc = df.groupby(["model", "quartile"]).correct.mean().unstack()
    acc = acc.loc[acc.mean(axis=1).sort_values(ascending=False).index]
    qs = list(acc.columns)

    fig, ax = plt.subplots(figsize=(2.0, 1.9))
    for model, row in acc.iterrows():
        ax.plot(qs, row.values, color="#bbb", lw=0.6, zorder=1)
    m = acc.mean(axis=0)
    ci = stats.t.ppf(0.975, len(acc) - 1) * acc.std(axis=0, ddof=1) / np.sqrt(len(acc))
    ax.errorbar(qs, m.values, yerr=ci.values, fmt="o-", ms=3, lw=1, capsize=0,
                color="#2a78d6", zorder=3, label=f"mean of {len(acc)} models")
    for label, y in ((acc.index[0], acc.iloc[0]), (acc.index[-1], acc.iloc[-1])):
        ax.annotate(label, (qs[-1], y.values[-1]), xytext=(3, 0), textcoords="offset points",
                    fontsize=6, va="center", color="#555")
    ax.set_xticks(qs, [f"Q{q}" for q in qs])
    ax.set_xlim(0.8, max(qs) + 0.2)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Item difficulty (IRT)"); ax.set_ylabel("Accuracy")
    ax.tick_params(length=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.2)
    offset(ax)
    out = Path("analysis/judge"); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{domain_name}_subject_accuracy.{ext}", dpi=300, bbox_inches="tight")
    print(acc.round(3).to_string())
    print(f"\nmean {m.round(3).to_dict()}\nwrote {out}/{domain_name}_subject_accuracy.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="coding")
    main(ap.parse_args().domain)
