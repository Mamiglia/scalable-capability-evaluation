"""Judge MSE against the number of train items: random item subsets of every judge run, refit, MSE.

    uv run python scripts/analyses/items_curve.py [science math coding ...]

One panel per domain, one line per protocol (direct, pairwise), reasoning shown. Each judge run
(domain x semester x judge) gets its MSE over the 8 targets averaged over DRAWS random subsets
per size; the band is the t-based 95% CI across runs. Output: analysis/results/mse_by_items.{csv,svg,png}.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import loading, matching
from plots import results as plots
from analysis.calibration import OfflineBT, OfflineDirect
from capability.evaluate import forecast

SIZES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DRAWS = 10
PROTOCOLS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pairwise", "direct"]
OUT = "analysis/results/mse_by_items"


def curve(dom, task, run, rng) -> list[dict]:
    items = task.references.items[["bench", "item"]].drop_duplicates()
    full = run.table
    rows = []
    for n in SIZES:
        for d in range(DRAWS if n < len(items) else 1):
            keep = items.iloc[rng.choice(len(items), size=min(n, len(items)), replace=False)]
            run.table = full.merge(keep)
            p = OfflineBT(run.table) if run.protocol == "pairwise" else OfflineDirect(run.table)
            f = forecast(task, p)
            rows.append(dict(domain=dom.name, semester=task.semester, protocol=run.protocol, judge=run.judge,
                             n=n, draw=d, mse=float(((f.theta_hat - f.theta) ** 2).mean())))
    run.table = full
    return rows


def collect(domains: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for name in domains:
        dom = loading.domain(name)
        for task in loading.tasks(dom):
            for run in loading.judge_runs(dom, task):
                if run.condition != "reasoning-shown" or run.protocol not in PROTOCOLS:  # skip hidden and the 40-item pairwise runs
                    continue
                if run.protocol == "pairwise":  # matching.match needs data/ transcripts; only the item cut is needed here
                    v = run.table.merge(task.references.items[["bench", "item"]]).sort_values("imported", kind="stable")
                    run.table = v.drop_duplicates(["bench", "item", "model_left", "model_right"], keep="first")
                else:
                    matching.match(run, task)
                if run.table.empty:
                    continue
                rows += curve(dom, task, run, rng)
                print(name, task.semester, run.protocol, run.judge, "verdicts", len(run.table), flush=True)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str) -> None:
    """Thin line per judge run (mean over draws), thick line for the mean over runs; the run is the unit."""
    per_run = df.groupby(["domain", "protocol", "semester", "judge", "n"]).mse.mean().reset_index()
    domains = list(dict.fromkeys(df.domain))
    fig, axes = plt.subplots(1, len(domains), figsize=(2.0 * len(domains), 1.8), sharey=True, squeeze=False)
    for ax, dom in zip(axes[0], domains):
        for proto in ("direct", "pairwise"):
            g = per_run[(per_run.domain == dom) & (per_run.protocol == proto)]
            if g.empty:
                continue
            for _, r in g.groupby(["semester", "judge"]):
                ax.plot(r.n, r.mse, lw=0.5, alpha=0.4, color=plots.COLORS[proto])
            m = g.groupby("n").mse.mean()
            ax.plot(m.index, m, lw=1, color=plots.COLORS[proto], label=f"{plots.LABELS[proto]}, mean of {g.groupby(['semester', 'judge']).ngroups} judge runs")
        ax.set_title(dom, loc="left", fontsize=7, color="grey")
        ax.set_xlabel("train items")
        ax.set_xticks([0, 50, 100])
        ax.set_xlim(0, 105)
        ax.spines["bottom"].set_bounds(0, 100)
    axes[0, 0].set_ylabel("MSE (θ units)")
    top = np.ceil(per_run.mse.max())
    axes[0, 0].set_yticks(np.arange(0, top + 1, 2 if top > 4 else 1))
    axes[0, 0].set_ylim(0, top + 0.2)
    for ax in axes[0]:
        plots.tidy(ax)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.savefig(path + ".svg", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")


if __name__ == "__main__":
    import os
    df = pd.read_csv(OUT + ".csv") if os.environ.get("REPLOT") else collect(sys.argv[1].split(",") if len(sys.argv) > 1 else ["science", "math", "coding"])
    df.to_csv(OUT + ".csv", index=False, float_format="%.4f")
    plot(df, OUT)
