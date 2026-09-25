"""Figure of scripts/analyses/item_count_effect.py (and reference_count_effect.py): prediction score or squared error
against the number of judged items (or references), one panel per semester, one line per domain."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.palette import DOMAIN  # domain markers

COLOR = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#4a3aa7"}  # as in reasoning_effect.py
XMIN = 10  # the figure starts here


def plot(df: pd.DataFrame, key: str = "linear", tag: str = "", x: str = "n", xlabel: str = "Judged items",
         xticks: tuple = (XMIN, 50, 100), out: Path = None, mse: bool = False) -> None:
    """Rows = judge rank among the references, one panel per semester, one line per domain (colour + palette marker): the task's median prediction score over draws,
    with the 2.5-97.5 percentile band over draws.

    A line stops at the task's number of judged items; scores below 0 sit on the floor. mse: plot the squared error (theta^2,
    log axis) instead, with the reference-mean baseline of each task as a dashed line in the domain's colour."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    df = df[df[x] >= xticks[0]]  # the axis starts at its first tick (items: fewer than XMIN is mostly draw noise)
    if "row" not in df:  # figure rows: the judge's rank among the references, unless the caller labels them
        df = df.assign(row="Top-" + df.judge_rank.astype(str) + " judge")
    semesters, ranks = sorted(df.semester.unique()), list(dict.fromkeys(df.sort_values("judge_rank", kind="stable").row))
    fig, axes = plt.subplots(len(ranks), len(semesters), figsize=(0.3 + 1.2 * len(semesters), 0.3 + 1.1 * len(ranks)),
                             sharex=True, sharey=True, squeeze=False, gridspec_kw={"wspace": 0.4, "hspace": 0.35})
    draws = (df[key] if mse else 100 - 100 * df[key] / df.baseline).groupby([df.row, df.semester, df.domain, df[x]])
    floor = (1e-9, np.inf) if mse else (0, 100)
    score, lo, hi = draws.median(), draws.quantile(0.025), draws.quantile(0.975)  # band: which items were judged
    for row, rank in zip(axes, ranks):
        for ax, sem in zip(row, semesters):
            for name, marker in DOMAIN.items():
                cell = (rank, sem, name.lower())
                if cell in score.index.droplevel(x):
                    y = score[cell].clip(*floor)
                    ax.fill_between(y.index, lo[cell].clip(*floor), hi[cell].clip(*floor), color=COLOR[name.lower()], alpha=0.15, lw=0)
                    if mse:
                        base = df[(df.row == rank) & (df.semester == sem) & (df.domain == name.lower())].baseline.iloc[0]
                        ax.plot(xticks[::len(xticks) - 1], [base] * 2, color=COLOR[name.lower()], lw=0.7, ls="--", alpha=0.7)
                    ax.plot(y.index, y, color=COLOR[name.lower()], lw=1, marker=marker, ms=2.5, mfc="white", mew=0.7, clip_on=False)
            if rank == ranks[0]:
                ax.text(0, 1.04, sem, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_position(("outward", 4))
        row[0].set_ylabel(rank)
    fig.supylabel("Squared error (θ²)" if mse else "Prediction score", fontsize=8, x=0.125 - 0.7 / fig.get_figwidth())  # left of the row labels, whatever the column count
    ax = axes[0, 0]
    ax.set_xlim(xticks[0], xticks[-1]), ax.set_xticks(xticks)
    if mse:
        ax.set_yscale("log"), ax.set_ylim(0.02, 6), ax.set_yticks([0.1, 1]), ax.set_yticklabels(["0.1", "1"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_ylim(0, 100), ax.set_yticks([0, 50, 100])
    fig.supxlabel(xlabel, fontsize=8, y=0.11 - 0.45 / fig.get_figheight())  # just under the tick labels, whatever the row count
    handles = [plt.Line2D([], [], color=COLOR[name.lower()], lw=1, marker=marker, ms=2.5, mfc="white", mew=0.7, label=name.capitalize())
               for name, marker in DOMAIN.items() if name.lower() in set(df.domain)]
    if mse:
        handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.7, ls="--", label="Baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.9 + 0.1 / len(ranks)),
               handletextpad=0.3, columnspacing=0.8, handlelength=1.4)
    out = out.with_name(out.name + tag)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
