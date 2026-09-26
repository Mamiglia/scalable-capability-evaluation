"""Figure of scripts/analyses/judge_ensemble.py: the 3-judge ensemble against the top-1 judge, per task."""

from pathlib import Path

import numpy as np
import pandas as pd

from plots.item_count import COLOR  # (sets the Agg backend)
from plots.palette import DOMAIN

import matplotlib.pyplot as plt


def offset(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 5))


def figure(both: pd.DataFrame, base: pd.Series, out: Path) -> None:
    """JudgeLinear, squared error (theta MSE over the task's targets; log axis), the task's mean_ref MSE `base` as a grey
    tick. Left: the top-1 judge and the 3-judge mean of forecasts per task (`both`: judge_ensemble.compare). Right: the
    paired difference per task. Writes out.{png,svg,pdf}."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig, (ax, dx) = plt.subplots(1, 2, figsize=(3.6, 1.5), gridspec_kw={"width_ratios": [6, 1], "wspace": 0.45})
    ax.plot(both.x, base, "_", ms=6, color="#9a9a9a", mew=1, ls="", clip_on=False)
    ax.vlines(both.x, both.se_top1, both.se_ens, color="#c8c8c8", lw=0.8)
    ax.plot(both.x, both.se_top1, "o", ms=3, color="#6e6e6e", mew=0, clip_on=False)
    for name, marker in DOMAIN.items():
        e = both[both.domain == name.lower()]
        ax.plot(e.x, e.se_ens, marker, ms=4, color=COLOR[name.lower()], mfc="white", mew=1, ls="", clip_on=False, label="Maths" if name.lower() == "math" else name.capitalize())
        dx.plot(np.linspace(-0.25, 0.25, len(e)), e.se_ens - e.se_top1, marker, ms=3, color=COLOR[name.lower()], mfc="white", mew=0.7, ls="", clip_on=False)
    ax.set_xticks(both.x, both.semester), ax.tick_params(axis="x", labelrotation=90)
    ax.set_yscale("log"), ax.set_ylim(0.05, 4), ax.set_yticks([0.1, 1], ["0.1", "1"]), ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("Squared error (θ²)")
    dx.axhline(0, color="#9a9a9a", lw=0.8, ls="--")
    dx.set_xlim(-0.5, 0.5), dx.set_xticks([]), dx.spines["bottom"].set_visible(False)
    dx.set_ylim(-1.3, 0.4), dx.set_yticks([-1, -0.5, 0]), dx.set_ylabel("Δ vs single judge (θ²)", labelpad=1)
    offset(ax), offset(dx)
    handles = [plt.Line2D([], [], marker="_", ms=6, color="#9a9a9a", mew=1, ls="", label="Baseline"),
               plt.Line2D([], [], marker="o", ms=3, color="#6e6e6e", mew=0, ls="", label="Single judge")] + ax.get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.92), handletextpad=0.1, columnspacing=0.6)
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches="tight")
