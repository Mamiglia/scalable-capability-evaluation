"""Figures of scripts/analyses/judge_choice.py: per-task judge scores, error against the judge-target gap, squared-error boxes."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from plots.style import INK2, MUTED, legend, save, style  # (sets the Agg backend)

import matplotlib.pyplot as plt

RANK_COLOR = {1: "#1f63b6", 2: "#7f9cc4", 3: "#b8b5ab"}  # judge rank by theta* among the references
BOX_COLOR = {"baseline": "#9a9890", "pairwise": "#2a78d6", "direct": "#1baf7a"}
BOX_COLUMNS = [("rank", 1, "Strongest ref."), ("rank", 2, "2nd strongest"), ("rank", 3, "3rd strongest")]


def figure(t: pd.DataFrame, protocols: list[str], out: Path):
    """Two rows (pairwise, direct): per task the score of each judge, colour = its theta* rank, ring = the judge
    the reference leave-one-out error would pick; a thin line spans the judges' range."""
    tasks = list(dict.fromkeys(t.task))
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True, sharey=True)
    for ax, p in zip(axes, protocols):
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 4))
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for x, task in enumerate(tasks):
            g = t[(t.task == task) & (t.protocol == p)]
            if g.empty:
                continue
            ax.plot([x, x], [g.score.min(), g.score.max()], color=MUTED, lw=0.8, zorder=2)
            pick = g.loo.idxmin()
            for i, r in g.iterrows():
                ax.plot(x, r.score, "o", ms=4, mfc=RANK_COLOR[r["rank"]], mec=INK2 if i == pick else RANK_COLOR[r["rank"]],
                        mew=0.9 if i == pick else 0.4, zorder=3)
        ax.text(0.0, 1.0, p, transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="bottom")
    axes[0].set_ylim(-2, 10)
    axes[0].set_yticks([-2, 0, 5, 10])
    fig.supylabel("Prediction score", fontsize=8, x=0.02)
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels([s.replace("science", "sci").replace("coding", "code") for s in tasks], rotation=90, ha="center")
    axes[1].set_xlim(-0.6, len(tasks) - 0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=c, mec=c, label=f"Judge θ* rank {k}") for k, c in RANK_COLOR.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=INK2, mew=0.9, label="Picked by reference LOO"))
    legend(axes[0], loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, handles=handles, columnspacing=1.0, fontsize=7)
    save(fig, out)


def gap_figure(f: pd.DataFrame, protocols: list[str], out: Path):
    """Per protocol: signed error of every target forecast against the target-minus-judge theta* gap, with the
    mean signed error in gap bins of 0.5 (bins with < 5 forecasts skipped)."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6), sharex=True, sharey=True)
    for ax, p in zip(axes, protocols):
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 4))
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        g = f[f.protocol == p]
        gap, err = g.theta - g.judge_theta, g.theta_hat - g.theta
        ax.plot(gap, err, "o", ms=2, mfc="#b8b5ab", mec="none", zorder=2)
        bins = np.arange(np.floor(gap.min() * 2) / 2, gap.max() + 0.5, 0.5)
        b = pd.cut(gap, bins)
        m = err.groupby(b, observed=True).agg(["mean", "size"])
        m = m[m["size"] >= 5]
        ax.plot([iv.mid for iv in m.index], m["mean"], "-o", color="#1f63b6", lw=1, ms=3, zorder=3)
        ax.text(0.0, 1.0, p, transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="bottom")
        ax.set_xlabel("Target − judge θ*")
    axes[0].set_ylabel("θ̂ − θ*")
    axes[0].set_xlim(-2, 3)
    axes[0].set_xticks([-2, -1, 0, 1, 2, 3])
    axes[0].set_ylim(-4, 2)
    axes[0].set_yticks([-4, -2, 0, 2])
    handles = [plt.Line2D([], [], marker="o", ls="", ms=2, mfc="#b8b5ab", mec="none", label="One target forecast"),
               plt.Line2D([], [], marker="o", ls="-", ms=3, lw=1, color="#1f63b6", label="Mean per 0.5 θ bin")]
    fig.tight_layout()
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7,
                     columnspacing=1.0, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out)


def box_figure(t: pd.DataFrame, base: pd.Series, protocols: list[str], out: Path):
    """Squared error per target model (N=16, 2025-II + 2026-I, averaged over its domains), one column per judge
    choice within each protocol (the judge of theta* rank 1, 2, 3 in each task):
    small points, the mean as a large circle, its 95% t-interval as a box; first column = the reference-mean constant
    forecast `base` (the no-information baseline). Log y-axis. `t`: judge_choice.box_table."""
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    style(ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_position(("outward", 4))
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3)
    ax.set_yticks([0.01, 0.1, 1, 3])
    ax.set_yticklabels(["0.01", "0.1", "1", "3"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    rng = np.random.default_rng(0)
    xs, labels, x = [], [], 0
    columns = [("baseline", "-", "Reference mean", base)] + [(p, "loo" if kind == "pick" else str(key), label, None)
                                                          for p in protocols for kind, key, label in BOX_COLUMNS]
    for p, col, label, e in columns:
        if e is None:
            e = t[(t.protocol == p) & (t.column == col)].set_index("model").e
        if e.empty:
            continue
        if col == "1":
            x += 0.6
        m, h = e.mean(), st.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / np.sqrt(len(e))
        c = BOX_COLOR[p]
        ax.add_patch(plt.Rectangle((x - 0.3, m - h), 0.6, 2 * h, color=c, alpha=0.3, lw=0, zorder=2))
        ax.plot(x + rng.uniform(-0.2, 0.2, len(e)), e, "o", ms=2, mfc=c, mec="none", alpha=0.7, zorder=3)
        ax.plot(x, m, "o", ms=5, mfc=c, mec=c, zorder=4)
        xs.append(x), labels.append(label)
        x += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.set_ylabel("Squared error (θ²)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=c, mec=c, label={"baseline": "Baseline", "pairwise": "Pairwise judge", "direct": "Absolute scoring"}[p])
               for p, c in BOX_COLOR.items()]
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, handles=handles, columnspacing=0.8, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.03)
    save(fig, out)
