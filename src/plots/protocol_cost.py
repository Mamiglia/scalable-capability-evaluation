"""Figure of scripts/analyses/protocol_cost.py: cost-error operating curves per domain."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = {"pairwise": ("Pairwise judging", "#2a78d6"), "direct": ("Absolute scoring", "#eb6834"),
          "glm_regex": ("Deterministic features", "#1baf7a"), "embedding": ("Embedding", "#4a3aa7")}
FREE = 1e-3  # where $0 sits on the log axis
YMAX = 3


def plot(s: pd.DataFrame, domains: list[str], out: Path) -> None:
    """Per domain: squared error against cost per semester, the LOO judge's pairwise and direct curves (band = 95%
    bootstrap) and the transcript protocols as points; dashed = the reference-mean forecast. `s`: protocol_cost.summarise."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25, "hspace": 0.45})
    for ax, name in zip(axes.flat, domains):
        d = s[s.domain == name]
        ax.axhline(d[d.protocol == "mean_ref"].mse.iloc[0], color="#9a9a9a", lw=0.8, ls="--", zorder=0)
        for key, (_, color) in SERIES.items():
            g = d[d.protocol == key].sort_values("fraction")
            x = g.cost.clip(lower=FREE)
            if key in ("pairwise", "direct"):
                ax.fill_between(x, g.lo, g.hi, color=color, alpha=0.18, lw=0)
                ax.plot(x, g.mse, color=color, lw=1, marker="o", ms=2.5, mfc="white", mew=0.7)
            else:
                if (g.mse > YMAX).any():  # off the chart (math/science length+prose blow up): a triangle at the edge + its value
                    ax.plot(x, [YMAX] * len(g), "^", color=color, ms=3.5, clip_on=False)
                    for xi, m in zip(x, g.mse):
                        ax.text(xi * 1.6, YMAX, f"{m:.0f}", fontsize=5, color=color, va="center")
                else:
                    ax.errorbar(x, g.mse, yerr=[g.mse - g.lo, g.hi - g.mse], fmt="o", color=color, ms=3.5, lw=1, capsize=0)
        ax.text(0, 1.04, name, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
    ax = axes[0, 0]
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(FREE / 1.5, 1000)
    ax.set_xticks([FREE, 0.01, 0.1, 1, 10, 100, 1000], ["0", ".01", ".1", "1", "10", "100", "1k"])
    ax.set_ylim(0.03, YMAX)
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    for a in axes[1]:
        a.tick_params(axis="x", labelrotation=90)
        a.set_xlabel("Cost per semester ($)")
    for a in axes[:, 0]:
        a.set_ylabel("Squared error (θ²)")
    handles = [plt.Line2D([], [], color=c, lw=1, marker="o", ms=2.5, mfc="white", mew=0.7, label=l) for k, (l, c) in SERIES.items() if k in ("pairwise", "direct")]
    handles += [plt.Line2D([], [], color=c, ls="", marker="o", ms=3.5, label=l) for k, (l, c) in SERIES.items() if k not in ("pairwise", "direct")]
    handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97),
               handletextpad=0.3, columnspacing=0.8, handlelength=1.4)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
