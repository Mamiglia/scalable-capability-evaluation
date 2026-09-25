"""Transcript truncation figure; the forecasts and their MSE come from analysis.challenges.truncation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.challenges.truncation import FULL, PROTOCOLS
from analysis.stats import BASELINE, DOMAIN_LABELS, LABELS
from plots.results import DOMAIN_COLORS, fix_axes_height


def figure(f: pd.DataFrame, baseline: pd.DataFrame, path: Path, cut: str = "ends") -> None:
    """MSE over target models, per protocol and domain, full transcripts against the `cut`:
    bars in the domain's colour at the MSE over the domain's targets (both semesters pooled,
    the unit is the target model) with a capless percentile-bootstrap 95% interval
    (stats.mse_row); the prediction from saturated items as a dashed grey segment over each
    domain's pair at its MSE over the same targets. `baseline`: the results forecasts rows of
    that prediction (theta_hat, theta per target)."""
    from analysis.stats import mse_row
    protocols = [p for p in PROTOCOLS if p in set(f.protocol)]
    domains = [d for d in DOMAIN_COLORS if d in set(f.domain)]
    fig, ax = plt.subplots(figsize=(2.6, 2.4), layout="constrained")
    x, ticks, centres = 0.0, [], []
    for p in protocols:
        for d in domains:
            for c in (FULL, cut):
                row = mse_row(f[(f.protocol == p) & (f.domain == d) & (f.cut == c)])
                ax.bar(x, row["mse"], color=DOMAIN_COLORS[d], width=0.72, zorder=1)
                ax.plot([x, x], [row["lo"], row["hi"]], color="black", lw=0.8, zorder=4)
                ticks.append((x, "Full" if c == FULL else "Truncated"))
                x += 1
            level = mse_row(baseline[baseline.domain == d])["mse"]
            ax.hlines(level, x - 2.45, x - 0.55, color="grey", lw=0.8, ls="--", zorder=0)
            x += 0.3
        centres.append(x - 0.3 - 2 * len(domains) / 2 - 0.5)
        x += 0.5
    top = ax.get_ylim()[1]
    for p, c in zip(protocols, centres):
        ax.text(c, top, LABELS[p], color="grey", fontsize=7, ha="center", va="bottom")
    ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Prediction Error\n(MSE θ)")
    ax.set_ylim(bottom=0)
    ax.spines["bottom"].set_bounds(ticks[0][0], ticks[-1][0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS[d], label=DOMAIN_LABELS[d]) for d in domains]
    handles.append(plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE]))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
