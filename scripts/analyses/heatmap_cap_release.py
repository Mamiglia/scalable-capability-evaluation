"""Judge accuracy over the capability-gap / release-gap plane (gpqa-pilot-rd2).

Three panels, colour = judge accuracy (0.5 = chance):
  1. raw capability gap (x) vs release-time gap (y)
  2. capability gap (x) vs release gap with capability regressed out (y)
  3. capability gap with release regressed out (x) vs release gap (y)

Panels 2/3 orthogonalise the two correlated axes so the colour gradient shows
each axis's independent link to judge accuracy (partial correlation annotated).

One point per (major, minor) pair; major is the more capable side (d_acc >= 0).
Untruncated cross-model runs only; colour is the pair's mean over its judges,
size is how many judges scored it.

Usage: uv run python scripts/analyses/heatmap_cap_release.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

from comparisons import MODELS
from plots import HEAT, INK, INK2, MUTED, plt, save, style

RESULTS = Path("judge_results/gpqa-pilot-rd2")
OUT = RESULTS / "figures"


def release_frac(model):
    d = pd.to_datetime(MODELS[model]["release"])
    return d.year + (d.dayofyear - 1) / 365.25


def resid(y, x):
    """Residual of y after a straight-line fit on x."""
    b, a = np.polyfit(x, y, 1)
    return y - (a + b * x)


def partial_r(a, b, ctrl):
    """Correlation of a and b with ctrl regressed out of both."""
    return pearsonr(resid(a, ctrl), resid(b, ctrl))[0]


def load_pairs():
    s = pd.read_csv(RESULTS / "summary.csv")
    c = s[(s.major != s.minor) & (s.truncate_frac.isna())].copy()
    c["rel_gap"] = [release_frac(m) - release_frac(n) for m, n in zip(c.major, c.minor)]
    pairs = (c.groupby(["major", "minor"])
             .agg(cap_gap=("d_acc", "first"), rel_gap=("rel_gap", "first"),
                  score=("score", "mean"), n=("score", "size"))
             .reset_index())
    return c, pairs


def panel(ax, x, y, pairs, xlabel, ylabel, title, norm, note=None):
    sizes = 60 + 45 * pairs.n
    sc = ax.scatter(x, y, c=pairs.score, cmap=HEAT, norm=norm, s=sizes,
                    edgecolors="white", linewidths=0.8, zorder=3)
    for xi, yi, (_, r) in zip(x, y, pairs.iterrows()):
        ax.annotate(f"{MODELS[r.major]['label']}\nv {MODELS[r.minor]['label']}",
                    (xi, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.0, color=MUTED, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK2, fontsize=8.5, loc="left")
    if note:
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
                color=MUTED, fontsize=6.5)
    style(ax, grid_axis="both")
    return sc


def main():
    rows, pairs = load_pairs()
    OUT.mkdir(exist_ok=True)

    # Orthogonalised axes (fit on the unique pairs).
    rel_resid = resid(pairs.rel_gap.to_numpy(), pairs.cap_gap.to_numpy())
    cap_resid = resid(pairs.cap_gap.to_numpy(), pairs.rel_gap.to_numpy())

    # Partial correlations of judge accuracy, over every judge x pair row.
    pr_cap = partial_r(rows.score.to_numpy(), rows.d_acc.to_numpy(), rows.rel_gap.to_numpy())
    pr_rel = partial_r(rows.score.to_numpy(), rows.rel_gap.to_numpy(), rows.d_acc.to_numpy())
    r_cr = pearsonr(pairs.cap_gap, pairs.rel_gap)[0]

    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.15, vmax=0.85)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")

    panel(axs[0], pairs.cap_gap, pairs.rel_gap, pairs,
          "capability gap (GPQA accuracy diff)", "release gap (years)",
          "1 · raw axes", norm,
          note=f"axes correlated: r = {r_cr:+.2f}")
    axs[0].axhline(0, color=MUTED, lw=0.7, ls=":", zorder=1)

    panel(axs[1], pairs.cap_gap, rel_resid, pairs,
          "capability gap (GPQA accuracy diff)", "release gap | capability (resid, years)",
          "2 · release gap, capability regressed out", norm,
          note=f"partial r(acc, release | capability) = {pr_rel:+.2f}")
    axs[1].axhline(0, color=MUTED, lw=0.7, ls=":", zorder=1)

    panel(axs[2], cap_resid, pairs.rel_gap, pairs,
          "capability gap | release (resid, GPQA acc)", "release gap (years)",
          "3 · capability gap, release regressed out", norm,
          note=f"partial r(acc, capability | release) = {pr_cap:+.2f}")
    axs[2].axvline(0, color=MUTED, lw=0.7, ls=":", zorder=1)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=HEAT),
                      ax=axs, shrink=0.85, pad=0.015)
    cb.set_label("judge accuracy (0.5 = chance)", color=INK2)
    cb.ax.axhline(0.5, color=INK, lw=0.8)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)

    # Size legend (judges per pair).
    for k in (2, 5, 8):
        axs[0].scatter([], [], s=60 + 45 * k, c="none", edgecolors=MUTED,
                       linewidths=0.9, label=f"{k} judges")
    axs[0].legend(frameon=False, loc="upper left", fontsize=6.5, labelcolor=INK2,
                  handletextpad=0.4, borderpad=0.2)

    fig.suptitle("Judge accuracy over the capability-gap / release-gap plane  ·  "
                 "gpqa-pilot-rd2  ·  one point per author pair, mean over judges",
                 color=INK2, fontsize=10, y=1.02)
    save(fig, OUT / "heatmap_cap_vs_release")
    print(f"corr(cap, release) = {r_cr:+.3f}")
    print(f"partial r(acc, release | cap) = {pr_rel:+.3f}")
    print(f"partial r(acc, cap | release) = {pr_cap:+.3f}")
    print(f"wrote {OUT/'heatmap_cap_vs_release'}.png/.svg")


if __name__ == "__main__":
    main()
