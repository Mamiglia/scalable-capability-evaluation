"""True vs predicted theta* of every semester's targets, one figure per transcript-GLM tier.

    PYTHONPATH=src .venv/bin/python scripts/plots/plot_glm_forecasts.py [glm_length glm_regex glm_structure] [--out analysis/glm]

Reads results/<domain>/eval_<protocol>.csv (written by `uv run evaluate`) for science, math,
coding and cyber, and writes <out>/<protocol>.{png,svg}. Targets are ordered by true theta*
within a semester; a dotted line joins each truth to its forecast. Forecasts beyond the y-range
are drawn as hollow triangles at the edge.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOMAINS = ["science", "math", "coding", "cyber"]
YLIM = (-4, 3)
GREEN = "#1baf7a"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})


def panel(ax, forecasts: pd.DataFrame, tag: str, color: str = GREEN) -> None:
    """One x per target (ordered by true theta within a semester); several forecast rows per target are all drawn."""
    semesters = list(dict.fromkeys(forecasts.semester))
    centres = []
    x = 0
    for semester in semesters:
        group = forecasts[forecasts.semester == semester]
        truth = group.groupby("model").theta.first().sort_values()
        xs = pd.Series(x + np.arange(len(truth)), index=truth.index)
        centres.append(xs.mean())
        gx = xs[group.model].to_numpy()
        for xi, t, p in zip(gx, group.theta, group.theta_hat.clip(*YLIM)):
            ax.plot([xi, xi], [t, p], ls=":", lw=0.6, color="0.6", zorder=1)
        ax.plot(xs, truth, "o", ms=3, color="black", zorder=3)
        inside = group.theta_hat.between(*YLIM).to_numpy()
        ax.plot(gx[inside], group.theta_hat[inside], "o", ms=3, color=color, zorder=2)
        for xi, p in zip(gx[~inside], group.theta_hat[~inside]):
            ax.plot(xi, YLIM[0] if p < YLIM[0] else YLIM[1], "v" if p < YLIM[0] else "^", ms=3.5,
                    mfc="white", mec=color, zorder=2)
        x += len(truth) + 2
    ax.set_xticks(centres, semesters)
    ax.set_xlim(-1.5, x - 1.5)
    ax.set_ylim(YLIM[0] - 0.3, YLIM[1] + 0.3)
    ax.set_yticks(range(YLIM[0], YLIM[1] + 1))
    ax.spines["left"].set_bounds(*YLIM)
    ax.spines["bottom"].set_bounds(-0.5, x - 2.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    ax.text(0.02, 1.02, tag, transform=ax.transAxes, color="0.4", va="bottom")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("protocols", nargs="*", default=["glm_length", "glm_regex", "glm_structure"])
    parser.add_argument("--out", type=Path, default=Path("analysis/glm"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for protocol in args.protocols:
        tables = {d: pd.read_csv(f"results/{d}/eval_{protocol}.csv") for d in DOMAINS
                  if Path(f"results/{d}/eval_{protocol}.csv").exists()}
        widths = [t.semester.nunique() for t in tables.values()]
        fig, axes = plt.subplots(1, len(tables), figsize=(7.2, 2.3), sharey=True,
                                 gridspec_kw={"width_ratios": widths, "wspace": 0.12})
        for ax, (domain, table) in zip(axes, tables.items()):
            panel(ax, table, domain)
        axes[0].set_ylabel("θ")
        handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color="black", label="true θ"),
                   plt.Line2D([], [], ls="", marker="o", ms=3, color=GREEN, label=f"predicted θ ({protocol})"),
                   plt.Line2D([], [], ls="", marker="^", ms=3.5, mfc="white", mec=GREEN, label="predicted, off-scale")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
        for suffix in ("png", "svg"):
            fig.savefig(args.out / f"{protocol}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        mse = pd.concat(t.assign(domain=d) for d, t in tables.items())
        mse["se"] = (mse.theta_hat - mse.theta) ** 2
        print(protocol, "MSE:\n" + mse.pivot_table(index="semester", columns="domain", values="se", aggfunc="mean")
              .round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
