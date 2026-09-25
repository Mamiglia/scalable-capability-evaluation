"""True vs predicted theta* of every semester's targets under direct judging, one figure per reasoning condition.

    PYTHONPATH=src .venv/bin/python scripts/plots/plot_direct_forecasts.py [--forecasts analysis/headline-nonagentic/forecasts.csv ...] [--out analysis/direct]

Reads the forecasts.csv tables written by scripts/analyses/headline.py and draws the direct rows:
every roster judge of a semester gives one forecast per target, so each target shows up to three
predicted points. Layout and styling follow scripts/plots/plot_glm_forecasts.py.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_glm_forecasts import DOMAINS, panel  # noqa: E402  (also applies the shared rcParams)

ORANGE = "#e8702a"
DEFAULT_TABLES = ["analysis/headline-nonagentic/forecasts.csv", "analysis/headline-agentic/forecasts.csv"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--forecasts", nargs="+", default=DEFAULT_TABLES)
    parser.add_argument("--out", type=Path, default=Path("analysis/direct"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    direct = pd.concat(pd.read_csv(p) for p in args.forecasts)
    direct = direct[direct.protocol == "direct"]
    for condition, table in direct.groupby("condition"):
        tables = {d: table[table.domain == d] for d in DOMAINS if (table.domain == d).any()}
        widths = [t.semester.nunique() for t in tables.values()]
        fig, axes = plt.subplots(1, len(tables), figsize=(7.2, 2.3), sharey=True,
                                 gridspec_kw={"width_ratios": widths, "wspace": 0.12})
        for ax, (domain, t) in zip(axes, tables.items()):
            panel(ax, t, domain, color=ORANGE)
        axes[0].set_ylabel("θ")
        handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color="black", label="true θ"),
                   plt.Line2D([], [], ls="", marker="o", ms=3, color=ORANGE, label=f"predicted θ (direct, {condition}, one point per roster judge)"),
                   plt.Line2D([], [], ls="", marker="^", ms=3.5, mfc="white", mec=ORANGE, label="predicted, off-scale")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
        for suffix in ("png", "svg"):
            fig.savefig(args.out / f"direct_{condition}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        se = table.assign(se=(table.theta_hat - table.theta) ** 2)
        print(condition, "MSE (mean over roster judges):\n"
              + se.pivot_table(index="semester", columns="domain", values="se", aggfunc="mean").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
