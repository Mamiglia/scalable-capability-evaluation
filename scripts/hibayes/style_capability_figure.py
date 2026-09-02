#!/usr/bin/env python3
"""Per-model style rates against capability, one panel per lexicon category.

Uses only the transcripts that reached the GLM frame, so the descriptive
picture and the model describe the same text. Each point is one answering
model; the trend is a Theil-Sen slope, which is rank-based like the Kendall
tau it accompanies, banded by the slope's 95% interval.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, theilslopes

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "analysis" / "hibayes" / "gpqa_glm.csv"
RATES = ROOT / "analysis" / "hibayes" / "style_rates.csv"
CAPABILITY_CONFIG = ROOT / "scripts" / "analyses" / "judge-elo" / "configs" / "gpqa.yaml"
OUT = ROOT / "analysis" / "hibayes" / "style_vs_capability"

RATE_KEYS = ("model", "item", "n_chars")
POINT_COLOUR = "#2a78d6"
TREND_COLOUR = "#333333"
COLUMNS = 3
PANEL_SIZE = (1.8, 1.7)  # inches per panel
# Headroom below zero, as a fraction of the panel's range, so a flat trend at a
# floor of zeros stays visible instead of merging into the bottom spine.
ZERO_CLEARANCE = 0.05


def per_model_rates(frame: pd.DataFrame, rates: pd.DataFrame, capabilities: dict) -> pd.DataFrame:
    """Mean rate per answering model over the transcripts the judges were shown."""
    sides = [
        frame[[f"{side}_model", "item"]].rename(columns={f"{side}_model": "model"})
        for side in ("left", "right")
    ]
    shown = pd.concat(sides).drop_duplicates().merge(rates, on=["model", "item"], how="left")
    if shown.isna().any().any():
        raise ValueError("Some transcripts in the frame have no style rates")
    categories = [column for column in rates.columns if column not in RATE_KEYS]
    per_model = shown.groupby("model")[categories].mean()
    per_model["capability"] = pd.Series(capabilities)
    missing = per_model["capability"].isna()
    if missing.any():
        raise ValueError(f"No capability estimate for {sorted(per_model.index[missing])}")
    return per_model


def plot(per_model: pd.DataFrame, categories: list[str], out: Path) -> pd.DataFrame:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = -(-len(categories) // COLUMNS)
    figure, axes = plt.subplots(
        rows,
        COLUMNS,
        figsize=(COLUMNS * PANEL_SIZE[0], rows * PANEL_SIZE[1]),
        sharex=True,
        squeeze=False,
    )
    capability = per_model["capability"].to_numpy()
    grid = np.linspace(capability.min(), capability.max(), 2)

    summary = []
    for index, category in enumerate(categories):
        axis = axes[index // COLUMNS][index % COLUMNS]
        rate = per_model[category].to_numpy()
        tau, p_value = kendalltau(capability, rate)
        slope, intercept, low_slope, high_slope = theilslopes(rate, capability)
        summary.append({"category": category, "kendall_tau": tau, "p_value": p_value,
                        "theil_sen_slope": slope, "slope_2.5%": low_slope, "slope_97.5%": high_slope})

        axis.scatter(capability, rate, s=6, color=POINT_COLOUR, alpha=0.7, linewidths=0,
                     label="Answering model" if index == 0 else None)
        centre = np.median(capability), np.median(rate)
        axis.fill_between(
            grid,
            centre[1] + low_slope * (grid - centre[0]),
            centre[1] + high_slope * (grid - centre[0]),
            color=TREND_COLOUR, alpha=0.15, linewidth=0,
            label="Theil-Sen slope, 95% interval" if index == 0 else None,
        )
        axis.plot(grid, centre[1] + slope * (grid - centre[0]), color=TREND_COLOUR, lw=1)
        axis.set_title(category.replace("_", " ").capitalize(), loc="left", fontsize=7, color="#555555")
        axis.annotate(f"tau {tau:+.2f}", xy=(0.97, 0.92), xycoords="axes fraction",
                      ha="right", va="top", fontsize=7, color="#555555")
        axis.set_ylim(bottom=-ZERO_CLEARANCE * rate.max())
        axis.locator_params(axis="y", nbins=4)
        if index % COLUMNS == 0:
            axis.set_ylabel("Rate per 1,000 words")
        if index // COLUMNS == rows - 1:
            axis.set_xlabel("Capability (ECI)")

    for empty in range(len(categories), rows * COLUMNS):
        axes[empty // COLUMNS][empty % COLUMNS].set_visible(False)

    figure.legend(loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300, bbox_inches="tight")
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=FRAME)
    parser.add_argument("--rates", type=Path, default=RATES)
    parser.add_argument("--capabilities", type=Path, default=CAPABILITY_CONFIG)
    parser.add_argument("--capability-field", default="eci")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    config = yaml.safe_load(args.capabilities.read_text())
    rates = pd.read_csv(args.rates)
    per_model = per_model_rates(pd.read_csv(args.frame), rates, config[args.capability_field])
    categories = [column for column in rates.columns if column not in RATE_KEYS]

    summary = plot(per_model, categories, args.out)
    per_model.to_csv(args.out.with_name(args.out.name + "_per_model.csv"))
    summary.to_csv(args.out.with_name(args.out.name + "_kendall.csv"), index=False)
    print(f"{len(per_model)} models, {len(categories)} categories")
    print(summary.round(4).to_string(index=False))
    print(f"Wrote {args.out}.png/.pdf and the two csvs beside them")


if __name__ == "__main__":
    main()
