#!/usr/bin/env python3
"""Checks to run on the style predictors before trusting their coefficients.

Three things, all cheap and all reported in the write-up:

correlation - collinear predictors split credit unpredictably, so categories
  correlating above the merge threshold have to be merged before any individual
  coefficient means anything.
sparsity - a category whose difference is almost always exactly zero is
  identified off a small slice of rows; the estimate can still be precise, but
  "per 1 SD" describes a jump the variable almost never makes.
between-model variance - the share of a rate's variance that sits between
  answering models rather than between their transcripts. A predictor that is
  mostly between-model is closer to a model-identity proxy than to a
  transcript-level feature.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "analysis" / "hibayes" / "gpqa_glm.csv"
RATES = ROOT / "analysis" / "hibayes" / "style_rates.csv"
RATE_KEYS = ("model", "item", "n_chars")
MERGE_THRESHOLD = 0.8  # |r| at or above this means two categories must be merged


def between_model_share(values: pd.Series, model: pd.Series) -> float:
    """Fraction of a rate's total variance that lies between answering models."""
    group = values.groupby(model)
    between = (group.count() * (group.mean() - values.mean()) ** 2).sum()
    return between / ((values - values.mean()) ** 2).sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=FRAME)
    parser.add_argument("--rates", type=Path, default=RATES)
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    frame = pd.read_csv(args.frame)
    rates = pd.read_csv(args.rates)
    categories = [column for column in rates.columns if column not in RATE_KEYS]
    style = [f"{category}_difference_sd" for category in categories]
    predictors = ["length_difference_sd", "capability_difference_sd", *style]

    correlation = frame[predictors].corr()
    print("Correlation of standardised predictors as they enter the model\n")
    print(correlation.round(3).to_string(), "\n")
    off_diagonal = correlation.loc[style, style].abs().to_numpy().copy()
    np.fill_diagonal(off_diagonal, 0)
    row, column = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
    print(f"largest |r| between style categories: {off_diagonal.max():.3f} "
          f"({style[row]} vs {style[column]}), merge threshold {MERGE_THRESHOLD}")
    if off_diagonal.max() >= MERGE_THRESHOLD:
        raise SystemExit("Merge those two categories before reading any coefficient")
    print("no merges needed\n")

    # Only the transcripts the judges were actually shown, matching the frame.
    sides = [frame[[f"{side}_model", "item"]].rename(columns={f"{side}_model": "model"})
             for side in ("left", "right")]
    shown = pd.concat(sides).drop_duplicates().merge(rates, on=["model", "item"])
    shown["log_length"] = np.log1p(shown["n_chars"])

    summary = pd.DataFrame(
        {
            "zero_difference_rows": [(frame[f"{c}_difference"] == 0).mean() for c in categories],
            "raw_difference_sd": [frame[f"{c}_difference"].std() for c in categories],
            "between_model_variance": [between_model_share(shown[c], shown["model"]) for c in categories],
        },
        index=categories,
    )
    print(f"Sparsity and between-model variance over {len(shown):,} transcripts, "
          f"{shown['model'].nunique()} models\n")
    print(summary.round(3).to_string())
    print(f"\nfor reference, log transcript length is "
          f"{between_model_share(shown['log_length'], shown['model']):.3f} between-model")


if __name__ == "__main__":
    main()
