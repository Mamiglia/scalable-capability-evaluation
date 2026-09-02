#!/usr/bin/env python3
"""Build the judge GLM frame with correctness as the outcome.

Same verdicts as prepare_from_verdicts.py, differently parameterised. The
outcome is whether the judge picked the stronger model, so the side holding the
stronger model becomes a predictor (truth_side) whose coefficient is the
position bias. Crossing it with total transcript length asks whether position
bias grows with the amount of text.

Every left-minus-right difference is re-oriented to stronger-minus-weaker.
Left-minus-right names a presented side, so it flips meaning when the order
flips; against a correctness outcome those columns average to nothing. The
re-oriented ones do not: positive means the stronger model had more of it.
Standardising happens after re-orienting.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_from_verdicts import (
    CAPABILITY_CONFIG,
    CONDITION_CONFIGS,
    ROOT,
    STYLE_KEYS,
    STYLE_RATES,
    VERDICTS,
    build,
    load_capabilities,
)
from prepare_gpqa import standardise

ORIENTED_SUFFIX = "_stronger"
CELL = ["judge", "model_pair", "item", "transcript_condition"]


def difference_columns(style_rates: pd.DataFrame) -> list[str]:
    """Left-minus-right columns that need re-orienting, from the same source build() used."""
    categories = [column for column in style_rates.columns if column not in STYLE_KEYS]
    return ["length_difference"] + [f"{category}_difference" for category in categories]


def to_correctness(
    frame: pd.DataFrame, differences: list[str], allow_reordering: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Left-choice frame -> correctness frame; returns it and the predictor names.

    allow_reordering lets the capability measure disagree with the exporter's
    major/minor labelling: the outcome is then "picked the higher-capability
    model" outright, not "picked the reference model". Needed for a corrected
    capability that reranks pairs; the default ECI ordering never disagrees.
    """
    truth_side = np.sign(frame["capability_difference"])
    if not truth_side.isin([-1.0, 1.0]).all():
        raise ValueError(
            "Some rows have a zero capability difference, so neither model is the stronger one "
            "and truth_side, the outcome and every re-oriented column are undefined"
        )
    frame = frame.copy()
    frame["truth_side"] = truth_side
    frame["is_correct"] = (frame["chosen_left"] == (truth_side > 0)).astype(int)

    # order is the position holding the major model and picked is the position chosen, so
    # picked == order is correctness derived without capability. It agrees only if major is
    # also the higher-capability model; if the two ever part company, every sign below flips.
    independent = (frame["picked"] == frame["order"]).astype(int)
    disagreeing = int((frame["is_correct"] != independent).sum())
    if disagreeing and not allow_reordering:
        raise ValueError(
            f"{disagreeing} rows where the major model is not the higher-capability one; "
            "truth_side and the outcome disagree about which model is stronger"
        )
    if disagreeing:
        print(
            f"  reordered by capability: {disagreeing:,} of {len(frame):,} rows where the "
            "corrected-stronger model is not the exporter's reference model"
        )

    frame["capability_gap"] = frame["capability_difference"].abs()
    frame["total_length"] = np.log1p(frame["left_chars"] + frame["right_chars"])
    per_cell = frame.groupby(CELL, observed=True)["total_length"].nunique()
    if (per_cell > 1).any():
        raise ValueError(
            "total_length differs between the two presentation orders of a cell; it is a sum "
            "over both transcripts and must not depend on which one came first"
        )

    oriented = []
    for column in differences:
        name = f"{column}{ORIENTED_SUFFIX}"
        frame[name] = frame[column] * truth_side
        oriented.append(name)
    # The left-minus-right standardised columns build() made are incoherent against this
    # outcome. Dropping them stops a config wiring one in by name.
    frame = frame.drop(columns=[f"{column}_sd" for column in differences + ["capability_difference"]])

    scaled = ["capability_gap", "total_length", *oriented]
    standardise(frame, scaled)
    predictors = [f"{column}_sd" for column in scaled]
    if not np.isfinite(frame[predictors].to_numpy()).all():
        raise ValueError(f"Standardised columns are not finite: {predictors}")

    frame["truth_side_x_total_length"] = frame["truth_side"] * frame["total_length_sd"]
    predictors = ["truth_side", "truth_side_x_total_length", *predictors, "same_model_family"]
    return frame, predictors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    parser.add_argument("--capabilities", type=Path, default=CAPABILITY_CONFIG)
    parser.add_argument("--capability-field", default="eci", help="Block in the capabilities config to read")
    parser.add_argument("--condition-configs", type=Path, default=CONDITION_CONFIGS)
    parser.add_argument("--bench", default="gpqa")
    parser.add_argument("--conditions", nargs="+", default=["a_baseline"])
    parser.add_argument("--style-rates", type=Path, default=STYLE_RATES)
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "hibayes" / "gpqa_correctness.csv")
    parser.add_argument("--allow-reordering", action="store_true",
                        help="Let the capability measure rerank pairs the exporter labelled the other way")
    args = parser.parse_args()
    print(f"Reading {args.verdicts}")

    style_rates = pd.read_csv(args.style_rates)
    frame, counts = build(
        pd.read_csv(args.verdicts),
        load_capabilities(args.capabilities, args.capability_field),
        args.bench,
        args.conditions,
        args.condition_configs,
        style_rates,
    )
    frame, predictors = to_correctness(frame, difference_columns(style_rates), args.allow_reordering)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}")
    for name, value in counts.items():
        print(f"  {name}: {value:,}")
    print(f"  predictors: {predictors}")
    print(f"  is_correct rate: {frame['is_correct'].mean():.4f}")
    print(f"  truth_side = +1 rows: {int((frame['truth_side'] > 0).sum()):,}")
    print(f"  accuracy by truth_side: {frame.groupby('truth_side')['is_correct'].mean().round(4).to_dict()}")


if __name__ == "__main__":
    main()
