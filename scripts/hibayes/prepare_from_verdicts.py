#!/usr/bin/env python3
"""Build the judge GLM frame from exported pairwise verdicts.

Replaces prepare_gpqa.py, whose eval-log roots are gone. Row semantics come from
the verdict exporter: `order` is the position (1 or 2) holding the major model,
`picked` is the position the judge chose, so position 1 is the first-named
transcript ("left") in both. Lengths are the characters the judge saw.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prepare_gpqa import model_family, standardise

LEFT_POSITION = 1


def load_capabilities(config_path: Path, field: str) -> dict[str, float]:
    """model -> capability score from the judge-elo bench config."""
    config = yaml.safe_load(config_path.read_text())
    if field not in config:
        raise ValueError(f"{config_path} has no '{field}' block")
    return {model: float(value) for model, value in config[field].items()}


def build(
    verdicts: pd.DataFrame,
    capabilities: dict[str, float],
    bench: str,
    conditions: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Verdict rows -> one model row per parsed verdict, with the GLM columns."""
    data = verdicts[(verdicts["bench"] == bench) & (verdicts["condition"].isin(conditions))].copy()
    counts = {"verdicts_selected": len(data)}
    if data.empty:
        raise ValueError(f"No verdicts for bench {bench} and conditions {list(conditions)}")

    picked = pd.to_numeric(data["picked"], errors="coerce")
    parsed = picked.isin([1, 2])
    measured = data["len_major"].notna() & data["len_minor"].notna()
    counts["dropped_unparsed_verdict"] = int((~parsed).sum())
    counts["dropped_missing_length"] = int((parsed & ~measured).sum())
    data, picked = data[parsed & measured].copy(), picked[parsed & measured]

    missing = sorted((set(data["major"]) | set(data["minor"])) - set(capabilities))
    if missing:
        raise ValueError(f"Missing capability estimate for {missing}")

    major_is_left = data["order"] == LEFT_POSITION
    left_model = data["major"].where(major_is_left, data["minor"])
    right_model = data["minor"].where(major_is_left, data["major"])
    length_left = data["len_major"].where(major_is_left, data["len_minor"])
    length_right = data["len_minor"].where(major_is_left, data["len_major"])
    left_capability = left_model.map(capabilities)
    right_capability = right_model.map(capabilities)

    frame = pd.DataFrame(
        {
            "chosen_left": (picked == LEFT_POSITION).astype(int),
            "judge": data["judge"],
            "item": data["item_id"],
            "transcript_condition": data["condition"],
            "bench": data["bench"],
            "model_pair": [
                "|".join(sorted(pair)) for pair in zip(data["major"], data["minor"], strict=True)
            ],
            "left_model": left_model,
            "right_model": right_model,
            "order": data["order"],
            "picked": picked.astype(int),
            "left_chars": length_left.astype(int),
            "right_chars": length_right.astype(int),
            "left_capability": left_capability,
            "right_capability": right_capability,
            "length_difference": np.log1p(length_left) - np.log1p(length_right),
            "capability_difference": left_capability - right_capability,
            "same_model_family": (
                (left_model != right_model)
                & (left_model.map(model_family) == right_model.map(model_family))
            ).astype(int),
            "self_comparison": (left_model == right_model).astype(int),
        }
    )
    standardise(frame, ("length_difference", "capability_difference"))
    counts["model_rows"] = len(frame)
    frame = frame.sort_values(["judge", "model_pair", "item", "transcript_condition", "order"])
    return frame.reset_index(drop=True), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=Path("judge_results/judging_variations/verdicts.csv"))
    parser.add_argument("--capabilities", type=Path, default=Path("scripts/analyses/judge-elo/configs/gpqa.yaml"))
    parser.add_argument("--capability-field", default="eci", help="Block in the capabilities config to read")
    parser.add_argument("--bench", default="gpqa")
    parser.add_argument("--conditions", nargs="+", default=["a_baseline"])
    parser.add_argument("--out", type=Path, default=Path("analysis/hibayes/gpqa_glm.csv"))
    args = parser.parse_args()

    frame, counts = build(
        pd.read_csv(args.verdicts),
        load_capabilities(args.capabilities, args.capability_field),
        args.bench,
        args.conditions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}")
    for name, value in counts.items():
        print(f"  {name}: {value:,}")
    print(
        f"  judges: {frame['judge'].nunique()}, items: {frame['item'].nunique()}, "
        f"pairs: {frame['model_pair'].nunique()}, conditions: {frame['transcript_condition'].nunique()}"
    )
    print(f"  chosen_left rate: {frame['chosen_left'].mean():.4f}")


if __name__ == "__main__":
    main()
