"""Write the hibayes input tables for the three GLMs from the design matrices.

    uv run python scripts/verdict_dissection/glm/prepare.py <output_dir> [--grouped]

`--grouped` reads the grouped design matrices and predictors-grouped.yaml instead. Each output has the predictors from predictors.yaml, the target, the grouping columns and a
boolean `test` column (confirmation items). Pair inputs rename the left-minus-right `d_`
columns to the bare feature names so all three fits share one predictor list.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TEST_SPLIT = "confirmation"
PAIR_IDENTITY = ["bench", "item", "model_left", "model_right", "epoch_left", "epoch_right",
                 "log_left", "log_right"]
ACCURACY_CLIP = 0.01          # logit needs accuracies strictly inside (0, 1)


def capability_from_transcript(transcript, predictors):
    """One row per transcript; target is the logit of the model's leave-item-out accuracy."""
    accuracy = transcript.model_accuracy.clip(ACCURACY_CLIP, 1 - ACCURACY_CLIP)
    return pd.DataFrame({
        "bench": transcript.bench, "item": transcript.item, "model": transcript.model,
        "logit_model_accuracy": np.log(accuracy / (1 - accuracy)),
        "test": transcript.split == TEST_SPLIT,
    }).join(transcript[predictors])


def stronger_model_from_pair(pair, predictors):
    """One row per distinct transcript pair (judges dropped), ties dropped."""
    distinct = pair.drop_duplicates(PAIR_IDENTITY)
    distinct = distinct[distinct.left_stronger.notna()]
    return pd.DataFrame({
        "bench": distinct.bench, "item": distinct.item,
        "left_stronger": (distinct.left_stronger > 0).astype(int),
        "test": distinct.split == TEST_SPLIT,
    }).join(_bare_names(distinct, predictors))


def judge_verdict_from_pair(pair, predictors):
    """One row per judge draw."""
    return pd.DataFrame({
        "bench": pair.bench, "item": pair.item, "judge": pair.judge,
        "pick_left": pair.pick_left, "accuracy_difference": pair.accuracy_difference,
        "test": pair.split == TEST_SPLIT,
    }).join(_bare_names(pair, predictors))


def _bare_names(pair, predictors):
    return pair[[f"d_{name}" for name in predictors]].set_axis(predictors, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    design_matrices = Path(yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())["output_root"])
    if args.grouped:
        design_matrices = design_matrices / "grouped"
    predictors = yaml.safe_load((HERE / ("predictors-grouped.yaml" if args.grouped else "predictors.yaml")).read_text())

    transcript = pd.read_parquet(design_matrices / "transcript.parquet")
    missing = [name for name in predictors if name not in transcript]
    if missing:
        raise ValueError(f"predictors not in the transcript design matrix: {missing}")
    pair = pd.read_parquet(design_matrices / "pair.parquet")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for build in (capability_from_transcript, stronger_model_from_pair, judge_verdict_from_pair):
        table = build(transcript if build is capability_from_transcript else pair, predictors)
        table.to_parquet(args.output_dir / f"{build.__name__}.parquet", index=False)
        print(f"{build.__name__}: {len(table)} rows, {int(table.test.sum())} test")


if __name__ == "__main__":
    main()
