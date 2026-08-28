from dataclasses import fields

import numpy as np
import pandas as pd

from .fit_common import FitResult
from .hierarchy import (
    HierarchyLevels,
    effective_coefficients,
    hierarchical_design,
    outer_roles,
)


FEATURES = ("feature_one", "feature_two")
PENALTIES = {
    "judge_deviation_penalty": 4.0,
    "benchmark_deviation_penalty": 9.0,
    "cell_deviation_penalty": 16.0,
    "item_intercept_penalty": 25.0,
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "judge": judge,
            "bench": bench,
            "item": item,
            "model_left": "seen_left",
            "model_right": "seen_right",
            "feature_one": 1.0,
            "feature_two": -1.0,
        }
        for judge in ("judge_a", "judge_b")
        for bench in ("bench_a", "bench_b")
        for item in ("item_a", "item_b")
    ])


def test_outer_roles_are_strict_for_new_models_and_new_items() -> None:
    frame = pd.DataFrame({
        "bench": ["bench_a", "bench_a", "bench_b", "bench_b", "bench_a", "bench_a"],
        "item": ["item_a", "item_b", "item_a", "item_b", "item_a", "item_b"],
        "model_left": ["held", "seen", "held", "held", "seen", "seen"],
        "model_right": ["seen", "seen", "seen", "seen", "seen", "held"],
    })

    assert outer_roles(frame, {"held"}, item_fold=0, item_folds=2).tolist() == [
        "test", "train", "test", "drop", "drop", "drop",
    ]


def test_hierarchical_design_has_exact_blocks_and_no_item_slopes() -> None:
    frame = _frame()
    levels = HierarchyLevels.from_training(frame)

    design, column_map = hierarchical_design(frame, FEATURES, levels, PENALTIES)

    assert design.shape == (len(frame), 31)
    assert column_map.block.value_counts().to_dict() == {
        "global_intercept": 1,
        "judge_intercept": 2,
        "benchmark_intercept": 2,
        "cell_intercept": 4,
        "item_intercept": 4,
        "population_slope": 2,
        "judge_slope": 4,
        "benchmark_slope": 4,
        "cell_slope": 8,
    }
    assert column_map.loc[column_map.block.str.endswith("slope"), "item"].isna().all()
    assert column_map.groupby("block").multiplier.first().to_dict() == {
        "global_intercept": 1.0,
        "judge_intercept": 0.5,
        "benchmark_intercept": 1 / 3,
        "cell_intercept": 0.25,
        "item_intercept": 0.2,
        "population_slope": 1.0,
        "judge_slope": 0.5,
        "benchmark_slope": 1 / 3,
        "cell_slope": 0.25,
    }


def test_unseen_test_item_has_no_item_intercept_column() -> None:
    frame = _frame()
    levels = HierarchyLevels.from_training(frame[~((frame.bench == "bench_b") & (frame.item == "item_b"))])

    design, column_map = hierarchical_design(frame, FEATURES, levels, PENALTIES)

    item_columns = column_map.loc[column_map.block == "item_intercept", "column"].to_numpy()
    row = frame.index[(frame.bench == "bench_b") & (frame.item == "item_b")][0]
    assert design[row, item_columns].nnz == 0


def test_effective_coefficients_sum_all_cell_deviations() -> None:
    frame = _frame()
    levels = HierarchyLevels.from_training(frame)
    _, column_map = hierarchical_design(frame, FEATURES, levels, {key: 1.0 for key in PENALTIES})
    raw = np.zeros(len(column_map))
    values = {
        "population_slope": 1.0,
        "judge_slope": 2.0,
        "benchmark_slope": 3.0,
        "cell_slope": 4.0,
    }
    for block, value in values.items():
        match = (column_map.block == block) & (column_map.feature == "feature_one")
        if block in {"judge_slope", "cell_slope"}:
            match &= column_map.judge == "judge_b"
        if block in {"benchmark_slope", "cell_slope"}:
            match &= column_map.benchmark == "bench_b"
        raw[column_map.loc[match, "column"].item()] = value

    result = effective_coefficients(raw, column_map).set_index(["feature", "judge", "benchmark"])

    assert result.loc[("feature_one", "judge_b", "bench_b")].to_dict() == {
        "population": 1.0,
        "judge_deviation": 2.0,
        "benchmark_deviation": 3.0,
        "cell_deviation": 4.0,
        "effective": 10.0,
    }


def test_fit_result_has_the_shared_backend_contract() -> None:
    assert [field.name for field in fields(FitResult)] == [
        "backend", "fold", "selected_features", "predictions", "coefficients", "tuning", "metrics", "diagnostics",
    ]
