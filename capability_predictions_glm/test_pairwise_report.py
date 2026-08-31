import pandas as pd
import pytest

from .pairwise_report import (
    backproject_pc_coefficients,
    expand_leaf_coefficients,
    judge_capability_overview,
)


def test_backprojection_uses_shared_loadings_and_benchmark_leaf_weights() -> None:
    coefficients = pd.DataFrame({
        "fold": ["f0", "f0"], "benchmark": ["a", "a"],
        "feature": ["pc_001", "pc_002"], "effective": [2.0, 3.0],
    })
    loadings = pd.DataFrame([
        {"component": "pc_001", "feature": "c1", "loading": 0.5},
        {"component": "pc_001", "feature": "c2", "loading": 1.0},
        {"component": "pc_002", "feature": "c1", "loading": -1.0},
        {"component": "pc_002", "feature": "c2", "loading": 0.25},
        {"component": "pc_003", "feature": "c1", "loading": 9.0},
        {"component": "pc_003", "feature": "c2", "loading": 9.0},
    ])
    composites = [
        {"output_name": "c1", "block": "mixed",
         "leaf_weights_by_benchmark": {"a": {"x": 0.5, "y": -0.5}}},
        {"output_name": "c2", "block": "surface",
         "leaf_weights_by_benchmark": {"a": {"y": 2.0}}},
    ]

    projected = backproject_pc_coefficients(coefficients, loadings)
    leaves = expand_leaf_coefficients(projected, composites)

    assert projected.set_index("feature").effective.to_dict() == {"c1": -2.0, "c2": 2.75}
    assert leaves.set_index("feature").effective.to_dict() == {"x": -1.0, "y": 6.5}


def test_judge_capability_overview_joins_prediction_and_representation_metrics() -> None:
    accuracy = pd.DataFrame([{
        "variant": "raw-l1", "judge": "j", "capability_accuracy": 0.68,
        "judge_accuracy": 0.60, "pair_count": 100,
    }])
    statistics = pd.DataFrame([{
        "variant": "raw-l1", "judge": "j", "benchmark": "all",
        "pearson": 0.4, "spearman": 0.3, "same_sign_share": 0.7,
    }])
    cosines = pd.DataFrame([{
        "variant": "raw-l1", "judge": "j", "benchmark": "all",
        "estimate": 0.5, "lower": 0.2, "upper": 0.8,
    }])
    eci = pd.DataFrame([{
        "judge": "j", "eci": 140.0, "eci_ci_low": 135.0, "eci_ci_high": 145.0,
    }])

    overview = judge_capability_overview(accuracy, statistics, cosines, eci)

    assert overview.to_dict("records") == [{
        "variant": "raw-l1", "judge": "j", "capability_accuracy": 0.68,
        "judge_accuracy": 0.60, "accuracy_difference": pytest.approx(0.08),
        "pair_count": 100, "pearson": 0.4, "spearman": 0.3,
        "same_sign_share": 0.7, "cosine": 0.5, "cosine_lower": 0.2,
        "cosine_upper": 0.8, "eci": 140.0, "eci_ci_low": 135.0,
        "eci_ci_high": 145.0,
    }]
