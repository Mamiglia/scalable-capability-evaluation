import pandas as pd

from .pairwise_report import backproject_pc_coefficients, expand_leaf_coefficients


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
