import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from .pairwise_report import (
    _benchmark_lesions,
    _correct_null_accuracy,
    _overall_lesions,
    backproject_pc_coefficients,
    expand_leaf_coefficients,
    judge_capability_overview,
    pc_score_scales,
    refit_matches_frozen,
    summarize_capability_predictions,
    validate_fold_manifest,
)


def test_capability_singular_accuracy_uses_same_row_null() -> None:
    rows = pd.DataFrame([
        {"fold": f"model-fold-{model}__item-{item}", "bench": "a", "item": "i",
         "kind": kind, "variable": "zero", "reference": reference,
         "candidate_accuracy": candidate, "reference_accuracy": reference_accuracy,
         "delta_accuracy": delta, "delta_log_loss": log_loss, "row_count": 4}
        for model in range(4) for item in range(4)
        for kind, reference, candidate, reference_accuracy, delta, log_loss in (
            ("singular-feature", "null", 0.75, 0.5, 0.25, 0.0),
            ("lesion-feature", "full", 0.65, 0.75, 0.1, 0.02),
        )
    ])
    null = pd.DataFrame([
        {"fold": f"model-fold-{model}__item-{item}", "bench": "a", "item": "i",
         "observed": observed, "probability": 0.5}
        for model in range(4) for item in range(4) for observed in (1, 1, 1, 0)
    ])

    corrected = _correct_null_accuracy(rows, null)
    summary = _benchmark_lesions(corrected).set_index("kind")
    overall = _overall_lesions(corrected).set_index("kind")

    assert summary.loc["singular-feature", "estimate"] == pytest.approx(0.0)
    assert summary.loc["lesion-feature", "estimate"] == pytest.approx(10.0)
    assert overall.loc["singular-feature", "estimate"] == pytest.approx(0.0)
    assert overall.loc["lesion-feature", "estimate"] == pytest.approx(10.0)


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
    score_scales = pd.Series({"pc_001": 2.0, "pc_002": 3.0})
    composites = [
        {"output_name": "c1", "block": "mixed",
         "leaf_weights_by_benchmark": {"a": {"x": 0.5, "y": -0.5}}},
        {"output_name": "c2", "block": "surface",
         "leaf_weights_by_benchmark": {"a": {"y": 2.0}}},
    ]

    projected = backproject_pc_coefficients(coefficients, loadings, score_scales)
    leaves = expand_leaf_coefficients(projected, composites)

    assert projected.set_index("feature").effective.to_dict() == {"c1": -0.5, "c2": 1.25}
    assert leaves.set_index("feature").effective.to_dict() == {"x": -0.25, "y": 2.75}


def test_pc_score_scales_use_saved_explained_variance() -> None:
    variance = pd.DataFrame({
        "component": ["pc_001", "pc_002"], "explained_variance": [4.0, 9.0],
    })

    assert pc_score_scales(variance).to_dict() == {"pc_001": 2.0, "pc_002": 3.0}


def test_refit_validation_includes_fitted_benchmark_ratio() -> None:
    frozen = {"C": 0.1, "l1_ratio": 1.0, "benchmark_deviation_penalty": 0.3}

    assert refit_matches_frozen(frozen | {"converged": True}, frozen)
    assert not refit_matches_frozen(frozen | {"benchmark_deviation_penalty": 3.0}, frozen)
    assert not refit_matches_frozen({"C": 0.1, "l1_ratio": 1.0}, frozen)


def test_fold_manifest_validation_checks_predecessor_and_artifact_hashes(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("valid")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "state": "completed",
        "artifacts": {"artifact.txt": hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }))
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert validate_fold_manifest(tmp_path, expected)["state"] == "completed"

    artifact.write_text("tampered")
    with pytest.raises(RuntimeError, match="invalid completed fold artifacts"):
        validate_fold_manifest(tmp_path, expected)


def test_judge_capability_overview_joins_prediction_and_representation_metrics() -> None:
    accuracy = pd.DataFrame([{
        "variant": "raw-l1", "judge": "j", "capability_accuracy": 0.68,
        "judge_accuracy": 0.60, "pair_count": 100,
    }])
    statistics = pd.DataFrame([{
        "variant": "raw-l1", "judge": "j", "benchmark": "all",
        "pearson": 0.4, "spearman": 0.3, "same_sign_share": 0.7,
        "coordinate_space": "original_leaf",
    }])
    cosines = pd.DataFrame([{
        "variant": "raw-l1", "judge": "j", "benchmark": "all",
        "estimate": 0.5, "lower": 0.2, "upper": 0.8,
        "coordinate_space": "original_leaf",
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
    with pytest.raises(ValueError, match="original-leaf coordinates"):
        judge_capability_overview(
            accuracy, statistics.assign(coordinate_space="native"), cosines, eci,
        )


def test_capability_prediction_summary_comes_from_saved_fold_predictions() -> None:
    predictions = pd.DataFrame([
        {"variant": "raw-l1", "bench": "a", "observed": 1, "probability": 0.8},
        {"variant": "raw-l1", "bench": "a", "observed": 0, "probability": 0.4},
        {"variant": "pc-l1", "bench": "b", "observed": 1, "probability": 0.4},
        {"variant": "pc-l1", "bench": "b", "observed": 0, "probability": 0.7},
    ])

    variants, benchmarks = summarize_capability_predictions(predictions)

    assert variants.set_index("variant").rows.to_dict() == {"pc-l1": 2, "raw-l1": 2}
    assert variants.set_index("variant").accuracy.to_dict() == {"pc-l1": 0.0, "raw-l1": 1.0}
    assert benchmarks.set_index(["variant", "bench"]).accuracy.to_dict() == {
        ("pc-l1", "b"): 0.0, ("raw-l1", "a"): 1.0,
    }
    assert variants.set_index("variant").loc["raw-l1", "log_loss"] == pytest.approx(
        (-np.log(0.8) - np.log(0.6)) / 2,
    )
