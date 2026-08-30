from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.judge_verdict_analysis.config import CONFIG

from .sparse_pairwise import (
    CAPABILITY_TASK,
    CapabilityLevels,
    capability_coefficients,
    capability_design,
    chance_predictions,
    fit_capability_cell,
    prepare_capability_cell,
    validate_response_source,
)


def _responses() -> pd.DataFrame:
    models = ("low", "mid", "high", "held")
    rows = []
    for item_index in range(16):
        # Four items per solve-rate quartile; the final quartile gives every
        # model a distinct accuracy.
        correct = {
            "low": item_index < 12,
            "mid": item_index < 13,
            "held": item_index < 14,
            "high": item_index < 15,
        }
        for model in models:
            rows.append({
                "bench": "b", "model": model, "item": f"i{item_index:02d}",
                "correct": int(correct[model]),
            })
    return pd.DataFrame(rows)


def _pairs() -> pd.DataFrame:
    pairs = []
    for item_index in range(4):
        for left, right in (("low", "mid"), ("high", "mid"), ("held", "mid")):
            for judge in ("j1", "j2"):
                value = {"low": -1.0, "mid": 0.0, "held": 1.0, "high": 2.0}
                pairs.append({
                    "bench": "b", "item": f"i{item_index:02d}", "judge": judge,
                    "judge_log": f"{judge}-log", "judge_sample": judge == "j2",
                    "judge_epoch": 1 if judge == "j1" else 2,
                    "model_left": left, "model_right": right,
                    "log_left": left, "log_right": right,
                    "pick_left": int(judge == "j1"),
                    "d_signal": value[left] - value[right],
                })
    return pd.DataFrame(pairs)


def _config(**changes):
    return replace(
        CONFIG, benchmarks=("b",), judges=("j1", "j2"), inner_folds=2,
        elastic_net_cs=(1.0,), ridge_cs=(1.0,), fit_n_jobs=1,
        glm_max_iter=1_000, feature_space="raw", penalty="ridge",
        execution_target="local_test", local_test_max_rows=200,
        **changes,
    )


def test_prepare_capability_cell_builds_judge_independent_targets() -> None:
    dictionary = pd.DataFrame({"feature": ["signal"], "block": ["surface"]})

    train, test, prepared_dictionary = prepare_capability_cell(
        _pairs(), _responses(), dictionary, {"held"}, "Q4", _config(),
    )

    judge_metadata = {"judge", "judge_log", "judge_sample", "judge_epoch", "pick_left"}
    assert judge_metadata.isdisjoint(train.columns)
    assert judge_metadata.isdisjoint(test.columns)
    assert train[["bench", "item", "model_left", "model_right"]].duplicated().sum() == 0
    assert set(train.left_stronger) == {0, 1}
    assert not (train.accuracy_left == train.accuracy_right).any()
    assert test.model_left.eq("held").any() or test.model_right.eq("held").any()
    assert "signal" in train and "d_signal" not in train
    pd.testing.assert_frame_equal(prepared_dictionary, dictionary)


def test_capability_design_is_antisymmetric_and_coefficients_are_natural_scale() -> None:
    frame = pd.DataFrame({
        "bench": ["a", "b"], "item": ["i", "j"],
        "model_left": ["l", "x"], "model_right": ["r", "y"],
        "signal": [2.0, -3.0],
    })
    reversed_frame = frame.rename(columns={
        "model_left": "model_right", "model_right": "model_left",
    }).assign(signal=-frame.signal)
    levels = CapabilityLevels(("a", "b"))
    config = _config(benchmark_deviation_penalty=4.0)

    design, column_map = capability_design(frame, ["signal"], levels, config)
    reversed_design, _ = capability_design(reversed_frame, ["signal"], levels, config)
    np.testing.assert_allclose(reversed_design.toarray(), -design.toarray())

    coefficients = capability_coefficients(np.array([2.0, 4.0, -2.0]), column_map)
    assert coefficients[["benchmark", "population", "benchmark_deviation", "effective"]].values.tolist() == [
        ["a", 2.0, 2.0, 4.0], ["b", 2.0, -1.0, 1.0],
    ]
    assert CAPABILITY_TASK.outcome == "left_stronger"


def test_chance_predictions_are_exactly_half() -> None:
    test = pd.DataFrame({
        "bench": ["b", "b"], "item": ["i", "j"], "fold": ["f", "f"],
        "left_stronger": [0, 1], "signal": [-1.0, 1.0],
    })

    predictions = chance_predictions(test, ["signal"])

    assert predictions.probability.tolist() == [0.5, 0.5]
    assert predictions.observed.tolist() == [0, 1]
    assert "signal" not in predictions


def test_fit_capability_cell_returns_one_prepared_frozen_fit() -> None:
    dictionary = pd.DataFrame({"feature": ["signal"], "block": ["surface"]})

    result = fit_capability_cell(
        _pairs(), _responses(), dictionary, {"held"}, "Q4", _config(),
    )

    assert result.full.diagnostics["C"] == result.frozen_hyperparameters["C"]
    assert result.full.diagnostics["l1_ratio"] is result.frozen_hyperparameters["l1_ratio"] is None
    assert result.null.probability.eq(0.5).all()
    judge_metadata = {"judge", "judge_log", "judge_sample", "judge_epoch", "pick_left"}
    assert judge_metadata.isdisjoint(result.full.predictions.columns)
    assert judge_metadata.isdisjoint(result.null.columns)
    assert result.prepared_dictionary.feature.tolist() == ["signal"]
    assert result.provenance["feature_space"] == "raw"


def test_response_source_requires_verified_eligibility_and_hashes(tmp_path) -> None:
    responses_path = tmp_path / "responses.parquet"
    manifest_path = tmp_path / "responses-manifest.json"
    responses = _responses()
    responses.to_parquet(responses_path, index=False)
    response_sha256 = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps({
        "canonical_only": True,
        "min_reasoning_chars": 100,
        "row_count": len(responses),
        "response_parquet_sha256": response_sha256,
    }))

    loaded, source = validate_response_source(responses_path, manifest_path)

    pd.testing.assert_frame_equal(loaded, responses)
    assert source == {
        "responses_path": str(responses_path),
        "responses_sha256": response_sha256,
        "response_manifest_path": str(manifest_path),
        "response_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps({
        "canonical_only": False,
        "min_reasoning_chars": 100,
        "row_count": len(responses),
        "response_parquet_sha256": response_sha256,
    }))
    with pytest.raises(ValueError, match="canonical_only"):
        validate_response_source(responses_path, manifest_path)
