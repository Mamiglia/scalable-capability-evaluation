from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from scripts.judge_verdict_analysis.config import CONFIG
from scripts.judge_verdict_analysis.fit_common import FitResult

from . import sparse_pairwise
from .sparse_pairwise import (
    CAPABILITY_TASK,
    CapabilityLevels,
    CellResult,
    aggregate_variant,
    build_specifications,
    capability_coefficients,
    capability_design,
    chance_predictions,
    fit_capability_cell,
    prepare_capability_cell,
    run_cell,
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


def _response_source(tmp_path):
    responses_path = tmp_path / "responses.parquet"
    manifest_path = tmp_path / "responses-manifest.json"
    responses = _responses()
    responses.to_parquet(responses_path, index=False)
    manifest_path.write_text(json.dumps({
        "canonical_only": True,
        "min_reasoning_chars": 100,
        "row_count": len(responses),
        "response_parquet_sha256": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
    }))
    return responses_path, manifest_path


def _matrix_source(tmp_path):
    root = tmp_path / "matrix"
    root.mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"state": "completed"}))
    (root / "design-matrix.yaml").write_text(f"output_root: {root}\n")
    return root, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _fake_fit(train, test, features, penalty, frozen, config, tuning=None, task=None):
    probability = np.where(test[features[0]].to_numpy(float) >= 0, 0.75, 0.25)
    predictions = test.drop(columns=[*features, "left_stronger"], errors="ignore").copy()
    predictions["observed"] = test.left_stronger.to_numpy(int)
    predictions["probability"] = probability
    coefficients = pd.DataFrame([
        {
            "feature": feature, "benchmark": benchmark, "population": 0.1,
            "benchmark_deviation": 0.0, "effective": 0.1,
        }
        for feature in features for benchmark in sorted(train.bench.unique())
    ])
    return FitResult(
        backend=f"{config.feature_space}-{penalty}", fold=str(train.fold.iloc[0]),
        selected_features=tuple(features), predictions=predictions,
        coefficients=coefficients,
        tuning=pd.DataFrame() if tuning is None else tuning.copy(),
        metrics=pd.DataFrame({"metric": ["accuracy", "log_loss"], "value": [0.5, 0.7]}),
        diagnostics={"C": frozen["C"], "l1_ratio": frozen["l1_ratio"], "converged": True},
    )


def test_checkpoint_tunes_once_and_freezes_every_raw_refit(monkeypatch, tmp_path) -> None:
    pairs = _pairs().assign(d_other=lambda frame: -frame.d_signal)
    dictionary = pd.DataFrame({
        "feature": ["signal", "other"], "block": ["surface", "structure"],
    })
    matrix_root, matrix_hash = _matrix_source(tmp_path)
    responses_path, manifest_path = _response_source(tmp_path)
    tuning_calls = 0
    fit_calls = []

    def fake_tune(*args, **kwargs):
        nonlocal tuning_calls
        tuning_calls += 1
        return {"C": 0.25, "l1_ratio": None}, pd.DataFrame({"selected": [True], "C": [0.25]})

    def fake_fit(*args, **kwargs):
        frozen = args[4]
        fit_calls.append(dict(frozen))
        return _fake_fit(*args, **kwargs)

    monkeypatch.setattr(sparse_pairwise, "_load_rows", lambda config: (pairs, dictionary))
    monkeypatch.setattr(sparse_pairwise, "tune_sparse", fake_tune)
    monkeypatch.setattr(sparse_pairwise, "fit_sparse", fake_fit)
    config = _config(
        model_folds=(("fold", ("held",)),), processed_matrix_root=matrix_root,
        processed_matrix_manifest_sha256=matrix_hash, resume_completed_folds=True,
    )

    output = run_cell(
        2, "all-readable", config, responses_path, tmp_path / "output",
        response_manifest_path=manifest_path,
    )
    specifications = build_specifications(
        dictionary, "raw", config.lesion_families,
    )

    assert output == tmp_path / "output/all-readable/raw-ridge/cells/fold__Q4"
    assert tuning_calls == 1
    assert {record["C"] for record in fit_calls} == {0.25}
    assert {record["l1_ratio"] for record in fit_calls} == {None}
    assert {spec.kind for spec in specifications} >= {
        "singular-feature", "lesion-feature", "singular-block", "lesion-block",
    }
    assert {spec.kind for spec in build_specifications(dictionary, "pc", ())} == {
        "singular-feature", "lesion-feature",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["response_source"] == {
        "responses_path": str(responses_path),
        "responses_sha256": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
        "response_manifest_path": str(manifest_path),
        "response_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    assert run_cell(
        2, "all-readable", config, responses_path, tmp_path / "output",
        response_manifest_path=manifest_path,
    ) == output
    assert tuning_calls == 1
    with pytest.raises(RuntimeError, match="settings mismatch"):
        run_cell(
            2, "all-readable", replace(config, benchmark_deviation_penalty=4.0),
            responses_path, tmp_path / "output", response_manifest_path=manifest_path,
        )
    damaged = output / "predictions/full.parquet"
    damaged.write_bytes(damaged.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="stale or corrupt"):
        run_cell(
            2, "all-readable", config, responses_path, tmp_path / "output",
            response_manifest_path=manifest_path,
        )


def test_aggregate_requires_twelve_matching_hash_valid_cells(monkeypatch, tmp_path) -> None:
    pairs = _pairs().assign(d_other=lambda frame: -frame.d_signal)
    dictionary = pd.DataFrame({
        "feature": ["signal", "other"], "block": ["surface", "structure"],
    })
    matrix_root, matrix_hash = _matrix_source(tmp_path)
    responses_path, manifest_path = _response_source(tmp_path)
    config = _config(
        model_folds=tuple((f"fold-{index}", (model,)) for index, model in enumerate(
            ("low", "mid", "high", "held")
        )),
        processed_matrix_root=matrix_root,
        processed_matrix_manifest_sha256=matrix_hash,
        resume_completed_folds=True,
        bootstrap_iterations=20,
    )

    def fake_cell(pairs, responses, dictionary, model_fold, target, config):
        fold = f"{next(iter(model_fold))}__{target}"
        train = pd.DataFrame({
            "bench": ["b", "b"], "item": ["i0", "i1"], "fold": [fold, fold],
            "signal": [-1.0, 1.0], "other": [1.0, -1.0], "left_stronger": [0, 1],
        })
        test = train.copy()
        frozen = {"C": 0.25, "l1_ratio": None}
        tuning = pd.DataFrame({"selected": [True], "C": [0.25]})
        full = _fake_fit(train, test, ["signal", "other"], "ridge", frozen, config, tuning)
        return CellResult(
            train, test, dictionary.copy(), {
                "feature_space": "raw", "features": ["signal", "other"],
                "loadings": pd.DataFrame(columns=["component", "feature", "loading"]),
                "explained_variance": pd.DataFrame(columns=[
                    "component", "explained_variance", "explained_variance_ratio",
                    "cumulative_explained_variance_ratio", "singular_value",
                ]),
            }, frozen, full, chance_predictions(test, ["signal", "other"]),
        )

    monkeypatch.setattr(sparse_pairwise, "_load_rows", lambda config: (pairs, dictionary))
    monkeypatch.setattr(sparse_pairwise, "fit_capability_cell", fake_cell)
    monkeypatch.setattr(sparse_pairwise, "fit_sparse", _fake_fit)
    for index in range(12):
        run_cell(
            index, "all-readable", config, responses_path, tmp_path / "output",
            response_manifest_path=manifest_path,
        )

    output = aggregate_variant("all-readable", "raw", "ridge", tmp_path / "output")

    assert output == tmp_path / "output/all-readable/raw-ridge/aggregate"
    assert {
        "metrics.csv", "target-metrics.csv", "benchmark-metrics.csv",
        "fold-metrics.csv", "paired-item-deltas.parquet", "delta-summary.csv",
        "selection.csv", "coefficients.csv", "artifact-index.csv", "manifest.json",
    } <= {path.name for path in output.iterdir()}
    assert len(pd.read_csv(output / "artifact-index.csv").cell.unique()) == 12
    aggregate_manifest = json.loads((output / "manifest.json").read_text())
    assert aggregate_manifest["response_source"]["responses_path"] == str(responses_path)
    damaged = tmp_path / "output/all-readable/raw-ridge/cells/fold-0__Q2/predictions/full.parquet"
    damaged.write_bytes(damaged.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="stale or corrupt"):
        aggregate_variant("all-readable", "raw", "ridge", tmp_path / "output")


def test_checkpoint_slurm_entrypoint_refuses_non_slurm_execution() -> None:
    script = Path(sparse_pairwise.__file__).with_suffix(".sbatch")
    environment = os.environ.copy()
    environment.pop("SLURM_JOB_ID", None)

    result = subprocess.run(
        ["bash", str(script), "--help"], env=environment,
        text=True, capture_output=True, check=False,
    )

    assert result.returncode != 0
    assert "SLURM_JOB_ID" in result.stderr

    environment["SLURM_JOB_ID"] = "1"
    environment["SLURM_ARRAY_TASK_ID"] = "1"
    aggregate = subprocess.run(
        ["bash", str(script), "--mode", "aggregate"], env=environment,
        text=True, capture_output=True, check=False,
    )
    assert aggregate.returncode == 0
    assert "skipping nonzero aggregate task" in aggregate.stdout


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
