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
from scripts.judge_verdict_analysis.hierarchy import outer_roles

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
    overall_model_accuracy,
    prepare_capability_fold,
    run_cell,
    validate_response_source,
)


def _responses() -> pd.DataFrame:
    models = ("low", "mid", "high", "held")
    rows = []
    for item_index in range(16):
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
    for item_index in range(16):
        for left, right in (
            ("low", "mid"), ("low", "high"), ("high", "mid"), ("held", "mid"),
        ):
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
    values = {
        "benchmarks": ("b",), "judges": ("j1", "j2"), "inner_folds": 2,
        "elastic_net_cs": (1.0,), "ridge_cs": (1.0,), "fit_n_jobs": 1,
        "glm_max_iter": 1_000, "feature_space": "raw", "penalty": "ridge",
        "execution_target": "local_test", "local_test_max_rows": 200,
    }
    values.update(changes)
    return replace(CONFIG, **values)


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


@pytest.mark.parametrize(
    ("feature_space", "penalty"),
    (("raw", "l1"), ("raw", "ridge"), ("pc", "l1"), ("pc", "ridge")),
)
def test_checkpoint_tunes_once_and_freezes_every_refit(
    monkeypatch, tmp_path, feature_space, penalty,
) -> None:
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
        ratio = 1.0 if args[2] == "l1" else None
        return {"C": 0.25, "l1_ratio": ratio}, pd.DataFrame({
            "selected": [True], "C": [0.25], "l1_ratio": [ratio],
        })

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
        feature_space=feature_space, penalty=penalty,
    )

    output = run_cell(
        2, "all-readable", config, responses_path, tmp_path / "output",
        response_manifest_path=manifest_path,
    )
    prepared_dictionary = pd.read_parquet(output / "feature-dictionary.parquet")
    specifications = build_specifications(
        prepared_dictionary, feature_space, config.lesion_families,
    )

    assert output == tmp_path / f"output/all-readable/{feature_space}-{penalty}/cells/fold__item-2"
    assert tuning_calls == 1
    assert {record["C"] for record in fit_calls} == {0.25}
    assert {record["l1_ratio"] for record in fit_calls} == {
        1.0 if penalty == "l1" else None,
    }
    if feature_space == "raw":
        assert {spec.kind for spec in specifications} >= {
            "singular-feature", "lesion-feature", "singular-block", "lesion-block",
        }
    else:
        assert {spec.kind for spec in specifications} == {
            "singular-feature", "lesion-feature",
        }
        assert all(spec.variable.startswith("pc_") for spec in specifications)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["response_source"] == {
        "responses_path": str(responses_path),
        "responses_sha256": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
        "response_manifest_path": str(manifest_path),
        "response_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    assert manifest["matrix_source"] == {
        "matrix_config_path": str(matrix_root / "design-matrix.yaml"),
        "matrix_config_sha256": hashlib.sha256(
            (matrix_root / "design-matrix.yaml").read_bytes(),
        ).hexdigest(),
        "matrix_manifest_path": str(matrix_root / "manifest.json"),
        "matrix_manifest_sha256": matrix_hash,
    }
    variant_settings = json.loads((output / "settings.json").read_text())["settings"]["variant"]
    expected_selection_settings = {
        "feature_blocks": list(config.feature_blocks),
        "regressors": list(config.regressors),
        "splits": list(config.splits),
        "judges": list(config.judges),
        "benchmarks": list(config.benchmarks),
    }
    assert {
        key: variant_settings[key] for key in expected_selection_settings
    } == expected_selection_settings
    assert variant_settings["item_folds"] == 4
    assert set(variant_settings) == {
        "schema_version", "scope", "feature_space", "penalty", "feature_blocks",
        "regressors", "splits", "judges", "benchmarks", "model_folds", "item_folds",
        "inner_folds", "fit_n_jobs", "fit_pca_components", "fit_pca_algorithm", "l1_cs",
        "ridge_cs", "benchmark_deviation_penalty", "glm_max_iter", "bootstrap_iterations",
        "random_seed", "lesion_families", "matrix_source", "response_source",
    }
    assert run_cell(
        2, "all-readable", config, responses_path, tmp_path / "output",
        response_manifest_path=manifest_path,
    ) == output
    assert tuning_calls == 1
    manifest_path_in_cell = output / "manifest.json"
    manifest_text = manifest_path_in_cell.read_text()
    manifest["matrix_source"] = {"invalid": "source"}
    manifest_path_in_cell.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="stale or corrupt"):
        run_cell(
            2, "all-readable", config, responses_path, tmp_path / "output",
            response_manifest_path=manifest_path,
        )
    manifest_path_in_cell.write_text(manifest_text)
    with pytest.raises(RuntimeError, match="settings mismatch"):
        run_cell(
            2, "all-readable", replace(config, splits=("different-split",)),
            responses_path, tmp_path / "output", response_manifest_path=manifest_path,
        )
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


@pytest.mark.parametrize(
    ("feature_space", "penalty"),
    (("raw", "l1"), ("raw", "ridge"), ("pc", "l1"), ("pc", "ridge")),
)
def test_aggregate_requires_sixteen_matching_hash_valid_cells(
    monkeypatch, tmp_path, feature_space, penalty,
) -> None:
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
        feature_space=feature_space,
        penalty=penalty,
    )

    def fake_cell(pairs, responses, dictionary, held_out_models, item_fold, config):
        fold = f"{next(iter(held_out_models))}__item-{item_fold}"
        train = pd.DataFrame({
            "bench": ["b", "b"], "item": ["i0", "i1"], "fold": [fold, fold],
            "signal": [-1.0, 1.0], "other": [1.0, -1.0], "left_stronger": [0, 1],
        })
        test = train.copy()
        frozen = {"C": 0.25, "l1_ratio": None}
        tuning = pd.DataFrame({"selected": [True], "C": [0.25]})
        full = _fake_fit(train, test, ["signal", "other"], penalty, frozen, config, tuning)
        return CellResult(
            train, test, dictionary.copy(), {
                "feature_space": feature_space, "features": ["signal", "other"],
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
    for index in range(16):
        run_cell(
            index, "all-readable", config, responses_path, tmp_path / "output",
            response_manifest_path=manifest_path,
        )

    output = aggregate_variant(
        "all-readable", feature_space, penalty, tmp_path / "output",
    )

    assert output == tmp_path / f"output/all-readable/{feature_space}-{penalty}/aggregate"
    assert {path.name for path in output.iterdir()} == {
        "settings.json", "predictions.parquet", "item-metrics.parquet", "metrics.csv",
        "benchmark-metrics.csv", "tuning.csv",
        "fold-metrics.csv", "paired-item-deltas.parquet", "delta-summary.csv",
        "selection.csv", "coefficients.csv", "artifact-index.csv", "manifest.json",
    }
    artifact_index = pd.read_csv(output / "artifact-index.csv")
    expected_cells = {
        f"fold-{model_fold}__item-{item_fold}"
        for model_fold in range(4) for item_fold in range(4)
    }
    assert set(artifact_index.cell) == expected_cells
    assert set(artifact_index) == {
        "cell", "model_fold", "item_fold", "specification", "kind", "variable",
        "reference", "artifact", "predictions", "coefficients", "item_metrics",
        "cell_manifest",
    }
    aggregate_manifest = json.loads((output / "manifest.json").read_text())
    assert aggregate_manifest["response_source"]["responses_path"] == str(responses_path)
    assert aggregate_manifest["matrix_source"]["matrix_manifest_sha256"] == matrix_hash
    other = tmp_path / "other-sources"
    other.mkdir()
    other_matrix, other_matrix_hash = _matrix_source(other)
    other_responses, other_manifest = _response_source(other)
    with pytest.raises(RuntimeError, match="supplied source provenance"):
        sparse_pairwise.main([
            "--mode", "aggregate",
            "--matrix-root", str(other_matrix),
            "--matrix-manifest-sha256", other_matrix_hash,
            "--responses", str(other_responses),
            "--response-manifest", str(other_manifest),
            "--response-manifest-sha256", hashlib.sha256(
                other_manifest.read_bytes(),
            ).hexdigest(),
            "--output-root", str(tmp_path / "output"),
            "--scope", "all-readable",
            "--feature-space", feature_space,
            "--penalty", penalty,
        ])
    aggregate_manifest_path = output / "manifest.json"
    aggregate_manifest_text = aggregate_manifest_path.read_text()
    aggregate_manifest["matrix_source"] = {"invalid": "source"}
    aggregate_manifest_path.write_text(json.dumps(aggregate_manifest))
    with pytest.raises(RuntimeError, match="stale or corrupt"):
        aggregate_variant(
            "all-readable", feature_space, penalty, tmp_path / "output",
        )
    aggregate_manifest_path.write_text(aggregate_manifest_text)
    damaged = (
        tmp_path / f"output/all-readable/{feature_space}-{penalty}"
        / "cells/fold-0__item-0/predictions/full.parquet"
    )
    damaged.write_bytes(damaged.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="stale or corrupt"):
        aggregate_variant(
            "all-readable", feature_space, penalty, tmp_path / "output",
        )


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


def test_overall_model_accuracy_weights_benchmark_items_equally() -> None:
    responses = pd.DataFrame([
        {"bench": "b", "model": "a", "item": "i0", "correct": 1},
        {"bench": "b", "model": "a", "item": "i0", "correct": 1},
        {"bench": "b", "model": "a", "item": "i0", "correct": 1},
        {"bench": "b", "model": "a", "item": "i1", "correct": 0},
        {"bench": "b", "model": "b", "item": "i0", "correct": 1},
        {"bench": "b", "model": "b", "item": "i1", "correct": 1},
    ])

    accuracy = overall_model_accuracy(responses)

    assert accuracy.to_dict("records") == [
        {"bench": "b", "model": "a", "accuracy": 0.5},
        {"bench": "b", "model": "b", "accuracy": 1.0},
    ]


def test_prepare_capability_fold_uses_overall_accuracy_and_all_pair_items() -> None:
    dictionary = pd.DataFrame({"feature": ["signal"], "block": ["surface"]})
    responses = _responses()
    responses.loc[(responses.model == "low") & (responses.item == "i12"), "correct"] = 1

    train, test, prepared_dictionary = prepare_capability_fold(
        _pairs(), responses, dictionary, {"held"}, 0, _config(),
    )

    judge_metadata = {"judge", "judge_log", "judge_sample", "judge_epoch", "pick_left"}
    assert judge_metadata.isdisjoint(train.columns)
    assert judge_metadata.isdisjoint(test.columns)
    assert train[["bench", "item", "model_left", "model_right"]].duplicated().sum() == 0
    assert set(train.left_stronger) == {0, 1}
    assert not (train.accuracy_left == train.accuracy_right).any()
    assert not ((train.model_left == "low") & (train.model_right == "mid")).any()
    assert test.model_left.eq("held").any() or test.model_right.eq("held").any()
    assert set(train.item) | set(test.item) == {f"i{index:02d}" for index in range(16)}
    assert set(train) == set(test) == {
        "bench", "item", "model_left", "model_right", "log_left", "log_right",
        "signal", "accuracy_left", "accuracy_right", "left_stronger", "fold",
    }
    assert "signal" in train and "d_signal" not in train
    pd.testing.assert_frame_equal(prepared_dictionary, dictionary)


def test_prepare_capability_fold_matches_shared_outer_roles_after_deduplication() -> None:
    dictionary = pd.DataFrame({"feature": ["signal"], "block": ["surface"]})
    pairs = sparse_pairwise.deduplicate_feature_pairs(_pairs())
    roles = outer_roles(pairs, {"held"}, item_fold=2, item_folds=4)

    train, test, _ = prepare_capability_fold(
        _pairs(), _responses(), dictionary, {"held"}, 2, _config(),
    )

    keys = ["bench", "item", "model_left", "model_right", "log_left", "log_right"]
    for actual, role in ((train, "train"), (test, "test")):
        expected = pairs.loc[roles == role, keys].sort_values(keys).reset_index(drop=True)
        observed = actual[keys].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, expected)


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
        _pairs(), _responses(), dictionary, {"held"}, 0, _config(),
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
