from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.judge_verdict_analysis.config import CONFIG
from scripts.judge_verdict_analysis.fit_common import prepare_features
from scripts.judge_verdict_analysis.hierarchy import outer_roles

from . import pairwise_wrapper
from .pairwise_wrapper import (
    CAPABILITY_TASK,
    _capability_design,
    chance_predictions,
    fit_fold,
    overall_model_accuracy,
    prepare_capability_fold,
    prepare_capability_rows,
)


def _responses() -> pd.DataFrame:
    return pd.DataFrame([
        {"bench": "b", "model": model, "item": f"i{item:02d}",
         "correct": int(item < solved), "is_canonical": True}
        for model, solved in (("m0", 8), ("m1", 6), ("m2", 4), ("m3", 2))
        for item in range(8)
    ])


def _pairs() -> pd.DataFrame:
    ability = {"m0": 3.0, "m1": 2.0, "m2": 1.0, "m3": 0.0}
    rows = []
    for item in range(8):
        for pair_index, (first, second) in enumerate(
            (("m0", "m1"), ("m0", "m2"), ("m0", "m3"),
             ("m1", "m2"), ("m1", "m3"), ("m2", "m3"))
        ):
            left, right = ((first, second) if (item + pair_index) % 2
                           else (second, first))
            for judge in ("j0", "j1"):
                difference = ability[left] - ability[right]
                rows.append({
                    "bench": "b", "item": f"i{item:02d}", "judge": judge,
                    "judge_log": judge, "pick_left": int(difference > 0),
                    "model_left": left, "model_right": right,
                    "log_left": left, "log_right": right,
                    "signal": difference, "other": difference * 0.5,
                })
    return pd.DataFrame(rows)


def _pc_basis(tmp_path: Path) -> Path:
    root = tmp_path / "pc-basis"
    root.mkdir(exist_ok=True)
    pd.DataFrame([
        {"component": "pc_001", "feature": "signal", "loading": 1.0},
        {"component": "pc_001", "feature": "other", "loading": 0.0},
        {"component": "pc_002", "feature": "signal", "loading": 0.0},
        {"component": "pc_002", "feature": "other", "loading": 1.0},
    ]).to_parquet(root / "loadings.parquet", index=False)
    pd.DataFrame({
        "component": ["pc_001", "pc_002"],
        "explained_variance": [2.0, 1.0],
        "explained_variance_ratio": [2 / 3, 1 / 3],
        "cumulative_explained_variance_ratio": [2 / 3, 1.0],
        "singular_value": [2.0, 1.0],
    }).to_csv(root / "explained-variance.csv", index=False)
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, feature_space: str = "raw", penalty: str = "ridge"):
    return replace(
        CONFIG, results_dir=tmp_path, benchmarks=("b",), judges=("j0", "j1"),
        model_folds=tuple((f"model-fold-{i}", (f"m{i}",)) for i in range(4)),
        item_folds=4, inner_folds=2, fit_n_jobs=1, fit_pca_components=2,
        elastic_net_cs=(1.0,), ridge_cs=(1.0,), glm_max_iter=1_000,
        pooling_svi_steps=150, pooling_svi_learning_rate=0.03,
        benchmark_deviation_penalty=123.0,
        feature_space=feature_space, penalty=penalty, execution_mode="prepare",
        pc_basis_root=_pc_basis(tmp_path) if feature_space == "pc" else None,
    )


def test_overall_accuracy_is_canonical_and_equal_item_weighted() -> None:
    responses = pd.DataFrame([
        {"bench": "b", "model": "a", "item": "i0", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "a", "item": "i0", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "a", "item": "i1", "correct": 0, "is_canonical": True},
        {"bench": "b", "model": "a", "item": "i1", "correct": 1, "is_canonical": False},
        {"bench": "b", "model": "b", "item": "i0", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "b", "item": "i1", "correct": 1, "is_canonical": True},
    ])

    assert overall_model_accuracy(responses).to_dict("records") == [
        {"bench": "b", "model": "a", "accuracy": 0.5},
        {"bench": "b", "model": "b", "accuracy": 1.0},
    ]
    for column in ("bench", "model", "item", "correct", "is_canonical"):
        with pytest.raises(ValueError, match="response table missing columns"):
            overall_model_accuracy(responses.drop(columns=column))


def test_preparation_deduplicates_judges_checks_support_and_uses_shared_roles(tmp_path) -> None:
    pairs = _pairs()
    dictionary = pd.DataFrame({"feature": ["signal", "other"], "block": "surface"})
    config = _config(tmp_path)

    prepared = prepare_capability_rows(pairs, _responses(), dictionary)
    roles = outer_roles(prepared, {"m0"}, 2, config.item_folds)
    train, test = prepare_capability_fold(
        pairs, _responses(), dictionary, {"m0"}, 2, config,
    )

    assert len(prepared) == len(pairs) // 2
    assert {"judge", "judge_log", "pick_left"}.isdisjoint(prepared)
    assert set(train.item) | set(test.item) == set(prepared.item)
    assert train.item.tolist() == prepared.loc[roles == "train", "item"].tolist()
    assert test.item.tolist() == prepared.loc[roles == "test", "item"].tolist()
    assert not (prepared.accuracy_left == prepared.accuracy_right).any()
    assert set(prepared.left_stronger) == {0, 1}

    missing = pairs.iloc[[0]].assign(model_left="unsupported")
    with pytest.raises(ValueError, match="missing overall accuracy for left"):
        prepare_capability_rows(missing, _responses(), dictionary)


def test_design_and_null_are_exactly_antisymmetric(tmp_path) -> None:
    frame = pd.DataFrame({
        "bench": ["b"], "item": ["i"], "model_left": ["l"],
        "model_right": ["r"], "signal": [2.0], "left_stronger": [1],
    })
    reversed_frame = frame.assign(
        model_left="r", model_right="l", signal=-frame.signal, left_stronger=0,
    )
    levels = ("b",)

    design, _ = _capability_design(frame, ["signal"], levels, _config(tmp_path))
    reversed_design, _ = _capability_design(
        reversed_frame, ["signal"], levels, _config(tmp_path),
    )
    np.testing.assert_array_equal(reversed_design.toarray(), -design.toarray())
    assert CAPABILITY_TASK.outcome == "left_stronger"
    assert CAPABILITY_TASK.group_columns == ("bench", "item")
    assert set(_capability_design(
        frame, ["signal"], levels, _config(tmp_path),
    )[1].block) == {"population_slope", "benchmark_slope"}
    assert chance_predictions(frame, ["signal"]).probability.tolist() == [0.5]


@pytest.mark.parametrize(("penalty", "multiplier"), (("l1", 0.25), ("ridge", 0.5)))
def test_capability_design_uses_penalty_specific_benchmark_scaling(
    tmp_path: Path, penalty: str, multiplier: float,
) -> None:
    frame = pd.DataFrame({
        "bench": ["b"], "item": ["i"], "model_left": ["l"],
        "model_right": ["r"], "signal": [2.0], "left_stronger": [1],
    })
    config = replace(
        _config(tmp_path, penalty=penalty), benchmark_deviation_penalty=4.0,
    )

    design, columns = _capability_design(frame, ["signal"], ("b",), config)

    assert columns.loc[columns.block == "benchmark_slope", "multiplier"].item() == multiplier
    assert design[0, 1] == pytest.approx(2 * multiplier)


@pytest.mark.parametrize("feature_space", ("raw", "pc"))
@pytest.mark.parametrize("penalty", ("l1", "ridge"))
def test_fit_fold_runs_every_supported_variant(tmp_path, feature_space, penalty) -> None:
    dictionary = pd.DataFrame({
        "feature": ["signal", "other"], "block": ["surface", "surface"],
    })
    config = _config(tmp_path, feature_space, penalty)
    manifest_provenance = {
        "source_paths": {"responses": "/inputs/responses.parquet"},
        "source_sha256": {"responses": "a" * 64},
    }
    if feature_space == "pc":
        manifest_provenance["source_paths"].update({
            "pc_basis": str(config.pc_basis_root),
            "pc_loadings": str(config.pc_basis_root / "loadings.parquet"),
            "pc_explained_variance": str(
                config.pc_basis_root / "explained-variance.csv"
            ),
        })
        manifest_provenance["source_sha256"].update({
            "pc_loadings": "b" * 64,
            "pc_explained_variance": "c" * 64,
        })

    output = fit_fold(
        _pairs(), _responses(), dictionary, 0, config,
        manifest_provenance=manifest_provenance,
    )

    assert output == config.fit_dir / f"{feature_space}-{penalty}" / "model-fold-0__item-0"
    predictions = pd.read_parquet(output / "predictions.parquet")
    manifest = json.loads((output / "manifest.json").read_text())
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    tuning = pd.read_csv(output / "tuning.csv")
    assert predictions.probability.between(0, 1).all()
    assert {"judge", "pick_left"}.isdisjoint(predictions)
    assert manifest["source_paths"] == manifest_provenance["source_paths"]
    assert manifest["source_sha256"] == manifest_provenance["source_sha256"]
    ratio = diagnostics["benchmark_deviation_penalty"]
    assert np.isfinite(ratio) and ratio > 0
    assert ratio != config.benchmark_deviation_penalty
    variance = tuning[tuning.record_type == "variance_component"]
    selected = tuning[tuning.selected]
    assert len(variance) == len(selected) == 1
    assert variance.benchmark_deviation_penalty.item() == pytest.approx(ratio)
    assert selected.benchmark_deviation_penalty.item() == pytest.approx(ratio)
    assert manifest["frozen_hyperparameters"]["benchmark_deviation_penalty"] == ratio
    if feature_space == "raw":
        assert not any(key.startswith("pc_") for key in manifest["source_paths"])
        assert not any(key.startswith("pc_") for key in manifest["source_sha256"])


def test_global_pc_projection_is_fixed_while_values_respond_to_rows(tmp_path) -> None:
    dictionary = pd.DataFrame({
        "feature": ["signal", "other"], "block": ["surface", "surface"],
    })
    train = pd.DataFrame({"signal": [1.0, 2.0], "other": [3.0, 4.0]})
    test = pd.DataFrame({"signal": [5.0], "other": [6.0]})
    changed_train, changed_test = train * 10, test * -2
    config = _config(tmp_path, "pc")

    prepared_train, prepared_test, _, provenance = prepare_features(
        train, test, dictionary, "pc", config,
    )
    changed_prepared_train, changed_prepared_test, _, changed_provenance = prepare_features(
        changed_train, changed_test, dictionary, "pc", config,
    )

    pd.testing.assert_frame_equal(provenance["loadings"], changed_provenance["loadings"])
    pd.testing.assert_frame_equal(
        provenance["explained_variance"], changed_provenance["explained_variance"],
    )
    assert {
        key: value for key, value in provenance.items()
        if key not in {"loadings", "explained_variance"}
    } == {
        key: value for key, value in changed_provenance.items()
        if key not in {"loadings", "explained_variance"}
    }
    assert not prepared_train.equals(changed_prepared_train)
    assert not prepared_test.equals(changed_prepared_test)


def test_cli_passes_global_pc_settings_and_raw_needs_no_basis(
    monkeypatch, tmp_path,
) -> None:
    pin = "a" * 64
    responses = tmp_path / "responses.parquet"
    responses.write_bytes(b"pinned responses")
    basis = _pc_basis(tmp_path)
    guarded, fitted = [], []
    monkeypatch.setattr(pairwise_wrapper, "_guard_execution", guarded.append)
    monkeypatch.setattr(
        pairwise_wrapper, "_load_rows",
        lambda config: (pd.DataFrame(), pd.DataFrame()),
    )
    reads = []
    monkeypatch.setattr(
        pairwise_wrapper.pd, "read_parquet",
        lambda path: reads.append(path) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        pairwise_wrapper, "fit_fold",
        lambda pairs, responses, dictionary, fold_index, config, **kwargs:
        fitted.append((config, kwargs["manifest_provenance"])),
    )

    common = [
        "--matrix", str(tmp_path / "matrix"),
        "--matrix-manifest-sha256", pin,
        "--responses", str(responses),
        "--responses-sha256", _sha256(responses),
        "--output", str(tmp_path / "output"),
        "--fold-index", "3",
    ]
    arguments = [
        *common, "--feature-space", "pc", "--penalty", "ridge",
        "--pc-basis-root", str(basis), "--pc-components", "2",
        "--pc-loadings-sha256", _sha256(basis / "loadings.parquet"),
        "--pc-explained-variance-sha256", _sha256(basis / "explained-variance.csv"),
    ]
    pairwise_wrapper.main(arguments)

    assert guarded[0].processed_matrix_manifest_sha256 == pin
    assert guarded[0].glm_max_iter == 20_000
    assert guarded[0].pc_basis_root == basis
    assert guarded[0].fit_pca_components == 2
    assert fitted[0][0] is guarded[0]
    assert fitted[0][1] == {
        "source_paths": {
            "responses": str(responses),
            "pc_basis": str(basis),
            "pc_loadings": str(basis / "loadings.parquet"),
            "pc_explained_variance": str(basis / "explained-variance.csv"),
        },
        "source_sha256": {
            "responses": _sha256(responses),
            "pc_loadings": _sha256(basis / "loadings.parquet"),
            "pc_explained_variance": _sha256(basis / "explained-variance.csv"),
        },
    }

    pairwise_wrapper.main([
        *arguments, "--max-iter", "25000",
    ])
    assert guarded[1].glm_max_iter == 25_000
    assert guarded[1].fit_pca_components == 2
    assert fitted[1][0] is guarded[1]

    with pytest.raises(SystemExit):
        pairwise_wrapper.main([*arguments, "--max-iter", "0"])
    with pytest.raises(SystemExit):
        pairwise_wrapper.main([
            *common, "--feature-space", "pc", "--penalty", "ridge",
            "--pc-basis-root", str(basis),
        ])
    for option in ("--pc-loadings-sha256", "--pc-explained-variance-sha256"):
        mismatched_basis = arguments.copy()
        mismatched_basis[mismatched_basis.index(option) + 1] = "d" * 64
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            pairwise_wrapper.main(mismatched_basis)
    without_response_hash = [*common[:6], *common[8:]]
    with pytest.raises(SystemExit):
        pairwise_wrapper.main([
            *without_response_hash,
            "--feature-space", "raw", "--penalty", "ridge",
        ])
    mismatched_response = common.copy()
    mismatched_response[7] = "d" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pairwise_wrapper.main([
            *mismatched_response,
            "--feature-space", "raw", "--penalty", "ridge",
        ])
    assert reads == [responses, responses]

    pairwise_wrapper.main([
        *common, "--feature-space", "raw", "--penalty", "ridge",
    ])
    assert guarded[-1].pc_basis_root is None
    assert guarded[-1].fit_pca_components == pairwise_wrapper.CONFIG.fit_pca_components
    assert fitted[-1][0] is guarded[-1]
    assert fitted[-1][1] == {
        "source_paths": {"responses": str(responses)},
        "source_sha256": {"responses": _sha256(responses)},
    }
    with pytest.raises(SystemExit):
        pairwise_wrapper.main([
            *common, "--feature-space", "raw", "--penalty", "ridge",
            "--pc-basis-root", str(basis), "--pc-components", "2",
            "--pc-loadings-sha256", _sha256(basis / "loadings.parquet"),
            "--pc-explained-variance-sha256", _sha256(
                basis / "explained-variance.csv"
            ),
        ])
