from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from .config import CONFIG
from . import fit_pca


FEATURES = ["f0", "f1", "f2"]


def _rows() -> pd.DataFrame:
    rows = []
    for item_index in range(8):
        for judge_index, judge in enumerate(("j0", "j1")):
            sign = -1.0 if (item_index + judge_index) % 2 else 1.0
            for model_left in ("held", "seen"):
                rows.append({
                    "bench": "b0", "judge": judge, "item": f"i{item_index}",
                    "model_left": model_left, "model_right": "seen",
                    "fold": "held-models__item-0", "pick_left": int(sign > 0),
                    "f0": sign, "f1": sign * (item_index + 1),
                    "f2": float((item_index % 3) - 1),
                })
    return pd.DataFrame(rows)


def _config(tmp_path=None):
    return replace(
        CONFIG,
        results_dir=tmp_path or CONFIG.results_dir,
        judges=("j0", "j1"), benchmarks=("b0",),
        model_folds=(("held-models", ("held",)),), item_folds=2,
        inner_folds=2, ridge_cs=(1.0,), fit_n_jobs=1, glm_max_iter=1_000,
        fit_pca_components=2, fit_pca_algorithm="randomized",
        execution_mode="execute", execution_target="local_test",
        local_test_max_rows=100,
    )


def _train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _rows()
    train = rows[rows.model_left == "seen"].reset_index(drop=True)
    test = rows[rows.model_left == "held"].reset_index(drop=True)
    return train, test


def test_outer_test_rows_cannot_change_fitted_components() -> None:
    train, test = _train_test()
    changed = test.copy()
    changed[FEATURES] = changed[FEATURES] * 1_000_000 + 12345
    config = _config()

    _, loadings, variance = fit_pca.fit_fold(
        train, test, FEATURES, 2, "randomized", config)
    _, changed_loadings, changed_variance = fit_pca.fit_fold(
        train, changed, FEATURES, 2, "randomized", config)

    pd.testing.assert_frame_equal(loadings, changed_loadings)
    pd.testing.assert_frame_equal(variance, changed_variance)


def test_uncentred_component_transform_is_exactly_antisymmetric() -> None:
    train, test = _train_test()
    model, _, _ = fit_pca.fit_pooled_svd(
        train, FEATURES, 2, "randomized", random_seed=7)
    reversed_test = test.copy()
    reversed_test[FEATURES] *= -1

    transformed = fit_pca.transform_components(test, FEATURES, model)
    reversed_transformed = fit_pca.transform_components(
        reversed_test, FEATURES, model)

    np.testing.assert_array_equal(
        reversed_transformed.to_numpy(), -transformed.to_numpy())


def test_fold_fit_uses_ridge_only() -> None:
    train, test = _train_test()

    result, loadings, variance = fit_pca.fit_fold(
        train, test, FEATURES, 2, "randomized", _config())

    assert result.backend == "pca-ridge"
    assert result.selected_features == ("pc_01", "pc_02")
    assert set(result.tuning.stage) == {"ridge"}
    assert "l1_ratio" not in result.tuning
    assert not any("elastic" in key or "hibayes" in key for key in result.diagnostics)
    assert loadings.shape == (2 * len(FEATURES), 3)
    assert variance.shape == (2, 5)
    assert len(result.predictions) == len(test)


def test_fold_checkpoints_and_aggregation_have_complete_artifacts(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    monkeypatch.setattr(fit_pca, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca, "_load_rows", lambda _: (rows, FEATURES))
    abandoned = (
        config.fit_dir / "pca-ridge" / "folds" /
        ".held-models__item-0.partial")
    abandoned.mkdir(parents=True)
    (abandoned / "interrupted.txt").write_text("preempted\n")

    fold_dir = fit_pca.run_fold(0, config)
    fit_pca.run_fold(1, config)

    assert sorted(path.name for path in fold_dir.iterdir()) == [
        "coefficients.csv", "diagnostics.json", "manifest.json", "metrics.csv",
        "predictions.parquet", "svd-explained-variance.csv", "svd-loadings.csv",
        "tuning.csv",
    ]
    manifest = json.loads((fold_dir / "manifest.json").read_text())
    assert manifest["state"] == "completed"
    assert manifest["backend"] == "pca-ridge"
    assert manifest["component_count"] == 2
    assert pd.read_csv(fold_dir / "svd-loadings.csv").shape == (6, 3)
    assert pd.read_csv(fold_dir / "svd-explained-variance.csv").shape == (2, 5)

    monkeypatch.setattr(
        fit_pca, "fit_fold", lambda *_: (_ for _ in ()).throw(
            AssertionError("completed fold was recomputed")))
    assert fit_pca.run_fold(0, config) == fold_dir

    summary_dir = fit_pca.aggregate(config)
    assert sorted(path.name for path in summary_dir.iterdir() if path.is_file()) == [
        "coefficients.csv", "diagnostics.json", "manifest.json", "metrics.csv",
        "predictions.parquet", "svd-explained-variance.csv", "svd-loadings.csv",
        "tuning.csv",
    ]
    assert len(pd.read_parquet(summary_dir / "predictions.parquet")) == 16
    assert json.loads((summary_dir / "manifest.json").read_text())["fold_count"] == 2


def test_resume_rejects_a_different_source_revision_or_feature_inventory(
    tmp_path, monkeypatch,
) -> None:
    config = replace(_config(tmp_path), source_hf_revision="a" * 40)
    rows = _rows().drop(columns="fold")
    inventory = [rows, FEATURES]
    monkeypatch.setattr(fit_pca, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca, "_load_rows", lambda _: tuple(inventory))
    fit_pca.run_fold(0, config)

    with pytest.raises(RuntimeError, match="settings"):
        fit_pca.run_fold(0, replace(config, source_hf_revision="b" * 40))

    inventory[0] = rows.assign(f3=rows.f0 * rows.f2)
    inventory[1] = [*FEATURES, "f3"]
    with pytest.raises(RuntimeError, match="settings"):
        fit_pca.run_fold(0, config)


@pytest.mark.parametrize("artifact", [
    "predictions.parquet", "metrics.csv", "coefficients.csv", "tuning.csv",
    "svd-explained-variance.csv", "svd-loadings.csv", "diagnostics.json",
    "missing-metrics",
])
def test_resume_rejects_corrupt_or_incomplete_fold_artifacts(
    tmp_path, monkeypatch, artifact,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    monkeypatch.setattr(fit_pca, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca, "_load_rows", lambda _: (rows, FEATURES))
    fold_dir = fit_pca.run_fold(0, config)
    path = fold_dir / artifact
    if artifact == "predictions.parquet":
        pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    elif artifact == "diagnostics.json":
        values = json.loads(path.read_text())
        values.pop("ridge_C")
        path.write_text(json.dumps(values))
    elif artifact == "missing-metrics":
        (fold_dir / "metrics.csv").unlink()
    else:
        table = pd.read_csv(path)
        if artifact == "metrics.csv":
            table = table.drop(columns="metric")
        elif artifact == "coefficients.csv":
            table = table.drop(columns="effective")
        elif artifact == "tuning.csv":
            table.loc[:, "stage"] = "elastic_net"
        elif artifact == "svd-explained-variance.csv":
            table = table.iloc[:-1]
        else:
            table = table.iloc[:-1]
        table.to_csv(path, index=False)

    with pytest.raises(RuntimeError):
        fit_pca.run_fold(0, config)


def test_aggregate_resume_rejects_content_that_differs_from_fold_shards(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    monkeypatch.setattr(fit_pca, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca, "_load_rows", lambda _: (rows, FEATURES))
    fit_pca.run_fold(0, config)
    fit_pca.run_fold(1, config)
    output = fit_pca.aggregate(config)
    metrics = pd.read_csv(output / "metrics.csv")
    metrics.loc[0, "value"] += 0.25
    metrics.to_csv(output / "metrics.csv", index=False)

    with pytest.raises(RuntimeError, match="aggregate"):
        fit_pca.aggregate(config)


@pytest.mark.parametrize("features", [
    list(reversed(FEATURES)), [*FEATURES, "f3"],
])
def test_aggregate_rejects_mixed_ordered_feature_inventories(
    tmp_path, monkeypatch, features,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    inventory = [rows, FEATURES]
    monkeypatch.setattr(fit_pca, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca, "_load_rows", lambda _: tuple(inventory))
    fit_pca.run_fold(0, config)
    inventory[0] = rows.assign(f3=rows.f0 * rows.f2)
    inventory[1] = features
    fit_pca.run_fold(1, config)

    with pytest.raises(RuntimeError, match="feature inventory"):
        fit_pca.aggregate(config)


@pytest.mark.parametrize(("artifact", "column", "value"), [
    ("predictions.parquet", "observed", 0.5),
    ("predictions.parquet", "probability", 1.1),
    ("metrics.csv", "accuracy", 1.1),
    ("metrics.csv", "log_loss", -0.1),
    ("tuning.csv", "mean_log_loss", -0.1),
    ("tuning.csv", "max_n_iter", 1.5),
    ("tuning.csv", "converged", 1),
    ("tuning.csv", "converged", False),
    ("tuning.csv", "max_n_iter", 1_000),
    ("svd-explained-variance.csv", "explained_variance", -0.1),
    ("diagnostics.json", "svd_explained_variance_ratio_sum", 1.1),
    ("diagnostics.json", "ridge_n_iter", 1.5),
    ("diagnostics.json", "ridge_converged", 1),
    ("diagnostics.json", "ridge_converged", False),
    ("diagnostics.json", "ridge_n_iter", 1_000),
])
def test_resume_rejects_coherent_but_impossible_artifact_values(
    tmp_path, monkeypatch, artifact, column, value,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    monkeypatch.setattr(fit_pca, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca, "_load_rows", lambda _: (rows, FEATURES))
    fold_dir = fit_pca.run_fold(0, config)
    path = fold_dir / artifact
    if artifact == "predictions.parquet":
        table = pd.read_parquet(path)
        if column == "observed":
            table[column] = table[column].astype(float)
        table.loc[0, column] = value
        table.to_parquet(path, index=False)
    elif artifact == "diagnostics.json":
        values = json.loads(path.read_text())
        values[column] = value
        path.write_text(json.dumps(values))
    else:
        table = pd.read_csv(path)
        if artifact == "metrics.csv":
            table.loc[table.metric == column, "value"] = value
        else:
            if column == "max_n_iter":
                table[column] = table[column].astype(float)
            elif column in {"converged", "selected"}:
                table[column] = table[column].astype(object)
            table.loc[0, column] = value
        table.to_csv(path, index=False)
    manifest_path = fold_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][artifact] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="artifact contents"):
        fit_pca.run_fold(0, config)
