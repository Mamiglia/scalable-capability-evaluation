from dataclasses import replace
import json

import numpy as np
import pandas as pd

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
