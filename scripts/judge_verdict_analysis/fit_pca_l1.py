"""Fold-local pooled PCA followed by a pure-L1 hierarchical verdict model."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, log_loss

from .config import AnalysisConfig, CONFIG
from .fit import _guard_execution, _load_rows
from .fit_common import FitResult
from .fit_frequentist import (
    _designs,
    _fit_model,
    _model,
    _parallel_map,
    _penalties,
    _splits,
)
from .fit_pca import (
    _atomic_write,
    _component_names,
    _file_sha256,
    _fold_name,
    _json_default,
    _project,
    _select_fold,
    _signature,
    fit_pooled_svd,
    transform_components,
)
from .hierarchy import HierarchyLevels, effective_coefficients, hierarchical_design


BACKEND = "pca-l1"
ARTIFACTS = (
    "predictions.parquet", "metrics.csv", "coefficients.csv", "selection.csv",
    "tuning.csv", "svd-loadings.csv", "svd-explained-variance.csv",
    "diagnostics.json", "manifest.json",
)
CSV_COLUMNS = {
    "metrics.csv": ["metric", "value"],
    "coefficients.csv": [
        "feature", "judge", "benchmark", "population", "judge_deviation",
        "benchmark_deviation", "cell_deviation", "effective",
    ],
    "selection.csv": ["feature", "selected"],
    "tuning.csv": [
        "stage", "record_type", "C", "l1_ratio", "mean_log_loss",
        "max_n_iter", "converged", "selected",
    ],
    "svd-loadings.csv": ["component", "feature", "loading"],
    "svd-explained-variance.csv": [
        "component", "explained_variance", "explained_variance_ratio",
        "cumulative_explained_variance_ratio", "singular_value",
    ],
}
DIAGNOSTIC_KEYS = {
    "l1_C", "l1_ratio", "l1_n_iter", "l1_converged",
    "l1_tuning_all_converged", "svd_algorithm", "svd_components",
    "svd_input_features", "svd_explained_variance_ratio_sum",
}
_DATA_SIGNATURE_KEYS = (
    "source_rows_sha256", "train_rows_sha256", "test_rows_sha256",
    "prediction_rows_sha256", "source_rows", "train_rows", "test_rows",
    "coefficient_cells",
)


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered values together with their column and dtype contract."""
    digest = hashlib.sha256(json.dumps([
        (str(column), str(dtype))
        for column, dtype in zip(frame.columns, frame.dtypes)
    ], separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(
        frame, index=False, categorize=True).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("fit_pca_l1.py", "fit_pca.py", "fit_frequentist.py", "hierarchy.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _data_signatures(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict[str, object]:
    expected_predictions = test.drop(
        columns=features + ["pick_left"], errors="ignore").copy()
    expected_predictions["observed"] = test.pick_left.to_numpy(dtype=int)
    cells = HierarchyLevels.from_training(train).cells
    return {
        "source_rows_sha256": _frame_sha256(rows),
        "train_rows_sha256": _frame_sha256(train),
        "test_rows_sha256": _frame_sha256(test),
        "prediction_rows_sha256": _frame_sha256(expected_predictions),
        "source_rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "coefficient_cells": [list(cell) for cell in cells],
    }


def _l1_model(config: AnalysisConfig, c: float):
    """Build the sklearn-version-compatible pure-L1 model."""
    return _model(config, c=c, l1_ratio=1.0)


def tune_l1(
    train: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
    levels: HierarchyLevels | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose pure-L1 C with grouped bench/item folds."""
    levels = levels or HierarchyLevels.from_training(train)
    y = train.pick_left.to_numpy(dtype=int)
    designs = [
        (fit_rows, check_rows, *_designs(
            train, fit_rows, check_rows, features, levels, config))
        for fit_rows, check_rows in _splits(train, config)
    ]

    def score(c: float) -> tuple[float, float, int, bool]:
        losses, iterations = [], []
        for fit_rows, check_rows, fit_x, check_x, _ in designs:
            model, n_iter, converged = _fit_model(
                _l1_model(config, c), fit_x, y[fit_rows])
            iterations.append(n_iter)
            losses.append(log_loss(
                y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
        return c, float(np.mean(losses)), max(iterations), all(
            n_iter < config.glm_max_iter for n_iter in iterations)

    scores = _parallel_map(score, config.elastic_net_cs, config.fit_n_jobs)
    best_c = min(scores, key=lambda result: result[1])[0]
    tuning = pd.DataFrame([
        {
            "stage": "l1", "record_type": "tuning", "C": c,
            "l1_ratio": 1.0, "mean_log_loss": loss,
            "max_n_iter": max_n_iter, "converged": converged,
            "selected": c == best_c,
        }
        for c, loss, max_n_iter, converged in scores
    ])
    return best_c, tuning


def fit_fixed_l1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
    c: float,
    tuning: pd.DataFrame,
) -> tuple[FitResult, pd.DataFrame]:
    """Fit one final pure-L1 model and report its selected PC slopes."""
    folds = pd.concat([train.fold, test.fold]).drop_duplicates().tolist()
    if len(folds) != 1:
        raise ValueError("train and test must contain exactly one shared fold")
    levels = HierarchyLevels.from_training(train)
    train_x, column_map = hierarchical_design(
        train, features, levels, _penalties(config), penalty="l1")
    test_x, _ = hierarchical_design(
        test, features, levels, _penalties(config), penalty="l1")
    model, n_iter, converged = _fit_model(
        _l1_model(config, c), train_x, train.pick_left.to_numpy(dtype=int))
    raw = model.coef_[0]
    active = set(column_map.loc[
        column_map.block.str.endswith("_slope") & (raw != 0), "feature"
    ].dropna())
    selection = pd.DataFrame({
        "feature": features,
        "selected": [feature in active for feature in features],
    })
    probability = model.predict_proba(test_x)[:, 1]
    observed = test.pick_left.to_numpy(dtype=int)
    predictions = test.drop(columns=features + ["pick_left"], errors="ignore").copy()
    predictions["observed"] = observed
    predictions["probability"] = probability
    result = FitResult(
        backend=BACKEND,
        fold=folds[0],
        selected_features=tuple(selection.loc[selection.selected, "feature"]),
        predictions=predictions,
        coefficients=effective_coefficients(raw, column_map),
        tuning=tuning,
        metrics=pd.DataFrame({
            "metric": ["accuracy", "log_loss"],
            "value": [
                accuracy_score(observed, probability >= 0.5),
                log_loss(observed, probability, labels=[0, 1]),
            ],
        }),
        diagnostics={
            "l1_C": c,
            "l1_ratio": 1.0,
            "l1_n_iter": n_iter,
            "l1_converged": converged,
            "l1_tuning_all_converged": bool(tuning.converged.all()),
        },
    )
    return result, selection


def fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_components: int,
    algorithm: str,
    config: AnalysisConfig,
) -> tuple[FitResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit outer-train SVD, tune L1 C, then keep L1 for the final fit."""
    model, loadings, variance = fit_pooled_svd(
        train, features, n_components, algorithm, config.random_seed)
    component_names = _component_names(n_components)
    projected_train = _project(train, features, model)
    projected_test = _project(test, features, model)
    best_c, tuning = tune_l1(projected_train, component_names, config)
    result, selection = fit_fixed_l1(
        projected_train, projected_test, component_names, config, best_c, tuning)
    result.diagnostics.update({
        "svd_algorithm": algorithm,
        "svd_components": n_components,
        "svd_input_features": len(features),
        "svd_explained_variance_ratio_sum": float(
            variance.explained_variance_ratio.sum()),
    })
    return result, selection, loadings, variance


def _settings(
    config: AnalysisConfig,
    fold_index: int,
    features: list[str],
    data_signatures: dict[str, object] | None = None,
) -> dict[str, object]:
    """Bind a fold to its source, inventory, hierarchy, and L1 settings."""
    settings = {
        "schema_version": 1,
        "implementation_sha256": _implementation_sha256(),
        "sklearn_version": sklearn.__version__,
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "require_pinned_source_for_fit": config.require_pinned_source_for_fit,
        "processed_matrix_manifest_sha256": config.processed_matrix_manifest_sha256,
        "fold_index": fold_index,
        "model_folds": [[name, list(models)] for name, models in config.model_folds],
        "item_folds": config.item_folds,
        "feature_blocks": list(config.feature_blocks),
        "regressors": list(config.regressors),
        "judges": list(config.judges),
        "benchmarks": list(config.benchmarks),
        "splits": list(config.splits),
        "fit_pca_components": config.fit_pca_components,
        "fit_pca_algorithm": config.fit_pca_algorithm,
        "inner_folds": config.inner_folds,
        "fit_n_jobs": config.fit_n_jobs,
        "elastic_net_cs": list(config.elastic_net_cs),
        "l1_ratio": 1.0,
        "judge_deviation_penalty": config.judge_deviation_penalty,
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "cell_deviation_penalty": config.cell_deviation_penalty,
        "item_intercept_penalty": config.item_intercept_penalty,
        "glm_max_iter": config.glm_max_iter,
        "random_seed": config.random_seed,
        "features": features,
        "feature_inventory_sha256": _signature({"features": features}),
    }
    if data_signatures is not None:
        if set(data_signatures) != set(_DATA_SIGNATURE_KEYS):
            raise ValueError("data_signatures has the wrong keys")
        settings.update(data_signatures)
    return settings


def _expected_fold_name(settings: dict[str, object]) -> str:
    model_index, item_fold = divmod(settings["fold_index"], settings["item_folds"])
    return f"{settings['model_folds'][model_index][0]}__item-{item_fold}"


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and set(value.lower()) <= set("0123456789abcdef")
    )


def _validate_fold(path: Path, settings: dict[str, object]) -> dict[str, object]:
    """Validate schemas, hashes, provenance, and model-specific semantics."""
    if missing := [name for name in ARTIFACTS if not (path / name).exists()]:
        raise RuntimeError(f"incomplete PCA-L1 fold checkpoint {path}; missing {missing}")
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        predictions = pd.read_parquet(path / "predictions.parquet")
        tables = {name: pd.read_csv(path / name) for name in CSV_COLUMNS}
        diagnostics = json.loads((path / "diagnostics.json").read_text())
    except Exception as error:
        raise RuntimeError(f"invalid PCA-L1 fold checkpoint: {path}") from error
    components = _component_names(int(settings["fit_pca_components"]))
    features = settings["features"]
    cells = settings.get("coefficient_cells")
    expected_rows = {
        "predictions.parquet": settings.get("test_rows"),
        "metrics.csv": 2,
        "coefficients.csv": len(components) * len(cells) if isinstance(cells, list) else -1,
        "selection.csv": len(components),
        "tuning.csv": len(settings["elastic_net_cs"]),
        "svd-loadings.csv": len(components) * len(features),
        "svd-explained-variance.csv": len(components),
    }
    hashes = manifest.get("artifact_sha256")
    prediction_columns = manifest.get("prediction_columns")
    if (
        manifest.get("state") != "completed"
        or manifest.get("backend") != BACKEND
        or manifest.get("fold") != _expected_fold_name(settings)
        or manifest.get("fold_index") != settings["fold_index"]
        or manifest.get("settings") != settings
        or manifest.get("settings_signature") != _signature(settings)
        or manifest.get("component_count") != len(components)
        or manifest.get("input_features") != features
        or any(not _valid_sha256(settings.get(key)) for key in (
            "implementation_sha256", "source_rows_sha256", "train_rows_sha256",
            "test_rows_sha256", "prediction_rows_sha256",
        ))
        or not isinstance(settings.get("sklearn_version"), str)
        or not settings["sklearn_version"]
        or not isinstance(manifest.get("train_rows"), int)
        or manifest["train_rows"] != settings.get("train_rows")
        or not isinstance(manifest.get("test_rows"), int)
        or manifest["test_rows"] != settings.get("test_rows")
        or not isinstance(settings.get("source_rows"), int)
        or settings["source_rows"] <= 0
        or not isinstance(settings.get("train_rows"), int)
        or settings["train_rows"] <= 0
        or not isinstance(settings.get("test_rows"), int)
        or settings["test_rows"] <= 0
        or not isinstance(cells, list)
        or not cells
        or any(not isinstance(cell, list) or len(cell) != 2
               or not all(isinstance(value, str) for value in cell)
               for cell in cells)
        or len({tuple(cell) for cell in cells}) != len(cells)
        or not isinstance(manifest.get("coefficient_rows"), int)
        or manifest["coefficient_rows"] != expected_rows["coefficients.csv"]
        or not isinstance(manifest.get("selected_features"), list)
        or not isinstance(prediction_columns, list)
        or predictions.columns.tolist() != prediction_columns
        or not {"observed", "probability"} <= set(prediction_columns)
        or not isinstance(hashes, dict)
        or hashes != {
            name: _file_sha256(path / name)
            for name in ARTIFACTS if name != "manifest.json"
        }
        or any(tables[name].columns.tolist() != columns
               for name, columns in CSV_COLUMNS.items())
        or len(predictions) != expected_rows["predictions.parquet"]
        or any(len(tables[name]) != expected_rows[name] for name in CSV_COLUMNS)
    ):
        raise RuntimeError(f"PCA-L1 fold checkpoint settings or schema mismatch: {path}")

    metrics = tables["metrics.csv"]
    coefficients = tables["coefficients.csv"]
    selection = tables["selection.csv"]
    tuning = tables["tuning.csv"]
    loadings = tables["svd-loadings.csv"]
    variance = tables["svd-explained-variance.csv"]
    try:
        observed = predictions.observed.to_numpy()
        probability = predictions.probability.to_numpy(dtype=float)
        metric_values = metrics.set_index("metric").value.astype(float)
        coefficient_values = coefficients[
            CSV_COLUMNS["coefficients.csv"][3:]
        ].to_numpy(float)
        tuning_loss = tuning.mean_log_loss.to_numpy(float)
        tuning_iterations = tuning.max_n_iter.to_numpy(float)
        tuning_cs = tuning.C.to_numpy(float)
        loading_values = loadings.loading.to_numpy(float)
        explained = variance.explained_variance.to_numpy(float)
        ratios = variance.explained_variance_ratio.to_numpy(float)
        cumulative = variance.cumulative_explained_variance_ratio.to_numpy(float)
        singular = variance.singular_value.to_numpy(float)
        diagnostic_ratio = float(diagnostics["svd_explained_variance_ratio_sum"])
        final_iterations = diagnostics["l1_n_iter"]
        final_c = float(diagnostics["l1_C"])
        expected_accuracy = accuracy_score(observed, probability >= 0.5)
        expected_log_loss = log_loss(observed, probability, labels=[0, 1])
        effective = coefficients[[
            "population", "judge_deviation", "benchmark_deviation",
            "cell_deviation",
        ]].sum(axis=1).to_numpy(float)
        loading_matrix = loadings.pivot(
            index="component", columns="feature", values="loading"
        ).loc[components, features].to_numpy(float)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid PCA-L1 fold artifact contents: {path}") from error
    selected = selection.loc[selection.selected, "feature"].tolist()
    active = coefficients.loc[
        (coefficients[[
            "population", "judge_deviation", "benchmark_deviation",
            "cell_deviation",
        ]] != 0).any(axis=1),
        "feature",
    ].drop_duplicates().tolist()
    tolerance = 1e-10
    if (
        not np.isin(observed, [0, 1]).all()
        or _frame_sha256(predictions.drop(columns="probability"))
        != settings["prediction_rows_sha256"]
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or set(metrics.metric) != {"accuracy", "log_loss"}
        or not np.isfinite(metric_values).all()
        or not 0 <= metric_values["accuracy"] <= 1
        or metric_values["log_loss"] < 0
        or not np.isclose(
            metric_values["accuracy"], expected_accuracy, atol=tolerance, rtol=0)
        or not np.isclose(
            metric_values["log_loss"], expected_log_loss, atol=tolerance, rtol=0)
        or set(coefficients.feature) != set(components)
        or coefficients[["feature", "judge", "benchmark"]].values.tolist()
        != [
            [component, judge, benchmark]
            for component in components for judge, benchmark in cells
        ]
        or not np.isfinite(coefficient_values).all()
        or not np.allclose(
            coefficients.effective.to_numpy(float), effective,
            atol=tolerance, rtol=0)
        or selection.feature.tolist() != components
        or not pd.api.types.is_bool_dtype(selection.selected)
        or selected != [component for component in components if component in active]
        or selected != manifest["selected_features"]
        or set(tuning.stage) != {"l1"}
        or set(tuning.record_type) != {"tuning"}
        or tuning.C.tolist() != settings["elastic_net_cs"]
        or set(tuning.l1_ratio.to_numpy(float)) != {1.0}
        or not pd.api.types.is_bool_dtype(tuning.converged)
        or not pd.api.types.is_bool_dtype(tuning.selected)
        or tuning.selected.sum() != 1
        or not np.array_equal(
            tuning.selected.to_numpy(),
            np.arange(len(tuning)) == int(np.argmin(tuning_loss)),
        )
        or not np.isfinite(tuning_loss).all()
        or np.any(tuning_loss < 0)
        or not np.isfinite(tuning_iterations).all()
        or np.any(tuning_iterations != np.floor(tuning_iterations))
        or np.any((tuning_iterations < 1) | (tuning_iterations > settings["glm_max_iter"]))
        or not np.array_equal(
            tuning.converged.to_numpy(),
            tuning_iterations < settings["glm_max_iter"],
        )
        or final_c != tuning_cs[tuning.selected.to_numpy()][0]
        or set(loadings.component) != set(components)
        or set(loadings.feature) != set(features)
        or loadings[["component", "feature"]].duplicated().any()
        or not np.isfinite(loading_values).all()
        or not np.allclose(
            loading_matrix @ loading_matrix.T, np.eye(len(components)),
            atol=1e-10, rtol=0)
        or variance.component.tolist() != components
        or not np.isfinite(explained).all()
        or np.any(explained < -tolerance)
        or not np.isfinite(ratios).all()
        or np.any((ratios < -tolerance) | (ratios > 1 + tolerance))
        or not np.isfinite(cumulative).all()
        or np.any((cumulative < -tolerance) | (cumulative > 1 + tolerance))
        or np.any(np.diff(cumulative) < -tolerance)
        or not np.allclose(cumulative, np.cumsum(ratios), atol=tolerance, rtol=0)
        or not np.isfinite(singular).all()
        or np.any(singular < -tolerance)
        or not isinstance(diagnostics, dict)
        or not DIAGNOSTIC_KEYS <= set(diagnostics)
        or diagnostics["svd_algorithm"] != settings["fit_pca_algorithm"]
        or diagnostics["svd_components"] != len(components)
        or diagnostics["svd_input_features"] != len(features)
        or final_c not in settings["elastic_net_cs"]
        or float(diagnostics["l1_ratio"]) != 1.0
        or not isinstance(diagnostics["l1_converged"], bool)
        or type(final_iterations) is not int
        or not 1 <= final_iterations <= settings["glm_max_iter"]
        or diagnostics["l1_converged"] != (
            final_iterations < settings["glm_max_iter"])
        or not isinstance(diagnostics["l1_tuning_all_converged"], bool)
        or diagnostics["l1_tuning_all_converged"] != bool(tuning.converged.all())
        or not np.isfinite(diagnostic_ratio)
        or not -tolerance <= diagnostic_ratio <= 1 + tolerance
        or not np.isclose(diagnostic_ratio, ratios.sum(), atol=tolerance, rtol=0)
    ):
        raise RuntimeError(f"invalid PCA-L1 fold artifact contents: {path}")
    return manifest


def _write_fold(
    path: Path,
    result: FitResult,
    selection: pd.DataFrame,
    loadings: pd.DataFrame,
    variance: pd.DataFrame,
    fold_index: int,
    train_rows: int,
    test_rows: int,
    features: list[str],
    settings: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    result.predictions.to_parquet(partial / "predictions.parquet", index=False)
    result.metrics.to_csv(partial / "metrics.csv", index=False)
    result.coefficients.to_csv(partial / "coefficients.csv", index=False)
    selection.to_csv(partial / "selection.csv", index=False)
    result.tuning.to_csv(partial / "tuning.csv", index=False)
    loadings.to_csv(partial / "svd-loadings.csv", index=False)
    variance.to_csv(partial / "svd-explained-variance.csv", index=False)
    (partial / "diagnostics.json").write_text(json.dumps(
        result.diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n")
    manifest = {
        "state": "completed", "backend": BACKEND, "fold": result.fold,
        "fold_index": fold_index, "train_rows": train_rows, "test_rows": test_rows,
        "component_count": len(selection), "input_features": features,
        "selected_features": list(result.selected_features),
        "coefficient_rows": len(result.coefficients),
        "prediction_columns": result.predictions.columns.tolist(),
        "settings": settings, "settings_signature": _signature(settings),
        "artifacts": list(ARTIFACTS),
        "artifact_sha256": {
            name: _file_sha256(partial / name)
            for name in ARTIFACTS if name != "manifest.json"
        },
    }
    (partial / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _validate_fold(partial, settings)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite fold checkpoint: {path}")
    os.replace(partial, path)


def run_fold(fold_index: int, config: AnalysisConfig = CONFIG) -> Path:
    """Run or validate one immutable model-by-item outer-fold checkpoint."""
    config = config.validate()
    fold = _fold_name(fold_index, config)
    output = config.fit_dir / BACKEND / "folds" / fold
    _guard_execution(config)
    rows, features = _load_rows(config)
    selected_fold, train, test = _select_fold(rows, fold_index, config)
    settings = _settings(
        config, fold_index, features,
        _data_signatures(rows, train, test, features))
    if output.exists():
        _validate_fold(output, settings)
        return output
    if selected_fold != fold:
        raise AssertionError("selected fold name differs from configured fold")
    result, selection, loadings, variance = fit_fold(
        train, test, features, config.fit_pca_components,
        config.fit_pca_algorithm, config)
    _write_fold(
        output, result, selection, loadings, variance, fold_index,
        len(train), len(test), features, settings)
    return output


def _aggregate_contents(
    fold_names: list[str], fold_dirs: list[Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    tables = {
        "predictions.parquet": pd.concat([
            pd.read_parquet(path / "predictions.parquet") for path in fold_dirs
        ], ignore_index=True),
    }
    for name in CSV_COLUMNS:
        frames = []
        for fold, path in zip(fold_names, fold_dirs):
            frame = pd.read_csv(path / name)
            if "fold" not in frame:
                frame.insert(0, "fold", fold)
            frames.append(frame)
        tables[name] = pd.concat(frames, ignore_index=True)
    diagnostics = {
        fold: json.loads((path / "diagnostics.json").read_text())
        for fold, path in zip(fold_names, fold_dirs)
    }
    return tables, diagnostics


def _validate_aggregate(
    output: Path,
    settings: dict[str, object],
    fold_names: list[str],
    fold_dirs: list[Path],
) -> dict[str, object]:
    if missing := [name for name in ARTIFACTS if not (output / name).exists()]:
        raise RuntimeError(
            f"incomplete PCA-L1 aggregate checkpoint {output}; missing {missing}")
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        actual = {
            "predictions.parquet": pd.read_parquet(output / "predictions.parquet"),
            **{name: pd.read_csv(output / name) for name in CSV_COLUMNS},
        }
        actual_diagnostics = json.loads((output / "diagnostics.json").read_text())
        expected, expected_diagnostics = _aggregate_contents(fold_names, fold_dirs)
        for name in actual:
            pd.testing.assert_frame_equal(
                actual[name], expected[name], check_dtype=False,
                check_exact=False, rtol=1e-12, atol=1e-12)
    except Exception as error:
        raise RuntimeError(f"invalid PCA-L1 aggregate checkpoint: {output}") from error
    artifact_rows = {name: len(table) for name, table in actual.items()}
    artifact_columns = {name: table.columns.tolist() for name, table in actual.items()}
    hashes = {
        name: _file_sha256(output / name)
        for name in ARTIFACTS if name != "manifest.json"
    }
    if (
        manifest.get("state") != "completed"
        or manifest.get("backend") != BACKEND
        or manifest.get("fold_count") != len(fold_dirs)
        or manifest.get("folds") != fold_names
        or manifest.get("settings") != settings
        or manifest.get("settings_signature") != _signature(settings)
        or manifest.get("artifact_rows") != artifact_rows
        or manifest.get("artifact_columns") != artifact_columns
        or manifest.get("artifact_sha256") != hashes
        or actual_diagnostics != expected_diagnostics
    ):
        raise RuntimeError(f"PCA-L1 aggregate contents mismatch: {output}")
    return manifest


def aggregate(config: AnalysisConfig = CONFIG) -> Path:
    """Combine every configured outer-fold shard into deterministic artifacts."""
    config = config.validate()
    output = config.fit_dir / BACKEND
    fold_names = [
        f"{model_name}__item-{item_fold}"
        for model_name, _ in config.model_folds
        for item_fold in range(config.item_folds)
    ]
    fold_dirs = [output / "folds" / name for name in fold_names]
    manifests = []
    try:
        for index, path in enumerate(fold_dirs):
            saved = json.loads((path / "manifest.json").read_text())
            features = saved.get("input_features")
            if not isinstance(features, list) or not all(
                isinstance(feature, str) for feature in features
            ):
                raise RuntimeError(f"invalid PCA-L1 fold feature inventory: {path}")
            saved_settings = saved.get("settings")
            if not isinstance(saved_settings, dict):
                raise RuntimeError(f"invalid PCA-L1 fold settings: {path}")
            signatures = {key: saved_settings.get(key) for key in _DATA_SIGNATURE_KEYS}
            manifests.append(_validate_fold(
                path, _settings(config, index, features, signatures)))
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("incomplete PCA-L1 fold inventory") from error
    inventories = [
        (
            manifest["input_features"],
            manifest["settings"]["feature_inventory_sha256"],
            manifest["settings"]["source_rows_sha256"],
        )
        for manifest in manifests
    ]
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise RuntimeError(
            "completed PCA-L1 folds do not share one ordered feature inventory")
    settings = {
        "schema_version": 1,
        "fold_settings_signatures": [
            manifest["settings_signature"] for manifest in manifests
        ],
        "features": inventories[0][0],
        "feature_inventory_sha256": inventories[0][1],
        "source_rows_sha256": inventories[0][2],
    }
    completion = output / "manifest.json"
    if completion.exists():
        _validate_aggregate(output, settings, fold_names, fold_dirs)
        return output
    output.mkdir(parents=True, exist_ok=True)
    tables, diagnostics = _aggregate_contents(fold_names, fold_dirs)
    _atomic_write(
        output / "predictions.parquet",
        lambda path: tables["predictions.parquet"].to_parquet(path, index=False))
    for name in CSV_COLUMNS:
        frame = tables[name]
        _atomic_write(
            output / name,
            lambda path, frame=frame: frame.to_csv(path, index=False))
    _atomic_write(
        output / "diagnostics.json",
        lambda path: path.write_text(json.dumps(
            diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n"))
    manifest = {
        "state": "completed", "backend": BACKEND, "fold_count": len(fold_dirs),
        "folds": fold_names, "settings": settings,
        "settings_signature": _signature(settings), "artifacts": list(ARTIFACTS),
        "artifact_rows": {name: len(table) for name, table in tables.items()},
        "artifact_columns": {
            name: table.columns.tolist() for name, table in tables.items()
        },
    }
    manifest["artifact_sha256"] = {
        name: _file_sha256(output / name)
        for name in ARTIFACTS if name != "manifest.json"
    }
    _atomic_write(
        completion,
        lambda path: path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"))
    _validate_aggregate(output, settings, fold_names, fold_dirs)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fold = commands.add_parser("fold", help="fit one model-by-item outer fold")
    fold.add_argument("--fold-index", type=int, required=True)
    fold.add_argument("--processed-matrix-root", type=Path, required=True)
    fold.add_argument("--results-dir", type=Path, required=True)
    combined = commands.add_parser(
        "aggregate", help="combine all completed fold shards")
    combined.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fold":
        run_fold(args.fold_index, replace(
            CONFIG, processed_matrix_root=args.processed_matrix_root,
            results_dir=args.results_dir, execution_mode="execute",
            execution_target="swc_hpc"))
    else:
        aggregate(replace(CONFIG, results_dir=args.results_dir))


if __name__ == "__main__":
    main()
