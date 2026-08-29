"""Fold-local pooled TruncatedSVD followed by hierarchical ridge logistic fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .config import AnalysisConfig, CONFIG
from .fit import _guard_execution, _load_rows, _outer_folds
from .fit_common import FitResult
from .fit_frequentist import fit_fixed_ridge, tune_ridge


BACKEND = "pca-ridge"
ARTIFACTS = (
    "predictions.parquet", "metrics.csv", "coefficients.csv", "tuning.csv",
    "svd-loadings.csv", "svd-explained-variance.csv", "diagnostics.json",
    "manifest.json",
)
CSV_COLUMNS = {
    "metrics.csv": ["metric", "value"],
    "coefficients.csv": [
        "feature", "judge", "benchmark", "population", "judge_deviation",
        "benchmark_deviation", "cell_deviation", "effective",
    ],
    "tuning.csv": [
        "stage", "record_type", "C", "mean_log_loss", "max_n_iter",
        "converged", "selected",
    ],
    "svd-loadings.csv": ["component", "feature", "loading"],
    "svd-explained-variance.csv": [
        "component", "explained_variance", "explained_variance_ratio",
        "cumulative_explained_variance_ratio", "singular_value",
    ],
}
DIAGNOSTIC_KEYS = {
    "ridge_C", "ridge_n_iter", "ridge_converged", "svd_algorithm",
    "svd_components", "svd_input_features",
    "svd_explained_variance_ratio_sum",
}


def _component_names(count: int) -> list[str]:
    return [f"pc_{index:02d}" for index in range(1, count + 1)]


def _values(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be non-empty and unique")
    if missing := sorted(set(features) - set(frame)):
        raise ValueError(f"frame missing features: {missing}")
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("component features must be finite")
    return values


def fit_pooled_svd(
    train: pd.DataFrame,
    features: list[str],
    n_components: int,
    algorithm: str,
    random_seed: int,
) -> tuple[TruncatedSVD, pd.DataFrame, pd.DataFrame]:
    """Fit one uncentred SVD to pooled outer-training differences only."""
    values = _values(train, features)
    if not 0 < n_components <= len(features):
        raise ValueError("n_components must be in range(1, len(features) + 1)")
    if algorithm == "arpack" and n_components == len(features):
        raise ValueError("arpack requires n_components < len(features)")
    if algorithm not in {"randomized", "arpack"}:
        raise ValueError("algorithm must be 'randomized' or 'arpack'")
    model = TruncatedSVD(
        n_components=n_components, algorithm=algorithm, random_state=random_seed,
    ).fit(values)
    names = _component_names(n_components)
    loadings = pd.DataFrame([
        {"component": component, "feature": feature, "loading": float(loading)}
        for component, row in zip(names, model.components_)
        for feature, loading in zip(features, row)
    ])
    ratios = model.explained_variance_ratio_
    variance = pd.DataFrame({
        "component": names,
        "explained_variance": model.explained_variance_,
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance_ratio": np.cumsum(ratios),
        "singular_value": model.singular_values_,
    })
    return model, loadings, variance


def transform_components(
    frame: pd.DataFrame, features: list[str], model: TruncatedSVD,
) -> pd.DataFrame:
    """Apply the fitted linear map without centring, preserving sign reversal."""
    scores = model.transform(_values(frame, features))
    return pd.DataFrame(
        scores, columns=_component_names(scores.shape[1]), index=frame.index)


def _project(
    frame: pd.DataFrame, features: list[str], model: TruncatedSVD,
) -> pd.DataFrame:
    metadata = frame.drop(columns=features).reset_index(drop=True)
    scores = transform_components(frame, features, model).reset_index(drop=True)
    return pd.concat([metadata, scores], axis=1)


def fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_components: int,
    algorithm: str,
    config: AnalysisConfig,
) -> tuple[FitResult, pd.DataFrame, pd.DataFrame]:
    """Fit SVD on one outer training fold, then tune and fit ridge only."""
    model, loadings, variance = fit_pooled_svd(
        train, features, n_components, algorithm, config.random_seed)
    component_names = _component_names(n_components)
    projected_train = _project(train, features, model)
    projected_test = _project(test, features, model)
    ridge_c, tuning = tune_ridge(projected_train, component_names, config)
    fitted = fit_fixed_ridge(
        projected_train, projected_test, component_names, config, ridge_c)
    result = FitResult(
        backend=BACKEND,
        fold=fitted.fold,
        selected_features=tuple(component_names),
        predictions=fitted.predictions,
        coefficients=fitted.coefficients,
        tuning=tuning,
        metrics=fitted.metrics,
        diagnostics=fitted.diagnostics | {
            "svd_algorithm": algorithm,
            "svd_components": n_components,
            "svd_input_features": len(features),
            "svd_explained_variance_ratio_sum": float(
                variance.explained_variance_ratio.sum()),
        },
    )
    return result, loadings, variance


def _fold_name(fold_index: int, config: AnalysisConfig) -> str:
    count = len(config.model_folds) * config.item_folds
    if not 0 <= fold_index < count:
        raise ValueError(f"fold_index must be in range({count})")
    model_index, item_fold = divmod(fold_index, config.item_folds)
    return f"{config.model_folds[model_index][0]}__item-{item_fold}"


def _select_fold(
    rows: pd.DataFrame, fold_index: int, config: AnalysisConfig,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    _fold_name(fold_index, config)
    for index, fold in enumerate(_outer_folds(rows, config)):
        if index == fold_index:
            return fold
    raise AssertionError("validated fold index was not generated")


def _settings(
    config: AnalysisConfig, fold_index: int, features: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
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
        "ridge_cs": list(config.ridge_cs),
        "judge_deviation_penalty": config.judge_deviation_penalty,
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "cell_deviation_penalty": config.cell_deviation_penalty,
        "item_intercept_penalty": config.item_intercept_penalty,
        "glm_max_iter": config.glm_max_iter,
        "random_seed": config.random_seed,
        "features": features,
        "feature_inventory_sha256": _signature({"features": features}),
    }


def _signature(values: dict[str, object]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_fold(path: Path, settings: dict[str, object]) -> dict[str, object]:
    if missing := [name for name in ARTIFACTS if not (path / name).exists()]:
        raise RuntimeError(f"incomplete PCA fold checkpoint {path}; missing {missing}")
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        predictions = pd.read_parquet(path / "predictions.parquet")
        tables = {name: pd.read_csv(path / name) for name in CSV_COLUMNS}
        diagnostics = json.loads((path / "diagnostics.json").read_text())
    except Exception as error:
        raise RuntimeError(f"invalid PCA fold checkpoint: {path}") from error
    components = _component_names(int(settings["fit_pca_components"]))
    features = settings["features"]
    model_index, item_fold = divmod(settings["fold_index"], settings["item_folds"])
    expected_fold = f"{settings['model_folds'][model_index][0]}__item-{item_fold}"
    expected_rows = {
        "predictions.parquet": manifest.get("test_rows"),
        "metrics.csv": 2,
        "coefficients.csv": manifest.get("coefficient_rows"),
        "tuning.csv": len(settings["ridge_cs"]),
        "svd-loadings.csv": len(components) * len(features),
        "svd-explained-variance.csv": len(components),
    }
    prediction_columns = manifest.get("prediction_columns")
    hashes = manifest.get("artifact_sha256")
    if (
        manifest.get("state") != "completed"
        or manifest.get("backend") != BACKEND
        or manifest.get("fold") != expected_fold
        or manifest.get("fold_index") != settings["fold_index"]
        or manifest.get("settings") != settings
        or manifest.get("settings_signature") != _signature(settings)
        or manifest.get("component_count") != len(components)
        or manifest.get("input_features") != features
        or not isinstance(manifest.get("train_rows"), int)
        or manifest["train_rows"] <= 0
        or not isinstance(manifest.get("test_rows"), int)
        or manifest["test_rows"] <= 0
        or not isinstance(manifest.get("coefficient_rows"), int)
        or manifest["coefficient_rows"] <= 0
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
        raise RuntimeError(f"PCA fold checkpoint settings or schema mismatch: {path}")
    observed = predictions.observed.to_numpy(dtype=int)
    probability = predictions.probability.to_numpy(dtype=float)
    metrics, coefficients = tables["metrics.csv"], tables["coefficients.csv"]
    tuning = tables["tuning.csv"]
    loadings = tables["svd-loadings.csv"]
    variance = tables["svd-explained-variance.csv"]
    if (
        not np.isin(observed, [0, 1]).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or set(metrics.metric) != {"accuracy", "log_loss"}
        or not np.isfinite(metrics.value.to_numpy(float)).all()
        or set(coefficients.feature) != set(components)
        or not np.isfinite(coefficients.select_dtypes(include="number")).all().all()
        or set(tuning.stage) != {"ridge"}
        or set(tuning.record_type) != {"tuning"}
        or set(tuning.C.to_numpy(float)) != set(settings["ridge_cs"])
        or tuning.selected.sum() != 1
        or not np.isfinite(tuning[["C", "mean_log_loss", "max_n_iter"]]).all().all()
        or set(loadings.component) != set(components)
        or set(loadings.feature) != set(features)
        or loadings[["component", "feature"]].duplicated().any()
        or not np.isfinite(loadings.loading.to_numpy(float)).all()
        or variance.component.tolist() != components
        or not np.isfinite(variance.select_dtypes(include="number")).all().all()
        or not isinstance(diagnostics, dict)
        or not DIAGNOSTIC_KEYS <= set(diagnostics)
        or diagnostics["svd_algorithm"] != settings["fit_pca_algorithm"]
        or diagnostics["svd_components"] != len(components)
        or diagnostics["svd_input_features"] != len(features)
        or float(diagnostics["ridge_C"]) not in settings["ridge_cs"]
    ):
        raise RuntimeError(f"invalid PCA fold artifact contents: {path}")
    return manifest


def _json_default(value: object) -> object:
    return value.item() if hasattr(value, "item") else str(value)


def _write_fold(
    path: Path,
    result: FitResult,
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
    result.tuning.to_csv(partial / "tuning.csv", index=False)
    loadings.to_csv(partial / "svd-loadings.csv", index=False)
    variance.to_csv(partial / "svd-explained-variance.csv", index=False)
    (partial / "diagnostics.json").write_text(json.dumps(
        result.diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n")
    manifest = {
        "state": "completed", "backend": BACKEND, "fold": result.fold,
        "fold_index": fold_index, "train_rows": train_rows, "test_rows": test_rows,
        "component_count": len(result.selected_features), "input_features": features,
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
    """Run or resume one deterministic model-by-item outer-fold checkpoint."""
    config = config.validate()
    fold = _fold_name(fold_index, config)
    output = config.fit_dir / BACKEND / "folds" / fold
    _guard_execution(config)
    rows, features = _load_rows(config)
    selected_fold, train, test = _select_fold(rows, fold_index, config)
    settings = _settings(config, fold_index, features)
    if output.exists():
        _validate_fold(output, settings)
        return output
    if selected_fold != fold:
        raise AssertionError("selected fold name differs from configured fold")
    result, loadings, variance = fit_fold(
        train, test, features, config.fit_pca_components,
        config.fit_pca_algorithm, config)
    _write_fold(
        output, result, loadings, variance, fold_index, len(train), len(test),
        features, settings)
    return output


def _atomic_write(path: Path, write) -> None:
    partial = path.with_name(f".{path.name}.partial")
    write(partial)
    os.replace(partial, path)


def _aggregate_contents(
    fold_names: list[str], fold_dirs: list[Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    tables = {
        "predictions.parquet": pd.concat([
            pd.read_parquet(path / "predictions.parquet")
            for path in fold_dirs
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
        raise RuntimeError(f"incomplete PCA aggregate checkpoint {output}; missing {missing}")
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
        raise RuntimeError(f"invalid PCA aggregate checkpoint: {output}") from error
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
        raise RuntimeError(f"PCA aggregate contents mismatch: {output}")
    return manifest


def aggregate(config: AnalysisConfig = CONFIG) -> Path:
    """Combine all completed fold shards into deterministic pooled artifacts."""
    config = config.validate()
    output = config.fit_dir / BACKEND
    fold_names = [
        f"{model_name}__item-{item_fold}"
        for model_name, _ in config.model_folds
        for item_fold in range(config.item_folds)
    ]
    fold_dirs = [output / "folds" / name for name in fold_names]
    manifests = []
    for index, path in enumerate(fold_dirs):
        saved = json.loads((path / "manifest.json").read_text())
        features = saved.get("input_features")
        if not isinstance(features, list) or not all(
            isinstance(feature, str) for feature in features
        ):
            raise RuntimeError(f"invalid PCA fold feature inventory: {path}")
        manifests.append(_validate_fold(path, _settings(config, index, features)))
    settings = {
        "schema_version": 1,
        "fold_settings_signatures": [manifest["settings_signature"] for manifest in manifests],
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
        _atomic_write(output / name, lambda path, frame=frame: frame.to_csv(path, index=False))
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
        lambda path: path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"))
    _validate_aggregate(output, settings, fold_names, fold_dirs)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fold = commands.add_parser("fold", help="fit one model-by-item outer fold")
    fold.add_argument("--fold-index", type=int, required=True)
    fold.add_argument("--processed-matrix-root", type=Path, required=True)
    fold.add_argument("--results-dir", type=Path, required=True)
    combined = commands.add_parser("aggregate", help="combine all completed fold shards")
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
