"""Fixed-selection, paired held-out feature lesions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from . import fit
from .config import AnalysisConfig, CONFIG
from .fit_common import FitResult
from .fit_frequentist import fit_fixed_ridge, tune_ridge


def _log_losses(observed: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probability = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return -(observed * np.log(probability) + (1 - observed) * np.log1p(-probability))


def _full_predictions(full_fit: FitResult, test: pd.DataFrame) -> pd.DataFrame:
    predictions = full_fit.predictions.reset_index(drop=True).copy()
    if {"observed", "probability"} - set(predictions):
        raise ValueError("full fit predictions need observed and probability columns")
    observed = test.pick_left.to_numpy(dtype=int)
    if len(predictions) != len(test) or not np.array_equal(predictions.observed.to_numpy(dtype=int), observed):
        raise ValueError("full fit predictions are not row-aligned with the supplied test rows")
    metadata = predictions.columns.difference(["observed", "probability"])
    for column in metadata.intersection(test.columns):
        if not predictions[column].equals(test[column].reset_index(drop=True)):
            raise ValueError(f"full fit predictions disagree with test rows in {column!r}")
    probability = predictions.probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("full fit probabilities must be finite values in [0, 1]")
    return predictions


def lesion_fold(
    full_fit: FitResult, train: pd.DataFrame, test: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Refit one fixed-selection lesion per retained base feature on one outer fold."""
    config.validate()
    if full_fit.backend != "frequentist":
        raise NotImplementedError("fixed-selection lesions currently require the frequentist backend")
    full = _full_predictions(full_fit, test)
    features = list(full_fit.selected_features)
    if missing := sorted(set(features) - set(train) | set(features) - set(test)):
        raise ValueError(f"selected features are missing from lesion rows: {missing}")
    try:
        ridge_c = float(full_fit.diagnostics["ridge_C"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frequentist lesion needs the full baseline's ridge_C") from error
    observed = full.observed.to_numpy(dtype=int)
    full_probability = full.probability.to_numpy(dtype=float)
    full_loss = _log_losses(observed, full_probability)
    rows = []
    for feature in features:
        lesion = fit_fixed_ridge(
            train, test, [name for name in features if name != feature], config, ridge_c)
        lesion_probability = lesion.predictions.probability.to_numpy(dtype=float)
        lesion_loss = _log_losses(observed, lesion_probability)
        result = full.copy()
        result["backend"] = full_fit.backend
        result["fold"] = full_fit.fold
        result["feature"] = feature
        result["lesion_probability"] = lesion_probability
        result["full_log_loss"] = full_loss
        result["lesion_log_loss"] = lesion_loss
        result["delta_log_loss"] = lesion_loss - full_loss
        result["full_accuracy"] = (full_probability >= 0.5) == observed
        result["lesion_accuracy"] = (lesion_probability >= 0.5) == observed
        result["delta_accuracy"] = result.full_accuracy.astype(float) - result.lesion_accuracy.astype(float)
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bootstrap_summary(rows: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=[
            "backend", "feature", "lesion_metric", "estimate", "lower", "upper",
            "delta_log_loss", "delta_log_loss_lower", "delta_log_loss_upper",
            "delta_accuracy", "delta_accuracy_lower", "delta_accuracy_upper",
            "fold_count", "row_count",
        ])
    required = {"backend", "feature", "bench", "item", "fold", "delta_log_loss", "delta_accuracy"}
    if missing := sorted(required - set(rows)):
        raise ValueError(f"lesion rows missing bootstrap columns: {missing}")
    metric = f"delta_{config.lesion_metric}"
    rng = np.random.default_rng(config.random_seed)
    summaries = []
    for (backend, feature), group in rows.groupby(["backend", "feature"], sort=True):
        clusters = [cluster for _, cluster in group.groupby(["bench", "item"], sort=False)]
        estimates = {name: float(group[name].mean()) for name in ("delta_log_loss", "delta_accuracy")}
        samples = {name: np.empty(config.bootstrap_iterations) for name in estimates}
        for index in range(config.bootstrap_iterations):
            sample = pd.concat([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
            for name in samples:
                samples[name][index] = sample[name].mean()
        intervals = {name: np.percentile(values, [2.5, 97.5]) for name, values in samples.items()}
        summaries.append({
            "backend": backend, "feature": feature, "lesion_metric": config.lesion_metric,
            "estimate": estimates[metric],
            "delta_log_loss": estimates["delta_log_loss"],
            "lower": intervals["delta_log_loss"][0], "upper": intervals["delta_log_loss"][1],
            "delta_log_loss_lower": intervals["delta_log_loss"][0], "delta_log_loss_upper": intervals["delta_log_loss"][1],
            "delta_accuracy": estimates["delta_accuracy"],
            "delta_accuracy_lower": intervals["delta_accuracy"][0],
            "delta_accuracy_upper": intervals["delta_accuracy"][1],
            "fold_count": group.fold.nunique(), "row_count": len(group),
        })
    return pd.DataFrame(summaries)


def _write_lesions(rows: pd.DataFrame, summary: pd.DataFrame, config: AnalysisConfig) -> None:
    output = config.lesion_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".lesions-", dir=output.parent))
    try:
        rows.to_parquet(stage / "paired_rows.parquet", index=False)
        summary.to_csv(stage / "lesions.csv", index=False)
        (stage / "manifest.json").write_text(json.dumps({
            "lesion_backends": list(config.lesion_backends),
            "lesion_feature_scope": config.lesion_feature_scope,
            "lesion_metric": config.lesion_metric,
            "bootstrap_iterations": config.bootstrap_iterations,
            "random_seed": config.random_seed,
            "paired_rows": "paired_rows.parquet",
            "summary": "lesions.csv",
        }, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
    except BaseException:
        for artifact in stage.iterdir() if stage.exists() else ():
            artifact.unlink()
        if stage.exists():
            stage.rmdir()
        raise


def run_lesions(config: AnalysisConfig = CONFIG) -> None:
    """Run configured paired lesions only on an approved execution target."""
    config = config.validate()
    if config.execution_mode != "execute":
        raise RuntimeError("lesions require execution_mode='execute'")
    fit._guard_execution(config)
    rows, features = fit._load_rows(config)
    if config.execution_target == "local_test" and len(rows) > config.local_test_max_rows:
        raise RuntimeError("refusing local lesions above local_test_max_rows; use SWC HPC or RunPod")
    lesion_rows = []
    for fold_name, train, test in fit._outer_folds(rows, config):
        for backend in config.lesion_backends:
            if backend != "frequentist":
                raise NotImplementedError("fixed-selection lesions currently require the frequentist backend")
            ridge_c, _ = tune_ridge(train, features, config)
            full_fit = fit_fixed_ridge(train, test, features, config, ridge_c)
            lesion_rows.append(lesion_fold(full_fit, train, test, config))
    combined = pd.concat(lesion_rows, ignore_index=True) if lesion_rows else pd.DataFrame()
    _write_lesions(combined, _bootstrap_summary(combined, config), config)


def main() -> None:
    """Run the configuration-owned fixed-selection lesion entry point."""
    run_lesions()


if __name__ == "__main__":
    main()
