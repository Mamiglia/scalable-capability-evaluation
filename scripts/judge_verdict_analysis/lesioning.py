"""Fixed-selection, paired held-out feature lesions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import AnalysisConfig, CONFIG
from .data import load_features
from .fit_common import FitResult
from .fit_frequentist import fit_fixed_ridge, tune_ridge
from .hierarchy import outer_roles


COEFFICIENT_COLUMNS = (
    "feature", "judge", "benchmark", "population", "judge_deviation",
    "benchmark_deviation", "cell_deviation", "effective",
)
ITEM_METRIC_COLUMNS = ("bench", "item", "log_loss", "accuracy", "row_count")
EXPECTED_BLOCKS = frozenset({
    "surface", "structure", "rubric", "exposition", "derivation",
    "intermediate-errors", "clause-descriptors", "clause-motifs", "clause-plan",
})
EXPECTED_FAMILIES = frozenset({
    "annotations", "hand-engineered-text", "clause", "clause-structure", "planning",
})


@dataclass(frozen=True)
class Specification:
    name: str
    kind: str
    variable: str
    reference: str
    features: tuple[str, ...]


def _slug(value: str) -> str:
    return value.replace("_", "-").replace("/", "-")


def build_specifications(
    dictionary: pd.DataFrame, config: AnalysisConfig = CONFIG,
) -> list[Specification]:
    """Return every labelled leave-out and alone specification in dictionary order."""
    if missing := {"feature", "block"} - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    if dictionary.feature.duplicated().any():
        raise ValueError("feature dictionary contains duplicate features")
    full = tuple(dictionary.feature.astype(str))
    blocks = {
        str(block): tuple(group.feature.astype(str))
        for block, group in dictionary.groupby("block", sort=False)
    }
    output: list[Specification] = []
    for feature in full:
        output.extend((
            Specification(
                f"lesion-feature-{_slug(feature)}", "lesion-feature", feature, "full",
                tuple(name for name in full if name != feature),
            ),
            Specification(
                f"singular-feature-{_slug(feature)}", "singular-feature", feature, "intercept-only", (feature,),
            ),
        ))
    for block, features in blocks.items():
        members = set(features)
        output.extend((
            Specification(
                f"lesion-block-{_slug(block)}", "lesion-block", block, "full",
                tuple(name for name in full if name not in members),
            ),
            Specification(
                f"singular-block-{_slug(block)}", "singular-block", block, "intercept-only", features,
            ),
        ))
    for family, configured_blocks in config.lesion_families:
        family_blocks = set(configured_blocks)
        features = tuple(
            row.feature for row in dictionary.itertuples(index=False)
            if row.block in family_blocks
        )
        if not features:
            continue
        members = set(features)
        lesion_name = "no-annotations" if family == "annotations" else f"lesion-family-{family}"
        output.extend((
            Specification(
                lesion_name, "lesion-family", family, "full",
                tuple(name for name in full if name not in members),
            ),
            Specification(
                f"singular-family-{family}", "singular-family", family, "intercept-only", features,
            ),
        ))
    return output


def reference_specifications(dictionary: pd.DataFrame) -> tuple[Specification, Specification]:
    """Return the full and intercept-only references used by paired comparisons."""
    full = tuple(dictionary.feature.astype(str))
    return (
        Specification("full", "reference", "full", "", full),
        Specification("intercept-only", "reference", "intercept-only", "", ()),
    )


def unique_specifications(
    specifications: list[Specification],
) -> tuple[list[Specification], dict[str, str]]:
    """Reuse the first label for duplicate feature sets and return later labels as aliases."""
    by_features: dict[tuple[str, ...], Specification] = {}
    aliases: dict[str, str] = {}
    for specification in specifications:
        canonical = by_features.setdefault(specification.features, specification)
        if canonical is not specification:
            aliases[specification.name] = canonical.name
    return list(by_features.values()), aliases


def _validate_specification_inventory(
    dictionary: pd.DataFrame,
    specifications: list[Specification],
    unique: list[Specification],
    aliases: dict[str, str],
    config: AnalysisConfig,
) -> None:
    if len(dictionary) != 117:
        raise ValueError(f"production lesions require exactly 117 features, found {len(dictionary)}")
    blocks = set(dictionary.block.astype(str))
    if blocks != EXPECTED_BLOCKS:
        raise ValueError(f"production lesions require the exact nine blocks, found {sorted(blocks)}")
    families = {name for name, _ in config.lesion_families}
    if families != EXPECTED_FAMILIES:
        raise ValueError(
            f"production lesions require the exact five families, found {sorted(families)}")
    expected_aliases = {
        "lesion-family-planning": "lesion-block-clause-plan",
        "singular-family-planning": "singular-block-clause-plan",
    }
    if aliases != expected_aliases:
        raise ValueError(f"production lesions require the exact two planning aliases, found {aliases}")
    if (len(specifications), len(unique)) != (262, 260):
        raise ValueError(
            f"expected 262 labelled and 260 unique specifications, got "
            f"{len(specifications)} and {len(unique)}")


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


def _paired_predictions(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    specification: Specification,
    fold: str,
    backend: str = "frequentist",
) -> pd.DataFrame:
    """Return paired held-out contribution rows for one fitted specification."""
    required = {"observed", "probability"}
    if missing := required - set(candidate) | required - set(reference):
        raise ValueError(f"prediction tables missing columns: {sorted(missing)}")
    candidate = candidate.reset_index(drop=True)
    reference = reference.reset_index(drop=True)
    compared = reference.columns.difference(["probability"])
    if len(candidate) != len(reference) or any(
        column not in candidate or not candidate[column].equals(reference[column])
        for column in compared
    ):
        raise ValueError("candidate predictions are not row-aligned with their reference")
    observed = reference.observed.to_numpy(dtype=int)
    candidate_probability = candidate.probability.to_numpy(dtype=float)
    reference_probability = reference.probability.to_numpy(dtype=float)
    candidate_loss = _log_losses(observed, candidate_probability)
    reference_loss = _log_losses(observed, reference_probability)
    candidate_accuracy = (candidate_probability >= 0.5) == observed
    reference_accuracy = (reference_probability >= 0.5) == observed
    output = reference.drop(columns="probability").copy()
    output["backend"] = backend
    output["fold"] = fold
    output["specification"] = specification.name
    output["kind"] = specification.kind
    output["variable"] = specification.variable
    output["reference"] = specification.reference
    output["reference_probability"] = reference_probability
    output["candidate_probability"] = candidate_probability
    if specification.reference == "full":
        output["delta_log_loss"] = candidate_loss - reference_loss
        output["delta_accuracy"] = reference_accuracy.astype(float) - candidate_accuracy.astype(float)
    elif specification.reference == "intercept-only":
        output["delta_log_loss"] = reference_loss - candidate_loss
        output["delta_accuracy"] = candidate_accuracy.astype(float) - reference_accuracy.astype(float)
    else:
        raise ValueError(f"unknown specification reference: {specification.reference}")
    return output


def _paired_item_metrics(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    specification: Specification,
    fold: str,
) -> pd.DataFrame:
    """Pair saved per-item metric inputs without reopening model predictions."""
    if candidate.columns.tolist() != list(ITEM_METRIC_COLUMNS) or (
        reference.columns.tolist() != list(ITEM_METRIC_COLUMNS)
    ):
        raise RuntimeError("item-metric schema is not recoverable")
    merged = candidate.merge(
        reference, on=["bench", "item"], suffixes=("_candidate", "_reference"),
        validate="one_to_one", how="outer", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("candidate and reference item keys differ")
    merged = merged.drop(columns="_merge")
    if not merged.row_count_candidate.equals(merged.row_count_reference):
        raise RuntimeError("candidate and reference item metrics have different row counts")
    output = merged[["bench", "item"]].copy()
    output["backend"] = "frequentist"
    output["fold"] = fold
    output["specification"] = specification.name
    output["kind"] = specification.kind
    output["variable"] = specification.variable
    output["reference"] = specification.reference
    output["candidate_log_loss"] = merged.log_loss_candidate
    output["reference_log_loss"] = merged.log_loss_reference
    output["candidate_accuracy"] = merged.accuracy_candidate
    output["reference_accuracy"] = merged.accuracy_reference
    output["row_count"] = merged.row_count_candidate.astype(int)
    if specification.reference == "full":
        output["delta_log_loss"] = output.candidate_log_loss - output.reference_log_loss
        output["delta_accuracy"] = output.reference_accuracy - output.candidate_accuracy
    elif specification.reference == "intercept-only":
        output["delta_log_loss"] = output.reference_log_loss - output.candidate_log_loss
        output["delta_accuracy"] = output.candidate_accuracy - output.reference_accuracy
    else:
        raise ValueError(f"unknown specification reference: {specification.reference}")
    return output


def _prediction_rows(predictions: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Keep the common held-out metadata and probabilities, never inactive regressors."""
    if missing := {"observed", "probability"} - set(predictions):
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    return predictions.drop(columns=features, errors="ignore").reset_index(drop=True)


def _validate_prediction_rows(
    predictions: pd.DataFrame, test: pd.DataFrame, features: list[str],
) -> pd.DataFrame:
    """Require a checkpoint to match this fold's exact held-out rows and schema."""
    predictions = predictions.reset_index(drop=True)
    metadata = test.drop(columns=[*features, "pick_left"], errors="ignore").reset_index(drop=True)
    expected_columns = [*metadata.columns, "observed", "probability"]
    if predictions.columns.tolist() != expected_columns or len(predictions) != len(test):
        raise RuntimeError("prediction schema does not match the held-out fold")
    if any(not predictions[column].equals(metadata[column]) for column in metadata):
        raise RuntimeError("prediction rows do not match the held-out fold")
    if not np.array_equal(
        predictions.observed.to_numpy(dtype=int), test.pick_left.to_numpy(dtype=int)
    ):
        raise RuntimeError("prediction outcomes do not match the held-out fold")
    probability = predictions.probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise RuntimeError("checkpoint probabilities must be finite values in [0, 1]")
    return predictions


def _validate_coefficient_rows(
    coefficients: pd.DataFrame, features: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Require a recoverable effective-coefficient table for the fitted feature set."""
    coefficients = coefficients.reset_index(drop=True)
    if coefficients.columns.tolist() != list(COEFFICIENT_COLUMNS):
        raise RuntimeError("coefficient schema does not match the fitted model")
    if set(coefficients.feature.astype(str)) != set(features):
        raise RuntimeError("coefficient features do not match the fitted model")
    numeric = coefficients[list(COEFFICIENT_COLUMNS[3:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("coefficients must be finite")
    return coefficients


def _item_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    observed = predictions.observed.to_numpy(dtype=int)
    probability = predictions.probability.to_numpy(dtype=float)
    rows = predictions[["bench", "item"]].copy()
    rows["log_loss"] = _log_losses(observed, probability)
    rows["accuracy"] = ((probability >= 0.5) == observed).astype(float)
    return rows.groupby(["bench", "item"], sort=False, as_index=False).agg(
        log_loss=("log_loss", "mean"), accuracy=("accuracy", "mean"),
        row_count=("accuracy", "size"),
    )


def _validate_item_metric_rows(metrics: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.reset_index(drop=True)
    if metrics.columns.tolist() != list(ITEM_METRIC_COLUMNS):
        raise RuntimeError("item-metric schema does not match the held-out fold")
    expected = test.groupby(["bench", "item"], sort=False).size().reset_index(name="row_count")
    actual = metrics[["bench", "item", "row_count"]]
    if not actual.equals(expected):
        raise RuntimeError("item metrics do not match the held-out fold")
    if (
        not np.isfinite(metrics[["log_loss", "accuracy"]].to_numpy(dtype=float)).all()
        or (metrics.log_loss < 0).any()
        or ((metrics.accuracy < 0) | (metrics.accuracy > 1)).any()
    ):
        raise RuntimeError("item metrics contain invalid values")
    return metrics


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
    metadata = ["backend", "specification", "kind", "variable", "reference"]
    if rows.empty:
        return pd.DataFrame(columns=[
            *metadata, "lesion_metric", "estimate", "lower", "upper",
            "delta_log_loss", "delta_log_loss_lower", "delta_log_loss_upper",
            "delta_accuracy", "delta_accuracy_lower", "delta_accuracy_upper",
            "fold_count", "row_count",
        ])
    if "specification" not in rows and "feature" in rows:
        rows = rows.assign(
            specification=rows.feature.astype(str), kind="lesion-feature",
            variable=rows.feature.astype(str), reference="full",
        )
    required = {*metadata, "bench", "item", "fold", "delta_log_loss", "delta_accuracy"}
    if missing := sorted(required - set(rows)):
        raise ValueError(f"lesion rows missing bootstrap columns: {missing}")
    metric = f"delta_{config.lesion_metric}"
    rng = np.random.default_rng(config.random_seed)
    summaries = []
    for keys, group in rows.groupby(metadata, sort=True):
        weights = (
            group.row_count.to_numpy(dtype=int)
            if "row_count" in group else np.ones(len(group), dtype=int)
        )
        estimates = {
            name: float(np.average(group[name].to_numpy(float), weights=weights))
            for name in ("delta_log_loss", "delta_accuracy")
        }
        weighted = group[["bench", "item"]].copy()
        weighted["count"] = weights
        for name in estimates:
            weighted[name] = group[name].to_numpy(float) * weights
        clusters = weighted.groupby(["bench", "item"], sort=False).sum()
        draws = rng.integers(
            len(clusters), size=(config.bootstrap_iterations, len(clusters)))
        counts = clusters["count"].to_numpy()[draws].sum(axis=1)
        samples = {
            name: clusters[name].to_numpy()[draws].sum(axis=1) / counts
            for name in estimates
        }
        intervals = {name: np.percentile(values, [2.5, 97.5]) for name, values in samples.items()}
        summaries.append({
            **dict(zip(metadata, keys)), "lesion_metric": config.lesion_metric,
            "estimate": estimates[metric],
            "delta_log_loss": estimates["delta_log_loss"],
            "lower": intervals[metric][0], "upper": intervals[metric][1],
            "delta_log_loss_lower": intervals["delta_log_loss"][0], "delta_log_loss_upper": intervals["delta_log_loss"][1],
            "delta_accuracy": estimates["delta_accuracy"],
            "delta_accuracy_lower": intervals["delta_accuracy"][0],
            "delta_accuracy_upper": intervals["delta_accuracy"][1],
            "fold_count": group.fold.nunique(), "row_count": int(weights.sum()),
        })
    return pd.DataFrame(summaries)


def _guard_execution(config: AnalysisConfig) -> None:
    """Require explicit execution and a real SWC allocation for remote work."""
    if config.execution_mode != "execute":
        raise RuntimeError("lesions require execution_mode='execute'")
    if config.execution_target == "local_test":
        return
    if config.execution_target != "swc_hpc":
        raise RuntimeError("judge-verdict lesions support only swc_hpc or local_test")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SWC lesions require a Slurm allocation")
    hostname = os.uname().nodename.lower()
    if hostname.startswith(("gateway", "sgw")):
        raise RuntimeError("refusing lesion execution on the SWC gateway")
    if not config.processed_matrix_manifest_sha256:
        raise RuntimeError("SWC lesions require a configured processed matrix manifest hash")
    matrix = yaml.safe_load(config.processed_matrix_config.read_text())
    manifest = Path(matrix["output_root"]) / "manifest.json"
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != config.processed_matrix_manifest_sha256:
        raise RuntimeError(
            f"processed matrix manifest hash mismatch: expected "
            f"{config.processed_matrix_manifest_sha256}, found {actual}")


def _load_rows(config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, dictionary = load_features(
        "pair", config.feature_blocks, features=config.regressors or None,
        splits=config.splits, config=config.processed_matrix_config,
    )
    rows = rows[
        rows.judge.isin(config.judges) & rows.bench.isin(config.benchmarks)
    ].reset_index(drop=True)
    rows = rows.rename(columns=dict(zip(dictionary.column, dictionary.feature)))
    required = {
        "judge", "bench", "item", "model_left", "model_right", "pick_left",
        *dictionary.feature,
    }
    if missing := sorted(required - set(rows)):
        raise ValueError(f"pair matrix missing fitting columns: {missing}")
    if rows.empty:
        raise ValueError("no pair rows remain after configured judge and benchmark filtering")
    return rows, dictionary


def _select_outer_fold(
    rows: pd.DataFrame, fold_index: int, config: AnalysisConfig,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    fold_count = len(config.model_folds) * config.item_folds
    if not 0 <= fold_index < fold_count:
        raise ValueError(f"fold_index must be in range({fold_count})")
    model_index, item_fold = divmod(fold_index, config.item_folds)
    model_name, models = config.model_folds[model_index]
    fold = f"{model_name}__item-{item_fold}"
    roles = outer_roles(rows, models, item_fold, config.item_folds)
    train = rows.loc[roles == "train"].assign(fold=fold).reset_index(drop=True)
    test = rows.loc[roles == "test"].assign(fold=fold).reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"outer fold {fold} has no {'train' if train.empty else 'test'} rows")
    return fold, train, test


def _checkpoint(
    path: Path,
    write,
    read,
) -> None:
    """Publish one deterministic artifact atomically; preserve failed partials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(path):
        if _resume_checkpoint_unlocked(path, read):
            return
        partial = path.with_name(f".{path.name}.partial")
        write(partial)
        read(partial)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint created concurrently: {path}")
        os.replace(partial, path)
        read(path)


def _resume_checkpoint(path: Path, read) -> bool:
    """Validate a final checkpoint or promote its valid deterministic partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(path):
        return _resume_checkpoint_unlocked(path, read)


@contextmanager
def _checkpoint_lock(path: Path):
    """Serialize check/write/publish for one target; lock files intentionally remain."""
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _resume_checkpoint_unlocked(path: Path, read) -> bool:
    if path.exists():
        read(path)
        return True
    partial = path.with_name(f".{path.name}.partial")
    if not partial.exists():
        return False
    try:
        read(partial)
    except Exception as error:
        raise RuntimeError(f"incomplete checkpoint preserved at {partial}") from error
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint created concurrently: {path}")
    os.replace(partial, path)
    read(path)
    return True


def _write_prediction(
    frame: pd.DataFrame, path: Path, test: pd.DataFrame, features: list[str],
) -> None:
    _checkpoint(
        path,
        lambda target: frame.to_parquet(target, index=False),
        lambda target: _validate_prediction_rows(pd.read_parquet(target), test, features),
    )


def _write_coefficients(
    frame: pd.DataFrame, path: Path, features: tuple[str, ...] | list[str],
) -> None:
    _checkpoint(
        path,
        lambda target: frame.to_parquet(target, index=False),
        lambda target: _validate_coefficient_rows(pd.read_parquet(target), features),
    )


def _write_item_metrics(frame: pd.DataFrame, path: Path, test: pd.DataFrame) -> None:
    _checkpoint(
        path,
        lambda target: frame.to_parquet(target, index=False),
        lambda target: _validate_item_metric_rows(pd.read_parquet(target), test),
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    _checkpoint(
        path,
        lambda target: frame.to_parquet(target, index=False),
        lambda target: pd.read_parquet(target),
    )


def _artifact_paths(
    fold_dir: Path, name: str, *, reference: bool,
) -> tuple[Path, Path, Path]:
    if reference:
        return (
            fold_dir / "references" / f"{name}.parquet",
            fold_dir / "reference-coefficients" / f"{name}.parquet",
            fold_dir / "reference-item-metrics" / f"{name}.parquet",
        )
    return (
        fold_dir / "specifications" / f"{name}.parquet",
        fold_dir / "coefficients" / f"{name}.parquet",
        fold_dir / "item-metrics" / f"{name}.parquet",
    )


def _artifacts_ready(
    paths: tuple[Path, Path, Path],
    test: pd.DataFrame,
    all_features: list[str],
    fitted_features: tuple[str, ...],
) -> bool:
    prediction_path, coefficient_path, item_metric_path = paths
    readers = (
        lambda target: _validate_prediction_rows(
            pd.read_parquet(target), test, all_features),
        lambda target: _validate_coefficient_rows(
            pd.read_parquet(target), fitted_features),
        lambda target: _validate_item_metric_rows(pd.read_parquet(target), test),
    )
    return all(
        _resume_checkpoint(path, read)
        for path, read in zip(paths, readers)
    )


def _write_fit_artifacts(
    result: FitResult,
    paths: tuple[Path, Path, Path],
    test: pd.DataFrame,
    all_features: list[str],
    fitted_features: tuple[str, ...],
) -> None:
    predictions = _prediction_rows(result.predictions, all_features)
    _write_prediction(predictions, paths[0], test, all_features)
    _write_coefficients(result.coefficients, paths[1], fitted_features)
    _write_item_metrics(_item_metric_rows(predictions), paths[2], test)


def _require_fit_artifacts(
    paths: tuple[Path, Path, Path],
    test: pd.DataFrame,
    all_features: list[str],
    fitted_features: tuple[str, ...],
) -> None:
    readers = (
        lambda target: _validate_prediction_rows(
            pd.read_parquet(target), test, all_features),
        lambda target: _validate_coefficient_rows(
            pd.read_parquet(target), fitted_features),
        lambda target: _validate_item_metric_rows(pd.read_parquet(target), test),
    )
    missing = [
        str(path) for path, read in zip(paths, readers)
        if not _resume_checkpoint(path, read)
    ]
    if missing:
        raise RuntimeError(f"missing completed checkpoint: {missing}")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _checkpoint(
        path,
        lambda target: frame.to_csv(target, index=False),
        lambda target: pd.read_csv(target),
    )


def _write_json(values: dict[str, object], path: Path) -> None:
    _checkpoint(
        path,
        lambda target: target.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n"),
        lambda target: json.loads(target.read_text()),
    )


def _settings_signature(settings: dict[str, object]) -> str:
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ensure_settings(path: Path, settings: dict[str, object], label: str) -> str:
    expected = {"signature": _settings_signature(settings), "settings": settings}
    _write_json(expected, path)
    actual = json.loads(path.read_text())
    if actual != expected:
        raise RuntimeError(f"{label} settings mismatch at {path}")
    return expected["signature"]


def _specification_table(
    specifications: list[Specification], aliases: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "specification": spec.name, "kind": spec.kind, "variable": spec.variable,
            "reference": spec.reference, "feature_count": len(spec.features),
            "artifact": aliases.get(spec.name, spec.name),
        }
        for spec in specifications
    ])


def _fold_settings(
    config: AnalysisConfig,
    fold_index: int,
    features: list[str],
    specifications: pd.DataFrame,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "processed_matrix_manifest_sha256": config.processed_matrix_manifest_sha256,
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "fold_index": fold_index,
        "model_folds": [[name, list(models)] for name, models in config.model_folds],
        "item_folds": config.item_folds,
        "feature_blocks": list(config.feature_blocks),
        "regressors": list(config.regressors),
        "features": features,
        "judges": list(config.judges),
        "benchmarks": list(config.benchmarks),
        "splits": list(config.splits),
        "inner_folds": config.inner_folds,
        "fit_n_jobs": config.fit_n_jobs,
        "ridge_cs": list(config.ridge_cs),
        "judge_deviation_penalty": config.judge_deviation_penalty,
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "cell_deviation_penalty": config.cell_deviation_penalty,
        "item_intercept_penalty": config.item_intercept_penalty,
        "glm_max_iter": config.glm_max_iter,
        "random_seed": config.random_seed,
        "lesion_backends": list(config.lesion_backends),
        "lesion_feature_scope": config.lesion_feature_scope,
        "lesion_families": [
            [name, list(blocks)] for name, blocks in config.lesion_families
        ],
        "specifications": specifications.to_dict(orient="records"),
    }


def _validate_ridge_checkpoint(
    ridge: dict[str, object], features: list[str], expected_c: float | None = None,
) -> float:
    try:
        ridge_c = float(ridge["ridge_C"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid ridge checkpoint") from error
    if (
        not np.isfinite(ridge_c)
        or ridge_c <= 0
        or ridge.get("features") != features
        or not isinstance(ridge.get("tuning"), list)
        or not ridge["tuning"]
        or any(not isinstance(row, dict) for row in ridge["tuning"])
        or (expected_c is not None and ridge_c != float(expected_c))
    ):
        raise RuntimeError("invalid ridge checkpoint")
    return ridge_c


def _require_completed_fold(
    fold_dir: Path,
    test: pd.DataFrame,
    features: list[str],
    references: tuple[Specification, Specification],
    specifications: list[Specification],
    expected_ridge_c: float,
) -> None:
    ridge_path = fold_dir / "ridge.json"
    if not _resume_checkpoint(
        ridge_path,
        lambda path: _validate_ridge_checkpoint(
            json.loads(path.read_text()), features, expected_ridge_c),
    ):
        raise RuntimeError(f"missing completed checkpoint: {ridge_path}")
    for specification in references:
        _require_fit_artifacts(
            _artifact_paths(fold_dir, specification.name, reference=True),
            test, features, specification.features,
        )
    for specification in specifications:
        _require_fit_artifacts(
            _artifact_paths(fold_dir, specification.name, reference=False),
            test, features, specification.features,
        )


def run_fold(fold_index: int, config: AnalysisConfig = CONFIG) -> Path:
    """Tune once and checkpoint every fixed-C reference and specification for one outer fold."""
    config = config.validate()
    _guard_execution(config)
    if tuple(config.lesion_backends) != ("frequentist",):
        raise ValueError("judge-verdict lesions require lesion_backends=('frequentist',)")
    rows, dictionary = _load_rows(config)
    if config.execution_target == "local_test" and len(rows) > config.local_test_max_rows:
        raise RuntimeError("refusing local lesions above local_test_max_rows; use SWC HPC")
    fold, train, test = _select_outer_fold(rows, fold_index, config)
    fold_dir = config.lesion_dir / "folds" / fold
    completion = fold_dir / "manifest.json"
    specifications = build_specifications(dictionary, config)
    unique, aliases = unique_specifications(specifications)
    _validate_specification_inventory(dictionary, specifications, unique, aliases, config)
    features = dictionary.feature.astype(str).tolist()
    table = _specification_table(specifications, aliases)
    fold_settings = _fold_settings(config, fold_index, features, table)
    plan_settings = dict(fold_settings)
    plan_settings.pop("fold_index")
    plan_signature = _settings_signature(plan_settings)
    settings_signature = _ensure_settings(
        fold_dir / "settings.json", fold_settings, "fold")
    table_path = fold_dir / "specifications.csv"
    _write_csv(table, table_path)
    if not pd.read_csv(table_path, keep_default_na=False).equals(table):
        raise RuntimeError(f"fold settings mismatch at {table_path}")
    completed: dict[str, object] = {}

    def read_completion(path: Path) -> dict[str, object]:
        completed.update(json.loads(path.read_text()))
        return completed

    if _resume_checkpoint(completion, read_completion):
        manifest = completed
        if (manifest.get("state"), manifest.get("settings_signature")) != (
            "completed", settings_signature,
        ):
            raise RuntimeError(f"invalid fold completion manifest: {completion}")
        try:
            manifest_ridge_c = float(manifest["ridge_C"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid fold completion manifest: {completion}") from error
        if not np.isfinite(manifest_ridge_c) or manifest_ridge_c <= 0:
            raise RuntimeError(f"invalid fold completion manifest: {completion}")
        _require_completed_fold(
            fold_dir, test, features, reference_specifications(dictionary), unique,
            manifest_ridge_c,
        )
        return fold_dir

    ridge_path = fold_dir / "ridge.json"
    resumed_ridge: dict[str, float] = {}

    def read_ridge(path: Path) -> dict[str, object]:
        ridge = json.loads(path.read_text())
        resumed_ridge["ridge_C"] = _validate_ridge_checkpoint(ridge, features)
        return ridge

    if _resume_checkpoint(ridge_path, read_ridge):
        ridge_c = resumed_ridge["ridge_C"]
    else:
        ridge_c, tuning = tune_ridge(train, features, config)
        _write_json(
            {"ridge_C": float(ridge_c), "features": features,
             "tuning": tuning.to_dict(orient="records")},
            ridge_path,
        )

    references = reference_specifications(dictionary)
    for specification in references:
        paths = _artifact_paths(fold_dir, specification.name, reference=True)
        if not _artifacts_ready(
            paths, test, features, specification.features,
        ):
            result = fit_fixed_ridge(
                train, test, list(specification.features), config, ridge_c)
            _write_fit_artifacts(
                result, paths, test, features, specification.features)

    for specification in unique:
        paths = _artifact_paths(fold_dir, specification.name, reference=False)
        if _artifacts_ready(paths, test, features, specification.features):
            continue
        result = fit_fixed_ridge(
            train, test, list(specification.features), config, ridge_c)
        _write_fit_artifacts(
            result, paths, test, features, specification.features)

    _require_completed_fold(fold_dir, test, features, references, unique, ridge_c)

    _write_json({
        "state": "completed", "backend": "frequentist", "fold": fold,
        "fold_index": fold_index, "ridge_C": ridge_c,
        "settings_signature": settings_signature, "plan_signature": plan_signature,
        "labelled_specifications": len(specifications),
        "unique_specifications": len(unique), "aliases": aliases,
        "train_rows": len(train), "test_rows": len(test),
    }, completion)
    return fold_dir


def _fold_names(config: AnalysisConfig) -> list[str]:
    return [
        f"{model_name}__item-{item_fold}"
        for model_name, _ in config.model_folds
        for item_fold in range(config.item_folds)
    ]


def _artifact_index(
    fold_names: list[str], fold_dirs: list[Path], specifications: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    references = specifications.reference.drop_duplicates().tolist()
    for fold, fold_dir in zip(fold_names, fold_dirs):
        for reference in references:
            paths = _artifact_paths(fold_dir, reference, reference=True)
            rows.append({
                "fold": fold, "specification": reference, "kind": "reference",
                "variable": reference, "reference": "", "artifact": reference,
                "predictions": str(paths[0]), "coefficients": str(paths[1]),
                "item_metrics": str(paths[2]),
            })
        for row in specifications.itertuples(index=False):
            paths = _artifact_paths(fold_dir, row.artifact, reference=False)
            rows.append({
                "fold": fold, "specification": row.specification, "kind": row.kind,
                "variable": row.variable, "reference": row.reference,
                "artifact": row.artifact, "predictions": str(paths[0]),
                "coefficients": str(paths[1]), "item_metrics": str(paths[2]),
            })
    return pd.DataFrame(rows)


def aggregate_lesions(config: AnalysisConfig = CONFIG) -> Path:
    """Aggregate completed fold shards without holding the full lesion grid in memory."""
    config = config.validate()
    output = config.lesion_dir / "lesions.csv"
    item_delta_output = config.lesion_dir / "item-deltas.parquet"
    artifact_index_output = config.lesion_dir / "artifact-index.csv"
    completion = config.lesion_dir / "summary-manifest.json"
    fold_names = _fold_names(config)
    fold_dirs = [config.lesion_dir / "folds" / fold for fold in fold_names]
    specification_table: pd.DataFrame | None = None
    fold_signatures = []
    plan_signatures = set()
    for fold_dir in fold_dirs:
        manifest = fold_dir / "manifest.json"
        values: dict[str, object] = {}
        manifest_ready = _resume_checkpoint(
            manifest, lambda path: values.update(json.loads(path.read_text())))
        if (
            not manifest_ready
            or values.get("state") != "completed"
            or not values.get("settings_signature")
            or not values.get("plan_signature")
        ):
            raise RuntimeError(f"fold is not complete: {fold_dir}")
        fold_signatures.append(values["settings_signature"])
        plan_signatures.add(values["plan_signature"])
        current = pd.read_csv(fold_dir / "specifications.csv", keep_default_na=False)
        if specification_table is None:
            specification_table = current
        elif not current.equals(specification_table):
            raise ValueError(f"fold specification table differs: {fold_dir}")
    assert specification_table is not None
    if len(plan_signatures) != 1:
        raise RuntimeError("completed folds do not share one common plan")
    summary_settings = {
        "schema_version": 1,
        "fold_signatures": fold_signatures,
        "plan_signature": plan_signatures.pop(),
        "lesion_metric": config.lesion_metric,
        "bootstrap_iterations": config.bootstrap_iterations,
        "random_seed": config.random_seed,
        "specifications": specification_table.to_dict(orient="records"),
    }
    settings_signature = _ensure_settings(
        config.lesion_dir / "summary-settings.json", summary_settings, "summary")
    completed: dict[str, object] = {}

    def read_completion(path: Path) -> dict[str, object]:
        completed.update(json.loads(path.read_text()))
        return completed

    if _resume_checkpoint(completion, read_completion):
        values = completed
        if (values.get("state"), values.get("settings_signature")) != (
            "completed", settings_signature,
        ):
            raise RuntimeError(f"summary settings mismatch at {completion}")
        pd.read_csv(output)
        pd.read_parquet(item_delta_output)
        pd.read_csv(artifact_index_output, keep_default_na=False)
        return output

    summaries = []
    item_deltas = []
    for row in specification_table.itertuples(index=False):
        specification = Specification(
            row.specification, row.kind, row.variable, row.reference, (),
        )
        paired = []
        for fold, fold_dir in zip(fold_names, fold_dirs):
            candidate = pd.read_parquet(
                fold_dir / "item-metrics" / f"{row.artifact}.parquet")
            reference = pd.read_parquet(
                fold_dir / "reference-item-metrics" / f"{row.reference}.parquet")
            paired.append(_paired_item_metrics(candidate, reference, specification, fold))
        item_deltas.extend(paired)
        summaries.append(_bootstrap_summary(pd.concat(paired, ignore_index=True), config))
    summary = pd.concat(summaries, ignore_index=True) if summaries else _bootstrap_summary(
        pd.DataFrame(), config)
    combined_item_deltas = pd.concat(item_deltas, ignore_index=True) if item_deltas else pd.DataFrame()
    artifact_index = _artifact_index(fold_names, fold_dirs, specification_table)
    _write_csv(summary, output)
    _write_parquet(combined_item_deltas, item_delta_output)
    _write_csv(artifact_index, artifact_index_output)
    _write_json({
        "state": "completed", "backend": "frequentist",
        "fold_count": len(fold_dirs), "labelled_specifications": len(specification_table),
        "settings_signature": settings_signature,
        "plan_signature": summary_settings["plan_signature"],
        "lesion_metric": config.lesion_metric,
        "bootstrap_iterations": config.bootstrap_iterations,
        "random_seed": config.random_seed, "summary": output.name,
        "item_deltas": item_delta_output.name, "artifact_index": artifact_index_output.name,
    }, completion)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fold = commands.add_parser("fold", help="run one model-by-item outer fold")
    fold.add_argument("--fold-index", type=int, required=True)
    fold.add_argument("--processed-matrix-root", type=Path, required=True)
    fold.add_argument("--results-dir", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="aggregate all completed folds")
    summarize.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fold":
        run_fold(args.fold_index, replace(
            CONFIG, processed_matrix_root=args.processed_matrix_root,
            results_dir=args.results_dir, execution_mode="execute", execution_target="swc_hpc",
        ))
    else:
        aggregate_lesions(replace(CONFIG, results_dir=args.results_dir))


if __name__ == "__main__":
    main()
