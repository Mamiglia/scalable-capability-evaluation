"""Pairwise capability sparse fitting, checkpoints, and aggregation."""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from scripts.judge_verdict_analysis.config import AnalysisConfig, CONFIG as JUDGE_CONFIG
from scripts.judge_verdict_analysis.fit import _guard_execution, _load_rows
from scripts.judge_verdict_analysis.fit_common import FitResult, prepare_features
from scripts.judge_verdict_analysis.fit_frequentist import SparseTask, fit_sparse, tune_sparse
from scripts.judge_verdict_analysis.lesioning import (
    Specification,
    _item_metrics,
    _paired_item_metrics,
    _settings_signature as _signature,
    build_specifications as _judge_build_specifications,
    unique_specifications,
)

from .config import TARGET_QUARTILES
from .preprocessing import (
    attach_pair_target,
    deduplicate_feature_pairs,
    item_quartiles,
    quartile_accuracy,
    split_pairs,
)


@dataclass(frozen=True)
class CapabilityLevels:
    benchmarks: tuple[str, ...]


@dataclass
class CellResult:
    prepared_train: pd.DataFrame
    prepared_test: pd.DataFrame
    prepared_dictionary: pd.DataFrame
    provenance: dict[str, object]
    frozen_hyperparameters: dict[str, float | None]
    full: FitResult
    null: pd.DataFrame


def capability_levels(frame: pd.DataFrame) -> CapabilityLevels:
    if "bench" not in frame:
        raise ValueError("capability frame missing column: bench")
    return CapabilityLevels(tuple(sorted(frame.bench.astype(str).unique())))


def capability_design(
    frame: pd.DataFrame,
    features: list[str],
    levels: CapabilityLevels,
    config: AnalysisConfig,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Build no-intercept population and shrunken benchmark slopes."""
    required = {"bench", *features}
    if missing := sorted(required - set(frame)):
        raise ValueError(f"capability frame missing columns: {missing}")
    multiplier = 1 / np.sqrt(float(config.benchmark_deviation_penalty))
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("benchmark_deviation_penalty must be finite and positive")

    blocks, records = [], []
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"feature {feature!r} contains non-finite values")
        blocks.append(sparse.csr_matrix(values[:, None]))
        records.append({
            "column": len(records), "block": "population_slope", "feature": feature,
            "benchmark": None, "multiplier": 1.0,
        })
        for benchmark in levels.benchmarks:
            active = frame.bench.astype(str).to_numpy() == benchmark
            blocks.append(sparse.csr_matrix((values * active * multiplier)[:, None]))
            records.append({
                "column": len(records), "block": "benchmark_slope", "feature": feature,
                "benchmark": benchmark, "multiplier": multiplier,
            })
    if not blocks:
        return sparse.csr_matrix((len(frame), 0)), pd.DataFrame(columns=[
            "column", "block", "feature", "benchmark", "multiplier",
        ])
    return sparse.hstack(blocks, format="csr"), pd.DataFrame(records)


def capability_coefficients(
    raw_coefficients: np.ndarray | pd.DataFrame,
    column_map: pd.DataFrame,
) -> pd.DataFrame:
    """Return natural-scale population, deviation, and effective slopes."""
    if isinstance(raw_coefficients, pd.DataFrame):
        if "coefficient" not in raw_coefficients:
            raise ValueError("raw coefficient table needs a 'coefficient' column")
        values = (column_map[["column"]].merge(
            raw_coefficients[["column", "coefficient"]], on="column",
            how="left", validate="one_to_one",
        ).coefficient.to_numpy(dtype=float) if "column" in raw_coefficients
                  else raw_coefficients.coefficient.to_numpy(dtype=float))
    else:
        values = np.asarray(raw_coefficients, dtype=float)
    if values.ndim != 1 or len(values) != len(column_map) or not np.isfinite(values).all():
        raise ValueError("raw coefficients must be finite and match the column map")

    mapped = column_map.assign(natural=values * column_map.multiplier.to_numpy(dtype=float))
    output = []
    for feature in mapped.loc[mapped.block == "population_slope", "feature"]:
        population = float(mapped.loc[
            (mapped.block == "population_slope") & (mapped.feature == feature), "natural"
        ].item())
        for row in mapped[(mapped.block == "benchmark_slope") &
                          (mapped.feature == feature)].itertuples(index=False):
            deviation = float(row.natural)
            output.append({
                "feature": feature, "benchmark": row.benchmark,
                "population": population, "benchmark_deviation": deviation,
                "effective": population + deviation,
            })
    return pd.DataFrame(output, columns=[
        "feature", "benchmark", "population", "benchmark_deviation", "effective",
    ])


CAPABILITY_TASK = SparseTask(
    outcome="left_stronger",
    group_columns=("bench", "item"),
    levels=capability_levels,
    design=capability_design,
    coefficients=capability_coefficients,
)


def _drop_judge_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column for column in frame
        if column == "pick_left" or column == "judge" or column.startswith("judge_")
    ]
    return frame.drop(columns=columns)


def prepare_capability_cell(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    dictionary: pd.DataFrame,
    model_fold: set[str],
    target_quartile: str,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct one held-out-model capability target cell from Q1 pairs."""
    del config  # The target population is independent of fitting configuration.
    features = dictionary.feature.astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("capability features must be non-empty and unique")
    held_out_models = set(model_fold)
    if not held_out_models:
        raise ValueError("model_fold must contain at least one model")
    quartiles = item_quartiles(responses)
    accuracies = quartile_accuracy(responses, quartiles)
    train, test = split_pairs(pairs, quartiles, held_out_models)
    train = attach_pair_target(deduplicate_feature_pairs(train), accuracies, target_quartile)
    test = attach_pair_target(deduplicate_feature_pairs(test), accuracies, target_quartile)

    rename = {f"d_{feature}": feature for feature in features}
    required = set(rename)
    if missing := sorted(required - set(train) | required - set(test)):
        raise ValueError(f"capability pair table missing columns: {missing}")
    fold = f"{'+'.join(sorted(held_out_models))}__{target_quartile}"
    prepared = []
    for frame in (train, test):
        output = _drop_judge_metadata(frame).rename(columns=rename)
        output["fold"] = fold
        prepared.append(output)
    return prepared[0], prepared[1], dictionary.copy()


def chance_predictions(test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the fixed no-predictor antisymmetric reference."""
    if "left_stronger" not in test:
        raise ValueError("capability test frame missing column: left_stronger")
    predictions = test.drop(columns=[*features, "left_stronger"], errors="ignore").copy()
    predictions["observed"] = test.left_stronger.to_numpy(dtype=int)
    predictions["probability"] = 0.5
    return predictions


def fit_capability_cell(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    dictionary: pd.DataFrame,
    model_fold: set[str],
    target_quartile: str,
    config: AnalysisConfig,
) -> CellResult:
    """Prepare, tune once, and fit one complete capability cell."""
    train, test, dictionary = prepare_capability_cell(
        pairs, responses, dictionary, model_fold, target_quartile, config,
    )
    prepared_train, prepared_test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, config.feature_space, config,
    )
    features = prepared_dictionary.feature.astype(str).tolist()
    frozen, tuning = tune_sparse(
        prepared_train, features, config.penalty, config, task=CAPABILITY_TASK,
    )
    full = fit_sparse(
        prepared_train, prepared_test, features, config.penalty,
        frozen, config, tuning, task=CAPABILITY_TASK,
    )
    return CellResult(
        prepared_train, prepared_test, prepared_dictionary, provenance,
        frozen, full, chance_predictions(prepared_test, features),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_response_source(
    responses_path: str | Path,
    response_manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load responses only when their explicit eligibility manifest matches."""
    responses_path, response_manifest_path = Path(responses_path), Path(response_manifest_path)
    response_sha256 = _sha256(responses_path)
    manifest = json.loads(response_manifest_path.read_text())
    expected = {
        "canonical_only": True,
        "min_reasoning_chars": 100,
        "row_count": None,
        "response_parquet_sha256": response_sha256,
    }
    for key, value in expected.items():
        if key not in manifest or (value is not None and manifest[key] != value):
            raise ValueError(f"response manifest requires {key}={value!r}")
    responses = pd.read_parquet(responses_path)
    if missing := sorted({"bench", "model", "item", "correct"} - set(responses)):
        raise ValueError(f"response table missing columns: {missing}")
    if manifest["row_count"] != len(responses):
        raise ValueError("response manifest row_count does not match the response table")
    return responses, {
        "responses_path": str(responses_path),
        "responses_sha256": response_sha256,
        "response_manifest_path": str(response_manifest_path),
        "response_manifest_sha256": _sha256(response_manifest_path),
    }


def build_specifications(
    dictionary: pd.DataFrame,
    feature_space: str,
    families: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[Specification]:
    """Reuse the judge inventory with capability's explicit null reference."""
    config = replace(
        JUDGE_CONFIG, feature_space=feature_space, lesion_families=tuple(families),
    )
    return [
        Specification(
            spec.name, spec.kind, spec.variable,
            "null" if spec.reference == "intercept-only" else spec.reference,
            spec.features,
        )
        for spec in _judge_build_specifications(dictionary, config)
    ]


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=lambda item: (
        item.item() if hasattr(item, "item") else str(item)
    )) + "\n"


def _default_response_manifest(responses_path: Path) -> Path:
    candidates = (
        responses_path.with_name(f"{responses_path.stem}-manifest.json"),
        responses_path.with_suffix(".manifest.json"),
    )
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError("response_manifest_path is required unless exactly one sidecar manifest exists")
    return found[0]


def _matrix_source(config: AnalysisConfig) -> dict[str, str]:
    matrix_config = Path(config.processed_matrix_config)
    values = yaml.safe_load(matrix_config.read_text())
    matrix_manifest = Path(values["output_root"]) / "manifest.json"
    digest = _sha256(matrix_manifest)
    if config.processed_matrix_manifest_sha256 and digest != config.processed_matrix_manifest_sha256:
        raise RuntimeError("processed matrix manifest SHA does not match the configured pin")
    return {
        "matrix_config_path": str(matrix_config),
        "matrix_config_sha256": _sha256(matrix_config),
        "matrix_manifest_path": str(matrix_manifest),
        "matrix_manifest_sha256": digest,
    }


def _variant_settings(
    scope: str, config: AnalysisConfig, matrix_source: dict[str, str],
    response_source: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": scope,
        "feature_space": config.feature_space,
        "penalty": config.penalty,
        "feature_blocks": list(config.feature_blocks),
        "regressors": list(config.regressors),
        "splits": list(config.splits),
        "judges": list(config.judges),
        "benchmarks": list(config.benchmarks),
        "model_folds": [[name, list(models)] for name, models in config.model_folds],
        "targets": list(TARGET_QUARTILES),
        "inner_folds": config.inner_folds,
        "fit_n_jobs": config.fit_n_jobs,
        "fit_pca_components": config.fit_pca_components,
        "fit_pca_algorithm": config.fit_pca_algorithm,
        "l1_cs": list(config.elastic_net_cs),
        "ridge_cs": list(config.ridge_cs),
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "glm_max_iter": config.glm_max_iter,
        "bootstrap_iterations": config.bootstrap_iterations,
        "random_seed": config.random_seed,
        "lesion_families": [[name, list(blocks)] for name, blocks in config.lesion_families],
        "matrix_source": matrix_source,
        "response_source": response_source,
    }


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    items = _item_metrics(predictions)
    weights = items.row_count.to_numpy(int)
    return pd.DataFrame({
        "metric": ["accuracy", "log_loss"],
        "value": [
            np.average(items.accuracy, weights=weights),
            np.average(items.log_loss, weights=weights),
        ],
        "row_count": [int(weights.sum())] * 2,
    })


def _specification_plan(
    dictionary: pd.DataFrame, feature_space: str,
    families: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[list[Specification], list[Specification], dict[str, str]]:
    features = tuple(dictionary.feature.astype(str))
    references = [
        Specification("full", "reference", "full", "", features),
        Specification("null", "reference", "null", "", ()),
    ]
    specifications = [*references, *build_specifications(dictionary, feature_space, families)]
    unique, aliases = unique_specifications(specifications)
    return specifications, unique, aliases


def _specification_table(
    specifications: list[Specification], aliases: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "specification": spec.name, "kind": spec.kind, "variable": spec.variable,
            "reference": spec.reference, "feature_count": len(spec.features),
            "features": json.dumps(spec.features), "artifact": aliases.get(spec.name, spec.name),
        }
        for spec in specifications
    ])


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _validate_external_source(source: dict[str, str], prefix: str) -> None:
    suffixes = ("matrix_config", "matrix_manifest") if prefix == "matrix" else (
        "responses", "response_manifest",
    )
    for suffix in suffixes:
        path_key, hash_key = f"{suffix}_path", f"{suffix}_sha256"
        path = Path(source.get(path_key, ""))
        if not path.is_file() or _sha256(path) != source.get(hash_key):
            raise RuntimeError(f"{prefix} source changed: {path}")


def _validate_cell(directory: Path, expected_settings: dict[str, object] | None = None) -> dict[str, object]:
    try:
        saved = json.loads((directory / "settings.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cell is not a completed checkpoint: {directory}") from error
    settings = saved.get("settings")
    if not isinstance(settings, dict) or saved.get("signature") != _signature(settings):
        raise RuntimeError(f"cell settings are corrupt: {directory}")
    if expected_settings is not None and settings != expected_settings:
        raise RuntimeError(f"cell settings mismatch: {directory}")
    hashes = manifest.get("artifact_sha256")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    required = {
        "settings.json", "feature-dictionary.parquet", "feature-provenance.json",
        "loadings.csv", "explained-variance.csv", "tuning.csv", "specifications.csv",
        "metrics.csv", "selection.csv", "diagnostics.json",
    }
    if (
        manifest.get("state") != "completed"
        or manifest.get("settings_signature") != saved["signature"]
        or manifest.get("variant_signature") != saved.get("variant_signature")
        or manifest.get("response_source") != settings.get("variant", {}).get("response_source")
        or manifest.get("matrix_source") != settings.get("variant", {}).get("matrix_source")
        or not isinstance(hashes, dict) or set(hashes) != actual or not required <= actual
        or any(_sha256(directory / name) != digest for name, digest in hashes.items())
    ):
        raise RuntimeError(f"cell checkpoint is stale or corrupt: {directory}")
    specifications = pd.read_csv(directory / "specifications.csv", keep_default_na=False)
    artifacts = set(specifications.artifact)
    expected_nested = {
        f"{kind}/{artifact}.parquet"
        for artifact in artifacts for kind in ("predictions", "coefficients", "item-metrics")
    }
    if not expected_nested <= actual:
        raise RuntimeError(f"cell artifact inventory is incomplete: {directory}")
    variant = settings["variant"]
    _validate_external_source(variant["matrix_source"], "matrix")
    _validate_external_source(variant["response_source"], "response")
    return manifest


def run_cell(
    cell_index: int,
    scope: str,
    config: AnalysisConfig,
    responses_path: str | Path,
    output_root: str | Path,
    *,
    response_manifest_path: str | Path | None = None,
    response_manifest_sha256: str | None = None,
) -> Path:
    """Write or validate one immutable model-fold by target checkpoint."""
    config = config.validate()
    _guard_execution(config)
    if not scope.strip():
        raise ValueError("scope cannot be empty")
    cells = [
        (name, set(models), target)
        for name, models in config.model_folds for target in TARGET_QUARTILES
    ]
    if not 0 <= cell_index < len(cells):
        raise ValueError(f"cell_index must be in range({len(cells)})")
    model_fold, held_out, target = cells[cell_index]
    responses_path = Path(responses_path)
    response_manifest_path = (
        Path(response_manifest_path) if response_manifest_path is not None
        else _default_response_manifest(responses_path)
    )
    if response_manifest_sha256 and _sha256(response_manifest_path) != response_manifest_sha256:
        raise RuntimeError("response manifest SHA does not match the supplied pin")
    responses, response_source = validate_response_source(responses_path, response_manifest_path)
    matrix_source = _matrix_source(config)
    variant = _variant_settings(scope, config, matrix_source, response_source)
    settings = {
        "variant": variant,
        "cell_index": cell_index,
        "model_fold": model_fold,
        "held_out_models": sorted(held_out),
        "target_quartile": target,
        "cell": f"{model_fold}__{target}",
    }
    output = Path(output_root) / scope / f"{config.feature_space}-{config.penalty}" / "cells" / settings["cell"]
    if output.exists():
        _validate_cell(output, settings)
        return output

    pairs, dictionary = _load_rows(config)
    cell = fit_capability_cell(pairs, responses, dictionary, held_out, target, config)
    specifications, unique, aliases = _specification_plan(
        cell.prepared_dictionary, config.feature_space, config.lesion_families,
    )
    by_name: dict[str, FitResult | pd.DataFrame] = {"full": cell.full, "null": cell.null}
    for spec in unique:
        if spec.name in by_name:
            continue
        if not spec.features:
            aliases[spec.name] = "null"
            continue
        by_name[spec.name] = fit_sparse(
            cell.prepared_train, cell.prepared_test, list(spec.features), config.penalty,
            cell.frozen_hyperparameters, config, task=CAPABILITY_TASK,
        )
    table = _specification_table(specifications, aliases)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{settings['cell']}-", dir=output.parent))
    try:
        for name in ("predictions", "coefficients", "item-metrics"):
            (stage / name).mkdir()
        payload = {
            "signature": _signature(settings),
            "variant_signature": _signature(variant),
            "settings": settings,
        }
        (stage / "settings.json").write_text(_json(payload))
        cell.prepared_dictionary.to_parquet(stage / "feature-dictionary.parquet", index=False)
        provenance = cell.provenance
        provenance["loadings"].to_csv(stage / "loadings.csv", index=False)
        provenance["explained_variance"].to_csv(stage / "explained-variance.csv", index=False)
        (stage / "feature-provenance.json").write_text(_json({
            key: value for key, value in provenance.items()
            if key not in {"loadings", "explained_variance"}
        }))
        cell.full.tuning.to_csv(stage / "tuning.csv", index=False)
        table.to_csv(stage / "specifications.csv", index=False)
        metric_rows, selection_rows, diagnostics = [], [], {}
        coefficient_columns = [
            "feature", "benchmark", "population", "benchmark_deviation", "effective",
        ]
        for artifact in table.artifact.drop_duplicates():
            value = by_name[artifact]
            if isinstance(value, FitResult):
                predictions, coefficients = value.predictions, value.coefficients
                selected, artifact_diagnostics = set(value.selected_features), value.diagnostics
            else:
                predictions = value
                coefficients = pd.DataFrame(columns=coefficient_columns)
                selected, artifact_diagnostics = set(), {
                    "C": cell.frozen_hyperparameters["C"],
                    "l1_ratio": cell.frozen_hyperparameters["l1_ratio"],
                    "reference": "fixed-0.5",
                }
            predictions = predictions.assign(model_fold=model_fold, target_quartile=target)
            predictions.to_parquet(stage / f"predictions/{artifact}.parquet", index=False)
            coefficients.to_parquet(stage / f"coefficients/{artifact}.parquet", index=False)
            _item_metrics(predictions).to_parquet(
                stage / f"item-metrics/{artifact}.parquet", index=False,
            )
            metric_rows.append(_prediction_metrics(predictions).assign(artifact=artifact))
            spec_features = next(spec.features for spec in unique if spec.name == artifact)
            selection_rows.extend({
                "artifact": artifact, "feature": feature, "selected": feature in selected,
            } for feature in spec_features)
            diagnostics[artifact] = artifact_diagnostics
        pd.concat(metric_rows, ignore_index=True).to_csv(stage / "metrics.csv", index=False)
        pd.DataFrame(selection_rows, columns=["artifact", "feature", "selected"]).to_csv(
            stage / "selection.csv", index=False,
        )
        (stage / "diagnostics.json").write_text(_json(diagnostics))
        manifest = {
            "state": "completed",
            "cell": settings["cell"],
            "settings_signature": payload["signature"],
            "variant_signature": payload["variant_signature"],
            "response_source": response_source,
            "matrix_source": matrix_source,
            "artifact_sha256": _artifact_hashes(stage),
        }
        (stage / "manifest.json").write_text(_json(manifest))
        _validate_cell(stage, settings)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def _summarize_predictions(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    probability = np.clip(rows.probability.to_numpy(float), 1e-12, 1 - 1e-12)
    observed = rows.observed.to_numpy(int)
    scored = rows.copy()
    scored["accuracy"] = ((probability >= 0.5) == observed).astype(float)
    scored["log_loss"] = -(observed * np.log(probability) + (1 - observed) * np.log1p(-probability))
    keys = ["specification", "kind", "variable", *groups]
    output = scored.groupby(keys, as_index=False, sort=True).agg(
        accuracy=("accuracy", "mean"), log_loss=("log_loss", "mean"),
        row_count=("accuracy", "size"),
    )
    return output.melt(
        id_vars=[*keys, "row_count"], value_vars=["accuracy", "log_loss"],
        var_name="metric", value_name="value",
    )


def _paired_item_deltas(
    item_metrics: dict[str, pd.DataFrame], specifications: pd.DataFrame,
    model_fold: str, target: str,
) -> pd.DataFrame:
    rows = []
    for spec in specifications[specifications.kind != "reference"].itertuples(index=False):
        specification = Specification(
            spec.specification, spec.kind, spec.variable, spec.reference, (),
        )
        rows.append(_paired_item_metrics(
            item_metrics[spec.artifact], item_metrics[spec.reference],
            specification, model_fold, backend="capability",
        ).drop(columns="backend").rename(columns={"fold": "model_fold"}).assign(
            target_quartile=target,
        ))
    return pd.concat(rows, ignore_index=True)


def _delta_summary(rows: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scoped = pd.concat([
        rows.assign(scope="overall", group_target="all", group_benchmark="all"),
        rows.assign(scope="target", group_target=rows.target_quartile, group_benchmark="all"),
        rows.assign(scope="benchmark", group_target="all", group_benchmark=rows.bench),
        rows.assign(
            scope="target×benchmark", group_target=rows.target_quartile,
            group_benchmark=rows.bench,
        ),
    ], ignore_index=True)
    keys = [
        "specification", "kind", "variable", "reference", "scope",
        "group_target", "group_benchmark",
    ]
    output = []
    for values, group in scoped.groupby(keys, sort=True):
        weights = group.row_count.to_numpy(int)
        estimates = {
            metric: float(np.average(group[metric].to_numpy(float), weights=weights))
            for metric in ("delta_accuracy", "delta_log_loss")
        }
        clusters = group[["bench", "item", "row_count"]].copy()
        for metric in estimates:
            clusters[metric] = group[metric].to_numpy(float) * weights
        clusters = clusters.groupby(["bench", "item"], as_index=False, sort=False).sum()
        draws = rng.integers(len(clusters), size=(iterations, len(clusters)))
        draw_weights = clusters.row_count.to_numpy()[draws].sum(axis=1)
        intervals = {
            metric: np.percentile(
                clusters[metric].to_numpy()[draws].sum(axis=1) / draw_weights, [2.5, 97.5],
            )
            for metric in estimates
        }
        output.append({
            **dict(zip(keys, values)),
            **estimates,
            "delta_accuracy_lower": intervals["delta_accuracy"][0],
            "delta_accuracy_upper": intervals["delta_accuracy"][1],
            "delta_log_loss_lower": intervals["delta_log_loss"][0],
            "delta_log_loss_upper": intervals["delta_log_loss"][1],
            "fold_count": group.model_fold.nunique(),
            "target_count": group.target_quartile.nunique(),
            "item_count": len(clusters),
            "row_count": int(weights.sum()),
        })
    return pd.DataFrame(output)


def _validate_aggregate(
    directory: Path, settings: dict[str, object], cell_hashes: dict[str, str],
) -> None:
    try:
        saved = json.loads((directory / "settings.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"aggregate is not complete: {directory}") from error
    hashes = manifest.get("artifact_sha256")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    expected = {
        "settings.json", "predictions.parquet", "item-metrics.parquet", "metrics.csv",
        "target-metrics.csv", "benchmark-metrics.csv", "fold-metrics.csv",
        "paired-item-deltas.parquet", "delta-summary.csv", "selection.csv",
        "coefficients.csv", "tuning.csv", "artifact-index.csv",
    }
    if (
        saved != {"signature": _signature(settings), "settings": settings}
        or manifest.get("state") != "completed"
        or manifest.get("settings_signature") != saved["signature"]
        or manifest.get("cell_manifest_sha256") != cell_hashes
        or manifest.get("response_source") != settings.get("response_source")
        or manifest.get("matrix_source") != settings.get("matrix_source")
        or not isinstance(hashes, dict) or set(hashes) != actual or actual != expected
        or any(_sha256(directory / name) != digest for name, digest in hashes.items())
    ):
        raise RuntimeError(f"aggregate is stale or corrupt: {directory}")


def aggregate_variant(
    scope: str,
    feature_space: str,
    penalty: str,
    output_root: str | Path,
    *,
    matrix_source: dict[str, str] | None = None,
    response_source: dict[str, str] | None = None,
) -> Path:
    """Aggregate exactly twelve compatible, hash-valid capability cells."""
    if feature_space not in {"raw", "pc"} or penalty not in {"l1", "ridge"}:
        raise ValueError("variant must be raw|pc and l1|ridge")
    root = Path(output_root) / scope / f"{feature_space}-{penalty}"
    cells_root = root / "cells"
    cell_dirs = sorted(path for path in cells_root.iterdir() if path.is_dir()) if cells_root.is_dir() else []
    if len(cell_dirs) != 12:
        raise RuntimeError(f"aggregation requires exactly 12 cells, found {len(cell_dirs)}")
    payloads = []
    for directory in cell_dirs:
        _validate_cell(directory)
        payloads.append(json.loads((directory / "settings.json").read_text()))
    variant_signatures = {payload["variant_signature"] for payload in payloads}
    variants = [payload["settings"]["variant"] for payload in payloads]
    if len(variant_signatures) != 1 or any(variant != variants[0] for variant in variants[1:]):
        raise RuntimeError("cell variant settings signatures differ")
    variant = variants[0]
    if (variant["scope"], variant["feature_space"], variant["penalty"]) != (
        scope, feature_space, penalty,
    ):
        raise RuntimeError("cell variant does not match requested aggregate")
    if (
        (matrix_source is not None and matrix_source != variant["matrix_source"])
        or (response_source is not None and response_source != variant["response_source"])
    ):
        raise RuntimeError("supplied source provenance does not match cell provenance")
    expected_cells = {
        f"{name}__{target}"
        for name, _ in variant["model_folds"] for target in TARGET_QUARTILES
    }
    settings_rows = [payload["settings"] for payload in payloads]
    if (
        len(expected_cells) != 12
        or {settings["cell"] for settings in settings_rows} != expected_cells
        or {settings["cell_index"] for settings in settings_rows} != set(range(12))
        or {directory.name for directory in cell_dirs} != expected_cells
    ):
        raise RuntimeError("capability cell inventory differs from the 12-cell plan")
    cell_hashes = {
        directory.name: _sha256(directory / "manifest.json") for directory in cell_dirs
    }
    settings = {
        "schema_version": 1,
        "scope": scope,
        "feature_space": feature_space,
        "penalty": penalty,
        "variant_signature": next(iter(variant_signatures)),
        "cells": sorted(expected_cells),
        "response_source": variant["response_source"],
        "matrix_source": variant["matrix_source"],
        "bootstrap_iterations": variant["bootstrap_iterations"],
        "random_seed": variant["random_seed"],
    }
    output = root / "aggregate"
    if output.exists():
        _validate_aggregate(output, settings, cell_hashes)
        return output

    predictions, coefficients, selections, tunings, item_rows, delta_rows, index_rows = (
        [], [], [], [], [], [], [],
    )
    for directory, cell_settings in zip(cell_dirs, settings_rows):
        model_fold, target = cell_settings["model_fold"], cell_settings["target_quartile"]
        specifications = pd.read_csv(directory / "specifications.csv", keep_default_na=False)
        item_by_artifact = {
            artifact: pd.read_parquet(directory / f"item-metrics/{artifact}.parquet")
            for artifact in specifications.artifact.unique()
        }
        delta_rows.append(_paired_item_deltas(item_by_artifact, specifications, model_fold, target))
        selection = pd.read_csv(directory / "selection.csv", keep_default_na=False)
        tuning = pd.read_csv(directory / "tuning.csv").assign(
            model_fold=model_fold, target_quartile=target,
        )
        tunings.append(tuning)
        for spec in specifications.itertuples(index=False):
            metadata = {
                "model_fold": model_fold, "target_quartile": target,
                "specification": spec.specification, "kind": spec.kind,
                "variable": spec.variable, "reference": spec.reference,
            }
            prediction_path = directory / f"predictions/{spec.artifact}.parquet"
            coefficient_path = directory / f"coefficients/{spec.artifact}.parquet"
            item_path = directory / f"item-metrics/{spec.artifact}.parquet"
            predictions.append(pd.read_parquet(prediction_path).assign(**metadata))
            coefficient = pd.read_parquet(coefficient_path)
            if not coefficient.empty:
                coefficients.append(coefficient.assign(**metadata))
            selected = selection[selection.artifact == spec.artifact].drop(columns="artifact")
            if not selected.empty:
                selections.append(selected.assign(**metadata))
            item_rows.append(item_by_artifact[spec.artifact].assign(**metadata))
            index_rows.append({
                "cell": cell_settings["cell"], **metadata, "artifact": spec.artifact,
                "predictions": str(prediction_path), "coefficients": str(coefficient_path),
                "item_metrics": str(item_path), "cell_manifest": str(directory / "manifest.json"),
            })
    all_predictions = pd.concat(predictions, ignore_index=True)
    references = all_predictions[all_predictions.kind == "reference"]
    paired = pd.concat(delta_rows, ignore_index=True)
    summary = _delta_summary(
        paired, int(variant["bootstrap_iterations"]), int(variant["random_seed"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".aggregate-", dir=output.parent))
    try:
        (stage / "settings.json").write_text(_json({
            "signature": _signature(settings), "settings": settings,
        }))
        all_predictions.to_parquet(stage / "predictions.parquet", index=False)
        pd.concat(item_rows, ignore_index=True).to_parquet(stage / "item-metrics.parquet", index=False)
        _summarize_predictions(references, []).to_csv(stage / "metrics.csv", index=False)
        _summarize_predictions(references, ["target_quartile"]).to_csv(
            stage / "target-metrics.csv", index=False,
        )
        _summarize_predictions(references, ["bench"]).rename(columns={"bench": "benchmark"}).to_csv(
            stage / "benchmark-metrics.csv", index=False,
        )
        _summarize_predictions(references, ["model_fold"]).to_csv(
            stage / "fold-metrics.csv", index=False,
        )
        paired.to_parquet(stage / "paired-item-deltas.parquet", index=False)
        summary.to_csv(stage / "delta-summary.csv", index=False)
        pd.concat(selections, ignore_index=True).to_csv(stage / "selection.csv", index=False)
        pd.concat(coefficients, ignore_index=True).to_csv(stage / "coefficients.csv", index=False)
        pd.concat(tunings, ignore_index=True).to_csv(stage / "tuning.csv", index=False)
        pd.DataFrame(index_rows).to_csv(stage / "artifact-index.csv", index=False)
        manifest = {
            "state": "completed",
            "settings_signature": _signature(settings),
            "cell_manifest_sha256": cell_hashes,
            "response_source": variant["response_source"],
            "matrix_source": variant["matrix_source"],
            "artifact_sha256": _artifact_hashes(stage),
        }
        (stage / "manifest.json").write_text(_json(manifest))
        _validate_aggregate(stage, settings, cell_hashes)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cell", "aggregate"), required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--matrix-manifest-sha256", required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--response-manifest", type=Path, required=True)
    parser.add_argument("--response-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--feature-space", choices=("raw", "pc"), required=True)
    parser.add_argument("--penalty", choices=("l1", "ridge"), required=True)
    parser.add_argument("--cell-index", type=int)
    args = parser.parse_args(argv)
    if _sha256(args.response_manifest) != args.response_manifest_sha256:
        raise RuntimeError("response manifest SHA does not match the supplied pin")
    config = replace(
        JUDGE_CONFIG,
        processed_matrix_root=args.matrix_root,
        processed_matrix_manifest_sha256=args.matrix_manifest_sha256,
        feature_space=args.feature_space,
        penalty=args.penalty,
        execution_mode="execute",
        execution_target="swc_hpc",
        resume_completed_folds=True,
    )
    if args.mode == "cell":
        index = args.cell_index
        if index is None:
            try:
                index = int(os.environ["SLURM_ARRAY_TASK_ID"])
            except (KeyError, ValueError) as error:
                raise RuntimeError("cell mode requires --cell-index or SLURM_ARRAY_TASK_ID") from error
        run_cell(
            index, args.scope, config, args.responses, args.output_root,
            response_manifest_path=args.response_manifest,
            response_manifest_sha256=args.response_manifest_sha256,
        )
    else:
        _, response_source = validate_response_source(args.responses, args.response_manifest)
        matrix_source = _matrix_source(config)
        aggregate_variant(
            args.scope, args.feature_space, args.penalty, args.output_root,
            matrix_source=matrix_source, response_source=response_source,
        )


if __name__ == "__main__":
    main()
