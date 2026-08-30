"""Pairwise capability adapter for the shared sparse judge fitter."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.judge_verdict_analysis.config import AnalysisConfig
from scripts.judge_verdict_analysis.fit_common import FitResult, prepare_features
from scripts.judge_verdict_analysis.fit_frequentist import SparseTask, fit_sparse, tune_sparse

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
        output = frame.rename(columns=rename).drop(columns="judge", errors="ignore")
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
