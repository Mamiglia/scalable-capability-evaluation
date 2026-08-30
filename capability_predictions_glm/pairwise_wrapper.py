"""Thin capability-target adapter for the shared sparse pairwise fitter."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.judge_verdict_analysis.config import AnalysisConfig, CONFIG
from scripts.judge_verdict_analysis.fit import (
    _guard_execution,
    _load_rows,
    _write_result,
)
from scripts.judge_verdict_analysis.fit_common import prepare_features
from scripts.judge_verdict_analysis.fit_frequentist import (
    SparseTask,
    fit_sparse,
    tune_sparse,
)
from scripts.judge_verdict_analysis.hierarchy import outer_roles

from .preprocessing import _model_item_rates, deduplicate_feature_pairs


def _capability_levels(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(frame.bench.astype(str).unique()))


def _capability_design(frame, features, levels, config):
    missing = sorted({"bench", *features} - set(frame))
    if missing:
        raise ValueError(f"capability frame missing columns: {missing}")
    values = frame[features].to_numpy(dtype=float)
    multiplier = 1 / np.sqrt(float(config.benchmark_deviation_penalty))
    if not np.isfinite(values).all() or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("capability design values and penalty must be finite and positive")
    blocks = [sparse.csr_matrix(values)]
    records = [
        {"column": index, "block": "population_slope", "feature": feature,
         "benchmark": None, "multiplier": 1.0}
        for index, feature in enumerate(features)
    ]
    for benchmark in levels:
        active = (frame.bench.astype(str).to_numpy() == benchmark)[:, None]
        blocks.append(sparse.csr_matrix(values * active * multiplier))
        records.extend(
            {"column": len(records), "block": "benchmark_slope", "feature": feature,
             "benchmark": benchmark, "multiplier": multiplier}
            for feature in features
        )
    return sparse.hstack(blocks, format="csr"), pd.DataFrame(records)


def _capability_coefficients(raw: np.ndarray, columns: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) != len(columns) or not np.isfinite(values).all():
        raise ValueError("raw coefficients must be finite and match the column map")
    mapped = columns.assign(natural=values * columns.multiplier.to_numpy(dtype=float))
    population = (mapped[mapped.block == "population_slope"]
                  .set_index("feature").natural)
    rows = mapped[mapped.block == "benchmark_slope"].copy()
    rows["population"] = rows.feature.map(population)
    rows = rows.rename(columns={"natural": "benchmark_deviation"})
    rows["effective"] = rows.population + rows.benchmark_deviation
    return rows[["feature", "benchmark", "population", "benchmark_deviation", "effective"]]


CAPABILITY_TASK = SparseTask(
    outcome="left_stronger",
    group_columns=("bench", "item"),
    levels=_capability_levels,
    design=_capability_design,
    coefficients=_capability_coefficients,
)


def overall_model_accuracy(responses: pd.DataFrame) -> pd.DataFrame:
    """Return equal-item canonical accuracy for every benchmark and model."""
    return (_model_item_rates(responses)
            .groupby(["bench", "model"], as_index=False, sort=False).correct_rate.mean()
            .rename(columns={"correct_rate": "accuracy"}))


def _attach_accuracy(pairs: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    output = pairs.drop(columns=[
        "accuracy_left", "accuracy_right", "accuracy_difference", "left_stronger",
    ], errors="ignore").copy()
    accuracies = overall_model_accuracy(responses)
    for side in ("left", "right"):
        model, accuracy = f"model_{side}", f"accuracy_{side}"
        output = output.merge(
            accuracies.rename(columns={"model": model, "accuracy": accuracy}),
            on=["bench", model], how="left", validate="many_to_one",
        )
        missing = (output.loc[output[accuracy].isna(), ["bench", model]]
                   .drop_duplicates().rename(columns={model: "model"})
                   .sort_values(["bench", "model"]))
        if not missing.empty:
            raise ValueError(
                f"missing overall accuracy for {side} benchmark/model keys: "
                f"{missing.to_dict('records')}"
            )
    return output


def _finish_rows(rows: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    features = dictionary.feature.astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("capability features must be non-empty and unique")
    if missing := sorted(set(features) - set(rows)):
        raise ValueError(f"capability pair table missing columns: {missing}")
    rows = rows[rows.accuracy_left != rows.accuracy_right].copy()
    rows["left_stronger"] = (rows.accuracy_left > rows.accuracy_right).astype(int)
    metadata = [
        column for column in rows
        if column in {"judge", "pick_left"} or column.startswith("judge_")
    ]
    return rows.drop(columns=metadata).reset_index(drop=True)


def prepare_capability_rows(
    pairs: pd.DataFrame, responses: pd.DataFrame, dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Deduplicate pair evidence and attach checked overall capability labels."""
    return _finish_rows(
        _attach_accuracy(deduplicate_feature_pairs(pairs), responses), dictionary,
    )


def prepare_capability_fold(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    dictionary: pd.DataFrame,
    held_out_models: set[str],
    item_fold: int,
    config: AnalysisConfig,
    fold_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the shared strict new-model/new-item split to labelled rows."""
    rows = _attach_accuracy(deduplicate_feature_pairs(pairs), responses)
    roles = outer_roles(rows, held_out_models, item_fold, config.item_folds)
    fold = fold_name or f"{'+'.join(sorted(held_out_models))}__item-{item_fold}"
    return tuple(
        _finish_rows(rows.loc[roles == role], dictionary).assign(fold=fold)
        for role in ("train", "test")
    )


def chance_predictions(test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the fixed 0.5 no-feature reference."""
    if "left_stronger" not in test:
        raise ValueError("capability test frame missing column: left_stronger")
    output = test.drop(columns=[*features, "left_stronger"], errors="ignore").copy()
    output["observed"] = test.left_stronger.to_numpy(dtype=int)
    output["probability"] = 0.5
    return output


def fit_fold(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    dictionary: pd.DataFrame,
    fold_index: int,
    config: AnalysisConfig = CONFIG,
) -> Path:
    """Fit and write one configured model-by-item capability fold."""
    config = config.validate()
    _guard_execution(config)
    folds = [
        (name, set(models), item_fold)
        for name, models in config.model_folds
        for item_fold in range(config.item_folds)
    ]
    if not 0 <= fold_index < len(folds):
        raise ValueError(f"fold_index must be in range({len(folds)})")
    model_fold, held_out, item_fold = folds[fold_index]
    fold = f"{model_fold}__item-{item_fold}"
    train, test = prepare_capability_fold(
        pairs, responses, dictionary, held_out, item_fold, config, fold,
    )
    if train.empty or test.empty:
        raise ValueError(f"outer fold {fold} has no {'train' if train.empty else 'test'} rows")
    train, test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, config.feature_space, config,
    )
    features = prepared_dictionary.feature.astype(str).tolist()
    frozen, tuning = tune_sparse(
        train, features, config.penalty, config, task=CAPABILITY_TASK,
    )
    result = fit_sparse(
        train, test, features, config.penalty, frozen, config, tuning,
        task=CAPABILITY_TASK,
    )
    _write_result(
        result, train, test, config,
        prepared_dictionary=prepared_dictionary,
        provenance=provenance | {"frozen_hyperparameters": frozen},
    )
    return config.fit_dir / result.backend / result.fold


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--feature-space", choices=("raw", "pc"), required=True)
    parser.add_argument("--penalty", choices=("l1", "ridge"), required=True,
                        help="L1 or ridge (L2)")
    args = parser.parse_args(argv)
    config = replace(
        CONFIG, processed_matrix_root=args.matrix, results_dir=args.output,
        feature_space=args.feature_space, penalty=args.penalty,
        execution_mode="execute", execution_target="swc_hpc",
    )
    _guard_execution(config)
    pairs, dictionary = _load_rows(config)
    fit_fold(pairs, pd.read_parquet(args.responses), dictionary, args.fold_index, config)


if __name__ == "__main__":
    main()
