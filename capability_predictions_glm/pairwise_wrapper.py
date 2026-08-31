"""Thin capability-target adapter for the shared sparse pairwise fitter."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

import jax.numpy as jnp
from jax import random
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
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
from scripts.judge_verdict_analysis.pooling import penalty_multiplier

from .data import _sha256
from .preprocessing import _model_item_rates, deduplicate_feature_pairs


DETERMINISTIC_EXCLUDED_BLOCKS = frozenset({"rubric", "intermediate-errors", "annotations"})


def deterministic_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    """Return the predeclared 90-composite deterministic-only feature set."""
    required = {"feature", "block", "leaf_members"}
    if missing := required - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")

    def contains_bare_answer(value: object) -> bool:
        try:
            members = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("leaf_members must be a JSON list of strings") from error
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            raise ValueError("leaf_members must be a JSON list of strings")
        return "bare_answer" in members

    selected = dictionary[
        ~dictionary.block.isin(DETERMINISTIC_EXCLUDED_BLOCKS)
        & ~dictionary.leaf_members.map(contains_bare_answer)
    ].reset_index(drop=True)
    if len(selected) != 90:
        raise ValueError(f"expected 90 deterministic composites, found {len(selected)}")
    return selected


def _capability_levels(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(frame.bench.astype(str).unique()))


def _capability_design(frame, features, levels, config):
    missing = sorted({"bench", *features} - set(frame))
    if missing:
        raise ValueError(f"capability frame missing columns: {missing}")
    values = frame[features].to_numpy(dtype=float)
    multiplier = penalty_multiplier(config.benchmark_deviation_penalty, config.penalty)
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


def _capability_scale_model(
    x: jnp.ndarray, y: jnp.ndarray, benchmark: jnp.ndarray,
    benchmark_count: int, penalty: str, scale_prior: float,
) -> None:
    """No-intercept capability hierarchy used only to estimate block scales."""
    family = dist.Laplace if penalty == "l1" else dist.Normal
    population_scale = numpyro.sample("population_scale", dist.HalfNormal(scale_prior))
    benchmark_scale = numpyro.sample("benchmark_scale", dist.HalfNormal(scale_prior))
    population = numpyro.sample(
        "population", family(0, population_scale).expand((x.shape[1],)).to_event(1),
    )
    deviations = numpyro.sample(
        "benchmark",
        family(0, benchmark_scale).expand((benchmark_count, x.shape[1])).to_event(2),
    )
    logits = jnp.sum(x * (population + deviations[benchmark]), axis=1)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def _estimate_capability_pooling_ratio(
    train: pd.DataFrame, features: list[str], penalty: str, config: AnalysisConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate the benchmark-deviation ratio from one outer-training fold."""
    if penalty not in {"l1", "ridge"}:
        raise ValueError("penalty must be 'l1' or 'ridge'")
    required = {"bench", "left_stronger", *features}
    if missing := sorted(required - set(train)):
        raise ValueError(f"capability pooling rows missing columns: {missing}")
    x = train[features].to_numpy(np.float32)
    y = train.left_stronger.to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("capability pooling features must be finite and outcomes binary")
    benchmark, levels = pd.factorize(train.bench.astype(str), sort=True)
    guide = AutoNormal(_capability_scale_model)
    result = SVI(
        _capability_scale_model, guide,
        Adam(config.pooling_svi_learning_rate), Trace_ELBO(),
    ).run(
        random.PRNGKey(config.random_seed), config.pooling_svi_steps,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(benchmark.astype(np.int32)),
        len(levels), penalty, float(config.pooling_scale_prior), progress_bar=False,
    )
    losses = np.asarray(result.losses, dtype=float)
    if not np.isfinite(losses).all():
        raise RuntimeError("capability variance-component fit produced non-finite loss")
    median = guide.median(result.params)
    population_scale = float(np.asarray(median["population_scale"]))
    benchmark_scale = float(np.asarray(median["benchmark_scale"]))
    if any(not np.isfinite(value) or value <= 0
           for value in (population_scale, benchmark_scale)):
        raise RuntimeError("capability variance-component fit produced invalid scales")
    power = 1 if penalty == "l1" else 2
    ratio = (population_scale / benchmark_scale) ** power
    diagnostics = pd.DataFrame([{
        "stage": "pooling", "record_type": "variance_component",
        "penalty": penalty, "selected": False, "converged": True,
        "svi_steps": config.pooling_svi_steps,
        "svi_initial_loss": losses[0], "svi_final_loss": losses[-1],
        "population_scale": population_scale, "benchmark_scale": benchmark_scale,
        "benchmark_deviation_penalty": ratio,
    }])
    return {"benchmark_deviation_penalty": ratio}, diagnostics


CAPABILITY_TASK = SparseTask(
    outcome="left_stronger",
    group_columns=("bench", "item"),
    levels=_capability_levels,
    design=_capability_design,
    coefficients=_capability_coefficients,
    pooling_parameters=("benchmark_deviation_penalty",),
    estimate_pooling=_estimate_capability_pooling_ratio,
)


def overall_model_accuracy(responses: pd.DataFrame) -> pd.DataFrame:
    """Return equal-item canonical accuracy for every benchmark and model."""
    required = {"bench", "model", "item", "correct", "is_canonical"}
    if missing := required - set(responses):
        raise ValueError(f"response table missing columns: {sorted(missing)}")
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


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _sha256_arg(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("must be a 64-character SHA-256")
    return value.lower()


def _verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned input: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")


def fit_fold(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    dictionary: pd.DataFrame,
    fold_index: int,
    config: AnalysisConfig = CONFIG,
    *,
    manifest_provenance: dict[str, object] | None = None,
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
    selected = tuning[tuning.selected]
    if not result.diagnostics["converged"] or selected.empty or not selected.converged.all():
        raise RuntimeError(f"full capability fit or selected C tuning did not converge: {fold}")
    _write_result(
        result, train, test, config,
        prepared_dictionary=prepared_dictionary,
        provenance=provenance | {"frozen_hyperparameters": frozen},
        manifest_provenance=manifest_provenance,
    )
    return config.fit_dir / result.backend / result.fold


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-manifest-sha256", type=_sha256_arg, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--responses-sha256", type=_sha256_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--feature-space", choices=("raw", "pc"), required=True)
    parser.add_argument("--penalty", choices=("l1", "ridge"), required=True,
                        help="L1 or ridge (L2)")
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--max-iter", type=_positive_int, default=20_000)
    parser.add_argument("--pc-basis-root", type=Path)
    parser.add_argument("--pc-components", type=_positive_int)
    parser.add_argument("--pc-loadings-sha256", type=_sha256_arg)
    parser.add_argument("--pc-explained-variance-sha256", type=_sha256_arg)
    args = parser.parse_args(argv)
    pc_arguments = {
        "--pc-basis-root": args.pc_basis_root,
        "--pc-components": args.pc_components,
        "--pc-loadings-sha256": args.pc_loadings_sha256,
        "--pc-explained-variance-sha256": args.pc_explained_variance_sha256,
    }
    if args.feature_space == "pc":
        if missing := [name for name, value in pc_arguments.items() if value is None]:
            parser.error(f"{', '.join(missing)} required for PC features")
    elif any(value is not None for value in pc_arguments.values()):
        parser.error("PC-only arguments require --feature-space pc")
    if args.deterministic_only and args.feature_space != "raw":
        parser.error("--deterministic-only requires --feature-space raw")
    config = replace(
        CONFIG, processed_matrix_root=args.matrix, results_dir=args.output,
        processed_matrix_manifest_sha256=args.matrix_manifest_sha256,
        feature_space=args.feature_space, penalty=args.penalty, glm_max_iter=args.max_iter,
        execution_mode="execute", execution_target="swc_hpc",
    )
    if args.feature_space == "pc":
        config = replace(
            config, pc_basis_root=args.pc_basis_root,
            fit_pca_components=args.pc_components,
        )
    _guard_execution(config)
    _verify_sha256(args.responses, args.responses_sha256)
    source_paths = {"responses": str(args.responses)}
    source_sha256 = {"responses": args.responses_sha256}
    if args.feature_space == "pc":
        loadings = args.pc_basis_root / "loadings.parquet"
        variance = args.pc_basis_root / "explained-variance.csv"
        _verify_sha256(loadings, args.pc_loadings_sha256)
        _verify_sha256(variance, args.pc_explained_variance_sha256)
        source_paths.update({
            "pc_basis": str(args.pc_basis_root),
            "pc_loadings": str(loadings),
            "pc_explained_variance": str(variance),
        })
        source_sha256.update({
            "pc_loadings": args.pc_loadings_sha256,
            "pc_explained_variance": args.pc_explained_variance_sha256,
        })
    pairs, dictionary = _load_rows(config)
    feature_subset = None
    if args.deterministic_only:
        dictionary = deterministic_dictionary(dictionary)
        feature_subset = {
            "kind": "deterministic-only",
            "feature_count": len(dictionary),
            "features": dictionary.feature.astype(str).tolist(),
            "excluded_blocks": sorted(DETERMINISTIC_EXCLUDED_BLOCKS),
            "excluded_leaf": "bare_answer",
        }
    manifest_provenance = {
        "source_paths": source_paths,
        "source_sha256": source_sha256,
    }
    if feature_subset is not None:
        manifest_provenance["feature_subset"] = feature_subset
    fit_fold(
        pairs, pd.read_parquet(args.responses), dictionary, args.fold_index, config,
        manifest_provenance=manifest_provenance,
    )


if __name__ == "__main__":
    main()
