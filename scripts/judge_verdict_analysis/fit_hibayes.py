"""HiBayes model, job generation, and saved-fit evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpy as np
import pandas as pd
import yaml
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features

if TYPE_CHECKING:
    from .config import AnalysisConfig
    from .fit_common import FitResult


def _deviations(name: str, shape: tuple[int, ...], prior_scale: float) -> jnp.ndarray:
    spread = numpyro.sample(f"{name}_spread", dist.HalfNormal(prior_scale).expand(shape[1:]).to_event(len(shape) - 1))
    offsets = numpyro.sample(f"{name}_offsets", dist.Normal(0, 1).expand(shape).to_event(len(shape)))
    return numpyro.deterministic(name, spread * offsets)


@model
def judge_verdict_hibayes(
    predictors: Sequence[str], prior_scales: Mapping[str, float]
) -> Model:
    """Joint crossed-hierarchy model for pairwise judge verdicts."""
    required_scales = {"population", "judge", "benchmark", "cell", "item"}
    if set(prior_scales) != required_scales or any(
        value <= 0 or not isfinite(value) for value in prior_scales.values()
    ):
        raise ValueError(f"prior_scales must contain positive values for {sorted(required_scales)}")

    def statistical_model(features: Features) -> None:
        check_features(features, [
            "obs", "judge_index", "num_judge", "bench_index", "num_bench",
            "cell_index", "num_cell", "item_key_index", "num_item_key", *predictors,
        ])
        n_predictors = len(predictors)
        population_intercept = numpyro.sample(
            "population_intercept", dist.Normal(0, prior_scales["population"])
        )
        population_slopes = numpyro.sample(
            "population_slopes",
            dist.Normal(0, prior_scales["population"]).expand((n_predictors,)).to_event(1),
        )
        judge_intercepts = _deviations(
            "judge_intercepts", (features["num_judge"],), prior_scales["judge"]
        )
        judge_slopes = _deviations(
            "judge_slope_deviations",
            (features["num_judge"], n_predictors),
            prior_scales["judge"],
        )
        benchmark_intercepts = _deviations(
            "benchmark_intercepts", (features["num_bench"],), prior_scales["benchmark"]
        )
        benchmark_slopes = _deviations(
            "benchmark_slope_deviations",
            (features["num_bench"], n_predictors),
            prior_scales["benchmark"],
        )
        cell_intercepts = _deviations(
            "cell_intercepts", (features["num_cell"],), prior_scales["cell"]
        )
        cell_slopes = _deviations(
            "cell_slope_deviations",
            (features["num_cell"], n_predictors),
            prior_scales["cell"],
        )
        item_intercepts = _deviations(
            "item_intercepts", (features["num_item_key"],), prior_scales["item"]
        )

        judge = features["judge_index"]
        benchmark = features["bench_index"]
        cell = features["cell_index"]
        item = features["item_key_index"]
        design = (jnp.stack([features[name] for name in predictors], axis=1) if predictors
                  else jnp.empty((features["obs"].shape[0], 0)))
        slopes = population_slopes + judge_slopes[judge] + benchmark_slopes[benchmark] + cell_slopes[cell]
        logits = (
            population_intercept
            + judge_intercepts[judge]
            + benchmark_intercepts[benchmark]
            + cell_intercepts[cell]
            + item_intercepts[item]
            + jnp.sum(design * slopes, axis=1)
        )
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=features["obs"])

    return statistical_model


def _prior_scales(config: AnalysisConfig) -> dict[str, float]:
    return {
        "population": config.hibayes_population_prior_scale,
        "judge": config.hibayes_judge_prior_scale,
        "benchmark": config.hibayes_benchmark_prior_scale,
        "cell": config.hibayes_cell_prior_scale,
        "item": config.hibayes_item_prior_scale,
    }


def _key(*parts: object) -> str:
    return json.dumps([str(part) for part in parts], separators=(",", ":"))


def write_hibayes_job(
    train_path: Path,
    test_path: Path,
    selected_features: Sequence[str],
    fold: str,
    config: AnalysisConfig,
) -> Path:
    """Write one self-contained HiBayes input and YAML config for an outer fold."""
    from .hierarchy import HierarchyLevels

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    required = {"judge", "bench", "item", "pick_left", *selected_features}
    for name, frame in (("train", train), ("test", test)):
        if missing := sorted(required - set(frame)):
            raise ValueError(f"{name} input is missing columns: {missing}")

    levels = HierarchyLevels.from_training(train)
    unseen = {
        (str(row.judge), str(row.bench))
        for row in test[["judge", "bench"]].itertuples(index=False)
    } - set(levels.cells)
    if unseen:
        raise ValueError(f"test input contains unseen judge-benchmark cells: {sorted(unseen)}")

    columns = ["judge", "bench", "item", "pick_left", *selected_features]
    frame = pd.concat([
        train[columns].assign(test=False),
        test[columns].assign(test=True),
    ], ignore_index=True)
    frame["cell"] = [_key(judge, bench) for judge, bench in zip(frame.judge, frame.bench)]
    frame["item_key"] = [_key(bench, item) for bench, item in zip(frame.bench, frame.item)]

    job_dir = config.fit_dir / "hibayes" / fold
    job_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [job_dir / "train.parquet", job_dir / "test.parquet"]
    frame[~frame.test].to_parquet(input_paths[0], index=False)
    frame[frame.test].to_parquet(input_paths[1], index=False)
    category_order = {
        "judge": list(levels.judges),
        "bench": list(levels.benchmarks),
        "cell": [_key(*cell) for cell in levels.cells],
        "item_key": sorted(frame.item_key.unique()),
    }
    job = {
        "data_loader": {"paths": {"extracted_data": list(map(str, input_paths))}},
        "data_process": {"processors": [
            {"extract_observed_feature": {"feature_name": "pick_left", "test": "test"}},
            {"extract_features": {
                "categorical_features": ["judge", "bench", "cell", "item_key"],
                "continuous_features": list(selected_features),
                "category_order": category_order,
                "test": "test",
            }},
        ]},
        "model": {
            "path": "scripts/judge_verdict_analysis/fit_hibayes.py",
            "models": [{"name": "judge_verdict_hibayes", "config": {
                "predictors": list(selected_features),
                "prior_scales": _prior_scales(config),
                "fit": {
                    "method": config.mcmc_method,
                    "samples": config.mcmc_samples,
                    "warmup": config.mcmc_warmup,
                    "chains": config.mcmc_chains,
                    "seed": config.random_seed,
                    "target_accept": config.mcmc_target_accept,
                    "max_tree_depth": config.mcmc_max_tree_depth,
                    "init_strategy": config.mcmc_init_strategy,
                },
            }}],
        },
        "check": {"checkers": ["divergences", "r_hat", "ess_bulk", "ess_tail", "bfmi"]},
        "communicate": {"communicators": ["summary_table"]},
        "platform": {"device_type": "cpu", "chain_method": config.mcmc_chain_method},
    }
    (job_dir / "config.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
    return job_dir


def _model_dir(job_dir: Path) -> Path:
    candidates = (
        job_dir / "output" / "models" / "judge_verdict_hibayes",
        job_dir / "models" / "judge_verdict_hibayes",
    )
    for candidate in candidates:
        if (candidate / "inference_data.nc").exists():
            return candidate
    raise FileNotFoundError(f"no completed HiBayes posterior under {job_dir}")


def _load_posterior(job_dir: Path) -> dict[str, np.ndarray]:
    import arviz as az

    posterior = az.from_netcdf(_model_dir(job_dir) / "inference_data.nc").posterior
    return {
        name: posterior[name].values.reshape(-1, *posterior[name].shape[2:])
        for name in posterior.data_vars
    }


def _posterior_value(posterior: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name in posterior:
        return np.asarray(posterior[name])
    spread = np.asarray(posterior[f"{name}_spread"])
    offsets = np.asarray(posterior[f"{name}_offsets"])
    return offsets * (spread[:, None] if offsets.ndim == 2 else spread[:, None, :])


def _diagnostics(job_dir: Path) -> dict[str, object]:
    candidates = (
        job_dir / "output" / "models" / "judge_verdict_hibayes" / "diagnostics" / "diagnostics.json",
        job_dir / "models" / "judge_verdict_hibayes" / "diagnostics" / "diagnostics.json",
    )
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return {}


def evaluate_hibayes(
    job_dir: Path, test_rows: pd.DataFrame, config: AnalysisConfig
) -> FitResult:
    """Evaluate a completed HiBayes fit from saved posterior draws only."""
    from .fit_common import FitResult
    from .hierarchy import HierarchyLevels

    config.validate()
    job = yaml.safe_load((job_dir / "config.yaml").read_text())
    model_config = job["model"]["models"][0]["config"]
    predictors = tuple(model_config["predictors"])
    category_order = job["data_process"]["processors"][1]["extract_features"]["category_order"]
    input_rows = pd.concat([
        pd.read_parquet(path) for path in job["data_loader"]["paths"]["extracted_data"]
    ], ignore_index=True)
    levels = HierarchyLevels.from_training(input_rows[~input_rows.test])
    rows = test_rows.copy()
    rows["cell"] = [_key(judge, bench) for judge, bench in zip(rows.judge, rows.bench)]
    rows["item_key"] = [_key(bench, item) for bench, item in zip(rows.bench, rows.item)]

    indexes = {}
    for name in ("judge", "bench", "cell", "item_key"):
        lookup = {value: index for index, value in enumerate(category_order[name])}
        if unknown := sorted(set(rows[name]) - set(lookup)):
            raise ValueError(f"test rows contain unknown {name} levels: {unknown}")
        indexes[name] = rows[name].map(lookup).to_numpy(dtype=int)

    posterior = _load_posterior(job_dir)
    population_intercept = np.asarray(posterior["population_intercept"])
    population_slopes = np.asarray(posterior["population_slopes"])
    judge_intercepts = _posterior_value(posterior, "judge_intercepts")
    judge_slopes = _posterior_value(posterior, "judge_slope_deviations")
    benchmark_intercepts = _posterior_value(posterior, "benchmark_intercepts")
    benchmark_slopes = _posterior_value(posterior, "benchmark_slope_deviations")
    cell_intercepts = _posterior_value(posterior, "cell_intercepts")
    cell_slopes = _posterior_value(posterior, "cell_slope_deviations")
    item_intercepts = _posterior_value(posterior, "item_intercepts")
    design = rows.loc[:, predictors].to_numpy(dtype=float) if predictors else np.empty((len(rows), 0))
    effective_slopes = (
        population_slopes[:, None, :]
        + judge_slopes[:, indexes["judge"], :]
        + benchmark_slopes[:, indexes["bench"], :]
        + cell_slopes[:, indexes["cell"], :]
    )
    logits = (
        population_intercept[:, None]
        + judge_intercepts[:, indexes["judge"]]
        + benchmark_intercepts[:, indexes["bench"]]
        + cell_intercepts[:, indexes["cell"]]
        + item_intercepts[:, indexes["item_key"]]
        + np.sum(effective_slopes * design[None, :, :], axis=2)
    )
    probabilities = (1 / (1 + np.exp(-logits))).mean(axis=0)

    coefficients = []
    cell_lookup = {value: index for index, value in enumerate(category_order["cell"])}
    judge_lookup = {value: index for index, value in enumerate(category_order["judge"])}
    bench_lookup = {value: index for index, value in enumerate(category_order["bench"])}
    for feature_index, feature in enumerate(predictors):
        for judge, benchmark in levels.cells:
            population = population_slopes[:, feature_index]
            judge_deviation = judge_slopes[:, judge_lookup[judge], feature_index]
            benchmark_deviation = benchmark_slopes[:, bench_lookup[benchmark], feature_index]
            cell_deviation = cell_slopes[:, cell_lookup[_key(judge, benchmark)], feature_index]
            effective = population + judge_deviation + benchmark_deviation + cell_deviation
            coefficients.append({
                "feature": feature,
                "judge": judge,
                "benchmark": benchmark,
                "population": float(population.mean()),
                "judge_deviation": float(judge_deviation.mean()),
                "benchmark_deviation": float(benchmark_deviation.mean()),
                "cell_deviation": float(cell_deviation.mean()),
                "effective": float(effective.mean()),
                "lower": float(np.quantile(effective, 0.025)),
                "upper": float(np.quantile(effective, 0.975)),
            })

    predictions = test_rows.drop(columns=[*predictors, "pick_left"], errors="ignore").copy()
    observed = rows.pick_left.to_numpy(dtype=int)
    predictions["observed"] = observed
    predictions["probability"] = probabilities
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    metrics = pd.DataFrame({
        "metric": ["accuracy", "log_loss"],
        "value": [
            float(np.mean((probabilities >= 0.5) == observed)),
            float(-np.mean(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped))),
        ],
    })
    diagnostics = _diagnostics(job_dir)
    diagnostics["posterior_draws"] = len(population_intercept)
    return FitResult(
        backend="hibayes",
        fold=job_dir.name,
        selected_features=predictors,
        predictions=predictions,
        coefficients=pd.DataFrame(coefficients, columns=[
            "feature", "judge", "benchmark", "population", "judge_deviation",
            "benchmark_deviation", "cell_deviation", "effective", "lower", "upper",
        ]),
        tuning=pd.DataFrame(),
        metrics=metrics,
        diagnostics=diagnostics,
    )
