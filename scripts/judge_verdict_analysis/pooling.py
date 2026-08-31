"""Training-fold variance components for relative hierarchical penalties."""
from __future__ import annotations

import jax.numpy as jnp
from jax import random
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
import pandas as pd

from .config import AnalysisConfig


POOLING_PARAMETERS = (
    "judge_deviation_penalty",
    "benchmark_deviation_penalty",
    "cell_deviation_penalty",
    "item_intercept_penalty",
)
def penalty_multiplier(relative_penalty: float, penalty: str) -> float:
    """Return the design-column multiplier for one relative penalty."""
    ratio = float(relative_penalty)
    if penalty not in {"l1", "ridge"}:
        raise ValueError("penalty must be 'l1' or 'ridge'")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("relative penalties must be finite and positive")
    return 1 / ratio if penalty == "l1" else 1 / np.sqrt(ratio)


def _hierarchical_scale_model(
    x: jnp.ndarray, y: jnp.ndarray, judge: jnp.ndarray, benchmark: jnp.ndarray,
    cell: jnp.ndarray, item: jnp.ndarray, counts: tuple[int, int, int, int],
    penalty: str, scale_prior: float,
) -> None:
    """Crossed logistic model with one learned scale per coefficient block."""
    family = dist.Laplace if penalty == "l1" else dist.Normal

    def coefficients(name: str, scale: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        return numpyro.sample(
            name, family(0, scale).expand(shape).to_event(len(shape)),
        )

    scales = {
        name: numpyro.sample(f"{name}_scale", dist.HalfNormal(scale_prior))
        for name in ("population", "judge", "benchmark", "cell", "item")
    }
    width = x.shape[1] + 1
    population = coefficients("population", scales["population"], (width,))
    judge_effect = coefficients("judge", scales["judge"], (counts[0], width))
    benchmark_effect = coefficients(
        "benchmark", scales["benchmark"], (counts[1], width))
    cell_effect = coefficients("cell", scales["cell"], (counts[2], width))
    item_effect = coefficients("item", scales["item"], (counts[3],))
    logits = (
        population[0] + judge_effect[judge, 0] + benchmark_effect[benchmark, 0]
        + cell_effect[cell, 0] + item_effect[item]
        + jnp.sum(x * population[1:], axis=1)
        + jnp.sum(x * judge_effect[judge, 1:], axis=1)
        + jnp.sum(x * benchmark_effect[benchmark, 1:], axis=1)
        + jnp.sum(x * cell_effect[cell, 1:], axis=1)
    )
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def estimate_pooling_ratios(
    train: pd.DataFrame, features: list[str], penalty: str, config: AnalysisConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate relative penalties from one outer-training likelihood only."""
    if penalty not in {"l1", "ridge"}:
        raise ValueError("penalty must be 'l1' or 'ridge'")
    required = {"judge", "bench", "item", "pick_left", *features}
    if missing := sorted(required - set(train)):
        raise ValueError(f"pooling rows missing columns: {missing}")
    x = train[features].to_numpy(np.float32)
    y = train.pick_left.to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("pooling features must be finite and outcomes binary")

    def factorize(columns: list[str]) -> tuple[np.ndarray, int]:
        codes, levels = pd.factorize(pd.MultiIndex.from_frame(train[columns]), sort=True)
        return codes.astype(np.int32), len(levels)

    judge, n_judge = factorize(["judge"])
    benchmark, n_benchmark = factorize(["bench"])
    cell, n_cell = factorize(["judge", "bench"])
    item, n_item = factorize(["bench", "item"])
    args = (
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(judge), jnp.asarray(benchmark),
        jnp.asarray(cell), jnp.asarray(item),
        (n_judge, n_benchmark, n_cell, n_item), penalty,
        float(config.pooling_scale_prior),
    )
    guide = AutoNormal(_hierarchical_scale_model)
    result = SVI(
        _hierarchical_scale_model, guide,
        Adam(config.pooling_svi_learning_rate), Trace_ELBO(),
    ).run(
        random.PRNGKey(config.random_seed), config.pooling_svi_steps, *args,
        progress_bar=False,
    )
    losses = np.asarray(result.losses, dtype=float)
    if not np.isfinite(losses).all():
        raise RuntimeError("hierarchy variance-component fit produced non-finite loss")
    median = guide.median(result.params)
    scales = {
        name: float(np.asarray(median[f"{name}_scale"]))
        for name in ("population", "judge", "benchmark", "cell", "item")
    }
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise RuntimeError("hierarchy variance-component fit produced invalid scales")
    power = 1 if penalty == "l1" else 2
    ratios = {
        parameter: (scales["population"] / scales[group]) ** power
        for parameter, group in zip(
            POOLING_PARAMETERS, ("judge", "benchmark", "cell", "item"))
    }
    diagnostics = pd.DataFrame([{
        "stage": "pooling", "record_type": "variance_component",
        "penalty": penalty, "selected": False, "converged": True,
        "svi_steps": config.pooling_svi_steps,
        "svi_initial_loss": losses[0], "svi_final_loss": losses[-1],
        **{f"{name}_scale": value for name, value in scales.items()},
        **ratios,
    }])
    return ratios, diagnostics
