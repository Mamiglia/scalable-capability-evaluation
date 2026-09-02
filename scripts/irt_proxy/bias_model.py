"""Hierarchical calibration of GPQA pairwise judges."""

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd


POSTERIOR_KEYS = ("alpha", "source", "condition", "discrimination", "beta")


def linear_predictor(
    alpha, source, condition, discrimination, delta_theta, beta, design,
    judge, source_index, condition_index,
):
    """Return left-choice logits for encoded calibration rows."""
    logits = alpha[judge] + source[judge, source_index]
    logits += condition[judge, condition_index]
    logits += discrimination[judge] * delta_theta
    return logits + jnp.sum(beta[judge] * design, axis=1)


def judge_bias_model(
    obs, judge, source_index, condition_index, delta_theta, design,
    n_judges, n_sources, n_conditions,
):
    """Non-centred judge hierarchy with identified source/condition baselines."""
    alpha_mean = numpyro.sample("alpha_mean", dist.Normal(0, 1))
    alpha_scale = numpyro.sample("alpha_scale", dist.HalfNormal(0.5))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0, 1).expand([n_judges]))
    alpha = numpyro.deterministic(
        "judge_alpha", alpha_mean + alpha_scale * alpha_raw,
    )

    d_mean = numpyro.sample("d_mean", dist.Normal(1, 0.5))
    d_scale = numpyro.sample("d_scale", dist.HalfNormal(0.5))
    d_raw = numpyro.sample("d_raw", dist.Normal(0, 1).expand([n_judges]))
    discrimination = numpyro.deterministic(
        "judge_discrimination", d_mean + d_scale * d_raw,
    )

    n_features = design.shape[1]
    if n_features:
        beta_mean = numpyro.sample(
            "beta_mean", dist.Normal(0, 0.5).expand([n_features]),
        )
        beta_scale = numpyro.sample(
            "beta_scale", dist.HalfNormal(0.5).expand([n_features]),
        )
        beta_raw = numpyro.sample(
            "beta_raw",
            dist.Normal(0, 1).expand([n_judges, n_features]).to_event(2),
        )
        beta = beta_mean + beta_scale * beta_raw
    else:
        beta = jnp.zeros((n_judges, 0))
    numpyro.deterministic("judge_beta", beta)

    source_free = jnp.zeros((n_judges, 0))
    if n_sources > 1:
        source_mean = numpyro.sample(
            "source_mean", dist.Normal(0, 0.25).expand([n_sources - 1]),
        )
        source_scale = numpyro.sample(
            "source_scale", dist.HalfNormal(0.25).expand([n_sources - 1]),
        )
        source_raw = numpyro.sample(
            "source_raw",
            dist.Normal(0, 1).expand([n_judges, n_sources - 1]).to_event(2),
        )
        source_free = source_mean + source_scale * source_raw
    condition_free = jnp.zeros((n_judges, 0))
    if n_conditions > 1:
        condition_mean = numpyro.sample(
            "condition_mean", dist.Normal(0, 0.25).expand([n_conditions - 1]),
        )
        condition_scale = numpyro.sample(
            "condition_scale", dist.HalfNormal(0.25).expand([n_conditions - 1]),
        )
        condition_raw = numpyro.sample(
            "condition_raw",
            dist.Normal(0, 1).expand([n_judges, n_conditions - 1]).to_event(2),
        )
        condition_free = condition_mean + condition_scale * condition_raw
    source = jnp.concatenate([jnp.zeros((n_judges, 1)), source_free], axis=1)
    condition = jnp.concatenate(
        [jnp.zeros((n_judges, 1)), condition_free], axis=1,
    )
    numpyro.deterministic("judge_source", source)
    numpyro.deterministic("judge_condition", condition)

    logits = linear_predictor(
        alpha, source, condition, discrimination, delta_theta, beta, design,
        judge, source_index, condition_index,
    )
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)


def _public_posterior(samples, n_judges, n_sources, n_conditions, n_features):
    """Convert NumPyro's raw non-centred sites to the stable public contract."""
    alpha = (
        np.asarray(samples["alpha_mean"])[..., None]
        + np.asarray(samples["alpha_scale"])[..., None]
        * np.asarray(samples["alpha_raw"])
    )
    discrimination = (
        np.asarray(samples["d_mean"])[..., None]
        + np.asarray(samples["d_scale"])[..., None] * np.asarray(samples["d_raw"])
    )
    draws = alpha.shape[0]
    if n_features:
        beta = (
            np.asarray(samples["beta_mean"])[:, None, :]
            + np.asarray(samples["beta_scale"])[:, None, :]
            * np.asarray(samples["beta_raw"])
        )
    else:
        beta = np.zeros((draws, n_judges, 0))
    source_free = (
        np.asarray(samples["source_mean"])[:, None, :]
        + np.asarray(samples["source_scale"])[:, None, :]
        * np.asarray(samples["source_raw"])
        if n_sources > 1 else np.zeros((draws, n_judges, 0))
    )
    condition_free = (
        np.asarray(samples["condition_mean"])[:, None, :]
        + np.asarray(samples["condition_scale"])[:, None, :]
        * np.asarray(samples["condition_raw"])
        if n_conditions > 1 else np.zeros((draws, n_judges, 0))
    )
    return {
        "alpha": alpha,
        "source": np.concatenate(
            [np.zeros((draws, n_judges, 1)), source_free], axis=2,
        ),
        "condition": np.concatenate(
            [np.zeros((draws, n_judges, 1)), condition_free], axis=2,
        ),
        "discrimination": discrimination,
        "beta": beta,
    }


def fit_calibration(frame, feature_columns, warmup, samples, seed, target_accept=0.8):
    """Fit four calibration chains and return posterior arrays plus diagnostics."""
    features = list(feature_columns)
    labels = {
        name: sorted(frame[name].unique().tolist())
        for name in ("judge", "source", "condition")
    }
    design = frame.loc[:, features].to_numpy(dtype=float)
    feature_mean = design.mean(axis=0)
    feature_scale = design.std(axis=0)
    invalid = np.flatnonzero(feature_scale == 0)
    if len(invalid):
        names = [features[index] for index in invalid]
        raise ValueError(f"zero variance features: {names}")
    design = (design - feature_mean) / feature_scale

    encoded = {
        name: pd.Categorical(frame[name], categories=values).codes
        for name, values in labels.items()
    }
    kernel = NUTS(judge_bias_model, target_accept_prob=target_accept)
    mcmc = MCMC(
        kernel, num_warmup=warmup, num_samples=samples, num_chains=4,
        chain_method="parallel",
    )
    mcmc.run(
        random.PRNGKey(seed), obs=frame["chosen_left"].to_numpy(dtype=float),
        judge=encoded["judge"], source_index=encoded["source"],
        condition_index=encoded["condition"],
        delta_theta=frame["delta_theta"].to_numpy(dtype=float), design=design,
        n_judges=len(labels["judge"]), n_sources=len(labels["source"]),
        n_conditions=len(labels["condition"]),
    )
    flat_samples = mcmc.get_samples()
    chain_samples = mcmc.get_samples(group_by_chain=True)
    artifact = _public_posterior(
        flat_samples, len(labels["judge"]), len(labels["source"]),
        len(labels["condition"]), len(features),
    )
    artifact.update({
        "judges": labels["judge"], "sources": labels["source"],
        "conditions": labels["condition"], "features": features,
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "divergences": int(np.asarray(
            mcmc.get_extra_fields(group_by_chain=True)["diverging"]
        ).sum()),
        "diagnostics": numpyro.diagnostics.summary(
            {key: value for key, value in chain_samples.items() if value.size},
            group_by_chain=True,
        ),
    })
    return artifact


def cut_target_model(
    obs, left_target, right_target, left_anchor, right_anchor, judge,
    source_index, condition_index, design, calibration, n_targets,
):
    """Infer target abilities while marginalizing frozen calibration draws."""
    theta = numpyro.sample("theta", dist.Normal(0, 2).expand([n_targets]))
    left = jnp.where(
        left_target >= 0, theta[jnp.maximum(left_target, 0)], left_anchor,
    )
    right = jnp.where(
        right_target >= 0, theta[jnp.maximum(right_target, 0)], right_anchor,
    )
    delta = left - right

    logits = calibration["alpha"][:, judge]
    logits += calibration["source"][:, judge, source_index]
    logits += calibration["condition"][:, judge, condition_index]
    logits += calibration["discrimination"][:, judge] * delta[None, :]
    logits += jnp.sum(
        calibration["beta"][:, judge, :] * design[None, :, :], axis=-1,
    )
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(
        obs[None, :],
    ).sum(axis=1)
    numpyro.factor(
        "cut_obs", logsumexp(log_likelihood) - jnp.log(log_likelihood.shape[0]),
    )


def _calibration_draw_count(calibration):
    n_judges = len(calibration["judges"])
    alpha_shape = np.shape(calibration["alpha"])
    if len(alpha_shape) != 2:
        raise ValueError(
            f"calibration alpha shape: expected (draws, {n_judges}), "
            f"got {alpha_shape}"
        )
    draws = alpha_shape[0]
    if draws == 0:
        raise ValueError("calibration has zero posterior draws")
    expected = {
        "alpha": (draws, n_judges),
        "source": (draws, n_judges, len(calibration["sources"])),
        "condition": (draws, n_judges, len(calibration["conditions"])),
        "discrimination": (draws, n_judges),
        "beta": (draws, n_judges, len(calibration["features"])),
    }
    for key, expected_shape in expected.items():
        shape = np.shape(calibration[key])
        if shape != expected_shape:
            raise ValueError(
                f"calibration {key} shape: expected {expected_shape}, got {shape}"
            )
    feature_shape = (len(calibration["features"]),)
    for key in ("feature_mean", "feature_scale"):
        shape = np.shape(calibration[key])
        if shape != feature_shape:
            raise ValueError(
                f"calibration {key} shape: expected {feature_shape}, got {shape}"
            )
    return draws


def infer_targets(
    target_frame, anchors, calibration, warmup, samples, seed,
    max_calibration_draws=200, collect_diagnostics=False, chains=1,
    inference_name=None, target_models=None,
):
    """Return target ability summaries from a frozen empirical calibration."""
    if max_calibration_draws <= 0:
        raise ValueError("max_calibration_draws must be positive")
    if "benchmark" in target_frame:
        target_frame = target_frame[target_frame["benchmark"].eq("gpqa")]
    anchor_theta = anchors.set_index("model")["theta"]
    endpoints = set(target_frame["left_model"]) | set(target_frame["right_model"])
    if target_models is None:
        targets = sorted(endpoints - set(anchor_theta.index))
    else:
        target_models = set(target_models)
        unanchored = endpoints - set(anchor_theta.index) - target_models
        if unanchored:
            raise ValueError(
                f"unanchored non-held-out endpoints: {sorted(unanchored)}"
            )
        targets = sorted(endpoints & target_models)
    if not targets:
        return pd.DataFrame(columns=[
            "model", "theta", "theta_p5", "theta_p95",
        ])
    draws = _calibration_draw_count(calibration)
    levels = {
        "judge": calibration["judges"],
        "source": calibration["sources"],
        "condition": calibration["conditions"],
    }
    encoded = {
        name: target_frame[name].map(
            {value: index for index, value in enumerate(values)},
        ).fillna(-1).to_numpy(dtype=int)
        for name, values in levels.items()
    }
    unseen = {
        name: sorted(target_frame.loc[codes < 0, name].unique().tolist())
        for name, codes in encoded.items() if np.any(codes < 0)
    }
    if unseen:
        reason = "; ".join(
            f"unseen {name} levels: {values}" for name, values in unseen.items()
        )
        raise ValueError(f"coverage failure: {reason}")
    target_index = {model: index for index, model in enumerate(targets)}

    def endpoint(column):
        models = target_frame[column]
        return (
            models.map(target_index).fillna(-1).to_numpy(dtype=int),
            models.map(anchor_theta).fillna(0).to_numpy(dtype=float),
        )

    left_target, left_anchor = endpoint("left_model")
    right_target, right_anchor = endpoint("right_model")
    features = calibration["features"]
    design = target_frame.loc[:, features].to_numpy(dtype=float)
    design = (
        design - np.asarray(calibration["feature_mean"])
    ) / np.asarray(calibration["feature_scale"])
    draw_index = np.arange(draws)
    if draws > max_calibration_draws:
        draw_index = np.linspace(
            0, draws - 1, max_calibration_draws, dtype=int,
        )
    frozen = {
        key: jnp.asarray(np.asarray(calibration[key])[draw_index])
        for key in POSTERIOR_KEYS
    }
    mcmc = MCMC(
        NUTS(cut_target_model), num_warmup=warmup, num_samples=samples,
        num_chains=chains,
        chain_method="parallel" if chains > 1 else "sequential",
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(seed),
        obs=target_frame["chosen_left"].to_numpy(dtype=float),
        left_target=left_target, right_target=right_target,
        left_anchor=left_anchor, right_anchor=right_anchor,
        judge=encoded["judge"], source_index=encoded["source"],
        condition_index=encoded["condition"], design=design,
        calibration=frozen, n_targets=len(targets),
    )
    theta = np.asarray(mcmc.get_samples()["theta"])
    result = pd.DataFrame({
        "model": targets,
        "theta": theta.mean(axis=0),
        "theta_p5": np.percentile(theta, 5, axis=0),
        "theta_p95": np.percentile(theta, 95, axis=0),
    })
    if collect_diagnostics:
        chain_samples = mcmc.get_samples(group_by_chain=True)
        result.attrs.update({
            "divergences": int(np.asarray(
                mcmc.get_extra_fields(group_by_chain=True)["diverging"]
            ).sum()),
            "diagnostics": numpyro.diagnostics.summary(
                chain_samples, group_by_chain=True,
            ),
        })
    return result


def simulate_calibration(seed=0, n_judges=3, n_rows=3000):
    """Generate a reproducible calibration frame with known judge effects."""
    rng = np.random.default_rng(seed)
    alpha = np.linspace(-0.3, 0.3, n_judges)
    discrimination = np.linspace(0.5, 1.5, n_judges)
    beta = np.array([0.8, -0.5])
    judge_index = rng.integers(n_judges, size=n_rows)
    design = rng.normal(size=(n_rows, 2))
    design = (design - design.mean(axis=0)) / design.std(axis=0)
    delta_theta = rng.normal(size=n_rows)
    logits = (
        alpha[judge_index] + discrimination[judge_index] * delta_theta
        + design @ beta
    )
    frame = pd.DataFrame({
        "judge": [f"judge-{index}" for index in judge_index],
        "source": rng.choice(["hf", "pilot"], size=n_rows),
        "condition": rng.choice(["full", "natural"], size=n_rows),
        "chosen_left": rng.binomial(1, 1 / (1 + np.exp(-logits))).astype(float),
        "delta_theta": delta_theta,
        "delta_log_chars": design[:, 0],
        "delta_personal_pronouns": design[:, 1],
    })
    truth = pd.DataFrame({
        "judge": [f"judge-{index}" for index in range(n_judges)],
        "alpha": alpha, "discrimination": discrimination,
        "delta_log_chars": np.full(n_judges, beta[0]),
        "delta_personal_pronouns": np.full(n_judges, beta[1]),
    })
    return frame, truth
