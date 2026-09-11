"""
Fit IRT models to a response table.

theta per model (capability), b per item (difficulty).
1PL / 2PL / MIRT share one fit(); pick with fit(model=.
"""

from pathlib import Path

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from irt.data import IRTData, encode, pick_epoch


def one_pl(model_idx, item_idx, n_models, n_items, y=None, guess=None):
    """
    Rasch (1PL): one ability per model, one difficulty per item.

    P(correct) = sigmoid(theta[model] - b[item])
    or, with a known guessing floor c per item (1/n_choices on MCQ, 0 on free-form):
    P(correct) = c[item] + (1 - c[item]) * sigmoid(theta[model] - b[item])

    theta: model ability, higher is stronger.
    b: item difficulty, higher is harder.
    model_idx, item_idx: the model and item behind each response.
    n_models, n_items: how many of each. y: observed 0/1, None to sample.
    guess: fixed per-item floor (array of n_items), None for plain Rasch.
    """
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([n_items]))
    logit = theta[model_idx] - b[item_idx]
    if guess is None:
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)
    else:
        c = guess[item_idx]
        numpyro.sample("y", dist.Bernoulli(probs=c + (1 - c) * jax.nn.sigmoid(logit)), obs=y)


def two_pl(model_idx, item_idx, n_models, n_items, y=None, guess=None):
    """
    2PL: adds a discrimination per item on top of 1PL.

    P(correct) = sigmoid(a[item] * (theta[model] - b[item]))
    or, with a known guessing floor c per item, c + (1 - c) * that.

    a: item discrimination (how sharply the item separates by ability).
    theta, b, guess and the arguments are as in one_pl.
    """
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([n_items]))
    a = numpyro.sample("a", dist.LogNormal(0.0, 0.5).expand([n_items]))
    logit = a[item_idx] * (theta[model_idx] - b[item_idx])
    if guess is None:
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)
    else:
        c = guess[item_idx]
        numpyro.sample("y", dist.Bernoulli(probs=c + (1 - c) * jax.nn.sigmoid(logit)), obs=y)


def fixed_2pl(model_idx, item_idx, n_models, n_items, y=None, a=None, b=None, guess=None):
    """
    2PL with every item parameter (a, b, guess: arrays of n_items) fixed; only theta is fitted.

    Puts new models on the scale of an earlier fit (the global fit's items).
    """
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models]))
    logit = a[item_idx] * (theta[model_idx] - b[item_idx])
    c = guess[item_idx]
    numpyro.sample("y", dist.Bernoulli(probs=c + (1 - c) * jax.nn.sigmoid(logit)), obs=y)


def mirt(model_idx, item_idx, n_models, n_items, dim=2, y=None):
    """
    MIRT: ability is a vector of dim skills, each item loads on them.

    P(correct) = sigmoid(sum(a[item] * theta[model]) - d[item])

    theta: per-model ability vector (dim skills). a: per-item loading vector.
    d: item intercept (baseline difficulty). dim: number of skills.

    """
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models, dim]))
    a = numpyro.sample("a", dist.Normal(0, 1).expand([n_items, dim]))
    d = numpyro.sample("d", dist.Normal(0, 1).expand([n_items]))
    logit = (a[item_idx] * theta[model_idx]).sum(-1) - d[item_idx]
    numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)

MODELS = {"1pl": one_pl, "2pl": two_pl, "fixed_2pl": fixed_2pl, "mirt": mirt}

def fit(data: IRTData, model="1pl", warmup=1000, samples=1000, seed=0, **kwargs):
    """Run NUTS (No U-Turn Sampler), return posterior samples."""
    mcmc = MCMC(NUTS(MODELS[model]), num_warmup=warmup, num_samples=samples, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(seed),
        model_idx=data.model_idx,
        item_idx=data.item_idx,
        n_models=len(data.models),
        n_items=len(data.items),
        y=data.y,
        **kwargs,
    )
    return mcmc.get_samples()

def summary(labels, samples) -> pd.DataFrame:
    """Per-label mean and 90% interval from posterior samples."""
    samples = np.asarray(samples)
    lo, mid, hi = np.percentile(samples, [5, 50, 95], axis=0)
    return pd.DataFrame(
        {"name": labels, "mean": samples.mean(0), "p5": lo, "p50": mid, "p95": hi}
    )


def predict(model, post, model_idx, item_idx):
    """Posterior-mean P(correct) for the given (model, item) cells."""
    theta = np.asarray(post["theta"])
    if model == "mirt":
        a, d = np.asarray(post["a"]), np.asarray(post["d"])
        logit = (a[:, item_idx, :] * theta[:, model_idx, :]).sum(-1) - d[:, item_idx]
    elif model == "2pl":
        a, b = np.asarray(post["a"]), np.asarray(post["b"])
        logit = a[:, item_idx] * (theta[:, model_idx] - b[:, item_idx])
    else:
        b = np.asarray(post["b"])
        logit = theta[:, model_idx] - b[:, item_idx]
    return (1.0 / (1.0 + np.exp(-logit))).mean(0)


def save_results(responses, models=("1pl", "2pl", "mirt"), out_dir="irt_results",
                 name="gpqa", warmup=1000, samples=1000):
    """Fit each model type; write one capabilities table and one difficulties table."""
    data = encode(pick_epoch(responses))
    caps = pd.DataFrame({"model": data.models})
    diffs = pd.DataFrame(data.items, columns=["bench", "item"])
    for m in models:
        post = fit(data, model=m, warmup=warmup, samples=samples)
        theta = np.asarray(post["theta"])
        if theta.ndim == 3:  # MIRT: ability is a vector of skills
            for d in range(theta.shape[2]):
                caps[f"theta_{m}{d}"] = theta[:, :, d].mean(0)
        else:
            caps[f"theta_{m}"] = theta.mean(0)
            caps[f"theta_{m}_lo"] = np.percentile(theta, 5, axis=0)
            caps[f"theta_{m}_hi"] = np.percentile(theta, 95, axis=0)
        b = np.asarray(post["d" if m == "mirt" else "b"])
        col = "d" if m == "mirt" else "b"
        diffs[f"{col}_{m}"] = b.mean(0)
        diffs[f"{col}_{m}_lo"] = np.percentile(b, 5, axis=0)
        diffs[f"{col}_{m}_hi"] = np.percentile(b, 95, axis=0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caps.to_csv(out / f"{name}_capabilities.csv", index=False)
    diffs.to_csv(out / f"{name}_difficulties.csv", index=False)
    return caps, diffs
