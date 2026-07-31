"""
Fit IRT models to a response table.

theta per model (capability), b per item (difficulty).
1PL / 2PL / MIRT share one fit(); pick with fit(model=.
"""

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from irt.data import IRTData


def one_pl(model_idx, item_idx, n_models, n_items, y=None):
    """
    Rasch (1PL): one ability per model, one difficulty per item.

    P(correct) = sigmoid(theta[model] - b[item])

    theta: model ability, higher is stronger.
    b: item difficulty, higher is harder.
    model_idx, item_idx: the model and item behind each response.
    n_models, n_items: how many of each. y: observed 0/1, None to sample.
    """
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([n_items]))
    numpyro.sample("y", dist.Bernoulli(logits=theta[model_idx] - b[item_idx]), obs=y)


def two_pl(model_idx, item_idx, n_models, n_items, y=None):
    """
    2PL: adds a discrimination per item on top of 1PL.

    P(correct) = sigmoid(a[item] * (theta[model] - b[item]))

    a: item discrimination (how sharply the item separates by ability).
    theta, b and the arguments are as in one_pl.
    """
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([n_items]))
    a = numpyro.sample("a", dist.LogNormal(0.0, 0.5).expand([n_items]))
    logit = a[item_idx] * (theta[model_idx] - b[item_idx])
    numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


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

MODELS = {"1pl": one_pl, "2pl": two_pl, "mirt": mirt}

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
