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


def fixed_three_pl(
    model_idx, item_idx, n_models, n_items, lower_asymptote, y=None,
):
    """3PL with a supplied format floor; only difficulty/discrimination vary."""
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([n_items]))
    a = numpyro.sample("a", dist.LogNormal(0.0, 0.5).expand([n_items]))
    logistic = jax.nn.sigmoid(a[item_idx] * (theta[model_idx] - b[item_idx]))
    floor = lower_asymptote[item_idx]
    numpyro.sample(
        "y", dist.Bernoulli(probs=floor + (1 - floor) * logistic), obs=y,
    )


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

MODELS = {
    "1pl": one_pl,
    "2pl": two_pl,
    "fixed_3pl": fixed_three_pl,
    "mirt": mirt,
}

def fit(
    data: IRTData, model="1pl", warmup=1000, samples=1000, seed=0,
    chains=1, collect_diagnostics=False, **kwargs,
):
    """Run NUTS (No U-Turn Sampler), return posterior samples."""
    mcmc = MCMC(
        NUTS(MODELS[model]), num_warmup=warmup, num_samples=samples,
        num_chains=chains, chain_method="parallel", progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        model_idx=data.model_idx,
        item_idx=data.item_idx,
        n_models=len(data.models),
        n_items=len(data.items),
        y=data.y,
        **kwargs,
    )
    result = dict(mcmc.get_samples())
    if collect_diagnostics:
        chain_samples = mcmc.get_samples(group_by_chain=True)
        result.update({
            "divergences": int(np.asarray(
                mcmc.get_extra_fields(group_by_chain=True)["diverging"]
            ).sum()),
            "diagnostics": numpyro.diagnostics.summary(
                chain_samples, group_by_chain=True,
            ),
        })
    return result

def summary(labels, samples) -> pd.DataFrame:
    """Per-label mean and 90% interval from posterior samples."""
    samples = np.asarray(samples)
    lo, mid, hi = np.percentile(samples, [5, 50, 95], axis=0)
    return pd.DataFrame(
        {"name": labels, "mean": samples.mean(0), "p5": lo, "p50": mid, "p95": hi}
    )


def predict(model, post, model_idx, item_idx, lower_asymptote=None):
    """Posterior-mean P(correct) for the given (model, item) cells."""
    theta = np.asarray(post["theta"])
    if model == "mirt":
        a, d = np.asarray(post["a"]), np.asarray(post["d"])
        logit = (a[:, item_idx, :] * theta[:, model_idx, :]).sum(-1) - d[:, item_idx]
    elif model in {"2pl", "fixed_3pl"}:
        a, b = np.asarray(post["a"]), np.asarray(post["b"])
        logit = a[:, item_idx] * (theta[:, model_idx] - b[:, item_idx])
    else:
        b = np.asarray(post["b"])
        logit = theta[:, model_idx] - b[:, item_idx]
    probability = 1.0 / (1.0 + np.exp(-logit))
    if model == "fixed_3pl":
        if lower_asymptote is None:
            raise ValueError("fixed_3pl prediction needs lower_asymptote")
        floor = np.asarray(lower_asymptote)[item_idx]
        probability = floor + (1 - floor) * probability
    return probability.mean(0)


def save_results(responses, models=("1pl", "2pl", "mirt"), out_dir="irt_results",
                 name="gpqa", warmup=1000, samples=1000):
    """Fit each model type; write one capabilities table and one difficulties table."""
    data = encode(pick_epoch(responses))
    caps = pd.DataFrame({"model": data.models})
    diffs = pd.DataFrame(data.items, columns=["benchmark", "item"])
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
