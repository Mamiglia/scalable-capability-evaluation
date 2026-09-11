"""Fit IRT models to a response table with NUTS.

theta per model (capability), b per item (difficulty), a per item (discrimination).
fit(model=) picks 1pl / 2pl / fixed_2pl, or takes any numpyro model with the same signature.
"""

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from irt.data import IRTData


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


MODELS = {"1pl": one_pl, "2pl": two_pl, "fixed_2pl": fixed_2pl}

def fit(data: IRTData, model="1pl", warmup=1000, samples=1000, seed=0, **kwargs):
    """Run NUTS (No U-Turn Sampler), return posterior samples. `model`: a MODELS key or a numpyro model."""
    mcmc = MCMC(NUTS(MODELS.get(model, model)), num_warmup=warmup, num_samples=samples, progress_bar=False)
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
