"""MIRT and the result helpers of the Epoch GPQA reproduction, on top of irt.fitting."""

from pathlib import Path

import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules of this analysis
from data import pick_epoch  # noqa: E402
from irt.data import encode  # noqa: E402
from irt.fitting import fit  # noqa: E402


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
        post = fit(data, model=mirt if m == "mirt" else m, warmup=warmup, samples=samples)
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
