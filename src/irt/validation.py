"""
Validation of IRT outputs.

Held-out prediction: mask (model, item) cells, fit on the rest, predict the masked
cells, and compare against three %-correct baselines and across model types.

overall_base predicts from the global %-correct (no model or item info); item_base
from the item's %-correct (knows difficulty, not ability); model_base from the
model's %-correct (knows ability, not difficulty). IRT over item_base = value of
model capability; IRT over model_base = value of item difficulty.
"""

import numpy as np
import pandas as pd

from irt.data import IRTData, encode, pick_epoch
from irt.fitting import fit, predict


def logloss(y, p, eps=1e-6):
    """Binary log-loss."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y, p):
    """Fraction correct after thresholding P at 0.5."""
    return float(((p > 0.5).astype(int) == y).mean())


def pass_rate(idx, y, n):
    """Per-index mean of y, global mean where an index has no train cells."""
    rate = np.full(n, y.mean())
    seen = pd.Series(y).groupby(idx).mean()
    rate[seen.index.to_numpy()] = seen.to_numpy()
    return rate


def heldout(responses, model="1pl", k=10, seed=0, warmup=500, samples=500, **kwargs):
    """Per-fold held-out accuracy and log-loss: IRT vs the item and model baselines."""
    data = encode(pick_epoch(responses))
    folds = np.random.default_rng(seed).permutation(len(data.y)) % k
    rows = []
    for f in range(k):
        test, train = folds == f, folds != f
        train_data = IRTData(
            data.model_idx[train], data.item_idx[train], data.y[train],
            data.models, data.items,
        )
        post = fit(train_data, model=model, warmup=warmup, samples=samples, **kwargs)
        p = predict(model, post, data.model_idx[test], data.item_idx[test])
        item_p = pass_rate(data.item_idx[train], data.y[train], len(data.items))[data.item_idx[test]]
        model_p = pass_rate(data.model_idx[train], data.y[train], len(data.models))[data.model_idx[test]]
        yt = data.y[test]
        overall_p = np.full(yt.shape, data.y[train].mean())
        rows.append({
            "fold": f,
            "irt_acc": accuracy(yt, p), "irt_ll": logloss(yt, p),
            "item_base_acc": accuracy(yt, item_p), "item_base_ll": logloss(yt, item_p),
            "model_base_acc": accuracy(yt, model_p), "model_base_ll": logloss(yt, model_p),
            "overall_base_acc": accuracy(yt, overall_p), "overall_base_ll": logloss(yt, overall_p),
        })
    return pd.DataFrame(rows)


def compare(responses, specs=None, k=5, resamples=20, seed=0, warmup=400, samples=400):
    """
    Held-out accuracy per irt model (1pl / 2pl / mirt-K) vs all three baselines
    (overall / model / item). Reseeds the CV `resamples` times; one row per
    (spec, resample), so the caller takes the mean and error margin.
    """
    if specs is None:
        specs = [("1pl", "1pl", {}), ("2pl", "2pl", {})]
        specs += [(f"mirt-K{d}", "mirt", {"dim": d}) for d in range(1, 6)]
    rows = []
    for name, model, kw in specs:
        for r in range(resamples):
            res = heldout(responses, model=model, k=k, seed=r, warmup=warmup, samples=samples, **kw)
            rows.append({
                "spec": name, "resample": r,
                "irt": res["irt_acc"].mean(),
                "item_base": res["item_base_acc"].mean(),
                "model_base": res["model_base_acc"].mean(),
                "overall_base": res["overall_base_acc"].mean(),
            })
    return pd.DataFrame(rows)


def estimate_required_model_count(responses, counts=(5, 10, 20, 50, 100, 182),
                                  model="1pl", k=5, resamples=20, seed=0,
                                  warmup=400, samples=400):
    """
    Held-out accuracy vs number of models (all items kept), to estimate how
    many models we'll need for IRT to be workable.
    Draws 'resamples' random model subsets
    """
    models = np.array(sorted(responses["model"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for n in counts:
        if n > len(models):
            continue
        for r in range(resamples):
            sub = responses[responses["model"].isin(rng.choice(models, n, replace=False))]
            res = heldout(sub, model=model, k=k, seed=r, warmup=warmup, samples=samples)
            rows.append({
                "n_models": n, "resample": r,
                "irt": res["irt_acc"].mean(),
                "item_base": res["item_base_acc"].mean(),
                "model_base": res["model_base_acc"].mean(),
                "overall_base": res["overall_base_acc"].mean(),
            })
    return pd.DataFrame(rows)
