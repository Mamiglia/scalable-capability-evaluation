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


def mae(rate, p):
    """Mean absolute error of predicted P against the empirical rate."""
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(rate))))


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def cell_rate(responses):
    """Empirical %-correct per (model, benchmark, item) over all non-blank epochs."""
    valid = responses[responses["answer"].fillna("").astype(str).str.strip() != ""]
    return valid.groupby(["model", "bench", "item"])["correct"].mean().to_dict()


def heldout(responses, model="1pl", k=10, seed=0, warmup=500, samples=500, **kwargs):
    """
    Per-fold held-out MAE: predicted P(correct) vs empirical %-correct over epochs.

    Trains on one epoch per cell; scores held-out cells against their rate over all
    epochs. The whole cell is held out, so no epoch of it leaks into training.
    """
    data = encode(pick_epoch(responses))
    rate = cell_rate(responses)
    target = np.array([rate[(data.models[m], *data.items[i])]
                       for m, i in zip(data.model_idx, data.item_idx)])
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
        overall_p = np.full(int(test.sum()), data.y[train].mean())
        both_add = np.clip(model_p + item_p - overall_p, 0, 1)
        both_logit = 1 / (1 + np.exp(-(_logit(model_p) + _logit(item_p) - _logit(overall_p))))
        t = target[test]
        rows.append({
            "fold": f,
            "irt_mae": mae(t, p),
            "item_base_mae": mae(t, item_p),
            "model_base_mae": mae(t, model_p),
            "overall_base_mae": mae(t, overall_p),
            "both_add_mae": mae(t, both_add),
            "both_logit_mae": mae(t, both_logit),
        })
    return pd.DataFrame(rows)


def compare(responses, specs=None, k=10, seed=0, warmup=400, samples=400):
    """
    Held-out MAE per irt model (1pl / 2pl / mirt-K) vs the three baselines, from a
    single k-fold CV (no subsampling on full data). One row per (spec, fold).
    """
    if specs is None:
        specs = [("1pl", "1pl", {}), ("2pl", "2pl", {})]
        specs += [(f"mirt-K{d}", "mirt", {"dim": d}) for d in range(1, 6)]
    rows = []
    for name, model, kw in specs:
        res = heldout(responses, model=model, k=k, seed=seed, warmup=warmup, samples=samples, **kw)
        for _, r in res.iterrows():
            rows.append({
                "spec": name, "fold": int(r["fold"]),
                "irt": r["irt_mae"], "item_base": r["item_base_mae"],
                "model_base": r["model_base_mae"], "overall_base": r["overall_base_mae"],
                "both_add": r["both_add_mae"], "both_logit": r["both_logit_mae"],
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
            res1 = heldout(sub, model="1pl", k=k, seed=r, warmup=warmup, samples=samples)
            res2 = heldout(sub, model="2pl", k=k, seed=r, warmup=warmup, samples=samples)
            rows.append({
                "n_models": n, "resample": r,
                "irt_1pl": res1["irt_mae"].mean(), "irt_2pl": res2["irt_mae"].mean(),
                "item_base": res2["item_base_mae"].mean(),
                "model_base": res2["model_base_mae"].mean(),
                "overall_base": res2["overall_base_mae"].mean(),
                "both_add": res2["both_add_mae"].mean(),
                "both_logit": res2["both_logit_mae"].mean(),
            })
    return pd.DataFrame(rows)


def estimate_required_item_count(responses, counts=(10, 20, 50, 100, 198),
                                 model="1pl", k=5, resamples=20, seed=0,
                                 warmup=400, samples=400):
    """
    Held-out accuracy vs number of items (all models kept): how many items we need.
    Draws resamples random item subsets per count; one row per (count, resample).
    """
    items = np.array(sorted(responses["item"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for n in counts:
        if n > len(items):
            continue
        for r in range(resamples):
            sub = responses[responses["item"].isin(rng.choice(items, n, replace=False))]
            res1 = heldout(sub, model="1pl", k=k, seed=r, warmup=warmup, samples=samples)
            res2 = heldout(sub, model="2pl", k=k, seed=r, warmup=warmup, samples=samples)
            rows.append({
                "n_items": n, "resample": r,
                "irt_1pl": res1["irt_mae"].mean(), "irt_2pl": res2["irt_mae"].mean(),
                "item_base": res2["item_base_mae"].mean(),
                "model_base": res2["model_base_mae"].mean(),
                "overall_base": res2["overall_base_mae"].mean(),
                "both_add": res2["both_add_mae"].mean(),
                "both_logit": res2["both_logit_mae"].mean(),
            })
    return pd.DataFrame(rows)


def estimate_grid(responses, model_counts=(5, 10, 20, 50, 100, 182),
                  item_counts=(10, 25, 50, 100, 198), model="1pl", k=5,
                  resamples=20, seed=0, warmup=400, samples=400):
    """
    IRT held-out accuracy over a grid of (n_models, n_items). One row per
    (n_models, n_items, resample); the caller means over resamples for a heatmap.
    """
    models = np.array(sorted(responses["model"].unique()))
    items = np.array(sorted(responses["item"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for nm in model_counts:
        if nm > len(models):
            continue
        for ni in item_counts:
            if ni > len(items):
                continue
            for r in range(resamples):
                ms = rng.choice(models, nm, replace=False)
                it = rng.choice(items, ni, replace=False)
                sub = responses[responses["model"].isin(ms) & responses["item"].isin(it)]
                res = heldout(sub, model=model, k=k, seed=r, warmup=warmup, samples=samples)
                rows.append({"n_models": nm, "n_items": ni, "resample": r,
                             "irt": res["irt_mae"].mean()})
    return pd.DataFrame(rows)
