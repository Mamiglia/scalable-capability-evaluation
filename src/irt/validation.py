"""
Validation of IRT outputs.

Held-out prediction: mask (model, item) cells, fit on the rest, predict the
masked cells, and compare log-loss against a %-correct baseline and across models.
"""

import numpy as np
import pandas as pd

from irt.data import IRTData, encode, pick_epoch
from irt.fitting import fit, predict


def logloss(y, p, eps=1e-6):
    """Binary log-loss."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def item_pass_rate(item_idx, y, n_items):
    """Per-item mean of y, global mean where an item has no train cells."""
    rate = np.full(n_items, y.mean())
    seen = pd.Series(y).groupby(item_idx).mean()
    rate[seen.index.to_numpy()] = seen.to_numpy()
    return rate


def heldout(responses, model="1pl", k=10, seed=0, warmup=500, samples=500):
    """Per-fold held-out log-loss: IRT vs the item %-correct baseline."""
    data = encode(pick_epoch(responses))
    folds = np.random.default_rng(seed).permutation(len(data.y)) % k
    rows = []
    for f in range(k):
        test, train = folds == f, folds != f
        train_data = IRTData(
            data.model_idx[train], data.item_idx[train], data.y[train],
            data.models, data.items,
        )
        post = fit(train_data, model=model, warmup=warmup, samples=samples)
        p = predict(model, post, data.model_idx[test], data.item_idx[test])
        base = item_pass_rate(data.item_idx[train], data.y[train], len(data.items))
        rows.append({
            "fold": f,
            "irt": logloss(data.y[test], p),
            "baseline": logloss(data.y[test], base[data.item_idx[test]]),
        })
    return pd.DataFrame(rows)
