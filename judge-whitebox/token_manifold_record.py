"""Per-token residual recording for output-conditioned manifold discovery."""

import numpy as np


def balanced_item_folds(df, seed):
    """Deterministically split whole items while greedily balancing row counts."""
    counts = df.groupby("item", sort=True).size()
    rng = np.random.default_rng(seed)
    tie = dict(zip(counts.index, rng.random(len(counts))))
    ordered = sorted(counts.index, key=lambda item: (-int(counts[item]), tie[item], str(item)))
    totals = {"A": 0, "B": 0}
    folds = {}
    for item in ordered:
        fold = min(totals, key=lambda name: (totals[name], name))
        folds[item] = fold
        totals[fold] += int(counts[item])
    return folds


def select_fold(df, folds, name):
    """Return one fold without reordering rows or splitting items."""
    assert name in {"A", "B"}
    assert set(df["item"]) <= set(folds)
    return df[df["item"].map(folds) == name].copy()
