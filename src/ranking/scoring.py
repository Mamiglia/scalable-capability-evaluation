"""
How good a ranking is: rank correlation between fitted ability and benchmark accuracy.

The correctness table passed in is the validation set: accuracy is taken over
every item in it, so pass the whole benchmark, a hard subset, or the items the
judges never saw. bootstrap() resamples the judged items to put intervals on
the abilities and the correlation; the validation set stays fixed.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ranking.bradley_terry import RIDGE, rank

ITEM = ["bench", "item"]
CORRELATIONS = {"kendall": kendalltau, "spearman": spearmanr}  # kendall is tau-b (tie-corrected)


def score(ranking: pd.DataFrame, correctness: pd.DataFrame, correlation: str = "kendall") -> dict:
    """
    Rank correlation between ability and accuracy, over the models present in both.

    ranking: output of rank(). correctness: bench, model, item, correct — the items to validate on.
    correlation: "kendall" (tau-b) or "spearman".
    Returns correlation, method, n_models, n_items (items the accuracy was computed on).
    """
    accuracy = correctness.groupby("model").correct.mean()
    ability = ranking.set_index("model").ability
    models = ability.index.intersection(accuracy.index)
    value = CORRELATIONS[correlation](ability[models], accuracy[models]).statistic
    return {"correlation": float(value), "method": correlation,
            "n_models": len(models), "n_items": len(correctness[ITEM].drop_duplicates())}


def resample_items(verdicts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    One bootstrap draw: judged items sampled with replacement.

    Each draw gets its own item id, so an item drawn twice counts twice in the fit and in tie detection.
    """
    items = verdicts[ITEM].drop_duplicates()
    draws = items.iloc[rng.integers(len(items), size=len(items))].assign(draw=lambda d: np.arange(len(d)).astype(str))
    return verdicts.merge(draws, on=ITEM).drop(columns="item").rename(columns={"draw": "item"})


def bootstrap(verdicts: pd.DataFrame, correctness: pd.DataFrame, n_resamples: int = 200, seed: int = 0,
              drop_ties: bool = False, ridge: float = RIDGE, correlation: str = "kendall") -> dict:
    """
    Item-bootstrap intervals for the abilities and the ranking's correlation with accuracy.

    Returns {"ability": DataFrame model, p5, p50, p95; "correlation": Series p5, p50, p95}.
    """
    rng = np.random.default_rng(seed)
    abilities, correlations = [], []
    for _ in range(n_resamples):
        ranking = rank(resample_items(verdicts, rng), drop_ties=drop_ties, ridge=ridge)
        abilities.append(ranking.set_index("model").ability)
        correlations.append(score(ranking, correctness, correlation=correlation)["correlation"])

    quantiles = {"p5": 0.05, "p50": 0.5, "p95": 0.95}
    ability = pd.concat(abilities, axis=1)  # models x resamples
    return {
        "ability": pd.DataFrame({name: ability.quantile(q, axis=1) for name, q in quantiles.items()})
                     .rename_axis("model").reset_index(),
        "correlation": pd.Series({name: np.quantile(correlations, q) for name, q in quantiles.items()}),
    }
