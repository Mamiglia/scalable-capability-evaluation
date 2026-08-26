"""
How good a ranking is: rank correlation between fitted ability and benchmark accuracy.

Accuracy is computed on the items that were actually judged, so ranking a subset
of items is scored against accuracy on that subset. bootstrap() resamples items
to put intervals on the abilities and the correlation.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ranking.bradley_terry import RIDGE, rank

ITEM = ["bench", "item"]
CORRELATIONS = {"kendall": kendalltau, "spearman": spearmanr}  # kendall is tau-b (tie-corrected)


def accuracy_on_judged_items(correctness: pd.DataFrame, verdicts: pd.DataFrame) -> pd.Series:
    """Per model, mean correctness over the (bench, item) pairs that appear in the verdicts."""
    judged = verdicts[ITEM].drop_duplicates()
    return correctness.merge(judged, on=ITEM).groupby("model").correct.mean()


def score(ranking: pd.DataFrame, correctness: pd.DataFrame, verdicts: pd.DataFrame,
          correlation: str = "kendall") -> dict:
    """
    Rank correlation between ability and accuracy, over the models present in both.

    ranking: output of rank(). correctness: bench, model, item, correct. verdicts: the table rank() was given.
    correlation: "kendall" (tau-b) or "spearman".
    Returns correlation, method, n_models, n_items (judged items the accuracy was computed on).
    """
    accuracy = accuracy_on_judged_items(correctness, verdicts)
    ability = ranking.set_index("model").ability
    models = ability.index.intersection(accuracy.index)
    value = CORRELATIONS[correlation](ability[models], accuracy[models]).statistic
    return {"correlation": float(value), "method": correlation,
            "n_models": len(models), "n_items": len(verdicts[ITEM].drop_duplicates())}


def resample_items(verdicts: pd.DataFrame, correctness: pd.DataFrame, rng: np.random.Generator):
    """
    One bootstrap draw: judged items sampled with replacement, in both tables.

    Each draw gets its own item id, so an item drawn twice counts twice in the fit,
    in tie detection and in accuracy.
    """
    items = verdicts[ITEM].drop_duplicates()
    draws = items.iloc[rng.integers(len(items), size=len(items))].assign(draw=lambda d: np.arange(len(d)).astype(str))

    def relabel(table):
        return table.merge(draws, on=ITEM).drop(columns="item").rename(columns={"draw": "item"})

    return relabel(verdicts), relabel(correctness)


def bootstrap(verdicts: pd.DataFrame, correctness: pd.DataFrame, n_resamples: int = 200, seed: int = 0,
              drop_ties: bool = False, ridge: float = RIDGE, correlation: str = "kendall") -> dict:
    """
    Item-bootstrap intervals for the abilities and the ranking's correlation with accuracy.

    Returns {"ability": DataFrame model, p5, p50, p95; "correlation": Series p5, p50, p95}.
    """
    rng = np.random.default_rng(seed)
    abilities, correlations = [], []
    for _ in range(n_resamples):
        judged, table = resample_items(verdicts, correctness, rng)
        ranking = rank(judged, drop_ties=drop_ties, ridge=ridge)
        abilities.append(ranking.set_index("model").ability)
        correlations.append(score(ranking, table, judged, correlation=correlation)["correlation"])

    quantiles = {"p5": 0.05, "p50": 0.5, "p95": 0.95}
    ability = pd.concat(abilities, axis=1)  # models x resamples
    return {
        "ability": pd.DataFrame({name: ability.quantile(q, axis=1) for name, q in quantiles.items()})
                     .rename_axis("model").reset_index(),
        "correlation": pd.Series({name: np.quantile(correlations, q) for name, q in quantiles.items()}),
    }
