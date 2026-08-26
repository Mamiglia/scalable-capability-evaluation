"""
How good a ranking is: rank correlation between fitted ability and benchmark accuracy.

Accuracy can be taken over every item in the correctness table, over the judged
items only, or over the held-out items the judges never saw. bootstrap()
resamples the judged items to put intervals on the abilities and the correlation.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ranking.bradley_terry import RIDGE, rank

ITEM = ["bench", "item"]
CORRELATIONS = {"kendall": kendalltau, "spearman": spearmanr}  # kendall is tau-b (tie-corrected)


def accuracy(correctness: pd.DataFrame, verdicts: pd.DataFrame, items: str = "all") -> pd.Series:
    """
    Per model, mean correctness over the chosen items.

    items: "all" = every item in the correctness table; "judged" = items that appear in the
    verdicts; "held_out" = items that do not.
    """
    judged = correctness[ITEM].apply(tuple, axis=1).isin(verdicts[ITEM].drop_duplicates().apply(tuple, axis=1))
    keep = {"all": pd.Series(True, index=correctness.index), "judged": judged, "held_out": ~judged}[items]
    return correctness[keep].groupby("model").correct.mean()


def score(ranking: pd.DataFrame, correctness: pd.DataFrame, verdicts: pd.DataFrame,
          correlation: str = "kendall", items: str = "all") -> dict:
    """
    Rank correlation between ability and accuracy, over the models present in both.

    ranking: output of rank(). correctness: bench, model, item, correct. verdicts: the table rank() was given.
    correlation: "kendall" (tau-b) or "spearman". items: which items the accuracy is over, see accuracy().
    Returns correlation, method, n_models, n_items (items the accuracy was computed on).
    """
    truth = accuracy(correctness, verdicts, items)
    ability = ranking.set_index("model").ability
    models = ability.index.intersection(truth.index)
    value = CORRELATIONS[correlation](ability[models], truth[models]).statistic
    n_items = {"all": len(correctness[ITEM].drop_duplicates()), "judged": len(verdicts[ITEM].drop_duplicates())}
    n_items["held_out"] = n_items["all"] - n_items["judged"]
    return {"correlation": float(value), "method": correlation, "n_models": len(models), "n_items": n_items[items]}


def resample_items(verdicts: pd.DataFrame, correctness: pd.DataFrame, rng: np.random.Generator):
    """
    One bootstrap draw: judged items sampled with replacement, in both tables.
    """
    items = verdicts[ITEM].drop_duplicates()
    draws = items.iloc[rng.integers(len(items), size=len(items))].assign(draw=lambda d: np.arange(len(d)).astype(str))

    def relabel(table):
        return table.merge(draws, on=ITEM).drop(columns="item").rename(columns={"draw": "item"})

    return relabel(verdicts), relabel(correctness)

def bootstrap(verdicts: pd.DataFrame, correctness: pd.DataFrame, n_resamples: int = 200, seed: int = 0,
              drop_ties: bool = False, ridge: float = RIDGE, correlation: str = "kendall", items: str = "all") -> dict:
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
        correlations.append(score(ranking, table, judged, correlation=correlation, items=items)["correlation"])

    quantiles = {"p5": 0.05, "p50": 0.5, "p95": 0.95}
    ability = pd.concat(abilities, axis=1)  # models x resamples
    return {
        "ability": pd.DataFrame({name: ability.quantile(q, axis=1) for name, q in quantiles.items()})
                     .rename_axis("model").reset_index(),
        "correlation": pd.Series({name: np.quantile(correlations, q) for name, q in quantiles.items()}),
    }
