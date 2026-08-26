"""
How good a ranking is: Kendall tau between fitted ability and benchmark accuracy.

Accuracy is computed on the items that were actually judged, so ranking a subset
of items is scored against accuracy on that subset.
"""

import pandas as pd
from scipy.stats import kendalltau

ITEM = ["bench", "item"]


def accuracy_on_judged_items(correctness: pd.DataFrame, verdicts: pd.DataFrame) -> pd.Series:
    """Per model, mean correctness over the (bench, item) pairs that appear in the verdicts."""
    judged = verdicts[ITEM].drop_duplicates()
    return correctness.merge(judged, on=ITEM).groupby("model").correct.mean()


def score(ranking: pd.DataFrame, correctness: pd.DataFrame, verdicts: pd.DataFrame) -> dict:
    """
    Kendall tau between ability and accuracy, over the models present in both.

    ranking: output of rank(). correctness: bench, model, item, correct. verdicts: the table rank() was given.
    Returns kendall_tau, n_models, n_items (judged items the accuracy was computed on).
    """
    accuracy = accuracy_on_judged_items(correctness, verdicts)
    ability = ranking.set_index("model").ability
    models = ability.index.intersection(accuracy.index)
    tau = kendalltau(ability[models], accuracy[models]).statistic
    return {"kendall_tau": float(tau), "n_models": len(models), "n_items": len(verdicts[ITEM].drop_duplicates())}
