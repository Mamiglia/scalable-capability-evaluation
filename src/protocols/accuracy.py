"""Benchmark-score baselines: a model's accuracy on the frozen items, linearly mapped to theta*.

    acc_easy  accuracy on the task's saturated (EASY) items: the score a leaderboard shows once the
              benchmark has saturated, the conventional-benchmark baseline of the paper.
    acc_all   accuracy on every EASY and HARD item of the domain: the pre-saturation leaderboard
              number. Partly circular, because theta* is fit on the HARD items.
    acc_train accuracy on the cohort's selected train items only (25 per difficulty quartile of
              the EASY set): the items the transcript protocols see, with the correctness they
              are denied. The fair "score on what the protocol saw" comparison.

Correctness comes from data/meta/<domain>/responses.csv (every repeat, averaged per item, then over
items). The domain is found from the cohort itself: the frozen models.csv that lists every cohort
model and has a path_<bench> column for every cohort bench. Like every TabularRegression the map
is fit on the training cohort's theta* and applied to the targets.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from capability.task import Cohort
from protocols.features import TabularRegression
from reference import tables


@lru_cache(maxsize=None)
def _frozen(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_csv(directory / "items.csv", dtype={"item": str})
    responses = pd.read_csv(directory / "responses.csv", dtype={"item": str})
    return items, responses


def frozen_dir(cohort: Cohort) -> Path | None:
    """The cohort's domain directory under data/meta: a frozen models.csv with every cohort bench and some cohort model."""
    benches = {f"path_{b}" for b in cohort.items.bench.unique()}
    for csv in sorted(Path(tables.META).glob("*/models.csv")):
        columns = set(pd.read_csv(csv, nrows=0).columns)
        if benches <= columns and not set(cohort.models.index).isdisjoint(pd.read_csv(csv, usecols=["model"]).model):
            return csv.parent
    return None


def accuracy_table(cohort: Cohort, which: str) -> pd.DataFrame:
    """One `accuracy` column per cohort model over the task's EASY items (which="easy") or EASY+HARD ("all")."""
    directory = frozen_dir(cohort)
    if directory is None:
        return pd.DataFrame(np.nan, index=cohort.models.index, columns=["accuracy"])
    items, responses = _frozen(directory)
    label = items[f"{cohort.semester}_set"]
    if which == "train":
        keep = cohort.items
    else:
        keep = items[label == "easy" if which == "easy" else label.isin(["easy", "hard"])]
    selected = responses.merge(keep[["bench", "item"]].astype(str), on=["bench", "item"])
    per_item = selected.groupby(["model", "bench", "item"]).correct.mean()  # repeats averaged
    return per_item.groupby("model").mean().rename("accuracy").to_frame().reindex(cohort.models.index)


class AccuracyRegression(TabularRegression):
    """theta* regressed on one number, the benchmark accuracy."""

    def __init__(self, name: str, which: str):
        if which not in ("easy", "all", "train"):
            raise ValueError(f"which must be 'easy', 'all' or 'train', not {which!r}")
        self.which = which
        super().__init__(name, ["accuracy"], LinearRegression())

    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        return accuracy_table(cohort, self.which)


PROTOCOLS: dict[str, AccuracyRegression] = {
    "acc_easy": AccuracyRegression("acc_easy", "easy"),
    "acc_all": AccuracyRegression("acc_all", "all"),
    "acc_train": AccuracyRegression("acc_train", "train"),
}
