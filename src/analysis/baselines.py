"""Baselines that read no transcript: easy-item prediction, and theta* calibrated on HARD-item accuracy.

Easy-item prediction: every target forecast as the references' mean theta*. Accuracy on the
saturated train items is 1 for every model, so a regression of theta* on it is exactly this
baseline; a condition that cannot beat it has learned nothing.

Oracle (accuracy_hard): OLS of theta* on mean accuracy over every repeat on this task's HARD
items, the responses theta* itself was fit on. Calibrated on the 8 references, so it is not
exact: with weak references and strong targets the line extrapolates.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from capability.task import Cohort, Task
from configs.domain import DomainConfig
from protocols import Protocol
from reference import tables


class EasyItemPrediction(Protocol):
    """Forecast every target as the mean theta* of the previous semester's references.

    Uses task.references only (not history), like the judge conversions.
    """

    name = "easy_items"

    def __call__(self, task: Task) -> pd.Series:
        self.fit(task.references)
        return self.predict(task.test)

    def fit(self, train: Cohort) -> None:
        """Store the mean theta* of `train` (NaN if none is finite)."""
        self.mean = float(train.theta.mean())

    def predict(self, test: Cohort) -> pd.Series:
        """The stored mean for every model of `test`, indexed like test.models."""
        return pd.Series(self.mean, index=test.models.index)


class AccuracyOracle(Protocol):
    """OLS of the training models' theta* on their HARD-item accuracy, applied to the targets.

    Accuracy per model = mean of `correct` over every repeat and every item whose
    `<semester>_set` is "hard" in the frozen items table. Protocol name accuracy_hard.
    """

    name = "accuracy_hard"

    def __init__(self, domain: DomainConfig):
        self.responses = tables.load_responses(domain)  # bench, item, model, repeat, correct
        self.items = tables.load(domain)[1]  # bench, item, ..., <semester>_set (easy / hard / "")
        self.estimator: LinearRegression | None = None

    def accuracy(self, cohort: Cohort) -> pd.Series:
        """HARD-item accuracy per model of `cohort` (index = cohort.models.index; NaN for a
        model with no HARD response on this task)."""
        hard = self.items[self.items[f"{cohort.semester}_set"] == "hard"][["bench", "item"]]
        return self.responses.merge(hard).groupby("model").correct.mean().reindex(cohort.models.index)

    def fit(self, train: Cohort) -> None:
        """Fit theta* ~ accuracy on the models of `train` with both known; fewer than two
        usable models leaves the estimator None (every forecast NaN)."""
        x, y = self.accuracy(train), train.theta
        known = np.isfinite(x) & np.isfinite(y)
        self.estimator = LinearRegression().fit(x[known].to_frame(), y[known]) if known.sum() >= 2 else None

    def predict(self, test: Cohort) -> pd.Series:
        """theta-hat per model of `test` (index = test.models.index); NaN where the model has
        no HARD accuracy or nothing was fit."""
        x = self.accuracy(test)
        prediction = pd.Series(np.nan, index=test.models.index)
        if self.estimator is not None and (known := np.isfinite(x)).any():
            prediction[known] = self.estimator.predict(x[known].to_frame())
        return prediction
