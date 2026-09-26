"""
The two predictors that read no transcript. The Baseline predicts every target at the references' mean theta*: every
model solves every saturated train item, so a regression of theta* on accuracy over those items reduces to exactly this,
and a protocol that cannot beat it has learned nothing from the transcripts. The HARD-item accuracy oracle
(accuracy_hard) is a least-squares line of theta* on each model's accuracy over the task's HARD items, the responses
theta* itself was fitted on; being fitted on the references only, it is not exact and extrapolates when the targets are
stronger than the references.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from capability.task import Cohort, Task
from configs.domain import DomainConfig
from protocols import Protocol
from reference import tables


class EasyItemPrediction(Protocol):
    """
    The Baseline: every target is predicted at the mean theta* of the references (the previous semester's models). It
    uses task.references only, not the historical models, like the judge conversions.

    Attributes:
        name: the protocol's code key, "easy_items".
        mean: the references' mean theta*, set by fit.
    """

    name = "easy_items"

    def __call__(self, task: Task) -> pd.Series:
        """
        Fits on the task's references and predicts every target.

        Inputs:
            task: the semester task; uses references and test.
        Outputs:
            Series of theta-hat indexed by target model, every value the references' mean theta*.
        """
        self.fit(task.references)
        return self.predict(task.test)

    def fit(self, train: Cohort) -> None:
        """
        Stores the mean theta* of the cohort it is fitted on.

        Inputs:
            train: the cohort to fit on (the references); uses theta.
        Outputs:
            none returned; sets self.mean (NaN when no theta* is finite).
        """
        self.mean = float(train.theta.mean())

    def predict(self, test: Cohort) -> pd.Series:
        """
        Predicts the stored mean for every model of the cohort.

        Inputs:
            test: the cohort to predict (the targets); uses models.index.
        Outputs:
            Series indexed by test.models.index, every value self.mean.
        """
        return pd.Series(self.mean, index=test.models.index)


class AccuracyOracle(Protocol):
    """
    The HARD-item accuracy oracle: a least-squares line of the training models' theta* on their accuracy over the
    task's HARD items, applied to the targets. A model's accuracy is the mean of `correct` over every repeat of every
    item whose `<semester>_set` is "hard" in the frozen items table.

    Attributes:
        name: the protocol's code key, "accuracy_hard".
        responses: the domain's frozen responses (bench, item, model, repeat, correct).
        items: the domain's frozen items table, with a `<semester>_set` column per semester ("easy", "hard" or "").
        estimator: the fitted LinearRegression; None before fit, or when fewer than two models were usable.
    """

    name = "accuracy_hard"

    def __init__(self, domain: DomainConfig):
        """
        Loads the domain's frozen response and item tables for the HARD-item accuracy fit.

        Inputs:
            domain: the domain configuration.
        Outputs:
            none returned; sets responses, items and estimator (None).
        """
        self.responses = tables.load_responses(domain)  # bench, item, model, repeat, correct
        self.items = tables.load(domain)[1]  # bench, item, ..., <semester>_set (easy / hard / "")
        self.estimator: LinearRegression | None = None

    def accuracy(self, cohort: Cohort) -> pd.Series:
        """
        Computes each model's accuracy on the task's HARD items.

        Inputs:
            cohort: the models to cover; uses semester and models.index.
        Outputs:
            Series of accuracy (mean of correct over every HARD item and repeat) indexed by cohort.models.index; NaN
            for a model with no HARD response on this task.
        """
        hard = self.items[self.items[f"{cohort.semester}_set"] == "hard"][["bench", "item"]]
        return self.responses.merge(hard).groupby("model").correct.mean().reindex(cohort.models.index)

    def fit(self, train: Cohort) -> None:
        """
        Fits a least-squares line of theta* on HARD-item accuracy over the cohort's models that have both.

        Inputs:
            train: the cohort to fit on (the references); uses theta, semester and models.index.
        Outputs:
            none returned; sets self.estimator, or sets it to None when fewer than two models have both values (every
            prediction is then NaN).
        """
        x, y = self.accuracy(train), train.theta
        known = np.isfinite(x) & np.isfinite(y)
        self.estimator = LinearRegression().fit(x[known].to_frame(), y[known]) if known.sum() >= 2 else None

    def predict(self, test: Cohort) -> pd.Series:
        """
        Applies the fitted line to each model's HARD-item accuracy.

        Inputs:
            test: the cohort to predict (the targets); uses semester and models.index.
        Outputs:
            Series of theta-hat indexed by test.models.index; NaN for a model without HARD accuracy, and for every
            model when nothing was fitted.
        """
        x = self.accuracy(test)
        prediction = pd.Series(np.nan, index=test.models.index)
        if self.estimator is not None and (known := np.isfinite(x)).any():
            prediction[known] = self.estimator.predict(x[known].to_frame())
        return prediction
