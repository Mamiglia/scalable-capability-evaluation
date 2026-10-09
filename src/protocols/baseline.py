"""Constant baselines: theta-hat = mean theta* of the previous-semester references.

The floor any protocol must beat: `mean_ref` gives every target the references' mean theta*
(`top` averages the TOP_K strongest instead).
"""

import pandas as pd

from capability.task import Cohort, Task
from protocols import Protocol


class MeanReference(Protocol):
    def __init__(self, name: str, top: int | None = None):
        self.name, self.top = name, top

    def __call__(self, task: Task) -> pd.Series:
        self.fit(task.references)  # references only: history would drag the mean into the past
        return self.predict(task.test)

    def fit(self, train: Cohort) -> None:
        theta = train.theta.dropna().sort_values(ascending=False)
        self.constant = theta.head(self.top).mean() if self.top else theta.mean()

    def predict(self, test: Cohort) -> pd.Series:
        return pd.Series(self.constant, index=test.models.index)


PROTOCOLS = {"mean_ref": MeanReference("mean_ref")}
