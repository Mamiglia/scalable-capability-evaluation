"""PairwiseBT: theta-hat from pairwise verdicts, the references pinned at their theta*.

A verdict compares two models' correct transcripts of the same train item and picks the more
capable model; one row per verdict:
    bench, judge, item, model_left, model_right, pick_left (1 = picked the left one)

A target is only placed relative to the references it was compared with, so fit keeps the reference
cohort and predict fits Bradley-Terry abilities (bradley_terry.rank) over references + targets on
the train items with the references pinned at their theta* — P(left wins) = sigmoid(theta_left -
theta_right) is the 2PL's own logit scale, so the targets' abilities are theta-hat directly.
A subclass supplies the verdicts.
"""

import numpy as np
import pandas as pd

from capability.task import Cohort, Task
from protocols import Protocol
from protocols.pairwise.bradley_terry import rank


class PairwiseBT(Protocol):
    def __call__(self, task: Task) -> pd.Series:
        self.fit(task.references)
        return self.predict(task.test)

    def verdicts(self, train: Cohort, test: Cohort) -> pd.DataFrame:
        """Comparisons between the two cohorts' models (a superset is fine: predict filters)."""
        raise NotImplementedError

    def fit(self, train: Cohort) -> None:
        self.train = train  # the anchors

    def predict(self, test: Cohort) -> pd.Series:
        models = self.train.models.index.union(test.models.index)
        verdicts = self.verdicts(self.train, test).merge(test.items[["bench", "item"]])
        verdicts = verdicts[verdicts.model_left.isin(models) & verdicts.model_right.isin(models)]
        nothing = pd.Series(np.nan, index=test.models.index)
        if verdicts.empty:
            return nothing
        if not (verdicts.model_left.isin(self.train.models.index) | verdicts.model_right.isin(self.train.models.index)).any():
            return nothing  # no reference in the verdicts: nothing pins the scale
        ability = rank(verdicts, anchors=self.train.theta).set_index("model").ability
        return ability.reindex(test.models.index)
