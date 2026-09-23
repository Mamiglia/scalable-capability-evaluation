"""DirectRegression: mean ratings calibrated to the training models' HARD-item theta*.

Implemented to roughly match structure of ../pairwise.

A subclass supplies direct scores on correct train transcripts. fit keeps the references
and predict obtains judgments for training models and
targets on demand. Only training theta* enters calibration; target theta is never read.
"""

import pandas as pd

from capability.task import Cohort
from protocols import Protocol
from protocols.direct_judge.scoring import calibrate


class DirectRegression(Protocol):
    def scores(self, train: Cohort, test: Cohort) -> pd.DataFrame:
        """Ratings: bench, model, item, score, semester (a superset is fine: predict filters)."""
        raise NotImplementedError

    def fit(self, train: Cohort) -> None:
        self.train = train

    def predict(self, test: Cohort) -> pd.Series:
        scores = self.scores(self.train, test)
        eligible = []
        for cohort in (self.train, test):
            items = set(zip(cohort.items.bench, cohort.items.item))
            eligible.extend((model, bench, item, cohort.semester)
                            for model in cohort.models.index
                            for bench, item in cohort.transcripts[model] if (bench, item) in items)
        keys = ["model", "bench", "item", "semester"]
        scores = scores.merge(pd.DataFrame(eligible, columns=keys).drop_duplicates(), on=keys)
        return calibrate(scores, anchors=self.train.theta).reindex(test.models.index)
