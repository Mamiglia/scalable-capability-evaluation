"""Transcript-statistics baselines: theta-hat from how much a model writes on the train items.

A model is one row of features — the mean log length of its correct train transcripts — and
a protocol is a sklearn estimator of theta* on some of those columns, fit on the references.
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin as SKlearnRegressor

from capability.task import Cohort
from protocols import Protocol

# extract_transcript wraps a model's reasoning in these markers (when include_reasoning is on).
THOUGHT = re.compile(r"<THOUGHT_START>.*?<THOUGHT_END>", re.S)


def answer_text(transcript) -> str:
    return THOUGHT.sub("", transcript.text)

# TODO: These features are meant to be improved!
FEATURES = {
    "chars": lambda t: len(t.text),
    "answer_chars": lambda t: len(answer_text(t)),
    "reasoning_chars": lambda t: len(t.text) - len(answer_text(t)),
    "tokens": lambda t: t.tokens,  # billed output tokens, reasoning included
}


def feature_table(cohort: Cohort) -> pd.DataFrame:
    """Rows = the cohort's models, columns = FEATURES: mean of log1p(feature) over the correct train transcripts."""
    rows = {}
    for model, transcripts in cohort.transcripts.items():
        table = pd.DataFrame([{name: f(t) for name, f in FEATURES.items()} for t in transcripts.values()],
                             columns=list(FEATURES))
        rows[model] = np.log1p(table).mean()
    return pd.DataFrame(rows).T.reindex(cohort.models.index)


class TabularRegression(Protocol):
    """A sklearn estimator of theta* on some feature columns, fit on the references."""

    def __init__(self, name: str, features: list[str], estimator : SKlearnRegressor):
        self.name, self.features, self.estimator = name, features, estimator
        
    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        return feature_table(cohort)[self.features]

    def fit(self, train: Cohort) -> None:
        x = self.get_features(train)
        self.estimator.fit(x, train.theta)

    def predict(self, test: Cohort) -> pd.Series:
        x = self.get_features(test)
        return pd.Series(self.estimator.predict(x), index=x.index)


PROTOCOLS: dict[str, Protocol] = {f: TabularRegression(f, [f], LinearRegression()) for f in FEATURES}
# Eight references against four collinear features: standardise and shrink, or the fit is arbitrary.
PROTOCOLS["ridge"] = TabularRegression("ridge", list(FEATURES), make_pipeline(StandardScaler(), Ridge(1.0)))
