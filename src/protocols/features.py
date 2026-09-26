"""Legacy length baselines and the shared TabularRegression wrapper.

A model is one row of features — the mean log length of its correct train transcripts — and
a protocol is a sklearn estimator of theta* on some of those columns, fit on the references and historical models.
The new length/regex/structure GLM feature sets live in transcript_features/features.py.
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin as SKlearnRegressor, clone

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
                             columns=list(FEATURES), dtype=float)
        rows[model] = np.log1p(table).mean()
    return pd.DataFrame(rows).T.reindex(cohort.models.index)


class TabularRegression(Protocol):
    """Regress task-specific theta* on feature columns; incomplete rows remain unpredicted."""

    def __init__(self, name: str, features: list[str], estimator : SKlearnRegressor):
        self.name, self.features, self.estimator = name, features, estimator
        self.fitted = False
        
    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        """Cohort -> float DataFrame of selected columns, with one row per model.

        The default uses legacy mean-log-length features; subclasses supply their
        own feature tables. Row labels must match cohort.models.index.
        """
        return feature_table(cohort)[self.features]

    def fit(self, train: Cohort) -> None:
        """Fit a fresh estimator using train's feature rows and matching theta labels.

        Rows with any nonfinite feature or label are excluded. Returns None;
        fewer than two usable references leaves the estimator marked unfitted.
        Each call clears the previous fit, including when the new data is unusable.
        """
        self.fitted = False
        self.estimator = clone(self.estimator)
        x = self.get_features(train)
        y = train.theta.reindex(x.index)
        known = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if known.sum() < 2:
            return
        self.estimator.fit(x.loc[known], y.loc[known])
        self.fitted = True

    def predict(self, test: Cohort) -> pd.Series:
        """Return predicted theta* as a float Series indexed like test.models.

        Uses features only, never test.theta. Nonfinite feature rows receive NaN;
        if fit has not produced a usable estimator, every prediction is NaN.
        """
        prediction = pd.Series(np.nan, index=test.models.index)
        if not self.fitted:
            return prediction
        x = self.get_features(test)
        known = np.isfinite(x).all(axis=1)
        if known.any():
            prediction.loc[x.index[known]] = self.estimator.predict(x.loc[known])
        return prediction


PROTOCOLS: dict[str, Protocol] = {f: TabularRegression(f, [f], LinearRegression()) for f in FEATURES}
# Small training cohorts against collinear features: standardise and shrink, or the fit is arbitrary.
PROTOCOLS["ridge"] = TabularRegression("ridge", list(FEATURES), make_pipeline(StandardScaler(), Ridge(1.0)))

from protocols.transcript_features.protocol import TranscriptRegression  # noqa: E402  (subclasses TabularRegression)

for variant in ("length", "regex", "structure"):
    PROTOCOLS[f"glm_{variant}"] = TranscriptRegression(variant=variant)

from protocols.embeddings import EmbeddingRegression  # noqa: E402  (subclasses TabularRegression)

PROTOCOLS["embedding"] = EmbeddingRegression()  # the "answer" variant is an analysis condition (analysis.calibration), not a CLI protocol
