"""Transcript-statistics baselines: theta-hat from how much a model writes on the train items.

Each protocol summarises a model by one number per feature — the mean log length of its correct
train transcripts — fits the references' theta* on that summary, and applies the fit to the targets.
"""

import re

import numpy as np
import pandas as pd

from capability.task import Task
from protocols import Protocol

# extract_transcript wraps a model's reasoning in these markers (when include_reasoning is on).
THOUGHT = re.compile(r"<THOUGHT_START>.*?<THOUGHT_END>", re.S)


def answer_text(transcript) -> str:
    return THOUGHT.sub("", transcript.text)


FEATURES = {
    "chars": lambda t: len(t.text),
    "answer_chars": lambda t: len(answer_text(t)),
    "reasoning_chars": lambda t: len(t.text) - len(answer_text(t)),
    "tokens": lambda t: t.tokens,  # billed output tokens, reasoning included
}


def model_features(task: Task, model: str) -> pd.Series:
    """One value per feature: mean of log1p(feature) over the model's correct train transcripts."""
    rows = []
    for transcripts in task.transcripts(model).values():
        first = transcripts[0]  # repeat 0
        rows.append({name: feature(first) for name, feature in FEATURES.items()})
    table = pd.DataFrame(rows, columns=list(FEATURES))
    return np.log1p(table).mean()


def feature_table(task: Task) -> pd.DataFrame:
    """Rows = every model of the task (references and targets), columns = FEATURES."""
    return pd.DataFrame({model: model_features(task, model) for model in task.models.index}).T


class FeatureRegression(Protocol):
    """Least squares of the references' theta* on one feature, applied to the targets."""

    def __init__(self, feature: str):
        self.feature = feature
        self.name = feature

    def __call__(self, task: Task) -> pd.Series:
        x = feature_table(task)[self.feature]
        x_ref = x[task.references]
        if x_ref.std() == 0:
            # e.g. reasoning_chars when no reference reasons: the feature says nothing about theta.
            return pd.Series(task.theta_ref.mean(), index=task.targets)
        slope, intercept = np.polyfit(x_ref, task.theta_ref, 1)
        return intercept + slope * x[task.targets]


class Ridge(Protocol):
    """Ridge regression on every feature, standardised on the references.

    Eight references against four collinear features: without shrinkage the fit is arbitrary.
    """

    name = "ridge"

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def __call__(self, task: Task) -> pd.Series:
        x = feature_table(task)
        x_ref = x.loc[task.references]
        z = (x - x_ref.mean()) / x_ref.std().replace(0, 1)  # a constant feature becomes all zeros
        z_ref = z.loc[task.references].to_numpy()
        y_ref = task.theta_ref - task.theta_ref.mean()
        weights = np.linalg.solve(z_ref.T @ z_ref + self.lam * np.eye(z_ref.shape[1]), z_ref.T @ y_ref)
        return task.theta_ref.mean() + z.loc[task.targets] @ weights
