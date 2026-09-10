"""Transcript-statistics baselines: theta-hat from the length of a model's correct train transcripts."""

import re

import numpy as np
import pandas as pd

from capability.task import Task
from protocols import Protocol

THOUGHT = re.compile(r"<THOUGHT_START>.*?<THOUGHT_END>", re.S)
FEATURES = {
    "chars": lambda t: len(t.text),
    "answer_chars": lambda t: len(THOUGHT.sub("", t.text)),
    "reasoning_chars": lambda t: len(t.text) - len(THOUGHT.sub("", t.text)),
    "tokens": lambda t: t.tokens,
}


def features(task: Task, model: str) -> pd.Series:
    """Mean log1p of each transcript feature over the model's correct train transcripts."""
    rows = [{k: f(t[0]) for k, f in FEATURES.items()} for t in task.transcripts(model).values()]
    return np.log1p(pd.DataFrame(rows, columns=list(FEATURES))).mean()


def table(task: Task) -> pd.DataFrame:
    """Feature row per reference and the target."""
    return pd.DataFrame({m: features(task, m) for m in task.references + [task.target]}).T


class FeatureRegression(Protocol):
    """Least squares of the references' theta* on one log feature, applied to the target."""

    def __init__(self, feature: str):
        self.name = self.feature = feature

    def __call__(self, task: Task) -> float:
        x = table(task)[self.feature]
        xr, yr = x.loc[task.references], task.theta_ref
        if xr.std() == 0:  # e.g. no reference reasons: the feature says nothing about theta
            return float(yr.mean())
        beta, alpha = np.polyfit(xr, yr, 1)
        return alpha + beta * x.loc[task.target]


class Ridge(Protocol):
    """Ridge over every standardised log feature (8 references, so shrinkage is not optional)."""

    name = "ridge"

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def __call__(self, task: Task) -> float:
        x, y = table(task), task.theta_ref
        ref = x.loc[task.references]
        z = (x - ref.mean()) / ref.std().replace(0, 1)
        zr = z.loc[task.references].to_numpy()
        w = np.linalg.solve(zr.T @ zr + self.lam * np.eye(zr.shape[1]), zr.T @ (y - y.mean()))
        return float(y.mean() + z.loc[task.target].to_numpy() @ w)
