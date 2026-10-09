"""A complete new protocol, offline: each judge's pairwise win rate against the references, mapped to theta by a line
fit on the references (their win rate among themselves vs their theta*), averaged over the task's judges. Copy it.

    uv run evaluate --protocol pairwise scripts/example_protocol.py:WinRate --semester 2025-II 2026-I
"""

import numpy as np
import pandas as pd

from capability.task import Task
from protocols import Protocol
from protocols.judges import judgments


def win_rate(verdicts: pd.DataFrame, opponents: pd.Index) -> pd.Series:
    """Per model, the share of its verdicts against `opponents` that it won."""
    left = verdicts[verdicts.model_right.isin(opponents)].assign(model=lambda v: v.model_left, won=lambda v: v.pick_left)
    right = verdicts[verdicts.model_left.isin(opponents)].assign(model=lambda v: v.model_right, won=lambda v: 1 - v.pick_left)
    return pd.concat([left, right]).groupby("model").won.mean()


class WinRate(Protocol):
    name = "winrate"

    def __call__(self, task: Task) -> pd.Series:
        refs = task.references.theta
        v = judgments("pairwise")
        v = v[(v.domain == task.domain) & (v.semester == task.semester) & v.core]  # one run per judge (README)
        estimates = []
        for _, run in v.groupby("judge"):
            rate = win_rate(run, refs.index)
            known = rate.reindex(refs.index).dropna()
            slope, intercept = np.polyfit(known, refs[known.index], 1)
            estimates.append(intercept + slope * rate.reindex(task.test.models.index))
        return pd.concat(estimates, axis=1).mean(axis=1) if estimates else pd.Series(np.nan, index=task.test.models.index)
