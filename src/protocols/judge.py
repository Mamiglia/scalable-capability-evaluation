"""Pairwise LLM-judge protocol: Bradley-Terry over one judge run's verdicts, anchored on the references.

A judge run is a `uv run judge` log dir (logs/rank/<domain>/<judge>/<condition hash>/): the judge
compared two correct transcripts of the same item and picked the more capable model
(ranking.verdicts.from_logs). One protocol pools the runs of one judge (one dir per semester);
only verdicts among the task's models on its train items count.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from capability.task import Task
from protocols import Protocol
from ranking.bradley_terry import rank
from ranking.verdicts import from_logs


class PairwiseJudge(Protocol):
    def __init__(self, log_dirs: list[str | Path], name: str = ""):
        self.log_dirs = [Path(d) for d in log_dirs]
        self.name = name or f"judge_{self.log_dirs[0].parent.name}"
        self._verdicts: pd.DataFrame | None = None

    @property
    def verdicts(self) -> pd.DataFrame:
        if self._verdicts is None:
            self._verdicts = pd.concat([from_logs(d) for d in self.log_dirs], ignore_index=True)
        return self._verdicts

    def __call__(self, task: Task) -> float:
        """BT ability of the target, affine-mapped to theta by the references' abilities vs theta*; NaN without coverage."""
        models = {*task.references, task.target}
        v = self.verdicts.merge(task.items[["bench", "item"]])
        v = v[v.model_left.isin(models) & v.model_right.isin(models)]
        if v.empty or task.target not in {*v.model_left, *v.model_right}:
            return np.nan
        ability = rank(v).set_index("model").ability
        ref = ability.index.intersection(task.references)
        if len(ref) < 2:
            return np.nan
        beta, alpha = np.polyfit(ability[ref], task.theta_ref[ref], 1)
        return alpha + beta * ability[task.target]
