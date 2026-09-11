"""Pairwise LLM-judge protocol: Bradley-Terry over a judge's verdicts, anchored on the references.

A judge run (`uv run judge`, log dir logs/rank/<domain>/<judge>/<condition hash>/) asked a judge
model to compare two correct transcripts of the same item and pick the more capable model.
ranking.verdicts.from_logs turns such a dir into one row per verdict:
    bench, judge, item, model_left, model_right, pick_left (1 = picked the left one)

One PairwiseJudge pools the runs of one judge (typically one dir per semester). For a task it
keeps the verdicts on the task's train items between two of the task's models, fits one set of
Bradley-Terry abilities (ranking.bradley_terry.rank), and maps ability to theta with a line
fitted on the references' theta*.
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
        if self._verdicts is None:  # read the logs once, on first use
            self._verdicts = pd.concat([from_logs(d) for d in self.log_dirs], ignore_index=True)
        return self._verdicts

    def __call__(self, task: Task) -> pd.Series:
        """theta-hat per target; NaN for targets (or tasks) the runs do not cover."""
        models = set(task.models.index)
        verdicts = self.verdicts.merge(task.items[["bench", "item"]])  # train items only
        verdicts = verdicts[verdicts.model_left.isin(models) & verdicts.model_right.isin(models)]
        nothing = pd.Series(np.nan, index=task.targets)
        if verdicts.empty:
            return nothing

        ability = rank(verdicts).set_index("model").ability
        references = ability.index.intersection(task.references)
        if len(references) < 2:  # a line needs two anchors
            return nothing
        slope, intercept = np.polyfit(ability[references], task.theta_ref[references], 1)
        return intercept + slope * ability.reindex(task.targets)
