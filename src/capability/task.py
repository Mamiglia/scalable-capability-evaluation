"""A semester task: one (semester, target) a protocol must forecast, built from the frozen tables.

The 8 reference models come with their local theta* (anchors); the target does not. Both sides
expose only correct repeat-0 transcripts of the train items (capability.semesters).
"""

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from capability import semesters
from configs.domain import DomainConfig
from judge.dataset import Transcript, read_transcripts


@dataclass
class Task:
    semester: str
    references: list[str]
    theta_ref: pd.Series  # local theta* of the references, by model
    target: str
    items: pd.DataFrame  # train items: bench, item, guess, a, b
    paths: pd.DataFrame  # repeat-0 log per model (rows) and bench (path_<bench> columns)

    def transcripts(self, model: str, include_reasoning: bool = True) -> dict[tuple[str, str], list[Transcript]]:
        """Correct repeat-0 transcripts of the train items: (bench, item) -> [Transcript]."""
        out = {}
        for bench, ids in self.items.groupby("bench").item:
            path = self.paths.loc[model, f"path_{bench}"]
            if isinstance(path, str):  # NaN = the model was never run on this bench
                read = read_transcripts(semesters.DATA / path, require_correct=True,
                                        include_reasoning=include_reasoning, items=set(ids))
                out.update({(bench, i): t for i, t in read.items()})
        return out


def tasks(domain: DomainConfig) -> Iterator[Task]:
    models, items = semesters.load(domain)
    paths = models.filter(like="path_")
    for sem in [c.removesuffix("_role") for c in models.columns if c.endswith("_role")]:
        role = models[f"{sem}_role"]
        ref = role.index[role == "reference"].tolist()
        train = items[items[f"{sem}_train"]][["bench", "item", "guess", "a", "b"]].reset_index(drop=True)
        for target in role.index[role == "target"]:
            yield Task(sem, ref, models.loc[ref, f"{sem}_theta"], target, train, paths)
