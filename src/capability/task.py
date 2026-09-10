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
    domain: DomainConfig
    semester: str
    references: list[str]
    theta_ref: pd.Series  # local theta* of the references, indexed by model
    target: str
    items: pd.DataFrame  # train items: bench, item, guess, a, b
    _models: pd.DataFrame  # frozen models.csv (for the log paths)

    def transcripts(self, model: str, include_reasoning: bool = True) -> dict[tuple[str, str], list[Transcript]]:
        """Correct repeat-0 transcripts of the train items: (bench, item) -> [Transcript]."""
        row = self._models.set_index("model").loc[model]
        out = {}
        for bench, ids in self.items.groupby("bench").item:
            path = row.get(f"path_{bench}", "")
            if not path:
                continue
            read = read_transcripts(semesters.DATA / path, require_correct=True,
                                    include_reasoning=include_reasoning, items=set(ids))
            out.update({(bench, i): t for i, t in read.items()})
        return out


def tasks(domain: DomainConfig) -> Iterator[Task]:
    models, items = semesters.load(domain)
    for sem in [c[: -len("_role")] for c in models.columns if c.endswith("_role")]:
        role, theta = models[f"{sem}_role"], models.set_index("model")[f"{sem}_theta"]
        ref = models.model[role == "reference"].tolist()
        train = items[items[f"{sem}_train"]][["bench", "item", "guess", "a", "b"]].reset_index(drop=True)
        for target in models.model[role == "target"]:
            yield Task(domain, sem, ref, theta.loc[ref], target, train, models)
