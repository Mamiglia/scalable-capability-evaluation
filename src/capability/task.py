"""A semester task: one (semester, target) a protocol must forecast, built from the frozen tables.

The task gives a protocol the 8 reference models with their theta* (the anchors), the target
without it, the train items, and the correct repeat-0 transcripts of those items for any of
the 9 models. Nothing about the HARD items is exposed.
"""

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from capability import semesters
from configs.domain import DomainConfig
from judge.dataset import Transcript, read_transcripts

TRAIN_COLUMNS = ["bench", "item", "guess", "a", "b"]


@dataclass
class Task:
    semester: str
    references: list[str]
    theta_ref: pd.Series  # theta* of the references, indexed by model
    target: str
    items: pd.DataFrame  # the train items: bench, item, guess, a, b
    paths: pd.DataFrame  # indexed by model, one column per bench (`path_<bench>`): the repeat-0 log

    def transcripts(self, model: str, include_reasoning: bool = True) -> dict[tuple[str, str], list[Transcript]]:
        """The model's correct repeat-0 transcripts of the train items: (bench, item) -> [Transcript]."""
        found = {}
        for bench, item_ids in self.items.groupby("bench").item:
            path = self.paths.loc[model, f"path_{bench}"]
            if not isinstance(path, str):  # NaN: the model was never run on this bench
                continue
            transcripts = read_transcripts(semesters.DATA / path, require_correct=True,
                                           include_reasoning=include_reasoning, items=set(item_ids))
            for item, texts in transcripts.items():
                found[(bench, item)] = texts
        return found


def tasks(domain: DomainConfig) -> Iterator[Task]:
    """Every (semester, target) of the domain, from the frozen tables."""
    models, items = semesters.load(domain)
    paths = models.filter(like="path_")
    semester_names = [column.removesuffix("_role") for column in models.columns if column.endswith("_role")]
    for semester in semester_names:
        role = models[f"{semester}_role"]
        references = role.index[role == "reference"].tolist()
        targets = role.index[role == "target"].tolist()
        theta_ref = models.loc[references, f"{semester}_theta"]
        train_items = items[items[f"{semester}_train"]][TRAIN_COLUMNS].reset_index(drop=True)
        for target in targets:
            yield Task(semester, references, theta_ref, target, train_items, paths)
