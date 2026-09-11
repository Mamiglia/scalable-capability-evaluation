"""A semester task: the reference models, whose theta* is known, and the target models to forecast.

Built from the frozen tables (capability.semesters). A protocol gets the references with their
theta* (the anchors), the targets without it, the train items, and the correct repeat-0
transcripts of those items for any of the task's models. Nothing about the HARD items is exposed.
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
    # Indexed by model: role (reference / target), theta (theta* for references, NaN for targets),
    # and one column per bench, `path_<bench>`, with the model's repeat-0 log.
    models: pd.DataFrame
    items: pd.DataFrame  # the train items: bench, item, guess, a, b

    @property
    def references(self) -> list[str]:
        return self.models.index[self.models.role == "reference"].tolist()

    @property
    def targets(self) -> list[str]:
        return self.models.index[self.models.role == "target"].tolist()

    @property
    def theta_ref(self) -> pd.Series:
        """theta* of the references, indexed by model."""
        return self.models.theta[self.references]

    def transcripts(self, model: str, include_reasoning: bool = True) -> dict[tuple[str, str], list[Transcript]]:
        """The model's correct repeat-0 transcripts of the train items: (bench, item) -> [Transcript]."""
        found = {}
        for bench, item_ids in self.items.groupby("bench").item:
            path = self.models.loc[model, f"path_{bench}"]
            if not isinstance(path, str):  # NaN: the model was never run on this bench
                continue
            transcripts = read_transcripts(semesters.DATA / path, require_correct=True,
                                           include_reasoning=include_reasoning, items=set(item_ids))
            for item, texts in transcripts.items():
                found[(bench, item)] = texts
        return found


def tasks(domain: DomainConfig) -> Iterator[Task]:
    """One Task per semester of the domain, from the frozen tables."""
    frozen_models, frozen_items = semesters.load(domain)
    paths = frozen_models.filter(like="path_")
    semester_names = [column.removesuffix("_role") for column in frozen_models.columns if column.endswith("_role")]
    for semester in semester_names:
        role = frozen_models[f"{semester}_role"]
        models = pd.DataFrame({"role": role, "theta": frozen_models[f"{semester}_theta"]}).join(paths)
        models = models[models.role.isin(["reference", "target"])]
        models.loc[models.role == "target", "theta"] = float("nan")  # what the protocol must forecast
        items = frozen_items[frozen_items[f"{semester}_train"]][TRAIN_COLUMNS].reset_index(drop=True)
        yield Task(semester, models, items)
