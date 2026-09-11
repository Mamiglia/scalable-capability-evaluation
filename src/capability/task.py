"""A semester task: a train cohort (the references, theta* known) and a test cohort (the targets).

Built from the frozen tables (reference.tables). Both cohorts sit on the same train items:
a protocol fits on the references' theta* and forecasts the targets, and it may read the correct
repeat-0 transcripts of either cohort on those items. The targets' theta* is kept on the task
for the metric only. Nothing about the HARD items is exposed.
"""

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from configs.domain import DomainConfig
from evals.transcripts import Transcript, read_transcripts
from reference import tables

TRAIN_COLUMNS = ["bench", "item", "guess", "a", "b"]


@dataclass
class Cohort:
    """Some models on the train items: their theta (NaN on the test cohort) and their transcripts."""

    # Indexed by model: theta, and one column per bench, `path_<bench>`, with the repeat-0 log.
    models: pd.DataFrame
    items: pd.DataFrame  # bench, item, guess, a, b
    semester: str  # the task this cohort belongs to (a judge run is stored per semester)

    @property
    def theta(self) -> pd.Series:
        return self.models.theta

    @cached_property
    def transcripts(self) -> dict[str, dict[tuple[str, str], Transcript]]:
        """model -> (bench, item) -> correct repeat-0 transcript. Read from the logs once."""
        found: dict[str, dict[tuple[str, str], Transcript]] = {}
        for model in self.models.index:
            found[model] = {}
            for bench, item_ids in self.items.groupby("bench").item:
                path = self.models.loc[model, f"path_{bench}"]
                if not isinstance(path, str):  # NaN: the model was never run on this bench
                    continue
                read = read_transcripts(tables.DATA / path, require_correct=True, items=set(item_ids))
                found[model].update({(bench, item): texts[0] for item, texts in read.items()})
        return found


@dataclass
class Task:
    semester: str
    train: Cohort  # references, theta = theta*
    test: Cohort  # targets, theta = NaN
    truth: pd.Series  # the targets' theta*, for the metric only

    @classmethod
    def from_semester(cls, semester: str, domain: DomainConfig) -> "Task":
        frozen_models, frozen_items = tables.load(domain)
        role = frozen_models[f"{semester}_role"]
        models = pd.DataFrame({"theta": frozen_models[f"{semester}_theta"]}).join(frozen_models.filter(like="path_"))
        items = frozen_items[frozen_items[f"{semester}_train"]][TRAIN_COLUMNS].reset_index(drop=True)
        train = Cohort(models[role == "reference"], items, semester)
        truth = models.theta[role == "target"]
        test = Cohort(models[role == "target"].assign(theta=float("nan")), items, semester)
        return cls(semester, train, test, truth)
