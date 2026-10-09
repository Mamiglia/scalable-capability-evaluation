"""A semester task: previous-semester references, older history, and current targets.

All cohorts use the same selected EASY items and correct repeat-0 transcripts. Supervised
protocols fit on train (references + historical); pinned pairwise BT uses references only.
Labels come from this task's HARD items. Target theta* is kept in truth for evaluation only;
neither target labels nor HARD responses are exposed through the cohorts.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from configs.domain import DomainConfig
from configs.logs import logger
from evals.transcripts import Transcript, read_transcripts
from reference import tables

TRAIN_COLUMNS = ["bench", "item", "guess", "a", "b"]


@dataclass
class Cohort:
    """Models on one task's selected EASY items, possibly from several release semesters.

    semester identifies the evaluation task, not the models' release dates. theta is NaN
    on the test cohort; labelled cohorts carry this task's HARD-item theta*.
    """

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
    references: Cohort  # previous semester, theta = theta*
    historical: Cohort  # semesters before the references, theta = this task's theta*
    test: Cohort  # targets, theta = NaN
    truth: pd.Series  # the targets' theta*, for the metric only

    @property
    def train(self) -> Cohort:
        """The labelled union of references and history: which transcripts exist. Protocols fit on
        the references only (protocols.Protocol.__call__)."""
        models = pd.concat([self.references.models, self.historical.models], verify_integrity=True)
        return Cohort(models, self.references.items, self.semester)

    @classmethod
    def from_semester(cls, semester: str, domain: DomainConfig) -> "Task":
        frozen_models, frozen_items = tables.load(domain)
        role = frozen_models[f"{semester}_role"]
        models = pd.DataFrame({"theta": frozen_models[f"{semester}_theta"]}).join(frozen_models.filter(like="path_"))
        items = frozen_items[frozen_items[f"{semester}_train"]][TRAIN_COLUMNS].reset_index(drop=True)
        references = Cohort(models[role == "reference"], items, semester)
        rosters = tables.load_semesters()
        semesters = list(rosters)
        earlier = semesters[:max(0, semesters.index(semester) - 1)]
        older_models = {model for past in earlier for model in rosters[past]}
        historical = (frozen_models.semester.isin(earlier) & models.index.isin(older_models)
                      & ~role.isin(["reference", "target"]))
        missing = historical & ~np.isfinite(models.theta)
        if missing.any():
            # known gap of the frozen tables (no HARD coverage for old models), not a per-run problem: info, one line
            logger.info("%s: %d historical models without a task theta*, left out of `historical`", semester, int(missing.sum()))
        historical = Cohort(models[historical & ~missing], items, semester)
        truth = models.theta[role == "target"]
        test = Cohort(models[role == "target"].assign(theta=float("nan")), items, semester)
        return cls(semester, references, historical, test, truth)
