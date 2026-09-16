"""Benchmark-free protocols: a Protocol maps a semester Task (capability.task) to theta-hat of its targets.

fit sees train (references + historical models with task-specific theta*), predict sees targets
with theta NaN. PairwiseBT overrides this routing to use only the previous-semester references.
To add one: subclass Protocol, give it a `name`, and add an instance to PROTOCOLS below;
`uv run evaluate` picks it up. Protocols that need arguments (JudgeBT needs a judge's logs)
are constructed by the CLI instead.
"""

import pandas as pd

from capability.task import Cohort, Task


class Protocol:
    name: str

    def fit(self, train: Cohort) -> None:
        pass

    def predict(self, test: Cohort) -> pd.Series:
        """theta-hat indexed by test model (NaN where the protocol has nothing to say)."""
        raise NotImplementedError

    def __call__(self, task: Task) -> pd.Series:
        self.fit(task.train)
        return self.predict(task.test)


# Imported after Protocol is defined, because the module subclasses it.
from protocols.features import PROTOCOLS  # noqa: E402, F401
from protocols.baseline import PROTOCOLS as _BASELINES  # noqa: E402
from protocols.embeddings import PROTOCOLS as _EMBEDDINGS  # noqa: E402  emb_kr / emb_knn / emb_ens (+ _answer)
from protocols.accuracy import PROTOCOLS as _ACCURACY  # noqa: E402  acc_train / acc_easy / acc_all: benchmark-score baselines

PROTOCOLS.update(_BASELINES)
PROTOCOLS.update(_EMBEDDINGS)
PROTOCOLS.update(_ACCURACY)
