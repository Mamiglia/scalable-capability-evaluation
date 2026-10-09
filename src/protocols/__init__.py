"""Benchmark-free protocols: a Protocol maps a semester Task (capability.task) to theta-hat of its targets.

fit sees the previous semester's references only (never the historical models, so every protocol
learns from the same labelled models as the judges and the saturated-item prediction), predict sees
targets with theta NaN.
To add one: subclass Protocol, give it a `name`, and add an instance to PROTOCOLS below;
`uv run evaluate` picks it up. The paper's judge protocols read the recorded judgments (protocols.judges);
JudgeBT and friends, which read one judge's logs, are built by `uv run evaluate --judge` instead.
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
        self.fit(task.references)
        return self.predict(task.test)


# Imported after Protocol is defined, because the module subclasses it.
from protocols.features import PROTOCOLS  # noqa: E402, F401
from protocols.baseline import PROTOCOLS as _BASELINES  # noqa: E402

from protocols.judges import PROTOCOLS as _JUDGES  # noqa: E402

PROTOCOLS.update(_BASELINES)
PROTOCOLS.update(_JUDGES)
