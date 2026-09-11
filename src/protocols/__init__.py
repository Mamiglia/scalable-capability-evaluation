"""Benchmark-free protocols: a Protocol maps a semester Task (capability.task) to theta-hat of its targets.

fit sees the train cohort (references with theta*), predict the test cohort (targets, theta NaN).
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
