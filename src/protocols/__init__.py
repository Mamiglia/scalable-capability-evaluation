"""Benchmark-free protocols: a Protocol maps a semester Task (capability.task) to theta-hat of its targets.

To add one: subclass Protocol, give it a `name`, and add an instance to PROTOCOLS below;
`uv run evaluate` picks it up. Protocols that need arguments (PairwiseJudge needs log dirs) are
constructed by the CLI instead.
"""

import pandas as pd

from capability.task import Task


class Protocol:
    name: str

    def __call__(self, task: Task) -> pd.Series:
        """theta-hat indexed by target model (NaN where the protocol has nothing to say)."""
        raise NotImplementedError


# Imported after Protocol is defined, because the modules below subclass it.
from protocols.features import FEATURES, FeatureRegression, Ridge  # noqa: E402

PROTOCOLS: dict[str, Protocol] = {}
for feature in FEATURES:
    PROTOCOLS[feature] = FeatureRegression(feature)
PROTOCOLS["ridge"] = Ridge()
