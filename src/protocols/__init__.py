"""Benchmark-free protocols: a Protocol maps a semester Task (capability.task) to theta-hat of its target.

Register a new protocol by adding an instance to PROTOCOLS; `uv run evaluate` picks it up.
"""

from capability.task import Task


class Protocol:
    name: str

    def __call__(self, task: Task) -> float:
        raise NotImplementedError


from protocols.features import FEATURES, FeatureRegression, Ridge  # noqa: E402

PROTOCOLS: dict[str, Protocol] = {p.name: p for p in [*(FeatureRegression(f) for f in FEATURES), Ridge()]}
