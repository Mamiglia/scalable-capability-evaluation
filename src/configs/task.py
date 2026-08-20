from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from inspect_ai import Epochs
from inspect_ai.solver import SolverSpec

from configs.base import DraccusConfig


@dataclass(kw_only=True)
class TaskConfig(DraccusConfig):
    """A benchmark to run: which inspect task, with which arguments.

    ``name`` is the label used for the data directory; ``task`` is the inspect task spec
    (e.g. ``inspect_evals/gpqa_diamond``) and defaults to ``name``.
    """

    task: str = ''
    task_args: Dict[str, Any] = field(default_factory=dict)
    # Override the task's built-in solver, as an inspect spec: a registered name or a
    # ``file.py@name`` path (resolved from the repo root). Empty = use the task default.
    solver: str = ''
    # No generate cache by default: repeats reuse the same prompts, so a cache turns the
    # second repeat into a replay of the first (seen on aime/mbpp; gpqa escaped only because
    # its options are reshuffled per run).
    cache: Any = None
    limit: Optional[int] = None
    epochs: int = 1
    epochs_reducer: Any = None

    def __post_init__(self):
        if not self.task:
            self.task = self.name
        if self.task_args is None:
            self.task_args = {}

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments passed to inspect_ai.eval_set."""
        kwargs = dict(self._kwargs)
        if self.cache is not None:
            kwargs["cache"] = self.cache
        if self.limit:
            kwargs["limit"] = self.limit
        if self.solver:
            kwargs["solver"] = SolverSpec(solver=self.solver)
        kwargs["epochs"] = (
            Epochs(self.epochs, self.epochs_reducer)
            if self.epochs_reducer is not None
            else self.epochs
        )
        return kwargs
