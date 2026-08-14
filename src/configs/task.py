from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from inspect_ai import Epochs
from inspect_ai.model import CachePolicy

from configs.base import DraccusConfig


@dataclass(kw_only=True)
class TaskConfig(DraccusConfig):
    """A benchmark to run: which inspect task, with which arguments.

    ``name`` is the label used for the data directory; ``task`` is the inspect task spec
    (e.g. ``inspect_evals/gpqa_diamond``) and defaults to ``name``.
    """

    task: str = ''
    task_args: Dict[str, Any] = field(default_factory=dict)
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
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
        kwargs["epochs"] = (
            Epochs(self.epochs, self.epochs_reducer)
            if self.epochs_reducer is not None
            else self.epochs
        )
        return kwargs
