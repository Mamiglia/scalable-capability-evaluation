from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from inspect_ai.model import CachePolicy

from configs.base import DraccusConfig


@dataclass(kw_only=True)
class TaskConfig(DraccusConfig):
    task_args: Dict[str, Any] = field(default_factory=dict)
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
    limit: Optional[int] = None
    epochs: int = 1
    eval_path: Optional[Path] = None

    @property
    def task(self) -> str:
        """Return the task name."""
        return self.name

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments passed to inspect_ai.eval."""
        kwargs: Dict[str, Any] = {}
        if self.cache is not None:
            kwargs["cache"] = self.cache
        if self.task_args is not None:
            kwargs["task_args"] = self.task_args
        if self.limit is not None:
            kwargs["limit"] = self.limit
        if self.epochs is not None:
            kwargs["epochs"] = self.epochs
        return kwargs
