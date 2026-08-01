"""Benchmark configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from inspect_ai.model import CachePolicy

from configs.base import DraccusConfig


@dataclass
class TaskConfig(DraccusConfig):
    task: str
    name: Optional[str] = None
    task_args: Optional[Dict[str, Any]] = None
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
    limit: Optional[int] = None
    epochs: Optional[int] = None
    score: Optional[bool] = None
    eval_path: Optional[Path] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.task.split("/")[-1]

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
        if self.score is not None:
            kwargs["score"] = self.score
        return kwargs
