from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from inspect_ai.model import CachePolicy

from configs.base import DraccusConfig


@dataclass(kw_only=True)
class TaskConfig(DraccusConfig):
    task: str = ''
    task_args: Dict[str, Any] = field(default_factory=dict)
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
    limit: Optional[int] = None
    epochs: int = 1
    eval_path: Optional[Path] = None
    
    def __post_init__(self):
        if not self.task:
            self.task = self.name

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments passed to inspect_ai.eval."""
        kwargs = self._kwargs or {}
        if self.cache is not None:
            kwargs["cache"] = self.cache
        if self.limit is not None:
            kwargs["limit"] = self.limit
        if self.epochs is not None:
            kwargs["epochs"] = self.epochs
        return kwargs
