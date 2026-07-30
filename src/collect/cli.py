"""CLI interface for collecting benchmark transcripts using draccus."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inspect_ai.model import CachePolicy
import draccus
from inspect_ai import eval_set


@dataclass
class BenchConfig:
    name: str = "gpqa"
    task: Optional[str] = None
    log_dir: Optional[str] = None
    task_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = f"data/{self.name}"
        if not self.task:
            self.task = f"inspect_evals/{self.name}"


@dataclass
class ModelConfig:
    model_name: str
    slug: Optional[str] = None
    model_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.slug:
            clean_name = self.model_name.removeprefix("openrouter/")
            self.slug = f"openrouter/{clean_name}"


@dataclass
class CollectConfig:
    bench: BenchConfig = field(default_factory=BenchConfig)
    models: List[ModelConfig] = field(default_factory=list)
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
    max_tasks: int = 30
    limit: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@draccus.wrap()
def main(cfg: CollectConfig):
    model = [model.slug for model in cfg.models]
    model_args = [model.model_args for model in cfg.models]

    kwargs = {}
    if cfg.limit is not None:
        kwargs["limit"] = cfg.limit
    if cfg.cache is not None:
        kwargs["cache"] = cfg.cache
    if cfg.max_tasks is not None:
        kwargs["max_tasks"] = cfg.max_tasks
    if cfg.bench.task_args is not None:
        kwargs["task_args"] = cfg.bench.task_args

    return eval_set(
        tasks=[cfg.bench.task],
        model=model,
        log_dir=cfg.bench.log_dir,
        model_args=model_args,
        **kwargs,
    )


if __name__ == "__main__":
    main()