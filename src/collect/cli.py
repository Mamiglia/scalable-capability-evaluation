"""CLI interface for collecting benchmark transcripts using draccus."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import draccus
from inspect_ai.model import CachePolicy
from draccus_kwargs import wrap
from inspect_ai import eval_set


@dataclass
class BenchConfig:
    name: str = "gpqa"
    task: Optional[str] = None
    log_dir: Optional[str] = None
    task_args: Optional[Dict[str, Any]] = None
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
    max_tasks: int = 30
    kwargs: Dict[str, Any] = field(default_factory=dict)

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

        if self.model_args is None:
            self.model_args = {
                'provider': {
                    'quantizations': 'bf16',
                    'allow_fallback': True,
                }
            }
            
    @staticmethod
    def from_cli(spec: str) -> "ModelConfig":
        path = Path(spec)
        if path.exists():
            return draccus.parse(ModelConfig, config_path=path, args=[])
        return ModelConfig(model_name=spec)


@wrap(repeatable={"model"}, config_path_aliases={"bench"})
def main(cfg: BenchConfig):
    kwargs = dict(cfg.kwargs)
    model_specs = kwargs.pop("model", [])
    if not model_specs:
        raise ValueError("At least one --model must be specified")
    models = [ModelConfig.from_cli(spec) for spec in model_specs]
    model = [item.slug for item in models]
    model_args = next((item.model_args for item in models if item.model_args), {})
    if cfg.cache is not None:
        kwargs["cache"] = cfg.cache
    if cfg.max_tasks is not None:
        kwargs["max_tasks"] = cfg.max_tasks
    if cfg.task_args is not None:
        kwargs["task_args"] = cfg.task_args

    success, _ = eval_set(
        tasks=[cfg.task],
        model=model,
        log_dir=cfg.log_dir,
        model_args=model_args,
        **kwargs,
    )
    return int(not success)


if __name__ == "__main__":
    main()
