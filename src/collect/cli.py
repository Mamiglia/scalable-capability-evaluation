"""CLI interface for collecting benchmark transcripts using draccus."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
from inspect_ai.model import CachePolicy
from draccus_kwargs import wrap
from inspect_ai import eval_set


@dataclass
class BenchConfig:
    name: str = "gpqa"
    task: Optional[str] = None
    log_dir: Optional[str] = None
    log_dir_allow_dirty: bool = True
    task_args: Optional[Dict[str, Any]] = None
    cache: Any = field(default_factory=lambda: CachePolicy(expiry="1M"))
    max_tasks: int = 30
    limit: Optional[int] = None
    epochs: Optional[int] = None
    score: Optional[bool] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = f"data/raw/{self.name}"
        if not self.task:
            self.task = f"inspect_evals/{self.name}"
            
    def eval_set_kwargs(self) -> Dict[str, Any]:
        """Return a dictionary of keyword arguments for eval_set."""
        kwargs = dict(self.kwargs)
        kwargs.pop('model', None)  # Remove model if present
        if self.cache is not None:
            kwargs["cache"] = self.cache
        if self.max_tasks is not None:
            kwargs["max_tasks"] = self.max_tasks
        if self.task_args is not None:
            kwargs["task_args"] = self.task_args
        if self.limit is not None:
            kwargs["limit"] = self.limit
        if self.epochs is not None:
            kwargs["epochs"] = self.epochs
        if self.score is not None:
            kwargs["score"] = self.score
        if self.log_dir_allow_dirty is not None:
            kwargs["log_dir_allow_dirty"] = self.log_dir_allow_dirty
        return kwargs


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
                    'quantizations': ['bf16', 'fp16'],
                }
            }
            
    @staticmethod
    def from_cli(spec: Union[str, dict, "ModelConfig"]) -> "ModelConfig":
        if isinstance(spec, ModelConfig):
            return spec
        if isinstance(spec, dict):
            return draccus.decode(ModelConfig, spec)
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

    success, _ = eval_set(
        tasks=[cfg.task],
        model=model,
        log_dir=cfg.log_dir,
        model_args=model_args,
        **cfg.eval_set_kwargs(),
    )
    return int(not success)


if __name__ == "__main__":
    main()
