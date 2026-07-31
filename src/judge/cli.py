"""Run reproducible judge experiments from YAML configs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from draccus_kwargs import wrap
from inspect_ai import eval_set

from judge.task import SYSTEM_PROMPT, build_judge_tasks
from registry.build import build_registry
from collect.cli import BenchConfig, ModelConfig


@dataclass
class ExperimentConfig:
    judge: Any
    major: str
    minor: str
    benchmark: BenchConfig
    name: Optional[str] = None  # name of experiment

    mode: Literal["paired", "combinatorial", "custom"] = "paired"
    pairs: Optional[List[Any]] = None

    require_both_correct: bool = True
    include_reverse: bool = True
    max_length: int = 65536
    system_prompt: str = SYSTEM_PROMPT

    data_dir: str = "data"
    dataset_dir: str = "data/dataset"
    log_dir: Optional[str] = None
    registry_path: Optional[str] = None
    refresh_registry: bool = True
    limit: Optional[int] = None
    task_args: Optional[Dict[str, Any]] = None
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        judge_cfg = ModelConfig.from_cli(self.judge)
        if not self.name:
            self.name = f"{self.benchmark.name}/{judge_cfg.slug.split('/')[-1]}={self.major.split('/')[-1]}|{self.minor.split('/')[-1]}"
        if not self.log_dir:
            self.log_dir = f"logs/judge/{self.name}"
            
    def eval_set_kwargs(self) -> Dict[str, Any]:
        """Return a dictionary of keyword arguments for eval_set."""
        kwargs = dict(self.kwargs or {})
        if self.limit is not None:
            kwargs["limit"] = self.limit
        return kwargs


@wrap
def main(cfg: ExperimentConfig) -> int:
    dataset_dir = Path(cfg.dataset_dir)
    if cfg.refresh_registry:
        build_registry(Path(cfg.data_dir), dataset_dir)

    exp_name = cfg.name

    tasks = build_judge_tasks(
        major=cfg.major,
        minor=cfg.minor,
        benchmark_name=cfg.benchmark.name,
        task_args=cfg.task_args,
        system_prompt=cfg.system_prompt,
        max_length=cfg.max_length,
        require_both_correct=cfg.require_both_correct,
        include_reverse=cfg.include_reverse,
        experiment_name=exp_name,
        dataset_dir=dataset_dir,
        registry_path=cfg.registry_path,
        mode=cfg.mode,
        pairs=cfg.pairs,
    )

    kwargs = cfg.eval_set_kwargs()
 
    judge_cfg = ModelConfig.from_cli(cfg.judge)
    if judge_cfg.model_args and "model_args" not in kwargs:
        kwargs["model_args"] = judge_cfg.model_args

    success, _ = eval_set(
        tasks=tasks,
        model=judge_cfg.slug,
        log_dir=cfg.log_dir,
        **kwargs,
    )
    return int(not success)


if __name__ == "__main__":
    main()
