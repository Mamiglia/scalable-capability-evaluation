"""Run reproducible judge experiments from YAML configs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from draccus_kwargs import wrap
from inspect_ai import eval as inspect_eval

from judge.task import SYSTEM_PROMPT, judge_task
from registry.build import build_registry


@dataclass
class ExperimentConfig:
    judge: str
    major: str
    minor: str
    benchmark_name: str
    name: Optional[str] = None # name of experiment
    
    require_both_correct: bool = True
    include_reverse: bool = True
    max_length: int = 65536
    system_prompt: str = SYSTEM_PROMPT
    
    data_dir: str = "data"
    registry_path: str = "data/registry.csv"
    refresh_registry: bool = True
    task_args: Optional[Dict[str, Any]] = None
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.judge}={self.major}-v-{self.minor}-{self.benchmark_name}"


@wrap
def main(cfg: ExperimentConfig) -> int:
    registry_path = Path(cfg.registry_path)
    if cfg.refresh_registry:
        build_registry(Path(cfg.data_dir), registry_path)

    exp_name = cfg.name

    task = judge_task(
        major=cfg.major,
        minor=cfg.minor,
        benchmark_name=cfg.benchmark_name,
        task_args=cfg.task_args,
        system_prompt=cfg.system_prompt,
        max_length=cfg.max_length,
        require_both_correct=cfg.require_both_correct,
        include_reverse=cfg.include_reverse,
        experiment_name=exp_name,
    )
    logs = inspect_eval(
        task,
        model=cfg.judge,
        limit=cfg.kwargs.pop("limit", None),
        **(cfg.kwargs or {}),
    )
    return int(any(log.status != "success" for log in logs))


if __name__ == "__main__":
    main()
