"""Judge experiment configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from configs.benchmark import TaskConfig
from configs.model import ModelConfig
from judge.task import SYSTEM_PROMPT


from configs.base import DraccusConfig
from configs.utils import get_eval_path


@dataclass
class ExperimentConfig(DraccusConfig):
    judge: ModelConfig | str
    major: ModelConfig | str
    minor: ModelConfig | str
    benchmark: TaskConfig | str

    name: Optional[str] = None  # name of experiment

    mode: Literal["paired", "combinatorial", "custom"] = "paired"
    pairs: Optional[List[Tuple[int, int]]] = None

    require_both_correct: bool = True
    include_reverse: bool = True
    max_length: int = 65536
    system_prompt: str = SYSTEM_PROMPT

    log_dir: Optional[str] = None
    limit: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def model_evals(self, model: ModelConfig) -> List[Path]:
        """Return a list of all model evaluation paths for a given <model, task> tuple."""
        eval_path = get_eval_path(
            task=self.benchmark,
            model=model,
        )
        eval_paths = sorted(eval_path.glob("*.eval"))
        return eval_paths
    
    @property
    def task_name(self) -> str:
        """Return a name for the task based on the benchmark and model names."""
        return f"{self.benchmark.name}/{self.major.shortname}|{self.minor.shortname}"

    def resolve_pairs(self) -> List[Tuple[Path, Path]]:
        major_evals = self.model_evals(self.major)
        minor_evals = self.model_evals(self.minor)
        if self.mode == "paired":
            if len(major_evals) != len(minor_evals):
                raise ValueError(
                    f"Number of major evals ({len(major_evals)}) does not match number of minor evals ({len(minor_evals)})"
                )
            return list(zip(major_evals, minor_evals))
        elif self.mode == "combinatorial":
            return [(m, n) for m in major_evals for n in minor_evals]
        elif self.mode == "custom":
            if not self.pairs:
                raise ValueError("Custom mode requires 'pairs' to be specified as a list of (major_index, minor_index) tuples.")
            return [(major_evals[i], minor_evals[j]) for i, j in self.pairs]

    def __post_init__(self):
        if isinstance(self.judge, str):
            self.judge = ModelConfig.from_cli(self.judge)
        if isinstance(self.major, str):
            self.major = ModelConfig.from_cli(self.major)
        if isinstance(self.minor, str): 
            self.minor = ModelConfig.from_cli(self.minor)
        if isinstance(self.benchmark, str):
            self.benchmark = TaskConfig.from_cli(self.benchmark)
        if not self.name:
            self.name = f"{self.benchmark.name}/{self.judge.shortname}:{self.major.shortname}|{self.minor.shortname}"
        if not self.log_dir:
            self.log_dir = f"logs/judge/{self.name}"

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for inspect_ai.eval."""
        kwargs = dict(self.kwargs or {})
        if self.limit is not None:
            kwargs["limit"] = self.limit
        return kwargs
