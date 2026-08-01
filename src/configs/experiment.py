"""Judge experiment configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from configs.task import TaskConfig
from configs.model import ModelConfig
from configs.base import DraccusConfig
from configs.utils import get_eval_path

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript A and Transcript B, both coming to the correct solution.
One of the two models is more capable than the other, and your task is to identify which is the smarter model (A or B). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response is better and why. 2-4 sentences.>
ANSWER: <A or B, indicating which response is better.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>
"""


@dataclass(kw_only=True)
class ExperimentConfig(DraccusConfig):
    judge: ModelConfig | str
    major: ModelConfig | str
    minor: ModelConfig | str
    benchmark: TaskConfig | str

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
        if isinstance(self.judge, (str, Path)):
            self.judge = ModelConfig.from_cli(self.judge)
        if isinstance(self.major, (str, Path)):
            self.major = ModelConfig.from_cli(self.major)
        if isinstance(self.minor, (str, Path)): 
            self.minor = ModelConfig.from_cli(self.minor)
        if isinstance(self.benchmark, (str, Path)):
            self.benchmark = TaskConfig.from_cli(self.benchmark)
        if not self.log_dir:
            self.log_dir = f"logs/judge/{self.name}"

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for inspect_ai.eval."""
        kwargs = dict(self.kwargs or {})
        if self.limit is not None:
            kwargs["limit"] = self.limit
        return kwargs
