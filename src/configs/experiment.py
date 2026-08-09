"""Judge experiment configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from inspect_ai.log import read_eval_log

from configs.task import TaskConfig
from configs.model import ModelConfig
from configs.base import DraccusConfig
from configs.utils import get_eval_path, is_successful_eval


SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript 1 and Transcript 2, both coming to the correct solution.

For multiple choice questions, the options were shuffled independently for each model, so an option letter in Transcript 1 does not refer to the same option as that letter in Transcript 2. Each transcript is preceded by the options as that model saw them. Read each letter against that transcript's own option list.

One of the two models is more capable than the other, and your task is to identify which is the more capable model (1 or 2). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response was produced by the more capable model and why. 2-4 sentences.>
ANSWER: <1 or 2, indicating which response was produced by the more capable model.>
CONFIDENCE: <An integer between 1 and 5, how sure you are that the model you picked is the more capable one: 1 = the two are indistinguishable and you are guessing, 5 = certain. Rate the choice, not the explanation.>
"""

# Appended to the system prompt on truncation-control runs, where both
# transcripts are cut to the same length and stop mid-reasoning.
TRUNCATION_NOTE = """
Both transcripts have been cut to the same number of characters and their final answers removed, so each one stops mid-reasoning. Judge the reasoning that is shown; do not penalise a transcript for being cut off or for not stating an answer."""


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
    include_reasoning: bool = True
    max_length: int = 65536
    system_prompt: str = SYSTEM_PROMPT

    log_dir: Optional[str] = None
    epochs: Optional[int] = None
    limit: Optional[int] = None
    
    def model_evals(self, model: ModelConfig) -> List[Path]:
        """Return a list of all model evaluation paths for a given <model, task> tuple."""
        eval_path = get_eval_path(
            task=self.benchmark,
            model=model,
        )
        eval_paths = dict()
        for eval in eval_path.rglob("*.eval"):
            if not eval.parent.name.isdigit():
                raise ValueError(f"Evaluation file {eval} is not in a numbered epoch directory. The data folder structure should be <benchmark>/<model>/<epoch_number>/<eval_file>.eval")
            repeat_num = int(eval.parent.name)
            
            # if eval is successful store it
            if is_successful_eval(eval):
                if repeat_num in eval_paths:
                    raise ValueError(f"Multiple successful evaluation files found for repeat {repeat_num} in {eval_path}.")
                eval_paths[repeat_num] = eval
                
            
        return [eval_paths[k] for k in sorted(eval_paths.keys())]
    
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
        kwargs = self._kwargs or {}
        if self.limit is not None:
            kwargs["limit"] = self.limit
        if self.epochs is not None:
            kwargs["epochs"] = self.epochs
        return kwargs
