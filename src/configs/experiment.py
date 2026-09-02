"""Judge experiment configuration dataclass."""

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from configs.task import TaskConfig
from configs.model import ModelConfig
from configs.base import DraccusConfig
from configs.logs import logger
from configs.paths import get_eval_path, is_successful_eval


def condition_hash(
    system_prompt: str,
    include_reasoning: bool,
    truncate_frac: float,
    max_length: int,
    require_both_correct: bool,
    include_reverse: bool,
) -> str:
    """8-char hash of everything that changes what the judge sees, besides the pair."""
    condition = (
        str(system_prompt), bool(include_reasoning), float(truncate_frac or 0.0), int(max_length),
        bool(require_both_correct), bool(include_reverse),
    )
    return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]

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

# Appended instead of TRUNCATION_NOTE when the truncation keeps each
# transcript's end rather than its start.
TRUNCATION_NOTE_END = """
Both transcripts have been cut to the same number of characters, keeping only the final portion, and their final answers removed — each one starts mid-reasoning. Judge the reasoning that is shown; do not penalise a transcript for starting abruptly or for not stating an answer."""


@dataclass(kw_only=True)
class ExperimentConfig(DraccusConfig):
    judge: ModelConfig | str
    major: ModelConfig | str
    minor: ModelConfig | str
    benchmark: TaskConfig | str

    mode: Literal["paired", "combinatorial", "custom"] = "paired"
    # Optional[...] breaks draccus on py3.14; 0 / empty / None-default plain types mean unset.
    pairs: List[Tuple[int, int]] = field(default_factory=list)

    require_both_correct: bool = True
    include_reverse: bool = True
    include_reasoning: bool = True
    max_length: int = 262144
    system_prompt: str = 'assets/system_prompt/default.txt'
    # Length control: cut both transcripts to this fraction of the shorter one. 0 = off.
    truncate_frac: float = 0.0

    # Optional[...] breaks draccus on py3.14; empty / zero means unset.
    log_dir: Path = Path('logs/judge/')
    epochs: int = 0
    limit: int = 0

    def model_evals(self, model: ModelConfig) -> List[Path]:
        """Successful eval logs for <benchmark, model>, ordered by repeat number."""
        eval_path = get_eval_path(
            task=self.benchmark,
            model=model,
        )
        by_repeat: Dict[int, Path] = {}
        for eval_file in eval_path.rglob("*.eval"):
            if not eval_file.parent.name.isdigit():
                raise ValueError(
                    f"Evaluation file {eval_file} is not in a numbered repeat directory. The data "
                    f"layout is <benchmark>/<task_args_hash>/<model>/<repeat>/<eval_file>.eval"
                )
            if not is_successful_eval(eval_file):
                continue
            repeat = int(eval_file.parent.name)
            if repeat in by_repeat:
                raise ValueError(
                    f"Multiple successful evaluation files found for repeat {repeat} in {eval_path}."
                )
            by_repeat[repeat] = eval_file

        return [by_repeat[repeat] for repeat in sorted(by_repeat)]

    @property
    def task_name(self) -> str:
        """Return a name for the task based on the benchmark and model names."""
        return f"{self.benchmark.name}/{self.major.shortname}|{self.minor.shortname}"

    def resolve_pairs(self) -> List[Tuple[Path, Path]]:
        """Pair up the major and minor eval logs to compare, according to ``mode``."""
        major_evals = self.model_evals(self.major)
        minor_evals = self.model_evals(self.minor)
        if not major_evals or not minor_evals:
            missing = [
                model.shortname
                for model, evals in ((self.major, major_evals), (self.minor, minor_evals))
                if not evals
            ]
            raise ValueError(
                f"No successful eval logs for {', '.join(missing)} on "
                f"{self.benchmark.name} — collect them first."
            )

        if self.major.name == self.minor.name:
            # Self-pair control: with one epoch per log, two distinct repeats is the only
            # way to get two transcripts of the same model for a question.
            if len(major_evals) < 2:
                raise ValueError(f"{self.task_name}: self-pair needs at least two repeats")
            return list(combinations(major_evals, 2))
        if self.mode == "paired":
            n = min(len(major_evals), len(minor_evals))
            if n != len(major_evals) or n != len(minor_evals):
                logger.warning("%s: trimming paired evals from major=%d, minor=%d to %d",
                               self.task_name, len(major_evals), len(minor_evals), n)
            return list(zip(major_evals[:n], minor_evals[:n]))
        if self.mode == "combinatorial":
            return [(m, n) for m in major_evals for n in minor_evals]
        if self.mode == "custom":
            if not self.pairs:
                raise ValueError("Custom mode requires 'pairs' to be specified as a list of (major_index, minor_index) tuples.")
            return [(major_evals[i], minor_evals[j]) for i, j in self.pairs]
        raise ValueError(f"Unknown mode {self.mode!r}")

    def __post_init__(self):
        if isinstance(self.judge, (str, Path)):
            self.judge = ModelConfig.from_cli(self.judge)
        if isinstance(self.major, (str, Path)):
            self.major = ModelConfig.from_cli(self.major)
        if isinstance(self.minor, (str, Path)):
            self.minor = ModelConfig.from_cli(self.minor)
        if isinstance(self.benchmark, (str, Path)):
            self.benchmark = TaskConfig.from_cli(self.benchmark)
        # <root>/<bench>/<judge>/<condition hash>: inspect can't tell two conditions apart
        # on its own (same task names, same judge model), so the condition is in the path.
        # All pairs of a condition share the directory; task names keep them apart.
        self.log_dir = (
            self.log_dir / self.benchmark.name / self.judge.shortname.lower() / self.condition_hash
        )

    @property
    def condition_hash(self) -> str:
        return condition_hash(
            self.system_prompt, self.include_reasoning, self.truncate_frac, self.max_length,
            self.require_both_correct, self.include_reverse,
        )

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for inspect_ai.eval."""
        kwargs = dict(self._kwargs)
        if self.limit:
            kwargs["limit"] = self.limit
        if self.epochs:
            kwargs["epochs"] = self.epochs
        return kwargs
