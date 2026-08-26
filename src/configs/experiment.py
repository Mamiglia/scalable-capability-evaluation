"""Judge experiment: every pair of a model set, judged on one difficulty quartile of a domain."""

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from configs.base import DraccusConfig
from configs.domain import DomainConfig
from configs.model import ModelConfig
from configs.paths import get_eval_path, is_successful_eval
from configs.task import TaskConfig


def successful_evals(task: TaskConfig, model: ModelConfig) -> List[Path]:
    """Successful eval logs for <task, model>, ordered by repeat number."""
    eval_path = get_eval_path(task=task, model=model)
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


@dataclass(kw_only=True)
class ExperimentConfig(DraccusConfig):
    domain: DomainConfig | str
    judge: ModelConfig | str
    # Which pairs to judge: ``pairs`` explicitly (model config paths), else every unordered
    # pair of ``models``. The default pair set is configs/experiments/pairs.yml.
    pairs: List[List[Any]] = field(default_factory=list)
    models: List[Any] = field(default_factory=list)
    q_src: int  # quartile whose items the judge sees (1 = easiest)
    q_dst: int = 4  # quartile whose accuracy is the ground truth for major/minor

    include_reverse: bool = True
    include_reasoning: bool = True
    max_length: int = 262144
    system_prompt: str = "assets/system_prompt/default.txt"
    truncate_frac: float = 0.0  # 0 = off
    temperature: float = 0.0
    cache: bool = True  # inspect generate cache, never expires: repeated pairs are free

    log_dir: Path = Path("logs/rank/")
    limit: int = 0

    def __post_init__(self):
        if isinstance(self.domain, (str, Path)):
            self.domain = DomainConfig.from_cli(self.domain)
        if isinstance(self.judge, (str, Path)):
            self.judge = ModelConfig.from_cli(self.judge)
        as_model = lambda m: m if isinstance(m, ModelConfig) else ModelConfig.from_cli(m)
        if self.pairs:
            if self.models:
                raise ValueError("give either pairs or models, not both")
            self.pairs = [[as_model(a), as_model(b)] for a, b in self.pairs]
            self.models = list({m.shortname.lower(): m for pair in self.pairs for m in pair}.values())
        else:
            self.models = [as_model(m) for m in self.models]
            if len(self.models) < 2:
                raise ValueError("an experiment needs at least two models")
            self.pairs = [list(pair) for pair in combinations(self.models, 2)]
        for q in (self.q_src, self.q_dst):
            if q not in (1, 2, 3, 4):
                raise ValueError(f"quartiles are 1..4, got {q}")
        self.log_dir = self.log_dir / self.domain.name / self.judge.shortname.lower() / self.condition_hash

    @property
    def condition_hash(self) -> str:
        condition = (
            int(self.q_src), int(self.q_dst), str(self.system_prompt), bool(self.include_reasoning),
            float(self.truncate_frac or 0.0), int(self.max_length), bool(self.include_reverse),
            float(self.temperature),
        )
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]

    def items(self, quartile: int) -> Dict[str, set]:
        """bench -> item ids in that quartile."""
        table = self.domain.load_items()
        table = table[table.quartile == quartile]
        return {bench: set(group.item) for bench, group in table.groupby("bench")}

    def accuracy(self, quartile: int) -> pd.Series:
        """Per model: mean correctness over the domain's items in that quartile (repeats averaged)."""
        resp = self.domain.load_responses().merge(self.domain.load_items()[["bench", "item", "quartile"]])
        resp = resp[resp.quartile == quartile]
        return resp.groupby(["model", "bench", "item"]).correct.mean().groupby("model").mean()

    def ordered_pairs(self) -> List[Tuple[ModelConfig, ModelConfig]]:
        """The pairs as (major, minor): major has the higher q_dst accuracy (ties by name)."""
        acc = self.accuracy(self.q_dst)
        missing = [m.shortname for m in self.models if m.shortname.lower() not in acc.index]
        if missing:
            raise ValueError(f"no responses in domain {self.domain.name} Q{self.q_dst} for {missing}")
        key = lambda m: (-acc[m.shortname.lower()], m.shortname.lower())
        return [tuple(sorted(pair, key=key)) for pair in self.pairs]

    def eval_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self._kwargs)
        if self.limit:
            kwargs["limit"] = self.limit
        return kwargs
