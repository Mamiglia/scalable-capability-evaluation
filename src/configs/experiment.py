"""Judge experiment: every pair of a model set, judged on one difficulty quartile of a domain —
or, with ``semester``, the 8 references + 8 targets of a frozen semester task on its train items."""

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
    # Semester task (capability.semesters, e.g. "2026-I"): models = its references + targets, items =
    # its train set, ground truth = its theta*. Otherwise q_src / q_dst pick the quartiles.
    semester: str = ""
    q_src: int = 0  # quartile whose items the judge sees (1 = easiest)
    q_dst: int = 4  # quartile whose accuracy is the ground truth for major/minor

    include_reverse: bool = True
    include_reasoning: bool = False  # hidden by default
    # Chars kept per transcript. Two transcripts must fit the judge's context with its output
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
        if self.semester:
            if self.pairs or self.models:
                raise ValueError("a semester task picks its own models: give neither pairs nor models")
            configs = {m.shortname.lower(): m for m in self.domain.model_configs()}
            self.models = [configs[m] for m in self.task_models().index]
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
        for q in (self.q_dst,) if self.semester else (self.q_src, self.q_dst):
            if q not in (1, 2, 3, 4):
                raise ValueError(f"quartiles are 1..4, got {q}")
        self.log_dir = self.log_dir / self.domain.name / self.judge.shortname.lower() / self.condition_hash

    @property
    def condition_hash(self) -> str:
        condition = (
            str(self.semester), int(self.q_src), int(self.q_dst), str(self.system_prompt), bool(self.include_reasoning),
            float(self.truncate_frac or 0.0), int(self.max_length), bool(self.include_reverse),
            float(self.temperature),
        )
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]

    def task_models(self) -> pd.Series:
        """Semester task: ground-truth theta* by model (references + targets)."""
        from capability.semesters import load  # local: capability imports configs

        models = load(self.domain)[0]
        role = models.get(f"{self.semester}_role")
        if role is None:
            raise ValueError(f"no semester task {self.semester!r} in the frozen tables of {self.domain.name}")
        return models.loc[role.isin(["reference", "target"]), f"{self.semester}_theta"]

    def items(self) -> Dict[str, set]:
        """bench -> item ids the judge sees: the semester's train items, else the q_src quartile."""
        if self.semester:
            from capability.semesters import load

            table = load(self.domain)[1]
            table = table[table[f"{self.semester}_train"]]
        else:
            table = self.domain.load_items()
            table = table[table.quartile == self.q_src]
        return {bench: set(group.item) for bench, group in table.groupby("bench")}

    def responses(self) -> pd.DataFrame:
        """bench, model, item, repeat, correct: the frozen fit input in semester mode, else the domain's table."""
        if self.semester:
            from capability.semesters import frozen_dir

            return pd.read_csv(frozen_dir(self.domain) / "responses.csv", dtype={"item": str})
        return self.domain.load_responses()

    def truth(self) -> pd.Series:
        """Ground truth by model: the semester's theta*, else mean correctness on the q_dst items."""
        if self.semester:
            return self.task_models()
        resp = self.domain.load_responses().merge(self.domain.load_items()[["bench", "item", "quartile"]])
        resp = resp[resp.quartile == self.q_dst]
        return resp.groupby(["model", "bench", "item"]).correct.mean().groupby("model").mean()

    def ordered_pairs(self) -> List[Tuple[ModelConfig, ModelConfig]]:
        """The pairs as (major, minor): major has the higher ground truth (ties by name)."""
        acc = self.truth()
        missing = [m.shortname for m in self.models if m.shortname.lower() not in acc.index]
        if missing:
            raise ValueError(f"no ground truth in domain {self.domain.name} for {missing}")
        key = lambda m: (-acc[m.shortname.lower()], m.shortname.lower())
        return [tuple(sorted(pair, key=key)) for pair in self.pairs]

    def eval_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self._kwargs)
        if self.limit:
            kwargs["limit"] = self.limit
        return kwargs
