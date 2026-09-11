"""A domain: the benchmarks (and item filters) whose items share one difficulty scale."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from configs.base import DraccusConfig
from configs.model import ModelConfig
from configs.task import TaskConfig

MODEL_CONFIGS = Path("configs/models")


@dataclass(kw_only=True)
class DomainBench:
    bench: str  # path to a configs/bench/*.yml
    # Keep samples whose metadata[key] is one of the listed values (e.g. HLE category).
    metadata: Dict[str, List[Any]] = field(default_factory=dict)
    hle_gold: bool = False  # keep only HLE-Verified gold items (uids from inspect_evals)
    system_prompt: str = ""  # judge prompt for this bench; empty = the experiment's default


@dataclass(kw_only=True)
class DomainConfig(DraccusConfig):
    benches: List[DomainBench]
    # Model configs whose logs feed the fit; empty = every configs/models/**/*.yml.
    models: List[str] = field(default_factory=list)
    source: str = ""  # path this config was parsed from

    @classmethod
    def from_cli(cls, spec, *args, **kwargs):
        cfg = super().from_cli(spec, *args, **kwargs)
        if isinstance(spec, (str, Path)) and Path(spec).exists():
            cfg.source = str(spec)
        return cfg

    def task(self, bench: DomainBench) -> TaskConfig:
        return TaskConfig.from_cli(bench.bench)

    def model_configs(self) -> List[ModelConfig]:
        paths = self.models or sorted(str(p) for p in MODEL_CONFIGS.rglob("*.yml"))
        return [ModelConfig.from_cli(p) for p in paths]
