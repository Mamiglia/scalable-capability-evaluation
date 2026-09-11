"""An LLM judge and its judging condition. A JudgeBT protocol (protocols.pairwise) is built from one."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from configs.base import DraccusConfig
from configs.model import ModelConfig


@dataclass(kw_only=True)
class JudgeConfig(DraccusConfig):
    judge: ModelConfig | str
    system_prompt: str = "assets/system_prompt/default.txt"  # a DomainBench.system_prompt overrides it per bench
    include_reasoning: bool = False  # show the models' reasoning blocks to the judge
    include_reverse: bool = True  # judge every pair in both shown orders (position bias)
    max_length: int = 262144  # chars kept per transcript; two must fit the judge's context with its output
    temperature: float = 0.0
    cache: bool = True  # inspect generate cache, never expires: a repeated pair is free
    log_dir: Path = Path("judge")  # runs land in <log_dir>/<domain>/<semester>/<judge>/<condition hash>/
    limit: int = 0  # samples per task, for smoke runs (such logs never count as complete)

    def __post_init__(self):
        if isinstance(self.judge, (str, Path)):
            self.judge = ModelConfig.from_cli(self.judge)

    @property
    def condition_hash(self) -> str:
        condition = (str(self.system_prompt), bool(self.include_reasoning), bool(self.include_reverse),
                     int(self.max_length), float(self.temperature))
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]
