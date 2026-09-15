"""An LLM judge and its pairwise or direct judging condition."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from configs.base import DraccusConfig
from configs.model import ModelConfig


FULL_LENGTH = 262144  # the untruncated default; a smaller max_length is a truncation condition


@dataclass(kw_only=True)
class JudgeConfig(DraccusConfig):
    # Pairwise judging
    judge: ModelConfig | str
    system_prompt: str = "assets/system_prompt/default.txt"  # a DomainBench.system_prompt overrides it per bench
    include_reasoning: bool = False  # show the models' reasoning blocks to the judge
    include_reverse: bool = True  # judge every pair in both shown orders (position bias)
    max_length: int = FULL_LENGTH  # chars kept per transcript; two must fit the judge's context with its output
    truncate: str = "head"  # which max_length chars to keep when a transcript is longer: head (prefix), tail (suffix) or ends (both halves, "..." between)
    temperature: float = 0.0
    cache: bool = True  # inspect generate cache, never expires: a repeated pair is free
    log_dir: Path = Path("logs/judge")  # runs land in <log_dir>/<domain>/<semester>/<judge>/<condition hash>/ (logs/ syncs to the HF judging repo)
    limit: int = 0  # samples per task, for smoke runs (such logs never count as complete)
    # judge only a stratified subset of the train items: this many per global-b quartile (0 = all),
    # the same items for every pair and judge of a task (seeded). Part of the condition.
    train_per_quartile: int = 0
    # models (data-dir shortnames) left out of every pair for now, e.g. while their logs are being
    # repaired; not part of the condition, so their pairs are added to the same run dir later.
    exclude_models: list[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.judge, (str, Path)):
            self.judge = ModelConfig.from_cli(self.judge)
        if self.truncate not in ("head", "tail", "ends"):
            raise ValueError(f"truncate must be head, tail or ends, got {self.truncate!r}")

    @property
    def condition_hash(self) -> str:
        condition = (str(self.system_prompt), bool(self.include_reasoning), bool(self.include_reverse),
                     int(self.max_length), float(self.temperature))
        if self.train_per_quartile:  # appended only when set, so the full-set hashes stay as they were
            condition += (int(self.train_per_quartile),)
        if self.truncate != "head":  # likewise: head truncation is the historical behaviour
            condition += (self.truncate,)
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]

    @property
    def truncation_label(self) -> str:
        """Readable log-directory suffix of a truncation condition, e.g. truncated-head-4000; empty when untruncated."""
        return f"truncated-{self.truncate}-{self.max_length}" if self.max_length < FULL_LENGTH else ""


@dataclass(kw_only=True)
class DirectJudgeConfig(JudgeConfig):
    """Direct judging defaults"""

    system_prompt: str = "assets/system_prompt/direct.txt"
    log_dir: Path = Path("logs")
    include_reasoning: bool = True
    include_reverse: bool = False  # accepted for compatibility; unused by direct judging

    @property
    def condition_hash(self) -> str:
        prompt = Path(self.system_prompt).read_text(encoding="utf-8")
        condition = ("direct", prompt, bool(self.include_reasoning),
                     int(self.max_length), float(self.temperature))
        if self.truncate != "head":
            condition += (self.truncate,)
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]
