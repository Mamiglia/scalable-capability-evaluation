"""An LLM judge and its pairwise or direct judging condition."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from configs.base import DraccusConfig
from configs.model import ModelConfig


FULL_LENGTH = 262144  # the untruncated default; a smaller max_length is a truncation condition
# head/tail/ends keep max_length chars of a longer transcript; thirds-<kept> shows only the named equal character
# thirds of every transcript ("..." where a third is dropped), max_length ignored (src/challenges/thirds)
TRUNCATE_MODES = re.compile(r"head|tail|ends|thirds-(1|2|3|12|13|23)")
CHANNELS = ("reasoning", "text", "calls", "outputs")  # what a rendered transcript is made of: CoT blocks, assistant prose, tool calls, tool outputs
CHANNEL_GROUPS = {"tools": ("calls", "outputs")}  # a spec name standing for several channels


def parse_window(spec: str) -> tuple[str, int, int]:
    """("keep"|"drop", k, n) of a window spec "keep-k/n" / "drop-k/n": the k-th n-th of a transcript by characters."""
    mode, _, fraction = spec.partition("-")
    k, _, n = fraction.partition("/")
    if mode not in ("keep", "drop") or not (k.isdigit() and n.isdigit()) or not 1 <= int(k) <= int(n):
        raise ValueError(f"window must be keep-k/n or drop-k/n with 1 <= k <= n, got {spec!r}")
    return mode, int(k), int(n)


def parse_channels(spec: str) -> frozenset[str]:
    """The channels a spec shows: all of CHANNELS for "", c for "only-<c>", all but c for "drop-<c>" (c may be a CHANNEL_GROUPS name)."""
    if not spec:
        return frozenset(CHANNELS)
    mode, _, channel = spec.partition("-")
    picked = frozenset(CHANNEL_GROUPS.get(channel, (channel,)))
    if mode not in ("only", "drop") or not picked <= set(CHANNELS):
        raise ValueError(f"channels must be only-<c> or drop-<c> with c in {CHANNELS + tuple(CHANNEL_GROUPS)}, got {spec!r}")
    return picked if mode == "only" else frozenset(CHANNELS) - picked


@dataclass(kw_only=True)
class JudgeConfig(DraccusConfig):
    # Pairwise judging
    judge: ModelConfig | str
    system_prompt: str = "assets/system_prompt/default.txt"  # a DomainBench.system_prompt overrides it per bench
    include_reasoning: bool = False  # show the models' reasoning blocks to the judge
    include_reverse: bool = True  # judge every pair in both shown orders (position bias)
    max_length: int = FULL_LENGTH  # chars kept per transcript; two must fit the judge's context with its output
    truncate: str = "head"  # head (prefix), tail (suffix), ends (both halves, "..." between) of max_length chars, or thirds-<kept> (TRUNCATE_MODES)
    window: str = ""  # a fraction of each transcript by characters, keep-k/n (show only the k-th n-th) or drop-k/n (hide it); applied before max_length
    # a pair whose two transcripts together exceed this many chars has each cut to half of it (0 = never):
    # fits a small-context judge without cutting the pairs that already fit
    pair_max_length: int = 0
    channels: str = ""  # only-<c> shows one channel of the transcript, drop-<c> hides it: reasoning, text, calls, outputs or tools (calls and outputs)
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
        if not TRUNCATE_MODES.fullmatch(self.truncate):
            raise ValueError(f"truncate must match {TRUNCATE_MODES.pattern}, got {self.truncate!r}")
        if self.window:
            parse_window(self.window)
        parse_channels(self.channels)

    @property
    def condition_hash(self) -> str:
        condition = (str(self.system_prompt), bool(self.include_reasoning), bool(self.include_reverse),
                     int(self.max_length), float(self.temperature))
        if self.train_per_quartile:  # appended only when set, so the full-set hashes stay as they were
            condition += (int(self.train_per_quartile),)
        if self.truncate != "head":  # likewise: head truncation is the historical behaviour
            condition += (self.truncate,)
        condition += self._partial_view()
        if self.pair_max_length:
            condition += (("pair_max_length", int(self.pair_max_length)),)
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]

    def _partial_view(self) -> tuple:
        """Hash entries of a window or channel condition; empty for the full transcript, so older hashes stand."""
        return (("window", self.window) if self.window else ()) + (("channels", self.channels) if self.channels else ())

    @property
    def condition_label(self) -> str:
        """Readable log-directory suffix of a partial-transcript condition, e.g. truncated-head-4000,
        window-keep-2of10, channels-only-tools (joined with "_" when combined); empty for full transcripts."""
        parts = []
        if self.max_length < FULL_LENGTH:
            parts.append(f"truncated-{self.truncate}-{self.max_length}")
        if self.window:
            parts.append(f"window-{self.window.replace('/', 'of')}")
        if self.channels:
            parts.append(f"channels-{self.channels}")
        return "_".join(parts)


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
        condition += self._partial_view()
        return hashlib.sha256(repr(condition).encode()).hexdigest()[:8]
