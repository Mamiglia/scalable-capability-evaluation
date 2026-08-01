"""Judge experiment configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from configs.model import ModelConfig
from judge.task import SYSTEM_PROMPT


@dataclass
class ExperimentConfig:
    judge: Any
    major_log: Optional[Union[str, List[str]]] = None
    minor_log: Optional[Union[str, List[str]]] = None
    major_log_path: Optional[Union[str, List[str]]] = None
    minor_log_path: Optional[Union[str, List[str]]] = None
    pairs: Optional[List[Any]] = None
    name: Optional[str] = None  # name of experiment

    mode: Literal["paired", "combinatorial", "custom"] = "paired"

    require_both_correct: bool = True
    include_reverse: bool = True
    max_length: int = 65536
    system_prompt: str = SYSTEM_PROMPT

    log_dir: Optional[str] = None
    limit: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def resolve_pairs(self) -> List[Tuple[Path, Path]]:
        if self.pairs:
            return [(Path(p[0]), Path(p[1])) for p in self.pairs]

        major = self.major_log or self.major_log_path
        minor = self.minor_log or self.minor_log_path

        if not major or not minor:
            raise ValueError("Either 'pairs' or both 'major_log' and 'minor_log' must be provided.")

        major_list = [Path(p) for p in (major if isinstance(major, list) else [major])]
        minor_list = [Path(p) for p in (minor if isinstance(minor, list) else [minor])]

        if self.mode == "paired":
            return list(zip(major_list, minor_list))
        elif self.mode == "combinatorial":
            return [(m, n) for m in major_list for n in minor_list]
        else:
            return list(zip(major_list, minor_list))

    def __post_init__(self):
        judge_cfg = ModelConfig.from_cli(self.judge)
        if not self.name:
            self.name = f"judge_{judge_cfg.slug.split('/')[-1]}"
        if not self.log_dir:
            self.log_dir = f"logs/judge/{self.name}"

    def eval_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for inspect_ai.eval."""
        kwargs = dict(self.kwargs or {})
        if self.limit is not None:
            kwargs["limit"] = self.limit
        return kwargs
