"""Model configuration dataclass."""

from dataclasses import dataclass, field
from typing import Any, Dict

from configs.base import DraccusConfig


@dataclass(kw_only=True)
class ModelConfig(DraccusConfig):
    # Optional[...] on these fields breaks draccus on py3.14; empty dict means unset.
    slug: str = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    generate_args: Dict[str, Any] = field(default_factory=dict)
    # Consumed by the Model context manager when enabled.
    runpod: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slug:
            clean_name = self.name.removeprefix("openrouter/")
            self.slug = f"openrouter/{clean_name}"

        if not self.model_args:
            self.model_args = {
                "provider": {
                    "quantizations": ["bf16", "fp16"],
                }
            }
        self.model_args |= self._kwargs or {}
