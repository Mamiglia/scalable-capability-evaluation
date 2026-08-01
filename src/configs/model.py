"""Model configuration dataclass."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from configs.base import DraccusConfig


@dataclass(kw_only=True)
class ModelConfig(DraccusConfig):
    slug: str = None
    model_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.slug:
            clean_name = self.name.removeprefix("openrouter/")
            self.slug = f"openrouter/{clean_name}"

        if self.model_args is None:
            self.model_args = {
                "provider": {
                    "quantizations": ["bf16", "fp16"],
                }
            }
