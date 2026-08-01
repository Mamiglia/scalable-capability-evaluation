"""Model configuration dataclass."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from configs.base import DraccusConfig


@dataclass
class ModelConfig(DraccusConfig):
    name: str = ""
    model_name: Optional[str] = None
    slug: Optional[str] = None
    model_args: Optional[Dict[str, Any]] = None
    
    @property
    def shortname(self) -> str:
        """Return a short name for the model."""
        return self.name.split("/")[-1]

    def __post_init__(self):
        if not self.name and self.model_name:
            self.name = self.model_name
        elif not self.model_name and self.name:
            self.model_name = self.name

        if not self.slug and self.name:
            clean_name = self.name.removeprefix("openrouter/")
            self.slug = f"openrouter/{clean_name}"

        if self.model_args is None:
            self.model_args = {
                "provider": {
                    "quantizations": ["bf16", "fp16"],
                }
            }
