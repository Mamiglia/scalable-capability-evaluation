"""Model configuration dataclass."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from configs.base import DraccusConfig

# Default OpenRouter routing pin: full-precision endpoints only, for reproducibility.
OPENROUTER_PROVIDER_PIN = {"provider": {"quantizations": ["bf16", "fp16"]}}


@dataclass(kw_only=True)
class ModelConfig(DraccusConfig):
    """A model to evaluate: its inspect slug plus model/generate arguments.

    ``name`` is the canonical model name (its last segment names the data directory);
    ``slug`` is the inspect model spec and defaults to ``openrouter/<name>``, or
    ``vllm/<name>`` when this config is served from a Runpod pod.
    """

    # Optional[...] on these fields breaks draccus on py3.14; None means unset.
    slug: str = ''
    release_date: Optional[str] = None
    model_args: Optional[Dict[str, Any]] = None
    generate_args: Optional[Dict[str, Any]] = None
    # Consumed by the Model context manager when enabled.
    runpod: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slug:
            clean_name = self.name.removeprefix("openrouter/").removeprefix("vllm/")
            self.slug = f"vllm/{clean_name}" if self.on_runpod else f"openrouter/{clean_name}"

        if self.model_args is None:
            # Provider pins are an OpenRouter concept; a vLLM pod has no routing to pin.
            self.model_args = {} if self.on_runpod else dict(OPENROUTER_PROVIDER_PIN)
        if self._kwargs:
            self.model_args = self.model_args | self._kwargs
        if self.generate_args is None:
            self.generate_args = {}

    @property
    def on_runpod(self) -> bool:
        """Whether this config asks to be served from a self-hosted Runpod pod."""
        return bool(self.runpod and self.runpod.get("use"))
