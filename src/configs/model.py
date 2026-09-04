"""Model configuration dataclass."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from configs.base import DraccusConfig

# Default OpenRouter routing pin: full-precision endpoints only, for reproducibility.
OPENROUTER_PROVIDER_PIN = {"provider": {"quantizations": ["bf16", "fp16"]}}

# Tried in order when the configured GPU has no capacity: cheapest first (secure-cloud
# prices, 2026-09), 48GB+ only so every fallback fits whatever a single ≤48GB pin fits.
# A substitution changes price and comparability, so it is logged loudly and recorded in
# the eval metadata; ``fallback_gpus: []`` turns it off.
FALLBACK_GPUS = [
    "NVIDIA A40",  # 48GB $0.44/h
    "NVIDIA RTX A6000",  # 48GB $0.53/h
    "NVIDIA L40",  # 48GB $0.82/h
    "NVIDIA RTX 6000 Ada Generation",  # 48GB $0.84/h
    "NVIDIA L40S",  # 48GB $0.99/h
    "NVIDIA A100 80GB PCIe",  # 80GB $1.39/h
    "NVIDIA A100-SXM4-80GB",  # 80GB $1.59/h
    "NVIDIA H100 PCIe",  # 80GB $2.89/h
    "NVIDIA H100 80GB HBM3",  # 80GB $3.29/h
]


@dataclass
class RunpodConfig:
    """A vLLM pod to serve the model from; consumed by the ``Model`` context manager."""

    use: bool = False
    model: str = ""  # HF repo vLLM loads
    gpu_id: str = ""  # Runpod GPU display name, e.g. "NVIDIA A40"
    name: str = ""  # pod name shown in the Runpod console
    image: str = "vllm/vllm-openai:v0.28.0"  # pinned: `latest` changes under a running study
    cloud_type: str = "COMMUNITY"
    gpu_count: int = 1
    container_disk_in_gb: int = 40
    port: int = 8000
    ready_timeout: float = 1800  # seconds to wait for vLLM to serve /v1/models
    # A pod whose container log has been silent this long before becoming ready is a bad host
    # (stalled weight download, hung NCCL): it is terminated and created once more elsewhere.
    # Good hosts pull 138GB in ~5 min. 0 disables.
    stall_timeout: float = 900
    model_args: Dict[str, Any] = field(default_factory=dict)  # inspect model_args for the pod
    vllm_args: list[str] = field(default_factory=list)  # appended to the vLLM command
    create_args: list[str] = field(default_factory=list)  # appended to `runpodctl pod create`
    env: Dict[str, str] = field(default_factory=dict)  # container env, merged over HF_TOKEN
    fallback_gpus: list[str] = field(default_factory=lambda: list(FALLBACK_GPUS))

    def __post_init__(self):
        self.cloud_type = self.cloud_type.upper()
        if self.cloud_type not in {"COMMUNITY", "SECURE"}:
            raise ValueError("runpod.cloud_type must be COMMUNITY or SECURE")
        self.vllm_args = [str(a) for a in self.vllm_args]
        self.create_args = [str(a) for a in self.create_args]
        if self.gpu_count < 1 or self.container_disk_in_gb < 1 or self.ready_timeout <= 0:
            raise ValueError("runpod gpu_count, container_disk_in_gb and ready_timeout must be positive")

    def require_pod_fields(self) -> None:
        """Raise unless the fields a pod needs are set (they are optional while ``use`` is false)."""
        if not self.model or not self.gpu_id:
            raise ValueError("runpod.model and runpod.gpu_id are required to start a pod")

    def container_key(self) -> dict[str, Any]:
        """What decides the container: runs that differ only elsewhere can share a pod."""
        return {k: v for k, v in asdict(self).items() if k not in {"use", "model_args"}}


@dataclass(kw_only=True)
class ModelConfig(DraccusConfig):
    """A model to evaluate: its inspect slug plus model/generate arguments.

    ``name`` is the canonical model name (its last segment names the data directory);
    ``slug`` is the inspect model spec and defaults to ``openrouter/<name>``. A config with
    ``runpod.use: true`` ignores the slug: ``Model`` points inspect at the pod instead.
    """

    # Optional[...] on these fields breaks draccus on py3.14; None means unset.
    slug: str = ''
    release_date: Optional[str] = None
    model_args: Optional[Dict[str, Any]] = None
    generate_args: Optional[Dict[str, Any]] = None
    runpod: RunpodConfig = field(default_factory=RunpodConfig)

    def __post_init__(self):
        if isinstance(self.runpod, dict):
            self.runpod = RunpodConfig(**self.runpod)
        if not self.slug:
            self.slug = f"openrouter/{self.name.removeprefix('openrouter/')}"

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
        return self.runpod.use
