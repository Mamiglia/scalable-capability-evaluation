"""Small, testable Runpod and vLLM lifecycle operations."""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence


LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_CLOUD_TYPE = "COMMUNITY"
DEFAULT_CONTAINER_DISK_IN_GB = 20
DEFAULT_GPU_COUNT = 1
DEFAULT_PORT = 8000
DEFAULT_READY_TIMEOUT = 1800
DEFAULT_TERMINATE_AFTER = "4h"
DEFAULT_VLLM_ARGS = ("--enable-prefix-caching",)


class RunpodCommandError(RuntimeError):
    """A runpodctl command failed with output useful to the caller."""


def _scalar_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, (str, int, float)) for item in value
    ):
        raise ValueError(f"runpod.{field} must be a list of arguments")
    return [str(item) for item in value]


def validate(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate user configuration and fill wrapper defaults."""
    if not isinstance(config, dict):
        raise ValueError("the model must contain a 'runpod' mapping")
    result = config.copy()

    use = result.get("use", False)
    if type(use) is not bool:
        raise ValueError("runpod.use must be a Boolean")
    result["use"] = use

    model = result.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("runpod.model must be a non-empty model slug")
    gpu_id = result.get("gpu_id")
    if not isinstance(gpu_id, str) or not gpu_id:
        raise ValueError(
            "runpod.gpu_id is required for a GPU Pod; "
            "use 'runpodctl gpu list' to find valid IDs"
        )

    cloud_type = str(result.get("cloud_type", DEFAULT_CLOUD_TYPE)).upper()
    if cloud_type not in {"COMMUNITY", "SECURE"}:
        raise ValueError("runpod.cloud_type must be COMMUNITY or SECURE")
    result["cloud_type"] = cloud_type
    result["image"] = str(result.get("image", DEFAULT_IMAGE))
    result["gpu_count"] = int(result.get("gpu_count", DEFAULT_GPU_COUNT))
    result["container_disk_in_gb"] = int(
        result.get("container_disk_in_gb", DEFAULT_CONTAINER_DISK_IN_GB)
    )
    result["port"] = int(result.get("port", DEFAULT_PORT))
    result["ready_timeout"] = float(result.get("ready_timeout", DEFAULT_READY_TIMEOUT))
    result["terminate_after"] = result.get("terminate_after", DEFAULT_TERMINATE_AFTER)
    result["vllm_args"] = _scalar_list(result.get("vllm_args", []), "vllm_args")
    result["create_args"] = _scalar_list(result.get("create_args", []), "create_args")
    model_args = result.get("model_args", {})
    if not isinstance(model_args, dict):
        raise ValueError("runpod.model_args must be a mapping")
    result["model_args"] = model_args.copy()

    if result["gpu_count"] < 1:
        raise ValueError("runpod.gpu_count must be at least 1")
    if result["container_disk_in_gb"] < 1:
        raise ValueError("runpod.container_disk_in_gb must be at least 1")
    if not 1 <= result["port"] <= 65535:
        raise ValueError("runpod.port must be between 1 and 65535")
    if result["ready_timeout"] <= 0:
        raise ValueError("runpod.ready_timeout must be positive")
    return result


def _safe_command(command: Sequence[str]) -> str:
    """Render a command for logs without leaking environment values."""
    rendered: list[str] = []
    redact_next = False
    for value in command:
        if redact_next:
            rendered.append("<redacted>")
            redact_next = False
        elif value == "--env":
            rendered.append(value)
            redact_next = True
        elif value.startswith("--env="):
            rendered.append("--env=<redacted>")
        elif value.startswith("--docker-args="):
            safe_value = re.sub(
                r"(--(?:api-key|hf-token)\s+)\S+",
                r"\1<redacted>",
                value,
                flags=re.IGNORECASE,
            )
            rendered.append(safe_value)
        else:
            rendered.append(value)
    return shlex.join(rendered)


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run runpodctl and raise an informative error on failure."""
    LOGGER.info("Executing Runpod command: %s", _safe_command(command))
    try:
        result = subprocess.run(
            list(command), check=True, text=True, capture_output=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "runpodctl was not found; install and authenticate it before using Runpod"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RunpodCommandError(
            f"runpodctl failed with exit code {exc.returncode}: {detail}"
        ) from exc
    LOGGER.debug("Runpod command completed: %s", _safe_command(command))
    return result


def termination_deadline(value: Any) -> str:
    """Convert convenient durations to the RFC 3339 value Runpod expects."""
    text = str(value)
    duration = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text)
    if not duration:
        return text
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[duration.group(2)]
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=float(duration.group(1)) * scale
    )
    return deadline.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pod_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        value = payload.get("id")
        if isinstance(value, str) and value:
            return value
        for child in payload.values():
            if found := _pod_id(child):
                return found
    elif isinstance(payload, list):
        for child in payload:
            if found := _pod_id(child):
                return found
    return None


def run(config: dict[str, Any]) -> str:
    """Create a Runpod Pod running the configured vLLM server."""
    config = validate(config)
    vllm_args = [*DEFAULT_VLLM_ARGS, *config["vllm_args"]]
    command = [
        "runpodctl",
        "pod",
        "create",
        f"--image={config['image']}",
        f"--cloud-type={config['cloud_type']}",
        f"--gpu-id={config['gpu_id']}",
        f"--gpu-count={config['gpu_count']}",
        f"--container-disk-in-gb={config['container_disk_in_gb']}",
        f"--ports={config['port']}/http",
        "--terminate-after=" + termination_deadline(config["terminate_after"]),
        "--docker-args=" + shlex.join(["--model", config["model"], *vllm_args]),
    ]
    if config.get("name"):
        command.append(f"--name={config['name']}")
    command.extend(config["create_args"])

    LOGGER.info(
        "Creating Runpod Pod: model=%s gpu=%s count=%s cloud=%s disk=%sGB",
        config["model"],
        config["gpu_id"],
        config["gpu_count"],
        config["cloud_type"],
        config["container_disk_in_gb"],
    )
    result = run_command(command)
    try:
        pod_id = _pod_id(json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise RunpodCommandError(
            f"runpodctl returned invalid JSON: {result.stdout!r}"
        ) from exc
    if not pod_id:
        raise RunpodCommandError(
            f"runpodctl response did not contain a Pod id: {result.stdout!r}"
        )
    LOGGER.info("Runpod Pod created: id=%s", pod_id)
    return pod_id


def wait(pod_id: str, port: int, timeout: float) -> str:
    """Wait for vLLM's model endpoint and return its OpenAI base URL."""
    base_url = f"https://{pod_id}-{port}.proxy.runpod.net/v1"
    health_url = f"{base_url}/models"
    started = time.monotonic()
    deadline = started + timeout
    next_progress_log = started + 30
    LOGGER.info(
        "Waiting for vLLM readiness: pod=%s url=%s timeout=%ss",
        pod_id,
        health_url,
        timeout,
    )
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=10) as response:
                if 200 <= response.status < 300:
                    LOGGER.info(
                        "vLLM ready: pod=%s elapsed=%.1fs",
                        pod_id,
                        time.monotonic() - started,
                    )
                    return base_url
                LOGGER.debug("vLLM readiness returned HTTP %s", response.status)
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 502, 503, 504}:
                raise RuntimeError(
                    f"vLLM readiness request failed with HTTP {exc.code}: {health_url}"
                ) from exc
            LOGGER.debug("vLLM proxy not ready yet: pod=%s status=%s", pod_id, exc.code)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            LOGGER.debug("vLLM not ready yet: pod=%s error=%s", pod_id, exc)

        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"vLLM Pod {pod_id} did not become ready within {timeout:g}s"
            )
        if now >= next_progress_log:
            LOGGER.info(
                "Still waiting for vLLM: pod=%s elapsed=%.1fs",
                pod_id,
                now - started,
            )
            next_progress_log = now + 30
        time.sleep(min(5, max(0, deadline - now)))


def terminate(pod_id: str) -> None:
    """Permanently delete a Pod so that GPU billing stops."""
    LOGGER.info("Terminating Runpod Pod: id=%s", pod_id)
    run_command(["runpodctl", "pod", "delete", pod_id])
    LOGGER.info("Runpod Pod terminated: id=%s", pod_id)
