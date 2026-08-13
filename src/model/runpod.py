"""Runpod CLI and vLLM readiness helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULTS = {
    "image": "vllm/vllm-openai:latest",
    "cloud_type": "COMMUNITY",
    "gpu_count": 1,
    "container_disk_in_gb": 40,
    "port": 8000,
    "ready_timeout": 1800,
    "terminate_after": "4h",
    "keep_alive": 0,
    "model_args": {},
    "vllm_args": [],
    "create_args": [],
}


def validate(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("runpod must be a mapping")
    result = DEFAULTS | config

    for field in ("model", "gpu_id"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"runpod.{field} must be a non-empty string")
    if str(result["cloud_type"]).upper() not in {"COMMUNITY", "SECURE"}:
        raise ValueError("runpod.cloud_type must be COMMUNITY or SECURE")
    result["cloud_type"] = str(result["cloud_type"]).upper()
    if not isinstance(result["model_args"], dict):
        raise ValueError("runpod.model_args must be a mapping")
    result["model_args"] = result["model_args"].copy()

    for field in ("vllm_args", "create_args"):
        value = result[field]
        if not isinstance(value, list) or not all(
            isinstance(item, (str, int, float)) for item in value
        ):
            raise ValueError(f"runpod.{field} must be a list of arguments")
        result[field] = [str(item) for item in value]

    for field in ("gpu_count", "container_disk_in_gb", "port"):
        try:
            result[field] = int(result[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runpod.{field} must be an integer") from exc
    try:
        result["ready_timeout"] = float(result["ready_timeout"])
    except (TypeError, ValueError) as exc:
        raise ValueError("runpod.ready_timeout must be a number") from exc
    try:
        result["keep_alive"] = float(result.get("keep_alive", 60))
    except (TypeError, ValueError) as exc:
        raise ValueError("runpod.keep_alive must be a number") from exc
    if result["gpu_count"] < 1 or result["container_disk_in_gb"] < 1:
        raise ValueError("runpod GPU count and container disk size must be positive")
    if not 1 <= result["port"] <= 65535:
        raise ValueError("runpod.port must be between 1 and 65535")
    if result["ready_timeout"] <= 0:
        raise ValueError("runpod.ready_timeout must be positive")
    if result["keep_alive"] < 0:
        raise ValueError("runpod.keep_alive cannot be negative")
    return result


def is_healthy(base_url: str) -> bool:
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        return True
    request = urllib.request.Request(
        f"{base_url}/models", headers={"User-Agent": "small-judge/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return 200 <= response.status < 300
    except Exception:
        return False



def _deadline(value: Any) -> str:
    text = str(value)
    duration = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text)
    if not duration:
        return text
    seconds = (
        float(duration.group(1))
        * {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }[duration.group(2)]
    )
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("runpodctl is not installed") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"runpodctl failed: {detail}") from exc


FALLBACK_GPUS = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB PCIe",
    "NVIDIA H100 SXM",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
]


def _create_pod_once(config: dict[str, Any], gpu_id: str) -> str:
    create_args = list(config["create_args"])
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and not any(arg.startswith("--env=") for arg in create_args):
        env_dict = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}
        create_args.append(f"--env={json.dumps(env_dict)}")

    command = [
        "runpodctl",
        "pod",
        "create",
        f"--image={config['image']}",
        f"--cloud-type={config['cloud_type']}",
        f"--gpu-id={gpu_id}",
        f"--gpu-count={config['gpu_count']}",
        f"--container-disk-in-gb={config['container_disk_in_gb']}",
        f"--ports={config['port']}/http",
        f"--terminate-after={_deadline(config['terminate_after'])}",
        "--docker-args="
        + shlex.join(
            [
                "--model",
                config["model"],
                "--enable-prefix-caching",
                *config["vllm_args"],
            ]
        ),
    ]
    if config.get("name"):
        command.append(f"--name={config['name']}")
    command.extend(create_args)
    LOGGER.info("Creating Runpod for %s on %s", config["model"], gpu_id)
    output = _run(command).stdout
    try:
        pod_id = json.loads(output)["id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"runpodctl returned no Pod id: {output!r}") from exc
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError(f"runpodctl returned no Pod id: {output!r}")
    return pod_id


def create(config: dict[str, Any]) -> str:
    gpu_candidates = [config["gpu_id"]] + [g for g in FALLBACK_GPUS if g != config["gpu_id"]]
    last_exc = None
    for gpu in gpu_candidates:
        try:
            return _create_pod_once(config, gpu)
        except Exception as exc:
            err_str = str(exc).lower()
            if "graphql" in err_str or "resources" in err_str or "something went wrong" in err_str or "failed to create pod" in err_str:
                LOGGER.warning("GPU %s failed for %s (%s), trying next fallback GPU...", gpu, config["model"], exc)
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to create pod with any GPU candidate")


def wait(pod_id: str, port: int, timeout: float) -> str:
    base_url = f"https://{pod_id}-{port}.proxy.runpod.net/v1"
    request = urllib.request.Request(
        f"{base_url}/models", headers={"User-Agent": "small-judge/1.0"}
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if 200 <= response.status < 300:
                    return base_url
        except urllib.error.HTTPError as exc:
            if exc.code not in {403, 404, 502, 503, 504}:
                raise RuntimeError(
                    f"Runpod readiness failed with HTTP {exc.code}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Runpod {pod_id} was not ready within {timeout:g}s")


def terminate(pod_id: str) -> None:
    LOGGER.info("Terminating Runpod %s", pod_id)
    _run(["runpodctl", "pod", "delete", pod_id])
