"""Runpod CLI and vLLM readiness helpers."""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from configs.logs import logger

DURATION = re.compile(r"(\d+(?:\.\d+)?)([smhd])")
DEFAULTS = {
    "image": "vllm/vllm-openai:latest",
    "cloud_type": "COMMUNITY",
    "gpu_count": 1,
    "container_disk_in_gb": 40,
    "port": 8000,
    "ready_timeout": 1800,
    "terminate_after": "12h",
    "keep_alive": 0,
    "model_args": {},
    "vllm_args": [],
    "create_args": [],
    "fallback_gpus": None,  # None = the module default list; [] = never substitute
}


def validate(config: dict[str, Any] | None) -> dict[str, Any]:
    """Fill in defaults and reject configurations runpodctl would mishandle."""
    if not isinstance(config, dict):
        raise ValueError("runpod must be a mapping")
    result = DEFAULTS | config
    if result["fallback_gpus"] is None:
        result["fallback_gpus"] = list(FALLBACK_GPUS)

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

    if not isinstance(result["fallback_gpus"], list) or not all(
        isinstance(item, str) for item in result["fallback_gpus"]
    ):
        raise ValueError("runpod.fallback_gpus must be a list of GPU names")

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
        result["keep_alive"] = float(result["keep_alive"])
    except (TypeError, ValueError) as exc:
        raise ValueError("runpod.keep_alive must be a number") from exc
    if not DURATION.fullmatch(str(result["terminate_after"])):
        raise ValueError(
            f"runpod.terminate_after must be a duration like '4h', got {result['terminate_after']!r}"
        )
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
    """Turn a duration like '4h' into the absolute UTC timestamp runpodctl expects."""
    duration = DURATION.fullmatch(str(value))
    if not duration:
        raise ValueError(f"terminate_after must be a duration like '4h', got {value!r}")
    seconds = float(duration.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[duration.group(2)]
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@functools.cache
def _supports_terminate_after() -> bool:
    """Whether the installed runpodctl still accepts --terminate-after (2.12 removed it)."""
    help_text = subprocess.run(
        ["runpodctl", "pod", "create", "--help"], text=True, capture_output=True
    )
    return "--terminate-after" in (help_text.stdout or "") + (help_text.stderr or "")


def _redact(command: list[str]) -> str:
    """``--env`` carries HF_TOKEN, so it must never reach a log, a traceback or a job file."""
    return shlex.join("--env=<redacted>" if arg.startswith("--env=") else arg for arg in command)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("runpodctl is not installed") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        # `from None`, not `from exc`: CalledProcessError stringifies the whole command, so
        # the chained traceback would print the token that --env carries.
        raise RuntimeError(f"runpodctl failed: {detail}\n  command: {_redact(command)}") from None


# Tried in order when the configured GPU has no capacity. Substituting silently would make
# runs unreproducible and can change the price, so every fallback is logged loudly and
# ``fallback_gpus: []`` in the config turns it off.
FALLBACK_GPUS = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB PCIe",
    "NVIDIA H100 SXM",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
]

# runpodctl reports "no capacity" through several unrelated-looking messages.
OUT_OF_CAPACITY = ("graphql", "resources", "something went wrong", "failed to create pod")


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
        f"--ports={config['port']}/http,{config['port']}/tcp",
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
    # runpodctl 2.12 dropped --terminate-after with no replacement. It is only a backstop --
    # session.py terminates the pod when the last user exits -- so run without it rather than
    # not at all, but say so: after a crash nothing stops the pod billing.
    if _supports_terminate_after():
        command.append(f"--terminate-after={_deadline(config['terminate_after'])}")
    else:
        logger.warning(
            "runpodctl has no --terminate-after; pod %s has no expiry backstop. If this run "
            "crashes, terminate it by hand (runpodctl pod list) or it bills until you do.",
            config.get("name") or config["model"],
        )
    if config.get("name"):
        command.append(f"--name={config['name']}")
    command.extend(create_args)
    logger.info("Creating Runpod for %s on %s", config["model"], gpu_id)
    output = _run(command).stdout
    try:
        pod_id = json.loads(output)["id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"runpodctl returned no Pod id: {output!r}") from exc
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError(f"runpodctl returned no Pod id: {output!r}")
    return pod_id


def create(config: dict[str, Any]) -> tuple[str, str]:
    """Create a pod, falling back to another GPU type if the configured one has no capacity.

    Returns ``(pod_id, gpu)``; the GPU is recorded in the eval metadata since a substitution
    changes what was actually run.
    """
    requested = config["gpu_id"]
    fallbacks = config.get("fallback_gpus", FALLBACK_GPUS)
    candidates = [requested] + [gpu for gpu in fallbacks if gpu != requested]

    last_exc: Exception | None = None
    for gpu in candidates:
        try:
            pod_id = _create_pod_once(config, gpu)
        except Exception as exc:
            if not any(marker in str(exc).lower() for marker in OUT_OF_CAPACITY):
                raise
            logger.warning("GPU %s unavailable for %s (%s)", gpu, config["model"], exc)
            last_exc = exc
            continue
        if gpu != requested:
            logger.warning(
                "Runpod %s runs %s on %s, not the configured %s — results are not "
                "comparable to runs pinned to %s",
                pod_id, config["model"], gpu, requested, requested,
            )
        return pod_id, gpu

    raise RuntimeError(
        f"No capacity for {config['model']} on any of {candidates}"
    ) from last_exc


def direct_url(pod_id: str, port: int) -> str | None:
    """The pod's public ``ip:port``, or None if Runpod has not mapped one.

    The ``*.proxy.runpod.net`` route goes through Cloudflare, which cuts requests that
    run for many minutes — long reasoning generations die mid-stream. A TCP mapping is
    a direct connection with no such limit.
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return None
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            pod = json.load(response)
    except Exception:
        return None
    public_port = (pod.get("portMappings") or {}).get(str(port))
    if not public_port or not pod.get("publicIp"):
        return None
    return f"http://{pod['publicIp']}:{public_port}/v1"


def wait(pod_id: str, port: int, timeout: float) -> str:
    """Block until the pod serves an OpenAI API, and return the URL that reached it.

    Declaring a TCP port stops Runpod's HTTP proxy from serving it, so the two routes
    are tried independently rather than one gating the other.
    """
    proxy_url = f"https://{pod_id}-{port}.proxy.runpod.net/v1"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        direct = direct_url(pod_id, port)
        if direct and is_healthy(direct):
            logger.info("Runpod %s reachable directly at %s", pod_id, direct)
            return direct
        if is_healthy(proxy_url):
            return proxy_url
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Runpod {pod_id} was not ready within {timeout:g}s")


def terminate(pod_id: str) -> None:
    logger.info("Terminating Runpod %s", pod_id)
    _run(["runpodctl", "pod", "delete", pod_id])
