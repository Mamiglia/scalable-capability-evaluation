"""Runpod CLI and vLLM readiness helpers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

from configs.logs import logger
from configs.model import RunpodConfig


def is_healthy(base_url: str, timeout: float = 3) -> bool:
    request = urllib.request.Request(
        f"{base_url}/models", headers={"User-Agent": "sce/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


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


# runpodctl reports "no capacity" through several unrelated-looking messages.
OUT_OF_CAPACITY = ("graphql", "resources", "something went wrong", "failed to create pod")


def _create_pod_once(config: RunpodConfig, gpu_id: str) -> str:
    env = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        env = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}
    env |= {k: str(v) for k, v in config.env.items()}
    create_args = list(config.create_args)
    if env:
        create_args.append(f"--env={json.dumps(env)}")

    command = [
        "runpodctl",
        "pod",
        "create",
        f"--image={config.image}",
        f"--cloud-type={config.cloud_type}",
        f"--gpu-id={gpu_id}",
        f"--gpu-count={config.gpu_count}",
        f"--container-disk-in-gb={config.container_disk_in_gb}",
        f"--ports={config.port}/http,{config.port}/tcp",
        "--docker-args="
        + shlex.join(["--model", config.model, "--enable-prefix-caching", *config.vllm_args]),
    ]
    if config.name:
        command.append(f"--name={config.name}")
    command.extend(create_args)
    logger.info("Creating Runpod for %s on %s", config.model, gpu_id)
    output = _run(command).stdout
    try:
        pod_id = json.loads(output)["id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"runpodctl returned no Pod id: {output!r}") from exc
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError(f"runpodctl returned no Pod id: {output!r}")
    return pod_id


def create(config: RunpodConfig) -> tuple[str, str]:
    """Create a pod, falling back to another GPU type if the configured one has no capacity.

    Returns ``(pod_id, gpu)``; the GPU is recorded in the eval metadata since a substitution
    changes what was actually run.
    """
    requested = config.gpu_id
    candidates = [requested] + [gpu for gpu in config.fallback_gpus if gpu != requested]

    last_exc: Exception | None = None
    for gpu in candidates:
        try:
            pod_id = _create_pod_once(config, gpu)
        except Exception as exc:
            if not any(marker in str(exc).lower() for marker in OUT_OF_CAPACITY):
                raise
            logger.warning("GPU %s unavailable for %s (%s)", gpu, config.model, exc)
            last_exc = exc
            continue
        if gpu != requested:
            logger.warning(
                "Runpod %s runs %s on %s, not the configured %s — results are not "
                "comparable to runs pinned to %s",
                pod_id, config.model, gpu, requested, requested,
            )
        return pod_id, gpu

    raise RuntimeError(f"No capacity for {config.model} on any of {candidates}") from last_exc


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
        # Cloudflare in front of the API returns 403 to urllib's default User-Agent.
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "sce/1.0"},
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


class Stalled(RuntimeError):
    """The pod's container log went silent for too long before it became ready: bad host."""


def last_log_time(pod_id: str) -> float | None:
    """Unix time of the newest container log line, ignoring vLLM's periodic "still hung" notice."""
    try:
        output = _run(["runpodctl", "pod", "logs", pod_id]).stdout
    except RuntimeError:
        return None
    newest = None
    for raw in output.splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "shm_broadcast" in entry.get("line", ""):
            continue
        # e.g. 2026-09-03T15:05:33.810646874Z: fromisoformat rejects nanoseconds
        stamp = entry.get("ts", "")[:26].rstrip("Z")
        try:
            moment = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        newest = moment if newest is None else max(newest, moment)
    return newest


def _graphql(query: str) -> dict | None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return None
    request = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        data=json.dumps({"query": query}).encode(),
        # Cloudflare in front of the API returns 403 to urllib's default User-Agent.
        headers={"content-type": "application/json", "User-Agent": "sce/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response).get("data")
    except Exception:
        return None


def placement(pod_id: str) -> dict[str, str | None]:
    """Which machine and data center Runpod put the pod on; failures cluster by these."""
    data = _graphql(
        f'{{ pod(input:{{podId:"{pod_id}"}}) {{ machineId machine {{ dataCenterId }} }} }}'
    )
    pod = (data or {}).get("pod") or {}
    return {
        "machine": pod.get("machineId"),
        "datacenter": (pod.get("machine") or {}).get("dataCenterId"),
    }


def diagnostics(pod_id: str, log_lines: int = 40) -> str:
    """What a stalled pod looked like, for the job log: placement, runtime stats, log tail."""
    data = _graphql(
        f'{{ pod(input:{{podId:"{pod_id}"}}) {{ machineId machine {{ dataCenterId gpuDisplayName }} '
        f"runtime {{ uptimeInSeconds container {{ cpuPercent memoryPercent }} "
        f"gpus {{ gpuUtilPercent memoryUtilPercent }} }} }} }}"
    )
    try:
        tail = _run(["runpodctl", "pod", "logs", pod_id]).stdout.splitlines()[-log_lines:]
    except RuntimeError as exc:
        tail = [str(exc)]
    return f"pod {pod_id}: {json.dumps((data or {}).get('pod'))}\n" + "\n".join(tail)


def wait(pod_id: str, port: int, timeout: float, stall_timeout: float = 0) -> str:
    """Block until the pod serves an OpenAI API, and return the URL that reached it.

    Declaring a TCP port stops Runpod's HTTP proxy from serving it, so the two routes
    are tried independently rather than one gating the other. With ``stall_timeout``, a
    container log silent for that long raises ``Stalled`` so the caller can try another host.
    """
    proxy_url = f"https://{pod_id}-{port}.proxy.runpod.net/v1"
    started = time.time()
    deadline = time.monotonic() + timeout
    next_stall_check = time.monotonic() + 60
    last_seen = started  # newest log line seen so far; a failed/empty log fetch must not reset it
    while time.monotonic() < deadline:
        direct = direct_url(pod_id, port)
        if direct and is_healthy(direct):
            logger.info("Runpod %s reachable directly at %s", pod_id, direct)
            return direct
        if is_healthy(proxy_url):
            return proxy_url
        if stall_timeout and time.monotonic() >= next_stall_check:
            next_stall_check = time.monotonic() + 60
            last_seen = max(last_seen, last_log_time(pod_id) or 0)
            idle = time.time() - last_seen
            if idle > stall_timeout:
                raise Stalled(
                    f"Runpod {pod_id}: no container log output for {idle:.0f}s and not ready "
                    f"(bad host: stalled download or hung NCCL)"
                )
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        f"Runpod {pod_id} was not ready within {timeout:g}s\n{diagnostics(pod_id)}"
    )


def terminate(pod_id: str) -> None:
    logger.info("Terminating Runpod %s", pod_id)
    _run(["runpodctl", "pod", "delete", pod_id])
