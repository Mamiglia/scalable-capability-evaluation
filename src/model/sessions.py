"""Cross-process registry of shared Runpod pods.

Several `collect`/`judge` processes may want the same pod, so the pod is registered in a
small JSON file under /tmp, guarded by a file lock, and reference-counted: whoever drops
the last reference terminates it.

The file is the only record of a running pod, so a read/write failure is raised rather
than swallowed — losing it silently orphans a pod that then bills until
``terminate_after`` expires.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOGGER = logging.getLogger(__name__)

STATE_FILE = Path("/tmp/small_judge_runpod_sessions.json")
LOCK_FILE = Path("/tmp/small_judge_runpod_sessions.lock")


@contextmanager
def locked_state() -> Iterator[dict[str, Any]]:
    """Yield the shared state under an exclusive lock; changes are written back on exit.

    Only quick, local work belongs inside this block: every other process waiting for a
    pod blocks on the same lock. In particular, do not make network calls here.
    """
    with open(LOCK_FILE, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            state = _read_state()
            yield state
            _write_state(state)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Runpod session state at {STATE_FILE} is unreadable ({exc}). It tracks live pods, "
            f"so check the Runpod console for orphans, then delete the file to start over."
        ) from exc


def _write_state(state: dict[str, Any]) -> None:
    tmp_file = STATE_FILE.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(state))
    os.replace(tmp_file, STATE_FILE)


def claim_or_join(state: dict[str, Any], key: str, ready_timeout: float) -> tuple[dict | None, bool]:
    """Decide what this process should do about the pod for ``key``.

    Returns ``(entry, is_creating)``:
    - ``(entry, False)`` — a pod is registered and this process now holds a reference to it
      (the caller must still health-check it and call ``drop`` if it is dead).
    - ``(None, True)`` — another live process is creating it; wait for it.
    - ``(None, False)`` — nothing there, so this process has claimed creation.
    """
    entry = state.get(key)
    if entry and entry.get("status") == "ready":
        entry["ref_count"] = entry.get("ref_count", 0) + 1
        return entry, False

    if entry and entry.get("status") == "creating" and _creator_alive(entry, ready_timeout):
        return None, True

    state[key] = {"status": "creating", "created_at": time.time(), "pid": os.getpid()}
    return None, False


def _creator_alive(entry: dict[str, Any], ready_timeout: float) -> bool:
    """Whether the process that claimed creation is still around and still in time."""
    pid = entry.get("pid")
    if pid:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
    return time.time() - entry.get("created_at", 0) < ready_timeout + 60


def register(key: str, pod_id: str, base_url: str) -> None:
    """Publish a ready pod, with this process as its first user."""
    with locked_state() as state:
        state[key] = {
            "status": "ready",
            "pod_id": pod_id,
            "base_url": base_url,
            "ref_count": 1,
        }


def drop(key: str) -> int:
    """Release one reference to ``key``'s pod and return the remaining count."""
    with locked_state() as state:
        entry = state.get(key)
        if not entry:
            return 0
        entry["ref_count"] = max(0, entry.get("ref_count", 1) - 1)
        return entry["ref_count"]


def unregister(key: str, only_if_unused: bool = False) -> bool:
    """Forget ``key``'s pod. With ``only_if_unused``, keep it if someone took a reference."""
    with locked_state() as state:
        entry = state.get(key)
        if entry is None:
            return False
        if only_if_unused and entry.get("ref_count", 0) > 0:
            return False
        del state[key]
        return True


def abort_creation(key: str) -> None:
    """Give up a creation claim so another process can retry."""
    with locked_state() as state:
        if state.get(key, {}).get("status") == "creating":
            del state[key]
