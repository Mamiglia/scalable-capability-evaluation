"""Model configuration plus optional Runpod lifecycle."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from configs import ModelConfig
from model import runpod
from model.multiton import Multiton

LOGGER = logging.getLogger(__name__)

STATE_FILE = Path("/tmp/small_judge_runpod_sessions.json")
LOCK_FILE = Path("/tmp/small_judge_runpod_sessions.lock")


class _FileLock:
    def __enter__(self):
        self.fp = open(LOCK_FILE, "a+")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        finally:
            self.fp.close()


def _load_shared_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_shared_state(state: dict[str, Any]) -> None:
    try:
        tmp_file = STATE_FILE.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(state, f)
        os.replace(tmp_file, STATE_FILE)
    except Exception:
        LOGGER.exception("Failed to save runpod session state")


def _acquire_shared_pod(
    key: str, ready_timeout: float = 1800
) -> tuple[str | None, str | None, bool]:
    """
    Returns (pod_id, base_url, is_creating).
    - If ready pod found: returns (pod_id, base_url, False)
    - If another process is currently creating: returns (None, None, True)
    - If no pod exists and no process is creating: claims creation, returns (None, None, False)
    """
    with _FileLock():
        state = _load_shared_state()
        entry = state.get(key)
        now = time.monotonic()

        if entry:
            status = entry.get("status")
            if status == "ready":
                pod_id = entry.get("pod_id")
                base_url = entry.get("base_url")
                if pod_id and base_url and runpod.is_healthy(base_url):
                    entry["ref_count"] = entry.get("ref_count", 0) + 1
                    _save_shared_state(state)
                    return pod_id, base_url, False
                else:
                    del state[key]
                    _save_shared_state(state)
            elif status == "creating":
                created_at = entry.get("created_at", 0)
                pid = entry.get("pid")
                pid_dead = False
                if pid:
                    try:
                        os.kill(pid, 0)
                    except (OSError, ProcessLookupError):
                        pid_dead = True
                if now - created_at < ready_timeout + 60 and not pid_dead:
                    return None, None, True
                else:
                    del state[key]
                    _save_shared_state(state)

        state[key] = {
            "status": "creating",
            "created_at": now,
            "pid": os.getpid(),
        }
        _save_shared_state(state)
        return None, None, False


def _wait_for_shared_pod(
    key: str, ready_timeout: float = 1800
) -> tuple[str | None, str | None]:
    """Poll state file while another process is creating the pod."""
    deadline = time.monotonic() + ready_timeout + 60
    while time.monotonic() < deadline:
        time.sleep(2)
        with _FileLock():
            state = _load_shared_state()
            entry = state.get(key)
            if not entry:
                return None, None
            status = entry.get("status")
            if status == "ready":
                pod_id = entry.get("pod_id")
                base_url = entry.get("base_url")
                if pod_id and base_url and runpod.is_healthy(base_url):
                    entry["ref_count"] = entry.get("ref_count", 0) + 1
                    _save_shared_state(state)
                    return pod_id, base_url
                else:
                    del state[key]
                    _save_shared_state(state)
                    return None, None
            elif status != "creating":
                return None, None
    return None, None


def _abort_shared_pod_creation(key: str) -> None:
    with _FileLock():
        state = _load_shared_state()
        entry = state.get(key)
        if entry and entry.get("status") == "creating":
            del state[key]
            _save_shared_state(state)


def _register_shared_pod(key: str, pod_id: str, base_url: str) -> None:
    with _FileLock():
        state = _load_shared_state()
        state[key] = {
            "status": "ready",
            "pod_id": pod_id,
            "base_url": base_url,
            "ref_count": 1,
        }
        _save_shared_state(state)


def _release_shared_pod(key: str) -> int:
    with _FileLock():
        state = _load_shared_state()
        entry = state.get(key)
        if entry:
            entry["ref_count"] = max(0, entry.get("ref_count", 1) - 1)
            rem = entry["ref_count"]
            _save_shared_state(state)
            return rem
        return 0


def _unregister_shared_pod(key: str) -> None:
    with _FileLock():
        state = _load_shared_state()
        if key in state:
            del state[key]
            _save_shared_state(state)


class _RunpodSession(metaclass=Multiton):
    """One reference-counted Runpod session per container configuration."""

    def __init__(self, key: str, config: dict[str, Any]):
        self.key = key
        self.config = config
        self.condition = threading.Condition()
        self.users = 0
        self.starting = False
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self._timer: threading.Timer | None = None

    def acquire(self) -> str:
        with self.condition:
            self.users += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            while self.starting:
                self.condition.wait()

            if self.base_url and runpod.is_healthy(self.base_url):
                return self.base_url

            self.starting = True

        ready_timeout = float(self.config.get("ready_timeout", 1800))

        while True:
            shared_pod_id, shared_base_url, is_creating = _acquire_shared_pod(
                self.key, ready_timeout
            )
            if shared_pod_id and shared_base_url:
                with self.condition:
                    self.pod_id = shared_pod_id
                    self.base_url = shared_base_url
                    self.starting = False
                    self.condition.notify_all()
                    return shared_base_url

            if is_creating:
                shared_pod_id, shared_base_url = _wait_for_shared_pod(
                    self.key, ready_timeout
                )
                if shared_pod_id and shared_base_url:
                    with self.condition:
                        self.pod_id = shared_pod_id
                        self.base_url = shared_base_url
                        self.starting = False
                        self.condition.notify_all()
                        return shared_base_url
                continue

            pod_id: str | None = None
            try:
                pod_id = runpod.create(self.config)
                base_url = runpod.wait(
                    pod_id, self.config["port"], self.config["ready_timeout"]
                )
            except BaseException:
                _abort_shared_pod_creation(self.key)
                try:
                    if pod_id:
                        runpod.terminate(pod_id)
                except Exception:
                    LOGGER.exception("Failed to terminate Runpod %s", pod_id)
                finally:
                    with self.condition:
                        self.users -= 1
                        self.starting = False
                        self.condition.notify_all()
                raise

            _register_shared_pod(self.key, pod_id, base_url)

            with self.condition:
                self.pod_id = pod_id
                self.base_url = base_url
                self.starting = False
                self.condition.notify_all()
                return base_url

    def release(self) -> None:
        with self.condition:
            self.users -= 1
            if self.users > 0:
                return
            pod_id, self.pod_id = self.pod_id, None
            self.base_url = None
            self.starting = True

        remaining_ref_count = _release_shared_pod(self.key)

        try:
            if pod_id and remaining_ref_count == 0:
                grace_period = float(self.config.get("keep_alive", 0))
                if grace_period > 0:
                    def _delayed_terminate():
                        with _FileLock():
                            state = _load_shared_state()
                            entry = state.get(self.key)
                            if not entry or entry.get("ref_count", 0) == 0:
                                try:
                                    runpod.terminate(pod_id)
                                except Exception:
                                    LOGGER.exception("Failed to terminate Runpod %s", pod_id)
                                finally:
                                    _unregister_shared_pod(self.key)

                    self._timer = threading.Timer(grace_period, _delayed_terminate)
                    self._timer.daemon = True
                    self._timer.start()
                else:
                    runpod.terminate(pod_id)
                    _unregister_shared_pod(self.key)
        finally:
            with self.condition:
                self.starting = False
                self.condition.notify_all()


class Model:
    """Context-managed model arguments for Inspect evaluations."""

    def __init__(self, config: ModelConfig, use_runpod: bool = False):
        self.config = config
        self.use_runpod = use_runpod
        self._base_url: str | None = None
        self._runpod_config: dict[str, Any] | None = None
        self._session: _RunpodSession | None = None
        self._previous_sigterm: Any = None

    def _enabled(self) -> bool:
        config = self.config.runpod
        if not config:
            if self.use_runpod:
                raise ValueError(
                    f"model '{self.config.name}' needs a runpod mapping with --runpod"
                )
            return False
        if not isinstance(config, dict):
            raise ValueError(f"model '{self.config.name}': runpod must be a mapping")
        use = config.get("use", False)
        if type(use) is not bool:
            raise ValueError(
                f"model '{self.config.name}': runpod.use must be a Boolean"
            )
        return self.use_runpod or use

    def __enter__(self) -> Model:
        if not self._enabled():
            return self

        self._runpod_config = runpod.validate(self.config.runpod)
        session_config = {
            key: value
            for key, value in self._runpod_config.items()
            if key not in {"use", "model_args", "keep_alive"}
        }
        key = json.dumps(session_config, sort_keys=True, default=str)
        self._session = _RunpodSession(key, self._runpod_config)
        if threading.current_thread() is threading.main_thread():
            self._previous_sigterm = signal.getsignal(signal.SIGTERM)

            def interrupt(_signum: int, _frame: FrameType | None) -> None:
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, interrupt)
        try:
            self._base_url = self._session.acquire()
            return self
        except BaseException:
            self._restore_signal()
            raise

    def __exit__(self, exc_type, _exc_value, _traceback) -> None:
        try:
            if self._session:
                self._session.release()
        except Exception:
            if exc_type is None:
                raise
            LOGGER.exception("Failed to terminate Runpod")
        finally:
            self._restore_signal()

    def _restore_signal(self) -> None:
        if self._previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
            self._previous_sigterm = None

    def eval_kwargs(self) -> dict[str, Any]:
        """Return the model arguments expected by ``inspect_ai.eval_set``."""
        if self._runpod_config:
            slug = f"openai-api/runpod/{self._runpod_config['model']}"
            model_args = {
                "api_key": "inspectai",
                "stream": True,
            } | self._runpod_config["model_args"]
            generate_args = {"max_retries": 2} | (self.config.generate_args or {})
        else:
            slug = self.config.slug
            model_args = (self.config.model_args or {}).copy()
            generate_args = (self.config.generate_args or {}).copy()

        kwargs: dict[str, Any] = {
            "model": slug,
            "model_args": model_args,
            **generate_args,
        }
        base_url = self._base_url or model_args.pop("base_url", None)
        if base_url:
            kwargs["model_base_url"] = base_url
        return kwargs
