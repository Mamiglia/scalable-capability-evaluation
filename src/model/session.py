"""Reference-counted Runpod sessions: one shared vLLM pod per container configuration."""

from __future__ import annotations

import threading
import time

from configs.logs import logger
from configs.model import RunpodConfig
from model import runpod, sessions


POLL_INTERVAL = 2.0  # seconds between checks while another process starts the pod

_sessions: dict[str, RunpodSession] = {}
_sessions_lock = threading.Lock()


def get_session(key: str, config: RunpodConfig) -> RunpodSession:
    """The one session per container key in this process."""
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = RunpodSession(key, config)
        return _sessions[key]


class RunpodSession:
    """One pod per container configuration, shared by every user in this process.

    ``acquire``/``release`` count users inside the process; across processes the pod is
    shared through ``model.sessions``. The last live user terminates it.
    """

    def __init__(self, key: str, config: RunpodConfig):
        self.key = key
        self.config = config
        self.condition = threading.Condition()
        self.users = 0
        self.starting = False
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self.gpu: str | None = None

    def acquire(self) -> str:
        """Return the base URL of a ready pod, starting or joining one as needed."""
        with self.condition:
            self.users += 1
            while self.starting:
                self.condition.wait()
            if self.base_url:
                return self.base_url
            self.starting = True

        try:
            base_url = self._obtain()
            with self.condition:
                self.base_url = base_url
            return base_url
        except BaseException:
            with self.condition:
                self.users -= 1
            raise
        finally:
            with self.condition:
                self.starting = False
                self.condition.notify_all()

    def release(self) -> None:
        """Drop this user; the last live user across processes terminates the pod."""
        with self.condition:
            self.users -= 1
            if self.users > 0:
                return
            pod_id, self.pod_id = self.pod_id, None
            self.base_url = None

        if pod_id is None:
            return
        if sessions.drop(self.key) == 0 and sessions.unregister(self.key, only_if_unused=True):
            self._terminate(pod_id)

    def _obtain(self) -> str:
        """Join the registered pod for this config, or create it."""
        deadline = time.monotonic() + self.config.ready_timeout + 120
        while time.monotonic() < deadline:
            with sessions.locked_state() as state:
                entry, is_creating = sessions.claim_or_join(state, self.key, self.config.ready_timeout)

            if entry is not None:
                # Health-checked outside the lock: it is a network call, and every other
                # process wanting a pod is blocked while we hold the lock. Generous timeout:
                # a pod busy serving another run must not be mistaken for a dead one.
                if runpod.is_healthy(entry["base_url"], timeout=15):
                    self.pod_id = entry["pod_id"]
                    self.gpu = entry.get("gpu")
                    return entry["base_url"]
                if sessions.drop(self.key) == 0 and sessions.unregister(self.key, only_if_unused=True):
                    logger.warning("Registered Runpod %s is unhealthy, replacing it", entry["pod_id"])
                    self._terminate(entry["pod_id"])
                    continue
                raise RuntimeError(
                    f"Runpod {entry['pod_id']} at {entry['base_url']} is unhealthy but other "
                    f"processes still use it. Wait for them to exit, or terminate it by hand "
                    f"(runpodctl pod delete {entry['pod_id']}) and delete {sessions.STATE_FILE}."
                )
            elif is_creating:
                self._await_other_process()
            else:
                return self._create()

        raise TimeoutError(f"Gave up waiting for a Runpod for {self.config.model}")

    def _await_other_process(self) -> None:
        """Wait while another process is creating the pod, then let the caller retry."""
        deadline = time.monotonic() + self.config.ready_timeout + 60
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            with sessions.locked_state() as state:
                entry = state.get(self.key)
            if entry is None or entry.get("status") != "creating":
                return

    def _create(self) -> str:
        """Create the pod this process claimed, and publish it for the others.

        A pod that stalls before becoming ready is a bad host: it is terminated and one
        more pod is created, which Runpod places wherever it has capacity.
        """
        pod_id: str | None = None
        try:
            for attempt in (1, 2):
                pod_id, gpu = runpod.create(self.config)
                try:
                    base_url = runpod.wait(
                        pod_id, self.config.port, self.config.ready_timeout, self.config.stall_timeout
                    )
                    break
                except runpod.Stalled as exc:
                    if attempt == 2:
                        raise
                    logger.warning("%s; terminating it and trying another host", exc)
                    self._terminate(pod_id)
                    pod_id = None
        except BaseException:
            sessions.abort_creation(self.key)
            if pod_id:
                self._terminate(pod_id)
            raise

        sessions.register(self.key, pod_id, base_url, gpu)
        self.pod_id, self.gpu = pod_id, gpu
        return base_url

    @staticmethod
    def _terminate(pod_id: str) -> None:
        try:
            runpod.terminate(pod_id)
        except Exception:
            logger.exception(
                "Failed to terminate Runpod %s — check the console, it is still billing", pod_id
            )
