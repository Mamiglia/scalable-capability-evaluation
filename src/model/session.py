"""Reference-counted Runpod sessions: one shared vLLM pod per container configuration."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from model import runpod, sessions
from model.multiton import Multiton

LOGGER = logging.getLogger(__name__)

POLL_INTERVAL = 2.0  # seconds between checks while another process starts the pod


class RunpodSession(metaclass=Multiton):
    """One pod per container configuration, shared by every user in this process.

    ``acquire``/``release`` count users inside the process; the pod itself is also
    reference-counted across processes in ``model.sessions``. The last user to leave
    terminates it, after ``keep_alive`` seconds if that is configured.
    """

    def __init__(self, key: str, config: dict[str, Any]):
        self.key = key
        self.config = config
        self.condition = threading.Condition()
        self.users = 0
        self.starting = False
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self._timer: threading.Timer | None = None

    @property
    def ready_timeout(self) -> float:
        return float(self.config["ready_timeout"])

    def acquire(self) -> str:
        """Return the base URL of a ready pod, starting or joining one as needed."""
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

        try:
            base_url = self._obtain()
        except BaseException:
            with self.condition:
                self.users -= 1
            raise
        finally:
            with self.condition:
                self.starting = False
                self.condition.notify_all()

        with self.condition:
            self.base_url = base_url
        return base_url

    def release(self) -> None:
        """Drop this user's reference, terminating the pod once nobody wants it."""
        with self.condition:
            self.users -= 1
            if self.users > 0:
                return
            pod_id, self.pod_id = self.pod_id, None
            self.base_url = None

        if pod_id is None:
            return
        if sessions.drop(self.key) > 0:
            return

        grace_period = float(self.config.get("keep_alive", 0))
        if grace_period > 0:
            self._timer = threading.Timer(grace_period, self._terminate_if_unused, args=(pod_id,))
            self._timer.daemon = True
            self._timer.start()
        else:
            self._terminate_if_unused(pod_id)

    def _obtain(self) -> str:
        """Join the registered pod for this config, or create it."""
        deadline = time.monotonic() + self.ready_timeout + 120
        while time.monotonic() < deadline:
            with sessions.locked_state() as state:
                entry, is_creating = sessions.claim_or_join(state, self.key, self.ready_timeout)

            if entry is not None:
                # Health-checked outside the lock: it is a network call, and every other
                # process wanting a pod is blocked while we hold the lock.
                if runpod.is_healthy(entry["base_url"]):
                    self.pod_id = entry["pod_id"]
                    return entry["base_url"]
                LOGGER.warning("Registered Runpod %s is unhealthy, discarding", entry["pod_id"])
                sessions.drop(self.key)
                sessions.unregister(self.key, only_if_unused=True)
            elif is_creating:
                self._await_other_process()
            else:
                return self._create()

        raise TimeoutError(f"Gave up waiting for a Runpod for {self.config['model']}")

    def _await_other_process(self) -> None:
        """Wait while another process is creating the pod, then let the caller retry."""
        deadline = time.monotonic() + self.ready_timeout + 60
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            with sessions.locked_state() as state:
                entry = state.get(self.key)
            if entry is None or entry.get("status") != "creating":
                return

    def _create(self) -> str:
        """Create the pod this process claimed, and publish it for the others."""
        pod_id: str | None = None
        try:
            pod_id = runpod.create(self.config)
            base_url = runpod.wait(pod_id, self.config["port"], self.ready_timeout)
        except BaseException:
            sessions.abort_creation(self.key)
            if pod_id:
                self._terminate(pod_id)
            raise

        sessions.register(self.key, pod_id, base_url)
        self.pod_id = pod_id
        return base_url

    def _terminate_if_unused(self, pod_id: str) -> None:
        """Terminate the pod unless another user has claimed it in the meantime."""
        if sessions.unregister(self.key, only_if_unused=True):
            self._terminate(pod_id)

    @staticmethod
    def _terminate(pod_id: str) -> None:
        try:
            runpod.terminate(pod_id)
        except Exception:
            LOGGER.exception(
                "Failed to terminate Runpod %s — check the console, it is still billing", pod_id
            )
