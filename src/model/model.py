"""Model configuration plus optional Runpod lifecycle."""

from __future__ import annotations

import json
import logging
import signal
import threading
from types import FrameType
from typing import Any

from configs import ModelConfig
from model import runpod
from model.multiton import Multiton


LOGGER = logging.getLogger(__name__)


class _RunpodSession(metaclass=Multiton):
    """One reference-counted Runpod session per container configuration."""

    def __init__(self, _key: str, config: dict[str, Any]):
        self.config = config
        self.condition = threading.Condition()
        self.users = 0
        self.starting = False
        self.pod_id: str | None = None
        self.base_url: str | None = None

    def acquire(self) -> str:
        with self.condition:
            self.users += 1
            try:
                while self.starting:
                    self.condition.wait()
            except BaseException:
                self.users -= 1
                self.condition.notify_all()
                raise
            if self.base_url:
                return self.base_url
            self.starting = True

        pod_id: str | None = None
        try:
            pod_id = runpod.create(self.config)
            base_url = runpod.wait(
                pod_id, self.config["port"], self.config["ready_timeout"]
            )
        except BaseException:
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

        with self.condition:
            self.pod_id = pod_id
            self.base_url = base_url
            self.starting = False
            self.condition.notify_all()
            return base_url

    def release(self) -> None:
        with self.condition:
            self.users -= 1
            if self.users:
                return
            pod_id, self.pod_id = self.pod_id, None
            self.base_url = None
            self.starting = True
        try:
            if pod_id:
                runpod.terminate(pod_id)
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
        if config is None:
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
            if key not in {"use", "model_args"}
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
            slug = f"vllm/{self._runpod_config['model']}"
            model_args = self._runpod_config["model_args"].copy()
        else:
            slug = self.config.slug
            model_args = (self.config.model_args or {}).copy()

        kwargs: dict[str, Any] = {
            "model": slug,
            "model_args": model_args,
            **self.config.generate_args,
        }
        base_url = self._base_url or model_args.pop("base_url", None)
        if base_url:
            kwargs["model_base_url"] = base_url
        return kwargs
