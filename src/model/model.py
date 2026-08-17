"""Model arguments for Inspect evaluations, with an optional Runpod pod behind them."""

from __future__ import annotations

import json
import logging
import signal
import threading
from types import FrameType
from typing import Any

from configs import ModelConfig
from model import runpod
from model.session import RunpodSession

LOGGER = logging.getLogger(__name__)

# Excluded from the session key: they do not change the container, so runs that differ
# only in these can still share a pod.
NON_CONTAINER_KEYS = {"use", "model_args", "keep_alive"}


class Model:
    """Context-managed model arguments for Inspect evaluations.

    For hosted models this only shapes ``eval_set`` keyword arguments. For models with
    ``runpod.use: true`` (or under ``--runpod``) entering the context also starts, or joins,
    a vLLM pod, and leaving it releases that pod.
    """

    def __init__(self, config: ModelConfig, use_runpod: bool = False):
        self.config = config
        self.use_runpod = use_runpod
        self._base_url: str | None = None
        self._runpod_config: dict[str, Any] | None = None
        self._session: RunpodSession | None = None
        self._previous_sigterm: Any = None

    def _enabled(self) -> bool:
        """Whether this model is served from a Runpod pod."""
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
        if not isinstance(use, bool):
            raise ValueError(f"model '{self.config.name}': runpod.use must be a Boolean")
        return self.use_runpod or use

    def __enter__(self) -> Model:
        if not self._enabled():
            return self

        self._runpod_config = runpod.validate(self.config.runpod)
        container = {
            key: value
            for key, value in self._runpod_config.items()
            if key not in NON_CONTAINER_KEYS
        }
        self._session = RunpodSession(json.dumps(container, sort_keys=True, default=str), self._runpod_config)

        # A SIGTERM (e.g. from the parent runner) must unwind the context so the pod is
        # released rather than left running.
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
            LOGGER.exception("Failed to release Runpod session")
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
            # OpenRouter provider pins mean nothing to a vLLM pod; pod args come from
            # runpod.model_args instead.
            if self.config.model_args:
                LOGGER.warning(
                    "model '%s' is served from Runpod, so model_args %s are ignored; "
                    "use runpod.model_args instead",
                    self.config.name,
                    sorted(self.config.model_args),
                )
            model_args = {"api_key": "inspectai", "stream": True} | self._runpod_config["model_args"]
            generate_args = {"max_retries": 2} | (self.config.generate_args or {})
        else:
            slug = self.config.slug
            model_args = dict(self.config.model_args or {})
            generate_args = dict(self.config.generate_args or {})

        base_url = self._base_url or model_args.pop("base_url", None)
        kwargs: dict[str, Any] = {
            "model": slug,
            "model_args": model_args,
            **generate_args,
        }
        if base_url:
            kwargs["model_base_url"] = base_url
        if self._session and self._session.gpu:
            # Which GPU actually served the run (may be a capacity fallback).
            kwargs["metadata"] = {"runpod_gpu": self._session.gpu, "runpod_pod_id": self._session.pod_id}
        return kwargs
