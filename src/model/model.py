"""Model arguments for Inspect evaluations, with an optional Runpod pod behind them."""

from __future__ import annotations

import json
import signal
import threading
from types import FrameType
from typing import Any

from configs import ModelConfig
from configs.logs import logger
from model.session import RunpodSession, get_session


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
        self._session: RunpodSession | None = None
        self._previous_sigterm: Any = None

    def _enabled(self) -> bool:
        """Whether this model is served from a Runpod pod."""
        if self.use_runpod:
            self.config.runpod.require_pod_fields()
            return True
        return self.config.runpod.use

    def __enter__(self) -> Model:
        if not self._enabled():
            return self

        runpod = self.config.runpod
        runpod.require_pod_fields()
        key = json.dumps(runpod.container_key(), sort_keys=True, default=str)
        self._session = get_session(key, runpod)

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
            logger.exception("Failed to release Runpod session")
        finally:
            self._restore_signal()

    def _restore_signal(self) -> None:
        if self._previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
            self._previous_sigterm = None

    def eval_kwargs(self) -> dict[str, Any]:
        """Return the model arguments expected by ``inspect_ai.eval_set``."""
        if self._session:
            runpod = self.config.runpod
            slug = f"openai-api/runpod/{runpod.model}"
            # OpenRouter provider pins mean nothing to a vLLM pod; pod args come from
            # runpod.model_args instead.
            if self.config.model_args:
                logger.warning(
                    "model '%s' is served from Runpod, so model_args %s are ignored; "
                    "use runpod.model_args instead",
                    self.config.name,
                    sorted(self.config.model_args),
                )
            # stream=False: vLLM's streamed chunks break under a reasoning or tool-call parser
            # (JSONDecodeError / "generator didn't stop"). Streaming used to keep the Runpod HTTP
            # proxy from cutting long responses at ~100s, but pods are reached over a direct TCP
            # mapping now, which has no such limit.
            model_args = {"api_key": "inspectai", "stream": False} | runpod.model_args
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
            kwargs["metadata"] = {
                "runpod_gpu": self._session.gpu,
                "runpod_pod_id": self._session.pod_id,
                "runpod_machine": self._session.placement.get("machine"),
                "runpod_datacenter": self._session.placement.get("datacenter"),
            }
        return kwargs
