"""Coordinate Runpod lifecycles around per-model collection work."""

from __future__ import annotations

import logging
import os
import signal
import sys
from collections.abc import Callable, Sequence
from typing import Any

from configs import ModelConfig
from collect.runpod import utils


LOGGER = logging.getLogger(__name__)


def is_enabled(model: ModelConfig, force: bool = False) -> bool:
    """Return whether Runpod is enabled, validating the Boolean activation field."""
    config = model.runpod
    if config is None:
        if force:
            raise ValueError(
                f"model '{model.name}' needs a runpod mapping when --runpod is used"
            )
        return False
    if not isinstance(config, dict):
        raise ValueError(f"model '{model.name}': runpod must be a mapping")
    use = config.get("use", False)
    if type(use) is not bool:
        raise ValueError(f"model '{model.name}': runpod.use must be a Boolean")
    return force or use


def _run_one(
    model: ModelConfig,
    collect_model: Callable[[ModelConfig], int],
) -> int:
    config = utils.validate(model.runpod)
    pod_id: str | None = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_api_key = os.environ.get("OPENAI_API_KEY")
    previous_slug = model.slug
    previous_model_args = model.model_args

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, interrupt)
        pod_id = utils.run(config)
        base_url = utils.wait(pod_id, config["port"], config["ready_timeout"])
        os.environ.setdefault("OPENAI_API_KEY", "runpod-local")
        model.slug = f"openai/{config['model']}"
        model.model_args = config["model_args"] | {"base_url": base_url}
        LOGGER.info(
            "Starting collection against Runpod: model=%s pod=%s", model.name, pod_id
        )
        return collect_model(model)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if previous_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        model.slug = previous_slug
        model.model_args = previous_model_args
        if pod_id:
            active_error = sys.exc_info()[0]
            try:
                utils.terminate(pod_id)
            except Exception as cleanup_error:
                if active_error is None:
                    raise
                LOGGER.exception(
                    "Failed to terminate Pod %s while another error was active: %s",
                    pod_id,
                    cleanup_error,
                )


def main(
    models: Sequence[ModelConfig],
    collect_model: Callable[[ModelConfig], int],
    *,
    force: bool = False,
) -> int | None:
    """Run the Runpod-aware loop, or return ``None`` when it is not needed."""
    enabled = [is_enabled(model, force) for model in models]
    if not any(enabled):
        return None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Starting model loop: models=%s force_runpod=%s", len(models), force)
    for model, use_runpod in zip(models, enabled, strict=True):
        if use_runpod:
            LOGGER.info("Runpod enabled for model '%s'", model.name)
            exit_code = _run_one(model, collect_model)
        else:
            LOGGER.info(
                "Runpod disabled for model '%s'; using configured provider", model.name
            )
            exit_code = collect_model(model)
        if exit_code:
            LOGGER.error(
                "Collection stopped for model '%s': exit_code=%s", model.name, exit_code
            )
            return exit_code
    LOGGER.info("Model loop completed successfully")
    return 0
