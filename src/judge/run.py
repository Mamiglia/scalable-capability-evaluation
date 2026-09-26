"""Run judging tasks as one inspect eval_set in one log dir (resumable)."""

from pathlib import Path

from inspect_ai import Task, eval_set
from inspect_ai.model import CachePolicy

from collect.tasks import openrouter_compat  # noqa: F401  (side effect: retry OpenRouter's flaky 400s)
from configs.judge import JudgeConfig
from configs.logs import log_to, logger
from model import Model


def run_tasks(tasks: list[Task], config: JudgeConfig, log_dir: Path, display: str | None = None) -> bool:
    """One ``eval_set`` over all tasks of one <judge, condition> log dir.

    A single eval_set shares one connection pool, retries, resumes and displays on its own;
    the log dir may already hold finished tasks, which eval_set skips.
    """
    log_to(log_dir)
    kwargs = dict(config._kwargs) | {"temperature": config.temperature}
    if config.limit:
        kwargs["limit"] = config.limit
    if config.cache:
        kwargs["cache"] = CachePolicy(expiry=None)  # repeated pairs must never re-bill
    # Judging is API-bound and every task is small: run them all at once, with a wide
    # connection pool. Both are overridable from the config's extra kwargs.
    concurrency = {"max_tasks": len(tasks), "max_connections": 64}
    with Model(config.judge) as model:
        success, _ = eval_set(
            tasks=tasks,
            log_dir=str(log_dir),
            log_dir_allow_dirty=True,
            display=display,
            embed_viewer=False,
            **concurrency | model.eval_kwargs() | kwargs,
        )
    if not success:
        logger.error("Judge run failed in %s", log_dir)
    return success
