"""Deterministic eval-log paths and the task_args hash they are keyed on.

Logs live at ``data/{benchmark}/{task_args_hash}/{model}/{repeat}/*.eval``. The hash
covers the task's *resolved* arguments (explicit ones plus the task function's
defaults), so changing a bench config's args moves new logs to a new directory while
old ones stay valid under the old hash.

Because the layout depends on that hash, resolution never falls back silently: a task
that cannot be loaded, or arguments that do not fit its signature, raise. A silent
fallback would compute a different hash, write to a fresh directory and re-run
(re-pay for) collection that is already on disk.
"""

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from inspect_ai._eval.loader import load_tasks
from inspect_ai._util.registry import registry_info, registry_lookup
from inspect_ai.log import read_eval_log

from configs.model import ModelConfig
from configs.task import TaskConfig


class TaskResolutionError(RuntimeError):
    """Raised when a task cannot be resolved, so its args hash cannot be trusted."""


def resolve_task_fn(task_spec: str) -> Callable[..., Any]:
    """Return the registered task function for ``task_spec`` (e.g. ``inspect_evals/mbpp``)."""
    fn = registry_lookup("task", task_spec)
    if fn is not None:
        return fn

    # Not imported yet: loading the task registers it, then look it up by registry name.
    try:
        loaded = load_tasks([task_spec])
        info = registry_info(loaded[0])
        fn = registry_lookup(info.type, info.name)
    except Exception as exc:
        raise TaskResolutionError(
            f"cannot resolve task {task_spec!r}: {exc}. The task args hash, and therefore "
            f"the data directory, depend on it — check the 'task' field of the bench config."
        ) from exc

    if fn is None:
        raise TaskResolutionError(f"task {task_spec!r} loaded but is not in the task registry")
    return fn


def task_args_with_defaults(task: TaskConfig) -> dict[str, Any]:
    """Resolve a task's arguments: the config's ``task_args`` plus the task's own defaults."""
    task_spec = task.task or task.name
    user_args = task.task_args or {}
    fn = resolve_task_fn(task_spec)
    sig = inspect.signature(fn)

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    unknown = [key for key in user_args if key not in sig.parameters]
    if unknown and not accepts_kwargs:
        raise TaskResolutionError(
            f"task {task_spec!r} does not accept task_args {unknown}; "
            f"its parameters are {sorted(sig.parameters)}"
        )

    try:
        bound = sig.bind_partial(**user_args)
    except TypeError as exc:
        raise TaskResolutionError(f"task_args do not fit {task_spec!r}: {exc}") from exc
    bound.apply_defaults()

    resolved: dict[str, Any] = {}
    for key, value in bound.arguments.items():
        param = sig.parameters[key]
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            resolved.update(value)
        else:
            resolved[key] = value
    return resolved


def hash_task_args(task: TaskConfig) -> str:
    """Deterministic 8-char hash over a task's resolved arguments."""
    payload = json.dumps(task_args_with_defaults(task), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def get_eval_path(
    task: TaskConfig,
    model: ModelConfig,
    base_dir: str | Path = "data",
) -> Path:
    """Return the storage directory ``{base_dir}/{benchmark}/{task_args_hash}/{model}``."""
    return Path(base_dir) / task.name.lower() / hash_task_args(task) / model.shortname.lower()


def is_successful_eval(eval_path: Path) -> bool:
    """Whether an eval log completed successfully.

    False for the artifacts a crashed run leaves behind (0-byte stubs, logs without a
    header, ``status: error``) so that re-running only fills the real gaps. Also false for
    partial runs (``--limit`` / ``--sample-id``): they are not in the path hash, so a
    validation run must not count as the collected data.
    """
    try:
        log = read_eval_log(str(eval_path), header_only=True)
    except Exception:
        return False
    config = log.eval.config
    return log.status == "success" and config.limit is None and not config.sample_id
