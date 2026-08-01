"""Utility functions for eval log paths and task_args hashing."""
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict

from inspect_ai._eval.loader import load_tasks
from inspect_ai._util.registry import registry_info, registry_lookup

from configs.model import ModelConfig
from configs.task import TaskConfig


def get_task_args_with_defaults(task: TaskConfig) -> Dict[str, Any]:
    """Load the task from inspect_ai and extract all task arguments including defaults."""
    task_spec = task.task or task.name
    user_args = task.task_args or {}

    fn = registry_lookup("task", task_spec)

    if fn is None:
        try:
            loaded = load_tasks([task_spec], user_args)
            if loaded:
                info = registry_info(loaded[0])
                fn = registry_lookup(info.type, info.name)
        except Exception:
            pass

    if fn is not None:
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_kw:
                bound = sig.bind_partial(**user_args)
            else:
                valid_kwargs = {k: v for k, v in user_args.items() if k in sig.parameters}
                bound = sig.bind_partial(**valid_kwargs)

            bound.apply_defaults()
            res = {}
            for k, v in bound.arguments.items():
                param = sig.parameters.get(k)
                if param and param.kind == inspect.Parameter.VAR_KEYWORD:
                    if isinstance(v, dict):
                        res.update(v)
                else:
                    res[k] = v

            for k, v in user_args.items():
                if k not in res:
                    res[k] = v
            return res
        except Exception:
            pass

    return user_args


def hash_task_args(task: TaskConfig) -> str:
    """Compute a deterministic hash from loaded task arguments."""
    task_args = get_task_args_with_defaults(task)
    s = json.dumps(task_args, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def get_eval_path(
    task: TaskConfig,
    model: ModelConfig,
    base_dir: str | Path = "data",
) -> Path:
    """Return storage path: data/{benchmark}/{task_args_hashing}/{model}/{N}.eval"""
    b_clean = task.name
    h_str = hash_task_args(task)
    m_clean = model.shortname
    return Path(base_dir) / b_clean / h_str / m_clean

