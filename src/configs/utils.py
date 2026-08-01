"""Utility functions for eval log paths and task_args hashing."""
import hashlib
import json
from pathlib import Path
from typing import Any

from configs.benchmark import TaskConfig
from configs.model import ModelConfig

def hash_task_args(task_args: dict[str, Any] | None) -> str:
    """Compute a deterministic hash for task_args dictionary."""
    s = json.dumps(task_args or {}, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]

def get_eval_path(
    task: TaskConfig,
    model: ModelConfig,
    base_dir: str | Path = "data",
) -> Path:
    """Return storage path: data/{benchmark}/{task_args_hashing}/{model}/{N}.eval"""
    b_clean = task.name 
    h_str = hash_task_args(task.task_args)
    m_clean = model.shortname
    return Path(base_dir) / b_clean / h_str / m_clean
