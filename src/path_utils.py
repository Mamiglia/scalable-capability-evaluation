"""Utility functions for eval log paths and task_args hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_task_args(task_args: dict[str, Any] | None) -> str:
    """Compute a deterministic hash for task_args dictionary."""
    s = json.dumps(task_args or {}, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def clean_model_name(model_name: str) -> str:
    """Convert model identifier to filesystem-safe name."""
    return model_name.replace("/", "_").replace(":", "_")


def clean_bench_name(bench_name: str) -> str:
    """Convert benchmark identifier to filesystem-safe name."""
    return bench_name.rsplit("/", 1)[-1].replace("-", "_")


def get_eval_path(
    benchmark: str,
    task_args: dict[str, Any] | None,
    model: str,
    n: int,
    base_dir: str | Path = "data",
) -> Path:
    """Return storage path: data/{benchmark}/{task_args_hashing}/{model}/{N}.eval"""
    b_clean = clean_bench_name(benchmark)
    h_str = hash_task_args(task_args)
    m_clean = clean_model_name(model)
    return Path(base_dir) / b_clean / h_str / m_clean / f"{n}.eval"
