"""Queries over the derived eval-log registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "data/registry.csv"


def load_registry(path: Path = DEFAULT_REGISTRY) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"registry not found at {path}; "
            "run `uv run python src/registry/build.py`"
        )
    return pd.read_csv(path, keep_default_na=False)


def query_registry(
    *,
    model: str | None = None,
    benchmark: str | None = None,
    dataset: str | None = None,
    status: str | None = "success",
    task_args: dict[str, Any] | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
) -> pd.DataFrame:
    frame = load_registry(registry_path)
    match = frame
    if model:
        match = match[match.model.map(_model_key) == _model_key(model)]
    if benchmark:
        match = match[
            match.benchmark.map(_benchmark_key) == _benchmark_key(benchmark)
        ]
    if dataset:
        match = match[match.dataset == dataset]
    if status:
        match = match[match.status == status]
    if task_args:
        match = match[
            match.task_args.map(
                lambda raw: _contains_task_args(raw, task_args)
            )
        ]
    return match.sort_values(["created", "mtime", "path"], kind="stable")


def resolve_eval_path(
    model: str,
    benchmark: str,
    task_args: dict[str, Any] | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
) -> Path:
    match = query_registry(
        model=model,
        benchmark=benchmark,
        task_args=task_args,
        registry_path=registry_path,
    )
    if task_args is None and match.task_args.nunique() > 1:
        raise ValueError(
            f"multiple task configurations found for {model}/{benchmark}; "
            "pass task_args (for example, {'difficulty': 'easy'})"
        )
    if match.empty:
        raise ValueError(
            f"no successful eval file found for {model}/{benchmark}"
            f" with task_args={task_args or {}}"
        )

    path = Path(match.iloc[-1].path)
    return path if path.is_absolute() else REPO_ROOT / path


def _model_key(value: str) -> str:
    return str(value).rsplit("/", 1)[-1].casefold()


def _benchmark_key(value: str) -> str:
    return str(value).rsplit("/", 1)[-1].casefold().replace("-", "_")


def _contains_task_args(raw: str, expected: dict[str, Any]) -> bool:
    try:
        actual = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())
