"""Queries over the derived dataset soft-link tree."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from inspect_ai.log import read_eval_log
import pandas as pd

from registry.build import format_task_args, _benchmark_key, _model_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "data/dataset"


def query_registry(
    *,
    model: str | None = None,
    benchmark: str | None = None,
    dataset: str | None = None,
    status: str | None = "success",
    task_args: dict[str, Any] | None = None,
    dataset_dir: Path | None = None,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    """Query soft-linked logs in dataset_dir and return matching entries."""
    if dataset_dir is None:
        if registry_path is not None:
            # If registry_path was passed (e.g. tmp_path / "registry.csv"), use its parent or dataset directory
            path_obj = Path(registry_path)
            dataset_dir = path_obj.parent if path_obj.suffix == ".csv" else path_obj
        else:
            dataset_dir = DEFAULT_DATASET_DIR

    dataset_dir = Path(dataset_dir)
    rows = []

    if dataset_dir.exists():
        for path in sorted(dataset_dir.rglob("*.eval")):
            try:
                log = read_eval_log(path, header_only=True)
                info = log.eval
                rows.append(
                    {
                        "path": str(path),
                        "eval_id": info.eval_id,
                        "status": str(log.status),
                        "model": str(info.model),
                        "benchmark": str(info.task),
                        "dataset": getattr(info.dataset, "name", None),
                        "task_args": json.dumps(
                            info.task_args or {}, sort_keys=True, default=str
                        ),
                        "created": info.created or "",
                        "samples": getattr(info.dataset, "samples", None),
                        "mtime": path.stat().st_mtime,
                    }
                )
            except Exception:
                continue

    frame = pd.DataFrame(
        rows,
        columns=[
            "path",
            "eval_id",
            "status",
            "model",
            "benchmark",
            "dataset",
            "task_args",
            "created",
            "samples",
            "mtime",
        ],
    )

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


def resolve_eval_paths(
    model: str,
    benchmark: str,
    task_args: dict[str, Any] | None = None,
    dataset_dir: Path | None = None,
    registry_path: Path | None = None,
) -> list[Path]:
    """Resolve all matching eval log paths from data/dataset/{benchmark}/{task_args}/{model}/."""
    if dataset_dir is None:
        if registry_path is not None:
            path_obj = Path(registry_path)
            dataset_dir = path_obj.parent if path_obj.suffix == ".csv" else path_obj
        else:
            dataset_dir = DEFAULT_DATASET_DIR

    dataset_dir = Path(dataset_dir)
    bench_dir = _resolve_bench_dir(dataset_dir, benchmark)
    if not bench_dir or not bench_dir.is_dir():
        raise ValueError(
            f"no successful eval file found for {model}/{benchmark}"
            f" with task_args={task_args or {}}"
        )

    model_key = _model_key(model)

    if task_args:
        task_args_str = format_task_args(task_args)
        task_args_dir = bench_dir / task_args_str
        if not task_args_dir.is_dir():
            raise ValueError(
                f"no successful eval file found for {model}/{benchmark}"
                f" with task_args={task_args or {}}"
            )
        candidate_dirs = [task_args_dir]
    else:
        subdirs = [d for d in bench_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise ValueError(
                f"no successful eval file found for {model}/{benchmark}"
                f" with task_args={task_args or {}}"
            )
        matching_dirs = [d for d in subdirs if (d / model_key).is_dir()]
        if len(matching_dirs) > 1:
            raise ValueError(
                f"multiple task configurations found for {model}/{benchmark}; "
                "pass task_args (for example, {'difficulty': 'easy'})"
            )
        if not matching_dirs:
            raise ValueError(
                f"no successful eval file found for {model}/{benchmark}"
                f" with task_args={task_args or {}}"
            )
        candidate_dirs = matching_dirs

    target_dir = candidate_dirs[0] / model_key
    if not target_dir.is_dir():
        raise ValueError(
            f"no successful eval file found for {model}/{benchmark}"
            f" with task_args={task_args or {}}"
        )

    eval_files = [
        p
        for p in sorted(
            target_dir.glob("*.eval"), key=lambda p: (_reproduction_num(p.name), p.name)
        )
        if _is_successful_eval(p)
    ]
    if not eval_files:
        raise ValueError(
            f"no successful eval file found for {model}/{benchmark}"
            f" with task_args={task_args or {}}"
        )

    return eval_files


def _is_successful_eval(path: Path) -> bool:
    try:
        log = read_eval_log(path, header_only=True)
        return log.status == "success"
    except Exception:
        return False


def resolve_eval_path(
    model: str,
    benchmark: str,
    task_args: dict[str, Any] | None = None,
    dataset_dir: Path | None = None,
    registry_path: Path | None = None,
) -> Path:
    """Resolve an eval log path from data/dataset/{benchmark}/{task_args}/{model}/."""
    return resolve_eval_paths(
        model=model,
        benchmark=benchmark,
        task_args=task_args,
        dataset_dir=dataset_dir,
        registry_path=registry_path,
    )[-1]


def _resolve_bench_dir(dataset_dir: Path, benchmark: str) -> Path | None:
    bench_dir = dataset_dir / benchmark
    return bench_dir if bench_dir.is_dir() else None


def _reproduction_num(filename: str) -> int:
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


def _contains_task_args(raw: str, expected: dict[str, Any]) -> bool:
    try:
        actual = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())

