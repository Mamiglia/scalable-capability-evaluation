import json
from pathlib import Path
import zipfile

import pytest

from registry.build import build_registry
from registry.query import query_registry, resolve_eval_path, resolve_eval_paths


def create_mock_eval_log(
    path: Path,
    eval_id: str,
    model: str,
    task: str,
    task_args: dict | None = None,
    created: str = "2026-01-01T00:00:00Z",
    status: str = "success",
):
    header = {
        "version": 2,
        "status": status,
        "eval": {
            "eval_id": eval_id,
            "run_id": eval_id,
            "created": created,
            "task": task,
            "task_id": "1",
            "model": model,
            "task_args": task_args or {},
            "dataset": {"name": "core_test"},
            "config": {},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("header.json", json.dumps(header))


def setup_test_dataset(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "dataset"

    old_log = raw_dir / "core_bench" / "old_log.eval"
    new_log = raw_dir / "core_bench" / "new_log.eval"

    create_mock_eval_log(
        old_log,
        eval_id="old",
        model="openrouter/google/Gemma-4-31B-it",
        task="inspect_evals/core_bench",
        task_args={"difficulty": "easy"},
        created="2026-01-01T00:00:00Z",
    )
    create_mock_eval_log(
        new_log,
        eval_id="new",
        model="openrouter/google/gemma-4-31b-it",
        task="inspect_evals/core_bench",
        task_args={"difficulty": "hard"},
        created="2026-01-02T00:00:00Z",
    )

    build_registry(data_dir=raw_dir, dataset_dir=dataset_dir)
    return dataset_dir


def test_build_registry_creates_symlinks(tmp_path):
    dataset_dir = setup_test_dataset(tmp_path)

    easy_symlink = (
        dataset_dir / "core_bench" / "difficulty=easy" / "gemma-4-31b-it" / "0.eval"
    )
    hard_symlink = (
        dataset_dir / "core_bench" / "difficulty=hard" / "gemma-4-31b-it" / "0.eval"
    )

    assert easy_symlink.is_symlink()
    assert hard_symlink.is_symlink()


def test_query_registry_filters_metadata_and_task_args(tmp_path):
    dataset_dir = setup_test_dataset(tmp_path)

    matches = query_registry(
        model="gemma-4-31b-it",
        benchmark="core_bench",
        task_args={"difficulty": "easy"},
        dataset_dir=dataset_dir,
    )

    assert matches.eval_id.tolist() == ["old"]


def test_build_registry_sorts_chronologically_oldest_first(tmp_path):
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "dataset"

    log1 = raw_dir / "bench" / "log_older.eval"
    log2 = raw_dir / "bench" / "log_newer.eval"

    create_mock_eval_log(
        log1,
        eval_id="older_id",
        model="test_model",
        task="test_task",
        created="2026-01-01T00:00:00Z",
    )
    create_mock_eval_log(
        log2,
        eval_id="newer_id",
        model="test_model",
        task="test_task",
        created="2026-01-02T00:00:00Z",
    )

    build_registry(data_dir=raw_dir, dataset_dir=dataset_dir)

    link0 = dataset_dir / "bench" / "default" / "test_model" / "0.eval"
    link1 = dataset_dir / "bench" / "default" / "test_model" / "1.eval"

    assert link0.resolve() == log1.resolve()
    assert link1.resolve() == log2.resolve()


def test_build_registry_numbers_only_successful_logs(tmp_path):
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "dataset"
    started_log = raw_dir / "bench" / "started.eval"
    successful_log = raw_dir / "bench" / "successful.eval"

    create_mock_eval_log(
        started_log,
        eval_id="started",
        model="test_model",
        task="test_task",
        status="started",
    )
    create_mock_eval_log(
        successful_log,
        eval_id="successful",
        model="test_model",
        task="test_task",
    )

    build_registry(data_dir=raw_dir, dataset_dir=dataset_dir)

    model_dir = dataset_dir / "bench" / "default" / "test_model"
    assert [path.name for path in model_dir.glob("*.eval")] == ["0.eval"]
    assert (model_dir / "0.eval").resolve() == successful_log.resolve()


def test_resolve_requires_task_args_for_multiple_configurations(tmp_path):
    dataset_dir = setup_test_dataset(tmp_path)

    with pytest.raises(ValueError, match="multiple task configurations"):
        resolve_eval_path(
            "gemma-4-31b-it",
            "core_bench",
            dataset_dir=dataset_dir,
        )


def test_resolve_eval_paths_returns_all_logs(tmp_path):
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "dataset"

    log1 = raw_dir / "bench" / "log_older.eval"
    log2 = raw_dir / "bench" / "log_newer.eval"

    create_mock_eval_log(
        log1,
        eval_id="older_id",
        model="test_model",
        task="test_task",
        created="2026-01-01T00:00:00Z",
    )
    create_mock_eval_log(
        log2,
        eval_id="newer_id",
        model="test_model",
        task="test_task",
        created="2026-01-02T00:00:00Z",
    )

    build_registry(data_dir=raw_dir, dataset_dir=dataset_dir)

    paths = resolve_eval_paths(
        "test_model",
        "bench",
        dataset_dir=dataset_dir,
    )

    assert len(paths) == 2
    assert paths[0].name == "0.eval"
    assert paths[1].name == "1.eval"
