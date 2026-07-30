import json
from pathlib import Path

import pandas as pd
import pytest

from registry.query import query_registry, resolve_eval_path


def write_registry(path: Path):
    pd.DataFrame(
        [
            {
                "path": "data/old.eval",
                "eval_id": "old",
                "status": "success",
                "model": "openrouter/google/Gemma-4-31B-it",
                "benchmark": "inspect_evals/core_bench",
                "dataset": "core_test",
                "task_args": json.dumps({"difficulty": "easy"}),
                "created": "2026-01-01",
                "samples": 10,
                "mtime": 1,
            },
            {
                "path": "data/new.eval",
                "eval_id": "new",
                "status": "success",
                "model": "openrouter/google/gemma-4-31b-it",
                "benchmark": "inspect_evals/core_bench",
                "dataset": "core_test",
                "task_args": json.dumps({"difficulty": "hard"}),
                "created": "2026-01-02",
                "samples": 10,
                "mtime": 2,
            },
        ]
    ).to_csv(path, index=False)


def test_query_registry_filters_metadata_and_task_args(tmp_path):
    registry = tmp_path / "registry.csv"
    write_registry(registry)

    matches = query_registry(
        model="gemma-4-31b-it",
        benchmark="core-bench",
        task_args={"difficulty": "easy"},
        registry_path=registry,
    )

    assert matches.eval_id.tolist() == ["old"]


def test_resolve_eval_path_chooses_latest_matching_log(tmp_path):
    registry = tmp_path / "registry.csv"
    write_registry(registry)

    result = resolve_eval_path(
        "gemma-4-31b-it",
        "core_bench",
        task_args={"difficulty": "hard"},
        registry_path=registry,
    )

    assert result.name == "new.eval"


def test_resolve_requires_task_args_for_multiple_configurations(tmp_path):
    registry = tmp_path / "registry.csv"
    write_registry(registry)

    with pytest.raises(ValueError, match="multiple task configurations"):
        resolve_eval_path(
            "gemma-4-31b-it",
            "core_bench",
            registry_path=registry,
        )
