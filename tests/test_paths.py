import hashlib
import json
from pathlib import Path

import pytest
from inspect_ai import Task, task as inspect_task
from inspect_ai.dataset import MemoryDataset, Sample

from configs.model import ModelConfig
from configs.paths import TaskResolutionError, get_eval_path, hash_task_args
from configs.task import TaskConfig


@inspect_task
def mock_bench_no_args():
    return Task(dataset=MemoryDataset([Sample(input="test")]))


@inspect_task
def mock_bench(shots: int = 5, reasoning: bool = False):
    return Task(dataset=MemoryDataset([Sample(input="test")]))


def test_hash_task_args_empty():
    task = TaskConfig(name="mock_bench_no_args", task_args={})
    expected_hash = hashlib.sha256(json.dumps({}, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    assert hash_task_args(task) == expected_hash
    assert len(hash_task_args(task)) == 8


def test_hash_task_args_deterministic_order():
    task1 = TaskConfig(name="mock_bench", task_args={"reasoning": True, "shots": 1})
    task2 = TaskConfig(name="mock_bench", task_args={"shots": 1, "reasoning": True})
    assert hash_task_args(task1) == hash_task_args(task2)


def test_hash_task_args_none():
    task = TaskConfig(name="mock_bench_no_args", task_args=None)
    expected_hash = hashlib.sha256(json.dumps({}, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    assert hash_task_args(task) == expected_hash


def test_get_eval_path():
    task = TaskConfig(name="gpqa_diamond", task="mock_bench", task_args={"shots": 5})
    model = ModelConfig(name="openrouter/openai/gpt-4o", slug="openrouter/openai/gpt-4o")

    path = get_eval_path(task, model, base_dir="data")
    expected_path = Path("data") / "gpqa_diamond" / hash_task_args(task) / "gpt-4o"
    assert path == expected_path


def test_hash_task_args_inspect_defaults():
    """Explicit args equal to the task's defaults must not change the hash."""
    task_empty = TaskConfig(name="mock_bench", task_args={})
    task_explicit_shots = TaskConfig(name="mock_bench", task_args={"shots": 5})
    task_explicit_all = TaskConfig(name="mock_bench", task_args={"shots": 5, "reasoning": False})
    task_override = TaskConfig(name="mock_bench", task_args={"shots": 3})

    assert hash_task_args(task_empty) == hash_task_args(task_explicit_shots)
    assert hash_task_args(task_empty) == hash_task_args(task_explicit_all)
    assert hash_task_args(task_empty) != hash_task_args(task_override)


def test_hash_matches_collected_data_layout():
    """The hash of the real bench configs must keep matching the directories in data/."""
    known = {"configs/bench/gpqa.yml": "b0a0d9c9", "configs/bench/mbpp.yml": "4871cf01"}
    for config_path, expected in known.items():
        assert hash_task_args(TaskConfig.from_cli(config_path)) == expected


def test_unresolvable_task_raises():
    """A task we cannot load would hash to a bogus directory, so it must fail loudly."""
    with pytest.raises(TaskResolutionError):
        hash_task_args(TaskConfig(name="no_such_task_anywhere"))


def test_unknown_task_arg_raises():
    with pytest.raises(TaskResolutionError):
        hash_task_args(TaskConfig(name="mock_bench", task_args={"nonexistent": 1}))


def test_is_successful_eval_on_broken_files(tmp_path):
    from configs.paths import is_successful_eval

    empty = tmp_path / "empty.eval"
    empty.touch()
    truncated = tmp_path / "truncated.eval"
    truncated.write_bytes(b"not a zip archive")

    assert not is_successful_eval(empty)
    assert not is_successful_eval(truncated)
    assert not is_successful_eval(tmp_path / "missing.eval")
