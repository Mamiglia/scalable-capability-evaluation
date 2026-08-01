import hashlib
import json
from pathlib import Path

from configs.model import ModelConfig
from configs.task import TaskConfig
from configs.utils import hash_task_args, get_eval_path


def test_hash_task_args_empty():
    task = TaskConfig(name="test_bench", task_args={})
    expected_hash = hashlib.sha256(json.dumps({}, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    assert hash_task_args(task) == expected_hash
    assert len(hash_task_args(task)) == 8


def test_hash_task_args_deterministic_order():
    task1 = TaskConfig(name="test_bench", task_args={"b": 2, "a": 1})
    task2 = TaskConfig(name="test_bench", task_args={"a": 1, "b": 2})
    assert hash_task_args(task1) == hash_task_args(task2)


def test_hash_task_args_none():
    task = TaskConfig(name="test_bench", task_args=None)
    expected_hash = hashlib.sha256(json.dumps({}, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    assert hash_task_args(task) == expected_hash


def test_get_eval_path():
    task = TaskConfig(name="gpqa_diamond", task_args={"shots": 5})
    model = ModelConfig(name="openrouter/openai/gpt-4o", slug="openrouter/openai/gpt-4o")
    
    path = get_eval_path(task, model, base_dir="data")
    h_str = hash_task_args(task)
    expected_path = Path("data") / "gpqa_diamond" / h_str / "gpt-4o"
    assert path == expected_path


def test_hash_task_args_inspect_defaults():
    from inspect_ai import task as inspect_task, Task
    from inspect_ai.dataset import MemoryDataset, Sample

    @inspect_task
    def mock_inspect_bench(shots: int = 5, reasoning: bool = False):
        return Task(dataset=MemoryDataset([Sample(input="test")]))

    task_empty = TaskConfig(name="mock_inspect_bench", task_args={})
    task_explicit_shots = TaskConfig(name="mock_inspect_bench", task_args={"shots": 5})
    task_explicit_all = TaskConfig(name="mock_inspect_bench", task_args={"shots": 5, "reasoning": False})
    task_override = TaskConfig(name="mock_inspect_bench", task_args={"shots": 3})

    assert hash_task_args(task_empty) == hash_task_args(task_explicit_shots)
    assert hash_task_args(task_empty) == hash_task_args(task_explicit_all)
    assert hash_task_args(task_empty) != hash_task_args(task_override)

