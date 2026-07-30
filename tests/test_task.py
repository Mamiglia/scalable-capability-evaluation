import pytest
from unittest.mock import patch, MagicMock

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from task import judge_task, SYSTEM_PROMPT


class TestJudgeTask:
    """Unit tests for judge_task function in task.py."""

    @patch(
        "task.resolve_eval_path",
        side_effect=["data/major.eval", "data/minor.eval"],
    )
    @patch("task.build_dataset")
    def test_judge_task_creation(self, mock_build_dataset, mock_resolve):
        """Test judge_task constructs Task object with correct parameters and metadata."""
        log_major = MagicMock()
        log_major.eval.model = "gemma-26b"
        log_major.eval.dataset.name = "gpqa_diamond"

        log_minor = MagicMock()
        log_minor.eval.model = "gemma-2b"

        dummy_dataset = MemoryDataset(samples=[Sample(input="dummy input")], name="test_dataset")
        mock_build_dataset.return_value = dummy_dataset

        task = judge_task(
            major_model="gemma-26b",
            minor_model="gemma-2b",
            benchmark_name="gpqa_diamond",
            system_prompt=SYSTEM_PROMPT,
            max_length=12000,
        )

        assert isinstance(task, Task)
        assert task.name == "gpqa_diamond=gemma-26b|gemma-2b"
        assert task.dataset == dummy_dataset
        assert task.metadata["log_major_path"] == "data/major.eval"
        assert task.metadata["log_minor_path"] == "data/minor.eval"
        assert task.metadata["name_major"] == "gemma-26b"
        assert task.metadata["name_minor"] == "gemma-2b"
        assert task.metadata["benchmark_name"] == "gpqa_diamond"

        mock_build_dataset.assert_called_once_with(
            log_major_path="data/major.eval",
            log_minor_path="data/minor.eval",
            max_len=12000,
            require_both_correct=True,
            include_reverse=True,
        )
        assert mock_resolve.call_args_list[0].args == (
            "gemma-26b",
            "gpqa_diamond",
            None,
        )

    @patch(
        "task.resolve_eval_path",
        side_effect=["major.eval", "minor.eval"],
    )
    @patch("task.build_dataset")
    def test_judge_task_custom_system_prompt(self, mock_build_dataset, mock_resolve):
        """Test judge_task with a custom system prompt."""
        log_major = MagicMock()
        log_major.eval.model = "model-1"
        log_major.eval.dataset.name = "bench"

        log_minor = MagicMock()
        log_minor.eval.model = "model-2"

        dummy_dataset = MemoryDataset(samples=[Sample(input="dummy input")])
        mock_build_dataset.return_value = dummy_dataset

        custom_prompt = "Custom judge instructions..."
        task = judge_task(
            major_model="model-1",
            minor_model="model-2",
            benchmark_name="bench",
            system_prompt=custom_prompt,
        )

        assert task.metadata["system_prompt"] == custom_prompt
