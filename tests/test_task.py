from pathlib import Path
from unittest.mock import patch, MagicMock

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from judge.task import (
    judge_task,
    build_judge_tasks,
    SYSTEM_PROMPT,
)


class TestJudgeTask:
    """Unit tests for judge_task function in task.py."""

    @patch("judge.task.read_eval_log")
    @patch("judge.task.build_dataset")
    def test_judge_task_creation(self, mock_build_dataset, mock_read_log):
        """Test judge_task constructs Task object with correct parameters and metadata."""
        log_major = MagicMock()
        log_major.eval.model = "gemma-26b"
        log_major.eval.task = "gpqa_diamond"

        log_minor = MagicMock()
        log_minor.eval.model = "gemma-2b"
        log_minor.eval.task = "gpqa_diamond"

        mock_read_log.side_effect = [log_major, log_minor]

        dummy_dataset = MemoryDataset(samples=[Sample(input="dummy input")], name="test_dataset")
        mock_build_dataset.return_value = dummy_dataset

        task = judge_task(
            major_path="data/major/1.eval",
            minor_path="data/minor/1.eval",
            system_prompt=SYSTEM_PROMPT,
            max_length=12000,
        )

        assert isinstance(task, Task)
        assert task.name == "gpqa_diamond=1|1"
        assert task.dataset == dummy_dataset
        assert task.metadata["major_path"] == "data/major/1.eval"
        assert task.metadata["minor_path"] == "data/minor/1.eval"
        assert task.metadata["name_major"] == "gemma-26b"
        assert task.metadata["name_minor"] == "gemma-2b"
        assert task.metadata["benchmark_name"] == "gpqa_diamond"

        mock_build_dataset.assert_called_once_with(
            major_path=Path("data/major/1.eval"),
            minor_path=Path("data/minor/1.eval"),
            max_len=12000,
            require_both_correct=True,
            include_reverse=True,
            include_reasoning=True,
            truncate_frac=None,
        )


@patch("judge.task.read_eval_log")
@patch("judge.task.build_dataset")
def test_build_judge_tasks_multi_pair(mock_build_dataset, mock_read_log):
    log_major = MagicMock()
    log_major.eval.model = "ibm"
    log_major.eval.task = "gpqa"

    log_minor = MagicMock()
    log_minor.eval.model = "llama"
    log_minor.eval.task = "gpqa"

    mock_read_log.side_effect = [log_major, log_minor, log_major, log_minor]

    mock_build_dataset.return_value = MemoryDataset(samples=[Sample(input="test")])

    mock_exp = MagicMock()
    mock_exp.resolve_pairs.return_value = [
        (Path("major/0.eval"), Path("minor/0.eval")),
        (Path("major/0.eval"), Path("minor/1.eval")),
    ]
    mock_exp.system_prompt = SYSTEM_PROMPT
    mock_exp.max_length = 65536
    mock_exp.require_both_correct = True
    mock_exp.include_reverse = True
    mock_exp.name = "test_exp"
    mock_exp.task_name = "test_task"

    tasks = build_judge_tasks(experiment=mock_exp)

    assert len(tasks) == 2
    assert tasks[0].name == "test_task"
    assert tasks[1].name == "test_task"
