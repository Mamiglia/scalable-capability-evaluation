from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from judge.task import (
    judge_task,
    build_judge_tasks,
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
            system_prompt="test prompt",
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
        assert task.metadata["max_length"] == 12000

        mock_build_dataset.assert_called_once_with(
            major_path=Path("data/major/1.eval"),
            minor_path=Path("data/minor/1.eval"),
            max_len=12000,
            require_both_correct=True,
            include_reverse=True,
            include_reasoning=True,
            truncate_frac=None,
            truncate_keep="first",
        )

    @patch("judge.task.read_eval_log")
    @patch("judge.task.build_dataset")
    def test_accepts_equivalent_task_labels(self, mock_build_dataset, mock_read_log):
        """Local and imported GPQA logs use different labels for the same task."""
        local, imported = MagicMock(), MagicMock()
        local.eval.task, local.eval.model = "inspect_evals/gpqa_diamond", "local"
        imported.eval.task, imported.eval.model = "GPQA Diamond", "imported"
        mock_read_log.side_effect = [local, imported]
        mock_build_dataset.return_value = MemoryDataset(
            samples=[Sample(input="test")],
        )

        task = judge_task("local.eval", "imported.eval")

        assert isinstance(task, Task)


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
    mock_exp.system_prompt = "test prompt"
    mock_exp.max_length = 65536
    mock_exp.require_both_correct = True
    mock_exp.include_reverse = True
    mock_exp.include_reasoning = True
    mock_exp.truncate_frac = 0.0
    mock_exp.name = "test_exp"
    mock_exp.task_name = "test_task"

    tasks = build_judge_tasks(experiment=mock_exp)

    assert len(tasks) == 2
    # Distinct names: same-named tasks in one eval_set write over each other's logs.
    assert tasks[0].name == "test_task+0"
    assert tasks[1].name == "test_task+1"
    assert tasks[0].metadata["truncate_frac"] is None


@patch("judge.task.read_eval_log")
@patch("judge.task.build_dataset")
def test_build_judge_tasks_single_pair_keeps_plain_name(mock_build_dataset, mock_read_log):
    log = MagicMock()
    log.eval.model = "ibm"
    log.eval.task = "gpqa"
    mock_read_log.side_effect = [log, log]
    mock_build_dataset.return_value = MemoryDataset(samples=[Sample(input="test")])

    mock_exp = MagicMock()
    mock_exp.resolve_pairs.return_value = [(Path("major/0.eval"), Path("minor/0.eval"))]
    mock_exp.system_prompt = "test prompt"
    mock_exp.max_length = 65536
    mock_exp.truncate_frac = 0.0
    mock_exp.task_name = "test_task"

    tasks = build_judge_tasks(experiment=mock_exp)
    assert [t.name for t in tasks] == ["test_task"]


@patch("judge.task.read_eval_log")
@patch("judge.task.build_dataset")
def test_truncation_switches_to_truncated_prompt(mock_build_dataset, mock_read_log):
    """Truncated runs must tell the judge the transcripts were cut, or it penalises them."""
    from judge.task import DEFAULT_SYSTEM_PROMPT, TRUNCATED_SYSTEM_PROMPT

    log = MagicMock()
    log.eval.model = "m"
    log.eval.task = "gpqa"
    mock_read_log.side_effect = [log, log]
    mock_build_dataset.return_value = MemoryDataset(samples=[Sample(input="test")])

    task = judge_task(
        major_path="a.eval", minor_path="b.eval", truncate_frac=0.5,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    assert task.metadata["system_prompt"] == TRUNCATED_SYSTEM_PROMPT


@patch("judge.task.read_eval_log")
def test_mismatched_benchmarks_raise(mock_read_log):
    log_major = MagicMock()
    log_major.eval.task = "inspect_evals/gpqa_diamond"
    log_minor = MagicMock()
    log_minor.eval.task = "inspect_evals/aime2024"
    mock_read_log.side_effect = [log_major, log_minor]

    with pytest.raises(ValueError, match="same task"):
        judge_task(major_path="a.eval", minor_path="b.eval")
