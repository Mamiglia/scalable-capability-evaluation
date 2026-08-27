from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from judge.task import (
    judge_task,
    build_tasks,
)


class TestJudgeTask:
    """Unit tests for judge_task function in task.py."""

    @patch("judge.task.read_header")
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

        mock_build_dataset.assert_called_once_with(
            major_path=Path("data/major/1.eval"),
            minor_path=Path("data/minor/1.eval"),
            max_len=12000,
            require_both_correct=True,
            include_reverse=True,
            include_reasoning=True,
            truncate_frac=None,
            items=None,
        )


@patch("judge.task.read_header")
@patch("judge.task.build_dataset")
def test_build_tasks_one_per_bench_and_pair(mock_build_dataset, mock_read_log, monkeypatch):
    import pandas as pd
    from configs.domain import DomainBench
    from configs.model import ModelConfig
    from configs.task import TaskConfig

    log = MagicMock()
    log.eval.model = "ibm"
    log.eval.task = "gpqa"
    mock_read_log.return_value = log
    mock_build_dataset.return_value = MemoryDataset(samples=[Sample(input="test")])
    monkeypatch.setattr("judge.task.successful_evals", lambda task, model: [Path(f"{model.name}/0/run.eval")])

    a, b = ModelConfig(name="A", slug="A"), ModelConfig(name="B", slug="B")
    exp = MagicMock()
    exp.models, exp.q_src, exp.q_dst, exp.name = [a, b], 1, 4, "exp"
    exp.system_prompt, exp.max_length, exp.truncate_frac = "prompt", 65536, 0.0
    exp.domain.name, exp.domain.benches = "dom", [DomainBench(bench="gpqa", system_prompt="mcq")]
    exp.domain.task.return_value = TaskConfig(name="gpqa")
    exp.items.return_value = {"gpqa": {"1", "2", "3"}}
    exp.domain.load_responses.return_value = pd.DataFrame(  # B never solved item 2; item 3 is Q1 but A failed it
        [("a", "1", True), ("a", "2", True), ("a", "3", False), ("b", "1", True), ("b", "2", False), ("b", "3", True)],
        columns=["model", "item", "correct"]).assign(bench="gpqa", repeat=0)
    exp.accuracy.return_value = pd.Series({"a": 0.9, "b": 0.4})
    exp.ordered_pairs.return_value = [(a, b)]

    tasks = build_tasks(exp)
    assert [t.name for t in tasks] == ["gpqa/a|b"]
    assert tasks[0].metadata["major"] == "a" and tasks[0].metadata["acc_minor"] == 0.4
    assert tasks[0].metadata["system_prompt"] == "mcq"  # the bench's own prompt wins
    assert mock_build_dataset.call_args.kwargs["items"] == {"1"}  # both-correct Q1 items only


@patch("judge.task.read_header")
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


@patch("judge.task.read_header")
def test_mismatched_benchmarks_raise(mock_read_log):
    log_major = MagicMock()
    log_major.eval.task = "inspect_evals/gpqa_diamond"
    log_minor = MagicMock()
    log_minor.eval.task = "inspect_evals/aime2024"
    mock_read_log.side_effect = [log_major, log_minor]

    with pytest.raises(ValueError, match="same task"):
        judge_task(major_path="a.eval", minor_path="b.eval")
