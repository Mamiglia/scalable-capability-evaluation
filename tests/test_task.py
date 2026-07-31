from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from judge.task import (
    judge_task,
    build_judge_tasks,
    resolve_transcript_pairs,
    SYSTEM_PROMPT,
)


class TestJudgeTask:
    """Unit tests for judge_task function in task.py."""

    @patch(
        "judge.task.resolve_eval_path",
        side_effect=["data/major.eval", "data/minor.eval"],
    )
    @patch("judge.task.build_dataset")
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
            major="gemma-26b",
            minor="gemma-2b",
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
            log_major_path=Path("data/major.eval"),
            log_minor_path=Path("data/minor.eval"),
            max_len=12000,
            require_both_correct=True,
            include_reverse=True,
        )
        assert mock_resolve.call_args_list[0].args[0:2] == (
            "gemma-26b",
            "gpqa_diamond",
        )

def test_resolve_transcript_pairs_paired():
    major_paths = [Path("a/0.eval"), Path("a/1.eval"), Path("a/2.eval")]
    minor_paths = [Path("b/0.eval"), Path("b/1.eval")]

    pairs = resolve_transcript_pairs(major_paths, minor_paths, mode="paired")
    assert pairs == [(Path("a/0.eval"), Path("b/0.eval")), (Path("a/1.eval"), Path("b/1.eval"))]


def test_resolve_transcript_pairs_combinatorial():
    major_paths = [Path("a/0.eval"), Path("a/1.eval")]
    minor_paths = [Path("b/0.eval"), Path("b/1.eval")]

    pairs = resolve_transcript_pairs(major_paths, minor_paths, mode="combinatorial")
    assert pairs == [
        (Path("a/0.eval"), Path("b/0.eval")),
        (Path("a/0.eval"), Path("b/1.eval")),
        (Path("a/1.eval"), Path("b/0.eval")),
        (Path("a/1.eval"), Path("b/1.eval")),
    ]


def test_resolve_transcript_pairs_custom():
    major_paths = [Path("a/0.eval"), Path("a/1.eval")]
    minor_paths = [Path("b/0.eval"), Path("b/1.eval")]

    # 1. Specified as integers/indices
    pairs1 = resolve_transcript_pairs(major_paths, minor_paths, mode="custom", pairs=[[0, 0], [0, 1]])
    assert pairs1 == [(Path("a/0.eval"), Path("b/0.eval")), (Path("a/0.eval"), Path("b/1.eval"))]

    # 2. Specified as filename strings
    pairs2 = resolve_transcript_pairs(
        major_paths, minor_paths, mode="custom", pairs=[["0.eval", "1.eval"]]
    )
    assert pairs2 == [(Path("a/0.eval"), Path("b/1.eval"))]

    # 3. Missing pairs parameter raises ValueError
    with pytest.raises(ValueError, match="pairs must be provided"):
        resolve_transcript_pairs(major_paths, minor_paths, mode="custom")


def test_resolve_transcript_pairs_invalid_mode():
    major_paths = [Path("a/0.eval")]
    minor_paths = [Path("b/0.eval")]
    with pytest.raises(ValueError, match="Unknown comparison mode"):
        resolve_transcript_pairs(major_paths, minor_paths, mode="invalid_mode")


@patch("judge.task.resolve_eval_paths")
@patch("judge.task.build_dataset")
def test_build_judge_tasks_multi_pair(mock_build_dataset, mock_resolve_paths):
    mock_resolve_paths.side_effect = [
        [Path("major/0.eval"), Path("major/1.eval")],
        [Path("minor/0.eval"), Path("minor/1.eval")],
    ]
    mock_build_dataset.return_value = MemoryDataset(samples=[Sample(input="test")])

    tasks = build_judge_tasks(
        major="ibm",
        minor="llama",
        benchmark_name="gpqa",
        mode="combinatorial",
    )

    assert len(tasks) == 4
    assert tasks[0].name == "gpqa=ibm:0|llama:0"
    assert tasks[1].name == "gpqa=ibm:0|llama:1"
    assert tasks[2].name == "gpqa=ibm:1|llama:0"
    assert tasks[3].name == "gpqa=ibm:1|llama:1"
