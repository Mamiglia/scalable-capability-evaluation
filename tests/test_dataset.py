import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score
from inspect_ai.dataset import MemoryDataset

from dataset import responses_to_sample, build_dataset, USER_PROMPT


class TestResponsesToSample:
    """Unit tests for responses_to_sample in dataset.py."""

    def test_responses_to_sample_success(self):
        """Test creating a judge sample from two valid model responses."""
        sample_a = SimpleNamespace(
            id="q1",
            input="What is capital of France?",
            output=ModelOutput.from_content(model="model-a", content="Paris is capital."),
        )
        sample_b = SimpleNamespace(
            id="q1",
            input="What is capital of France?",
            output=ModelOutput.from_content(model="model-b", content="The capital is Paris."),
        )

        res = responses_to_sample(sample_a, sample_b, target="A")

        assert res.id == "q1_A"
        assert res.target == "A"
        assert "What is capital of France?" in res.input
        assert "Paris is capital." in res.input
        assert "The capital is Paris." in res.input

    def test_responses_to_sample_max_len(self):
        """Test truncation of responses exceeding max_len."""
        long_text = "A" * 100
        sample_a = SimpleNamespace(
            id="q1",
            input="Question",
            output=ModelOutput.from_content(model="model-a", content=long_text),
        )
        sample_b = SimpleNamespace(
            id="q1",
            input="Question",
            output=ModelOutput.from_content(model="model-b", content="Short answer"),
        )

        res = responses_to_sample(sample_a, sample_b, target="A", max_len=10)

        assert "Transcript A:\n---\n" + ("A" * 10) + "\n---" in res.input

    def test_responses_to_sample_id_mismatch(self):
        """Test that mismatched sample IDs raise AssertionError."""
        sample_a = SimpleNamespace(
            id="q1",
            input="Question",
            output=ModelOutput.from_content(model="model-a", content="Ans A"),
        )
        sample_b = SimpleNamespace(
            id="q2",
            input="Question",
            output=ModelOutput.from_content(model="model-b", content="Ans B"),
        )

        with pytest.raises(AssertionError, match="Sample IDs must match"):
            responses_to_sample(sample_a, sample_b)

    def test_responses_to_sample_input_mismatch(self):
        """Test that mismatched inputs raise AssertionError."""
        sample_a = SimpleNamespace(
            id="q1",
            input="Question 1",
            output=ModelOutput.from_content(model="model-a", content="Ans A"),
        )
        sample_b = SimpleNamespace(
            id="q1",
            input="Question 2",
            output=ModelOutput.from_content(model="model-b", content="Ans B"),
        )

        with pytest.raises(AssertionError, match="Sample inputs must match"):
            responses_to_sample(sample_a, sample_b)


class TestBuildDataset:
    """Unit tests for build_dataset in dataset.py."""

    @patch("dataset.read_eval_log")
    def test_build_dataset_success(self, mock_read_eval_log):
        """Test building dataset from two eval logs with matching correct samples."""
        log_major = MagicMock()
        log_major.eval.model = "gemma-26b"
        log_major.eval.dataset.name = "gpqa"

        s1_a = SimpleNamespace(
            id="q1",
            input="Q1",
            output=ModelOutput.from_content(model="gemma-26b", content="Ans 1A"),
            score=Score(value="C"),
        )
        s2_a = SimpleNamespace(
            id="q2",
            input="Q2",
            output=ModelOutput.from_content(model="gemma-26b", content="Ans 2A"),
            score=Score(value="C"),
        )
        log_major.samples = [s1_a, s2_a]

        log_minor = MagicMock()
        log_minor.eval.model = "gemma-2b"

        s1_b = SimpleNamespace(
            id="q1",
            input="Q1",
            output=ModelOutput.from_content(model="gemma-2b", content="Ans 1B"),
            score=Score(value="C"),
        )
        s2_b = SimpleNamespace(
            id="q2",
            input="Q2",
            output=ModelOutput.from_content(model="gemma-2b", content="Ans 2B"),
            score=Score(value="C"),
        )
        log_minor.samples = [s1_b, s2_b]

        mock_read_eval_log.side_effect = [log_major, log_minor]

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))

        assert isinstance(dataset, MemoryDataset)
        assert len(dataset.samples) == 4  # 2 questions x 2 targets (A and B)
        assert "gemma-26b" in dataset.name
        assert "gemma-2b" in dataset.name
        assert "gpqa" in dataset.name

    @patch("dataset.read_eval_log")
    def test_build_dataset_unmatched_samples_skipped(self, mock_read_eval_log):
        """Test that samples with non-matching IDs between logs are skipped gracefully."""
        log_major = MagicMock()
        log_major.eval.model = "model-major"
        log_major.eval.dataset.name = "bench"

        s1_a = SimpleNamespace(id="q1", input="Q1", output=ModelOutput.from_content(model="m1", content="A1"), score=Score(value="C"))
        s2_a = SimpleNamespace(id="q2", input="Q2", output=ModelOutput.from_content(model="m1", content="A2"), score=Score(value="C"))
        log_major.samples = [s1_a, s2_a]

        log_minor = MagicMock()
        log_minor.eval.model = "model-minor"
        s1_b = SimpleNamespace(id="q1", input="Q1", output=ModelOutput.from_content(model="m2", content="B1"), score=Score(value="C"))
        log_minor.samples = [s1_b]  # Only q1 matches

        mock_read_eval_log.side_effect = [log_major, log_minor]

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))
        assert len(dataset.samples) == 2  # Only q1 paired for target A and B

    @patch("dataset.read_eval_log")
    def test_build_dataset_filtering_incorrect_samples(self, mock_read_eval_log):
        """Test that samples where either model answered incorrectly ('I') are filtered out."""
        log_major = MagicMock()
        log_major.eval.model = "model-major"
        log_major.eval.dataset.name = "bench"

        s1_a = SimpleNamespace(
            id="q1", input="Q1", output=ModelOutput.from_content(model="m1", content="A1"), score=Score(value="C")
        )
        s2_a = SimpleNamespace(
            id="q2", input="Q2", output=ModelOutput.from_content(model="m1", content="A2"), score=Score(value="I")
        )
        s3_a = SimpleNamespace(
            id="q3", input="Q3", output=ModelOutput.from_content(model="m1", content="A3"), score=Score(value="C")
        )

        log_minor = MagicMock()
        log_minor.eval.model = "model-minor"

        s1_b = SimpleNamespace(
            id="q1", input="Q1", output=ModelOutput.from_content(model="m2", content="B1"), score=Score(value="C")
        )
        s2_b = SimpleNamespace(
            id="q2", input="Q2", output=ModelOutput.from_content(model="m2", content="B2"), score=Score(value="C")
        )
        s3_b = SimpleNamespace(
            id="q3", input="Q3", output=ModelOutput.from_content(model="m2", content="B3"), score=Score(value="I")
        )

        log_major.samples = [s1_a, s2_a, s3_a]
        log_minor.samples = [s1_b, s2_b, s3_b]

        mock_read_eval_log.side_effect = [log_major, log_minor]

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))

        # Only q1 had 'C' for both models. q2 had 'I' for major, q3 had 'I' for minor.
        assert len(dataset.samples) == 2  # q1 with target A and target B
        assert dataset.samples[0].id == "q1_A"
        assert dataset.samples[1].id == "q1_B"
