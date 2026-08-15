from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from dataset import (
    USER_PROMPT,
    Transcript,
    build_dataset,
    has_reasoning,
    read_transcripts,
    responses_to_sample,
)


def transcript(text, id="q1", question="Question", options=(), epoch=1):
    return Transcript(id=id, question=question, text=text, options=tuple(options), epoch=epoch)


def eval_sample(sample_id, content, correct=True, question=None, choices=None, epoch=1):
    """A stand-in for an EvalSample as read_eval_log_samples yields them."""
    return SimpleNamespace(
        id=sample_id,
        input=question or f"Q{sample_id}",
        epoch=epoch,
        choices=choices,
        output=ModelOutput.from_content(model="m", content=content),
        scores={"scorer": Score(value="C" if correct else "I")},
    )


class TestResponsesToSample:
    """Unit tests for responses_to_sample in dataset.py."""

    def test_responses_to_sample_success(self):
        res = responses_to_sample(
            transcript("Paris is capital.", question="What is capital of France?"),
            transcript("The capital is Paris.", question="What is capital of France?"),
            target="1",
        )

        assert res.id == "q1_1"
        assert res.target == "1"
        assert "What is capital of France?" in res.input
        assert "Paris is capital." in res.input
        assert "The capital is Paris." in res.input

    def test_responses_to_sample_max_len(self):
        res = responses_to_sample(
            transcript("A" * 100), transcript("Short answer"), target="1", max_len=10
        )
        assert "Transcript 1:\n---\n" + ("A" * 10) + "\n---" in res.input

    def test_responses_to_sample_options_per_transcript(self):
        """Each transcript is shown the options in the order its own model saw them."""
        res = responses_to_sample(
            transcript("ANSWER: A", question="Pick one.", options=["blue", "red", "green", "grey"]),
            transcript("ANSWER: C", question="Pick one.", options=["red", "grey", "blue", "green"]),
            target="1",
        )

        assert "Multiple choice options as shown to Model 1:\nA) blue" in res.input
        assert "Multiple choice options as shown to Model 2:\nA) red" in res.input
        # each block sits directly above its own transcript
        assert res.input.index("A) blue") < res.input.index("Transcript 1:")
        assert res.input.index("A) red") < res.input.index("Transcript 2:")
        assert res.input.index("Transcript 1:") < res.input.index("A) red")

    def test_responses_to_sample_no_options(self):
        res = responses_to_sample(transcript("4"), transcript("It is 4."), target="1")
        assert "Multiple choice options" not in res.input
        assert "---\n\nTranscript 1:" in res.input

    def test_responses_to_sample_truncation(self):
        res = responses_to_sample(
            transcript("reasoning one goes here\nANSWER: A"),
            transcript("short reasoning\nANSWER: B"),
            target="1",
            truncate_frac=0.5,
        )
        assert "ANSWER" not in res.input.split("Which of the two")[0]

    def test_responses_to_sample_id_mismatch(self):
        with pytest.raises(AssertionError, match="Sample IDs must match"):
            responses_to_sample(transcript("Ans A", id="q1"), transcript("Ans B", id="q2"))

    def test_responses_to_sample_input_mismatch(self):
        with pytest.raises(AssertionError, match="Sample inputs must match"):
            responses_to_sample(
                transcript("Ans A", question="Question 1"),
                transcript("Ans B", question="Question 2"),
            )


class TestHasReasoning:
    """Unit tests for has_reasoning in dataset.py."""

    def test_answer_only_fails(self):
        assert not has_reasoning("ANSWER: A", min_chars=10)

    def test_reasoning_passes(self):
        text = "First I considered X, then Y, then concluded Z.\nANSWER: A"
        assert has_reasoning(text, min_chars=10)

    def test_boundary_at_exactly_min_chars(self):
        text = ("x" * 10) + "\nANSWER: A"
        assert has_reasoning(text, min_chars=10)
        assert not has_reasoning(text, min_chars=11)

    def test_empty_string_fails(self):
        assert not has_reasoning("", min_chars=10)


class TestReadTranscripts:
    """read_transcripts streams a log into per-question transcripts."""

    @patch("dataset.read_eval_log_samples")
    def test_filters_incorrect_and_orders_epochs(self, mock_samples):
        mock_samples.return_value = iter([
            eval_sample("q1", "second epoch", epoch=2),
            eval_sample("q1", "first epoch", epoch=1),
            eval_sample("q2", "wrong", correct=False),
        ])

        by_id = read_transcripts(Path("log.eval"))

        assert list(by_id) == ["q1"]
        assert [t.text for t in by_id["q1"]] == ["first epoch", "second epoch"]

    @patch("dataset.read_eval_log_samples")
    def test_keeps_incorrect_when_not_required(self, mock_samples):
        mock_samples.return_value = iter([eval_sample("q1", "wrong", correct=False)])
        assert list(read_transcripts(Path("log.eval"), require_correct=False)) == ["q1"]


class TestBuildDataset:
    """Unit tests for build_dataset in dataset.py."""

    @staticmethod
    def _headers(major="gemma-26b", minor="gemma-2b", benchmark="gpqa"):
        header_major, header_minor = MagicMock(), MagicMock()
        header_major.eval.model = major
        header_major.eval.dataset.name = benchmark
        header_minor.eval.model = minor
        return [header_major, header_minor]

    @patch("dataset.read_eval_log")
    @patch("dataset.read_eval_log_samples")
    def test_build_dataset_success(self, mock_samples, mock_header):
        mock_samples.side_effect = [
            iter([eval_sample("q1", "Ans 1A"), eval_sample("q2", "Ans 2A")]),
            iter([eval_sample("q1", "Ans 1B"), eval_sample("q2", "Ans 2B")]),
        ]
        mock_header.side_effect = self._headers()

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))

        assert isinstance(dataset, MemoryDataset)
        assert len(dataset.samples) == 4  # 2 questions x 2 targets (1 and 2)
        assert "gemma-26b" in dataset.name
        assert "gemma-2b" in dataset.name
        assert "gpqa" in dataset.name

    @patch("dataset.read_eval_log")
    @patch("dataset.read_eval_log_samples")
    def test_build_dataset_unmatched_samples_skipped(self, mock_samples, mock_header):
        mock_samples.side_effect = [
            iter([eval_sample("q1", "A1"), eval_sample("q2", "A2")]),
            iter([eval_sample("q1", "B1")]),  # only q1 matches
        ]
        mock_header.side_effect = self._headers()

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))
        assert len(dataset.samples) == 2  # only q1, for target 1 and 2

    @patch("dataset.read_eval_log")
    @patch("dataset.read_eval_log_samples")
    def test_build_dataset_filtering_incorrect_samples(self, mock_samples, mock_header):
        mock_samples.side_effect = [
            iter([
                eval_sample("q1", "A1"),
                eval_sample("q2", "A2", correct=False),
                eval_sample("q3", "A3"),
            ]),
            iter([
                eval_sample("q1", "B1"),
                eval_sample("q2", "B2"),
                eval_sample("q3", "B3", correct=False),
            ]),
        ]
        mock_header.side_effect = self._headers()

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))

        # Only q1 was correct for both models.
        assert len(dataset.samples) == 2
        assert dataset.samples[0].id == "q1_1"
        assert dataset.samples[1].id == "q1_2"

    @patch("dataset.read_eval_log")
    @patch("dataset.read_eval_log_samples")
    def test_build_dataset_no_reverse(self, mock_samples, mock_header):
        mock_samples.side_effect = [
            iter([eval_sample("q1", "A1")]),
            iter([eval_sample("q1", "B1")]),
        ]
        mock_header.side_effect = self._headers()

        dataset = build_dataset(
            Path("path/major.eval"), Path("path/minor.eval"), include_reverse=False
        )
        assert [s.target for s in dataset.samples] == ["1"]

    @patch("dataset.read_eval_log")
    @patch("dataset.read_eval_log_samples")
    def test_self_pair_needs_two_epochs(self, mock_samples, mock_header):
        """The self-pair control reads one log once and pairs two correct epochs."""
        mock_samples.side_effect = [
            iter([
                eval_sample("q1", "epoch 1", epoch=1),
                eval_sample("q1", "epoch 2", epoch=2),
                eval_sample("q2", "only one epoch", epoch=1),
            ]),
        ]
        mock_header.side_effect = self._headers(major="m", minor="m")

        dataset = build_dataset(Path("path/same.eval"), Path("path/same.eval"))

        assert mock_samples.call_count == 1
        assert [s.id for s in dataset.samples] == ["q1_1", "q1_2"]
        assert "epoch 1" in dataset.samples[0].input


def test_user_prompt_placeholders():
    for field in ("question", "options_1", "options_2", "transcript_1", "transcript_2"):
        assert "{" + field + "}" in USER_PROMPT
