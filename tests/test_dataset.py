from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from judge.dataset import (
    response_chars,
    USER_PROMPT,
    build_dataset,
    has_reasoning,
    responses_to_sample,
)
from evals.transcripts import Transcript, read_transcripts


PAD = " " + "." * 30  # past MIN_RESPONSE_CHARS: the pairing tests are not about short answers


def transcript(text, id="q1", question="Question", options=(), epoch=1, pad=True):
    return Transcript(id=id, question=question, text=text + (PAD if pad else ""), options=tuple(options), epoch=epoch)


def eval_sample(sample_id, content, correct=True, question=None, choices=None, epoch=1):
    """A stand-in for an EvalSample as read_eval_log_samples yields them."""
    return SimpleNamespace(
        id=sample_id,
        input=question or f"Q{sample_id}",
        epoch=epoch,
        choices=choices,
        output=ModelOutput.from_content(model="m", content=content + PAD),
        scores={"scorer": Score(value="C" if correct else "I")},
    )


def test_read_transcripts_with_items_reads_only_those_samples(monkeypatch):
    import evals.transcripts as module

    read = []
    monkeypatch.setattr(module, "_read_samples",
                        lambda path, ids: (read.append(i) or eval_sample(i, f"ANSWER: {i}", correct=i != "q3") for i in ids))
    monkeypatch.setattr(module, "read_eval_log_samples", lambda *a, **k: pytest.fail("streamed"))

    got = read_transcripts("log.eval", items={"q1", "q3"})
    assert read == ["q1", "q3"] and list(got) == ["q1"]  # q3 read but wrong
    # Cached per log and filled incrementally: another pair's subset reads only what is new.
    assert sorted(read_transcripts("log.eval", items={"q3", "q1", "q2"})) == ["q1", "q2"]
    assert read == ["q1", "q3", "q2"]


def test_build_dataset_keeps_pairs_with_reasoning_on_one_side_only(monkeypatch):
    import judge.dataset as module

    with_cot = "<THOUGHT_START>\n think\n<THOUGHT_END>\nANSWER: 1 and the derivation follows here in some detail."
    plain = "The answer is 1 because of the derivation given here in some detail.\nANSWER: 1"
    logs = {"a.eval": {"q1": [transcript(with_cot)]}, "b.eval": {"q1": [transcript(plain)]}}
    monkeypatch.setattr(module, "read_transcripts", lambda path, *a, **k: logs[str(path)])
    monkeypatch.setattr(module, "read_header", lambda path: MagicMock())
    # a reasoning model vs a non-reasoning one: the trace is shown on the side that has it
    samples = build_dataset(Path("a.eval"), Path("b.eval"), include_reasoning=True)
    assert len(samples) == 2 and "<THOUGHT_START>" in samples[0].input and "<THOUGHT_START>" in samples[1].input


def test_question_text_takes_user_text_of_chat_inputs():
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    from evals.transcripts import question_text

    chat = [ChatMessageSystem(content="format"), ChatMessageUser(content="What is 2+2?")]
    assert question_text(chat) == "What is 2+2?"
    assert question_text("plain") == "plain"
    # Message ids differ per run; the same question must still compare equal across logs.
    again = [ChatMessageSystem(content="format"), ChatMessageUser(content="What is 2+2?")]
    assert chat[1].id != again[1].id and question_text(chat) == question_text(again)


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
        assert "Transcript 1:\n---\n\n" + ("A" * 10) + "\n\n---" in res.input
        assert res.metadata["truncated_1"] and res.metadata["chars_1"] == 10  # shown, not raw

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

    @patch("evals.transcripts.read_eval_log_samples")
    def test_filters_incorrect_and_orders_epochs(self, mock_samples):
        mock_samples.return_value = iter([
            eval_sample("q1", "second epoch", epoch=2),
            eval_sample("q1", "first epoch", epoch=1),
            eval_sample("q2", "wrong", correct=False),
        ])

        by_id = read_transcripts(Path("log.eval"))

        assert list(by_id) == ["q1"]
        assert [t.text.rstrip(". ") for t in by_id["q1"]] == ["first epoch", "second epoch"]

    @patch("evals.transcripts.read_eval_log_samples")
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

    @patch("evals.transcripts.read_eval_log")
    @patch("evals.transcripts.read_eval_log_samples")
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

    @patch("evals.transcripts.read_eval_log")
    @patch("evals.transcripts.read_eval_log_samples")
    def test_build_dataset_unmatched_samples_skipped(self, mock_samples, mock_header):
        mock_samples.side_effect = [
            iter([eval_sample("q1", "A1"), eval_sample("q2", "A2")]),
            iter([eval_sample("q1", "B1")]),  # only q1 matches
        ]
        mock_header.side_effect = self._headers()

        dataset = build_dataset(Path("path/major.eval"), Path("path/minor.eval"))
        assert len(dataset.samples) == 2  # only q1, for target 1 and 2

    @patch("evals.transcripts.read_eval_log")
    @patch("evals.transcripts.read_eval_log_samples")
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

    @patch("evals.transcripts.read_eval_log")
    @patch("evals.transcripts.read_eval_log_samples")
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

    @patch("evals.transcripts.read_eval_log")
    @patch("evals.transcripts.read_eval_log_samples")
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


def test_response_chars_ignores_role_prefixes_and_short_answers_are_skipped(monkeypatch):
    assert response_chars("[Assistant]: ANSWER: B") == 9
    assert response_chars("[User]: follow-up\n[Tool Output (bash)]: ok\n[Assistant]: done") == len("follow-up\nok\ndone")
    logs = {"a.eval": {"q1": [transcript("[Assistant]: ANSWER: B", pad=False)], "q2": [transcript("[Assistant]: " + "x" * 30, id="q2", pad=False)]},
            "b.eval": {"q1": [transcript("[Assistant]: " + "y" * 30, pad=False)], "q2": [transcript("[Assistant]: " + "y" * 30, id="q2", pad=False)]}}
    import judge.dataset as module
    monkeypatch.setattr(module, "read_transcripts", lambda path, *a, **k: logs[str(path)])
    monkeypatch.setattr(module, "read_header", lambda path: MagicMock())
    dataset = build_dataset(Path("a.eval"), Path("b.eval"), include_reasoning=False)
    assert [s.id for s in dataset.samples] == ["q2_1", "q2_2"]  # q1: one side answer-only
    assert dataset.samples[0].metadata["response_chars_1"] == 30
