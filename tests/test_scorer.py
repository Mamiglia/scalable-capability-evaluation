import pytest
from inspect_ai.scorer import Target, CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput

from scorer import parse_judge_output, judge_scorer


class TestParseJudgeOutput:
    """Unit tests for parse_judge_output function in scorer.py."""

    def test_parse_empty_or_none(self):
        """Test parsing when input is None or empty."""
        assert parse_judge_output(None) == (None, None, None)
        assert parse_judge_output("") == (None, None, None)
        assert parse_judge_output("   \n  ") == (None, None, None)

    def test_parse_standard_format(self):
        """Test parsing complete output with standard formatting."""
        text = (
            "EXPLANATION: Response A provides a more thorough explanation with code examples.\n"
            "ANSWER: A\n"
            "CONFIDENCE: 5"
        )
        answer, confidence, explanation = parse_judge_output(text)
        assert answer == "A"
        assert confidence == 5
        assert explanation == "Response A provides a more thorough explanation with code examples."

    def test_parse_answer_variations(self):
        """Test parsing different variations of ANSWER formatting."""
        cases = [
            ("ANSWER: A", "A"),
            ("ANSWER: B", "B"),
            ("ANSWER: Model A", "A"),
            ("ANSWER: Model B", "B"),
            ("ANSWER: *A*", "A"),
            ("ANSWER: *B*", "B"),
            ("answer: a", "A"),
            ("answer: model b", "B"),
        ]
        for input_text, expected_answer in cases:
            ans, _, _ = parse_judge_output(input_text)
            assert ans == expected_answer, f"Failed for input: {input_text}"

    def test_parse_confidence_formats(self):
        """Test parsing 1-5 scale formats for CONFIDENCE."""
        # Integer values 1-5
        _, conf1, _ = parse_judge_output("CONFIDENCE: 5")
        assert conf1 == 5

        _, conf2, _ = parse_judge_output("CONFIDENCE: 1")
        assert conf2 == 1

        # Float value in 1-5 range
        _, conf3, _ = parse_judge_output("CONFIDENCE: 3.5")
        assert conf3 == 3.5

        # Out of bounds (< 1 or > 5)
        _, conf4, _ = parse_judge_output("CONFIDENCE: 0")
        assert conf4 is None

        _, conf5, _ = parse_judge_output("CONFIDENCE: 6")
        assert conf5 is None

        # Non-numeric confidence string
        _, conf6, _ = parse_judge_output("CONFIDENCE: High")
        assert conf6 is None

    def test_parse_explanation_multiline(self):
        """Test parsing multiline EXPLANATION text."""
        text = (
            "EXPLANATION: Model B is much clearer.\n"
            "It structures the output logically and avoids code duplication.\n"
            "Therefore, Model B is superior.\n"
            "ANSWER: B\n"
            "CONFIDENCE: 4"
        )
        ans, conf, expl = parse_judge_output(text)
        assert ans == "B"
        assert conf == 4
        assert "Model B is much clearer." in expl
        assert "Therefore, Model B is superior." in expl

    def test_parse_missing_fields(self):
        """Test parsing when certain fields are missing."""
        # Missing confidence and explanation
        text1 = "ANSWER: A"
        assert parse_judge_output(text1) == ("A", None, None)

        # Missing answer
        text2 = "EXPLANATION: Good reasoning\nCONFIDENCE: 4"
        assert parse_judge_output(text2) == (None, 4, "Good reasoning")

class TestJudgeScorer:
    """Unit tests for judge_scorer in scorer.py."""

    @pytest.mark.asyncio
    async def test_scorer_correct_answer(self):
        """Test scorer when judge output matches target."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare A and B",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="EXPLANATION: Model A is better.\nANSWER: A\nCONFIDENCE: 5",
            ),
        )
        target = Target("A")
        score = await scorer_fn(state, target)

        assert score.value == CORRECT
        assert score.answer == "A"
        assert score.explanation == "Model A is better."
        assert score.metadata == {"confidence": 5}

    @pytest.mark.asyncio
    async def test_scorer_incorrect_answer(self):
        """Test scorer when judge output differs from target."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare A and B",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="EXPLANATION: Model B is better.\nANSWER: B\nCONFIDENCE: 4",
            ),
        )
        target = Target("A")
        score = await scorer_fn(state, target)

        assert score.value == INCORRECT
        assert score.answer == "B"
        assert score.explanation == "Model B is better."
        assert score.metadata == {"confidence": 4}

    @pytest.mark.asyncio
    async def test_scorer_no_answer(self):
        """Test scorer when judge fails to provide an answer or output is empty."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare A and B",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="I cannot decide which model is better.",
            ),
        )
        target = Target("A")
        score = await scorer_fn(state, target)

        assert score.value == NOANSWER
        assert score.answer is None
        assert score.explanation is None
        assert score.metadata is None

    @pytest.mark.asyncio
    async def test_scorer_target_as_list(self):
        """Test scorer when target.text is a list of strings."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare A and B",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="ANSWER: A",
            ),
        )
        target = Target(["A"])
        score = await scorer_fn(state, target)

        assert score.value == CORRECT
        assert score.answer == "A"
