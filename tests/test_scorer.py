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
            "EXPLANATION: Response 1 provides a more thorough explanation with code examples.\n"
            "ANSWER: 1\n"
            "CONFIDENCE: 5"
        )
        answer, confidence, explanation = parse_judge_output(text)
        assert answer == "1"
        assert confidence == 5
        assert explanation == "Response 1 provides a more thorough explanation with code examples."

    def test_parse_answer_variations(self):
        """Test parsing different variations of ANSWER formatting."""
        cases = [
            ("ANSWER: 1", "1"),
            ("ANSWER: 2", "2"),
            ("ANSWER: Model 1", "1"),
            ("ANSWER: Model 2", "2"),
            ("ANSWER: Transcript 1", "1"),
            ("ANSWER: Transcript 2", "2"),
            ("ANSWER: *1*", "1"),
            ("ANSWER: *2*", "2"),
            ("answer: 1", "1"),
            ("answer: model 2", "2"),
            ("ANSWER: $1$", "1"),
            ("ANSWER: $2$", "2"),
            ('"ANSWER": 2', "2"),          # JSON output (ling)
            ('"ANSWER": "2"', "2"),
            ('{"EXPLANATION": "x", "ANSWER": "1", "CONFIDENCE": 4}', "1"),
            ("**ANSWER**: 1", "1"),        # colon outside the bold (r1)
            ("ANSWER : 2", "2"),
            ("ANSWER:\n2", "2"),
            ("ANSWER: 'Transcript 2'", "2"),
            ("**ANSWER:** **Model 1**", "1"),
            ("### Answer\n2", "2"),           # heading then a bare digit (r1)
            ("### ANSWER  \n1", "1"),
            ("**Answer**\n\n2", "2"),
            ("Answer:\n1", "1"),
            ("reasoning.\n\n## Answer\n1\n", "1"),
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
            "EXPLANATION: Model 2 is much clearer.\n"
            "It structures the output logically and avoids code duplication.\n"
            "Therefore, Model 2 is superior.\n"
            "ANSWER: 2\n"
            "CONFIDENCE: 4"
        )
        ans, conf, expl = parse_judge_output(text)
        assert ans == "2"
        assert conf == 4
        assert "Model 2 is much clearer." in expl
        assert "Therefore, Model 2 is superior." in expl

    def test_parse_missing_fields(self):
        """Test parsing when certain fields are missing."""
        # Missing confidence and explanation
        text1 = "ANSWER: 1"
        assert parse_judge_output(text1) == ("1", None, None)

        # Missing answer
        text2 = "EXPLANATION: Good reasoning\nCONFIDENCE: 4"
        assert parse_judge_output(text2) == (None, 4, "Good reasoning")

    def test_quoted_transcript_answers_are_ignored(self):
        """A digit run like 'ANSWER: 128' is not a verdict, and the last verdict wins."""
        assert parse_judge_output("Model 1 wrote ANSWER: 128.\nANSWER: 2")[0] == "2"
        assert parse_judge_output("ANSWER: 128")[0] is None
        assert parse_judge_output('"ANSWER": 12')[0] is None
        assert parse_judge_output("**ANSWER**: 21\nANSWER: 1")[0] == "1"
        assert parse_judge_output("### Answer\n128")[0] is None
        assert parse_judge_output("Model 1 wrote:\nAnswer\n128\n\n### Answer\n2")[0] == "2"
        assert parse_judge_output("### Answer\n1\n...\nANSWER: 2")[0] == "2"

    def test_verdict_without_format(self):
        """A pick stated in prose without the ANSWER line is not a verdict."""
        assert parse_judge_output("Transcript 1 is verbose. Transcript 2 is stronger.")[0] is None
        assert parse_judge_output("My answer is 2.")[0] is None
        assert parse_judge_output("ANSWER 2")[0] is None
        assert parse_judge_output("ANSWER: none\nCONFIDENCE: 1")[0] is None
        assert parse_judge_output("### Answer\n1 is the more capable model")[0] is None
        assert parse_judge_output("The answer\n2")[0] is None


def test_pair_metrics():
    from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score
    from judge.scorer import consistent_accuracy, length_corr, position_bias

    def sample(sid, value, answer, chars):
        return SampleScore(sample_id=sid, score=Score(value=value, answer=answer),
                           sample_metadata={"chars_1": chars[0], "chars_2": chars[1]})

    scores = [  # metrics see reduced values: 1.0 / 0.0, and None answer for no verdict
        sample("q1_1", CORRECT, "1", (10, 5)), sample("q1_2", CORRECT, "2", (5, 10)),      # right both orders
        sample("q2_1", CORRECT, "1", (10, 5)), sample("q2_2", INCORRECT, "1", (5, 10)),    # position 1 both times
        sample("q3_1", NOANSWER, None, (1, 2)), sample("q3_2", NOANSWER, None, (2, 1)),    # no verdict
    ]
    assert consistent_accuracy()(scores) == pytest.approx(1 / 3)
    assert position_bias()(scores) == pytest.approx(-0.5)   # 3x "1", 1x "2" over 4 answered
    assert length_corr()(scores) == pytest.approx(0.5773502691896258)  # phi of x=[1,0,1,0], y=[1,0,1,1]
    import math
    assert math.isnan(length_corr()([]))       # no data != no length bias
    assert math.isnan(position_bias()([]))

    # epochs > 1: the mean reducer blanks ``answer`` unless every epoch agreed, so the
    # metrics must read the pick rate off the value, not filter on the answer.
    reduced = [SampleScore(sample_id="q4_1", score=Score(value=0.8, answer=None),
                           sample_metadata={"chars_1": 10, "chars_2": 5}),
               SampleScore(sample_id="q4_2", score=Score(value=0.2, answer=None),
                           sample_metadata={"chars_1": 5, "chars_2": 10})]
    assert position_bias()(reduced) == pytest.approx(-0.6)  # picks 1 in 80% of both orders


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
            input="Compare 1 and 2",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="EXPLANATION: Model 1 is better.\nANSWER: 1\nCONFIDENCE: 5",
            ),
        )
        target = Target("1")
        score = await scorer_fn(state, target)

        assert score.value == CORRECT
        assert score.answer == "1"
        assert score.explanation == "Model 1 is better."
        assert score.metadata == {"confidence": 5}

    @pytest.mark.asyncio
    async def test_scorer_incorrect_answer(self):
        """Test scorer when judge output differs from target."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare 1 and 2",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="EXPLANATION: Model 2 is better.\nANSWER: 2\nCONFIDENCE: 4",
            ),
        )
        target = Target("1")
        score = await scorer_fn(state, target)

        assert score.value == INCORRECT
        assert score.answer == "2"
        assert score.explanation == "Model 2 is better."
        assert score.metadata == {"confidence": 4}

    @pytest.mark.asyncio
    async def test_scorer_no_answer(self):
        """Test scorer when judge fails to provide an answer or output is empty."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare 1 and 2",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="I cannot decide which model is better.",
            ),
        )
        target = Target("1")
        score = await scorer_fn(state, target)

        assert score.value == NOANSWER
        assert score.answer is None
        assert score.explanation is None
        assert score.metadata == {}

    @pytest.mark.asyncio
    async def test_scorer_target_as_list(self):
        """Test scorer when target.text is a list of strings."""
        scorer_fn = judge_scorer()
        state = TaskState(
            model="judge-model",
            sample_id="sample-1",
            epoch=1,
            input="Compare 1 and 2",
            messages=[],
            output=ModelOutput.from_content(
                model="judge-model",
                content="ANSWER: 1",
            ),
        )
        target = Target(["1"])
        score = await scorer_fn(state, target)

        assert score.value == CORRECT
        assert score.answer == "1"


def test_answer_rate_metric():
    """Unparseable verdicts are wrong for accuracy, so they need their own number."""
    from inspect_ai.scorer import NOANSWER, CORRECT, INCORRECT, SampleScore, Score

    from judge.scorer import answer_rate

    scores = [  # values arrive epoch-reduced (floats); the verdict lives in ``answer``
        SampleScore(score=Score(value=1.0, answer="1")),
        SampleScore(score=Score(value=0.0, answer="2")),
        SampleScore(score=Score(value=0.0, answer=None)),
        SampleScore(score=Score(value=0.0, answer=None)),
    ]
    assert answer_rate()(scores) == 0.5
    assert answer_rate()([]) == 0.0
