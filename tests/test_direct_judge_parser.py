"""Final-text parsing for the inline ratings observed in MiniMax judge outputs."""

from types import SimpleNamespace

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target

from judge.scorer import direct_judge_scorer, parse_direct_judge_output


async def test_inline_rating_does_not_require_a_newline():
    # These are the final-text fields, without borrowing from the reasoning channel.
    scorer = direct_judge_scorer()
    for text, expected in [
        ("logical explanation showing solid understanding of basic lab safety, "
         "which is above the average person's knowledge but not at the level of "
         "a specialized expert. ANSWER: 6", 6),
        ("manipulation is typical of top-tier contest participants. ANSWER: 6", 6),
        (". ANSWER: 8", 8),
        ("Clear mathematical reasoning.ANSWER: 8.25", 8.25),
        ("Clear mathematical reasoning. **ANSWER:** $7.5$", 7.5),
        ("Clear mathematical reasoning!\tANSWER: 7/10", 7),
    ]:
        score = await scorer(SimpleNamespace(output=ModelOutput.from_content("judge", text)), Target(""))
        assert score.answer is not None and float(score.answer) == score.value == expected, text


def test_inline_rating_keeps_final_field_and_range_guards():
    for text in [
        "The transcript says ANSWER: 2",
        'The transcript says "ANSWER: 2".',
        "Strong reasoning. ANSWER: 907",
        "Strong reasoning. ANSWER: 10.5",
        "Strong reasoning. ANSWER: 7/11",
        "ANSWER: 7\nStrong reasoning. ANSWER: 11",
        "ANSWER: 7\nStrong reasoning. ANSWER:",
        "WER: 7",
        "",
    ]:
        assert parse_direct_judge_output(text)[0] is None, text
