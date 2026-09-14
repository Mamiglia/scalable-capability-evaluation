"""Direct judging: correct first attempts and numeric ratings."""

from types import SimpleNamespace

import numpy as np
import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ContentReasoning, ContentText, ModelOutput
from inspect_ai.scorer import Score, Target

from judge.dataset import build_direct_dataset
from judge.scorer import direct_judge_scorer


def source_log(tmp_path):
    path = tmp_path / "source.eval"
    samples = [EvalSample(
        id=item, epoch=epoch, input="Pick a colour.", target="A", choices=["blue", "red"],
        messages=[ChatMessageUser(content="Pick a colour."), ChatMessageAssistant(content=[
            ContentReasoning(reasoning="Private reasoning about the colours."), ContentText(text="ANSWER: A"),
        ])], scores={"correctness": Score(value="C" if correct else "I")},
    ) for item, epoch, correct in [("short", 1, True), ("wrong", 1, False),
                                   ("wrong", 2, True), ("unselected", 1, True)]]
    log = EvalLog(status="success", eval=EvalSpec(created="2026-09-14T12:00:00Z", task="b",
                  dataset=EvalDataset(name="b", samples=len(samples)), model="source-model", config=EvalConfig()),
                  samples=samples)
    write_eval_log(log, path)
    return path



def test_dataset_uses_correct_first_attempts_with_options_and_reasoning_control(tmp_path):
    path = source_log(tmp_path)
    dataset = build_direct_dataset(path, items={"short", "wrong"}, include_reasoning=False)
    assert [sample.id for sample in dataset] == ["short"]
    sample = dataset[0]
    assert sample.target == ""
    assert "Pick a colour." in sample.input and "A) blue\nB) red" in sample.input
    assert "ANSWER: A" in sample.input and "Private reasoning" not in sample.input
    assert "Transcript 2" not in sample.input and "source.eval" not in sample.input
    assert sample.metadata["response_chars"] == 9 and not sample.metadata["truncated"]

    shown = build_direct_dataset(path, items={"short"})[0]
    assert "Private reasoning" in shown.input
    shortened = build_direct_dataset(path, items={"short"}, max_len=5)[0]
    assert shortened.metadata["chars"] == 5 and shortened.metadata["truncated"]
    assert len(build_direct_dataset(path, items={"wrong"})) == 0
    assert len(build_direct_dataset(path, items=set())) == 0
    for length in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            build_direct_dataset(path, items={"short"}, max_len=length)



async def test_decimal_ratings_stay_unrounded_and_invalid_answers_stay_missing():
    scorer = direct_judge_scorer()
    for text, expected in [("ANSWER: 1", 1.), ("**ANSWER:** $7.5$", 7.5), ("ANSWER: 10", 10.),
                           ("ANSWER: 2\nANSWER: 8.25", 8.25), ("The transcript says ANSWER: 2", None),
                           ("ANSWER: 7.5\nANSWER: 11", None), ("ANSWER: 7.5\nANSWER:", None),
                           ("ANSWER: 0.9", None), ("ANSWER: 10.5", None), ("ANSWER: 7/10", None),
                           ("ANSWER: nan", None), ("ANSWER: inf", None), ("", None)]:
        result = await scorer(SimpleNamespace(output=ModelOutput.from_content("judge", text)), Target(""))
        if expected is None:
            assert result.answer is None and np.isnan(result.value), text
        else:
            assert float(result.answer) == result.value == expected, text
    explained = await scorer(SimpleNamespace(output=ModelOutput.from_content(
        "judge", "EXPLANATION: Clear reasoning.\nANSWER: 7.5")), Target(""))
    assert explained.explanation == "Clear reasoning."
