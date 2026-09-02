"""The dashboard exporter reads the judge prompt back; this catches template drift."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analyses"))

from export_rank_dashboard import parse_prompt  # noqa: E402
from judge.dataset import Transcript, responses_to_sample  # noqa: E402


def transcript(text, options=()):
    return Transcript(id="q1", question="What is 2 + 2?\n\n---\n\nnot a delimiter", text=text,
                      options=tuple(options))


def test_round_trip_plain():
    # A transcript containing the section delimiter must not truncate the parse.
    major = "step one\n\n---\n\nstep two\nANSWER: 4"
    minor = "ANSWER: 4"
    parsed = parse_prompt(responses_to_sample(transcript(major), transcript(minor)).input)
    assert parsed["question"] == "What is 2 + 2?\n\n---\n\nnot a delimiter"
    assert parsed["transcript_1"] == major
    assert parsed["transcript_2"] == minor
    assert parsed["options_1"] == parsed["options_2"] == []


def test_round_trip_mcq():
    first, second = transcript("ANSWER: A", ["four", "five"]), transcript("ANSWER: B", ["five", "four"])
    parsed = parse_prompt(responses_to_sample(first, second, target="2").input)
    assert parsed["options_1"] == ["four", "five"]
    assert parsed["options_2"] == ["five", "four"]


def order(pick, slot_major, chars, reasoning=""):
    return {"slot_of_major": slot_major, "pick": pick,
            "picked_major": None if pick is None else int(pick == slot_major),
            "confidence": None, "explanation": None, "reasoning": reasoning, "chars": chars}


def test_metrics_match_the_scorer_definitions():
    from export_rank_dashboard import metrics

    # Two items, both orders each. Item A: right in both orders. Item B: always picks
    # Transcript 1 (right in one order, wrong in the other), and 1 is the longer one.
    cells = [
        [order("1", "1", [100, 50]), order("2", "2", [50, 100])],
        [order("1", "1", [100, 50]), order("1", "2", [100, 50])],
    ]
    m = metrics(cells)
    assert m["n"] == 4 and m["answer_rate"] == 1.0
    assert m["accuracy"] == m["score"] == 0.75
    assert m["consistent_accuracy"] == 0.5          # only item A is right both ways
    assert m["disagree_rate"] == 0.5
    assert m["position_bias"] == -0.5               # 3 of 4 verdicts name Transcript 1
    assert m["length_corr"] == 1.0                  # "1 is longer" tracks "picked 1" exactly

    # A missing verdict is wrong for accuracy but excluded from score and answer_rate.
    m = metrics(cells + [[order(None, "1", [10, 10]), None]])
    assert m["answer_rate"] == 0.8 and m["score"] == 0.75 and m["accuracy"] == 0.6
