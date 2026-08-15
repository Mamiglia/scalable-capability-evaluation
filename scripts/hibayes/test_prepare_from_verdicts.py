"""Orientation guard for the GLM frame: order 1 means the major model is on the left."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_from_verdicts import build

CAPABILITIES = {"strong": 140.0, "weak": 120.0}


def verdicts(**overrides):
    row = dict(
        bench="gpqa", condition="a_baseline", judge="j", major="strong", minor="weak",
        item_id="i1", order=1, picked=1, len_major=100, len_minor=400, parse_ok=True,
    )
    return pd.DataFrame([row | overrides])


def test_order_one_puts_major_on_the_left():
    frame, _ = build(verdicts(order=1, picked=1), CAPABILITIES, "gpqa", ["a_baseline"])
    row = frame.iloc[0]
    assert row["chosen_left"] == 1
    assert (row["left_model"], row["right_model"]) == ("strong", "weak")
    assert (row["left_chars"], row["right_chars"]) == (100, 400)
    assert row["capability_difference"] == 20.0
    assert row["length_difference"] < 0  # left is the shorter transcript


def test_order_two_puts_minor_on_the_left():
    frame, _ = build(verdicts(order=2, picked=1), CAPABILITIES, "gpqa", ["a_baseline"])
    row = frame.iloc[0]
    assert row["chosen_left"] == 1
    assert (row["left_model"], row["right_model"]) == ("weak", "strong")
    assert (row["left_chars"], row["right_chars"]) == (400, 100)
    assert row["capability_difference"] == -20.0
    assert row["length_difference"] > 0


def test_picked_two_is_not_chosen_left():
    frame, _ = build(verdicts(picked=2), CAPABILITIES, "gpqa", ["a_baseline"])
    assert frame.iloc[0]["chosen_left"] == 0


def test_unparsed_verdicts_are_dropped():
    rows = pd.concat([verdicts(), verdicts(picked="", parse_ok=False)], ignore_index=True)
    frame, counts = build(rows, CAPABILITIES, "gpqa", ["a_baseline"])
    assert len(frame) == 1
    assert counts == {"verdicts_selected": 2, "dropped_unparsed_verdict": 1,
                      "dropped_missing_length": 0, "model_rows": 1}


def test_family_and_self_comparison():
    caps = CAPABILITIES | {"strong-v2": 145.0}
    rows = pd.concat(
        [verdicts(minor="strong-v2"), verdicts(minor="strong"), verdicts(len_minor=300)],
        ignore_index=True,
    )
    frame, _ = build(rows, caps, "gpqa", ["a_baseline"])
    frame = frame.set_index("right_model")
    assert frame.loc["strong-v2", "same_model_family"] == 1
    assert frame.loc["strong-v2", "self_comparison"] == 0
    assert frame.loc["strong", "self_comparison"].max() == 1
    assert frame.loc["weak", "same_model_family"] == 0
