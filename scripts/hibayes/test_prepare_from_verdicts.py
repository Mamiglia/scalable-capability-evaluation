"""Orientation guard for the GLM frame: order 1 means the major model is on the left."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prepare_from_verdicts import build

CAPABILITIES = {"strong": 140.0, "mid": 130.0, "weak": 120.0}


def verdicts(**overrides):
    row = dict(
        bench="gpqa", condition="a_baseline", judge="j", major="strong", minor="weak",
        item_id="i1", order=1, picked=1, len_major=100, len_minor=400, parse_ok=True,
    )
    return pd.DataFrame([row | overrides])


def ballast():
    """A row whose length and capability gaps differ from every case below."""
    return verdicts(item_id="ballast", major="mid", minor="weak",
                    len_major=250, len_minor=150)


def built(*frames):
    """Standardising needs spread, so every case carries one contrasting row."""
    rows = pd.concat([*frames, ballast()], ignore_index=True)
    frame, counts = build(rows, CAPABILITIES, "gpqa", ["a_baseline"])
    return frame.set_index("item"), counts


def test_order_one_puts_major_on_the_left():
    row = built(verdicts(order=1, picked=1))[0].loc["i1"]
    assert row["chosen_left"] == 1
    assert (row["left_model"], row["right_model"]) == ("strong", "weak")
    assert (row["left_chars"], row["right_chars"]) == (100, 400)
    assert row["capability_difference"] == 20.0
    assert row["length_difference"] < 0  # left is the shorter transcript


def test_order_two_puts_minor_on_the_left():
    row = built(verdicts(order=2, picked=1))[0].loc["i1"]
    assert row["chosen_left"] == 1
    assert (row["left_model"], row["right_model"]) == ("weak", "strong")
    assert (row["left_chars"], row["right_chars"]) == (400, 100)
    assert row["capability_difference"] == -20.0
    assert row["length_difference"] > 0


def test_string_order_maps_the_same_as_integer_order():
    """A re-export writing "1"/"2" must not flip every row to minor-on-the-left."""
    numeric = built(verdicts(order=2, picked=1))[0].loc["i1"]
    text = built(verdicts(order="2", picked=1))[0].loc["i1"]
    assert (text["left_model"], text["right_model"]) == ("weak", "strong")
    assert text["left_chars"] == numeric["left_chars"]
    assert text["capability_difference"] == numeric["capability_difference"]


def test_order_outside_one_and_two_is_rejected():
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        built(verdicts(order=3))


def test_picked_two_is_not_chosen_left():
    assert built(verdicts(picked=2))[0].loc["i1", "chosen_left"] == 0


def test_unparsed_verdicts_are_dropped():
    _, counts = built(verdicts(), verdicts(item_id="i2", picked="", parse_ok=False))
    assert counts == {"verdicts_selected": 3, "dropped_unparsed_verdict": 1,
                      "dropped_missing_length": 0, "model_rows": 2}


def test_family_and_self_comparison():
    caps = CAPABILITIES | {"strong-v2": 145.0}
    rows = pd.concat(
        [verdicts(item_id="family", minor="strong-v2"),
         verdicts(item_id="self", minor="strong"),
         verdicts(item_id="other", len_minor=300)],
        ignore_index=True,
    )
    frame, _ = build(pd.concat([rows, ballast()], ignore_index=True), caps, "gpqa", ["a_baseline"])
    frame = frame.set_index("item")
    assert frame.loc["family", "same_model_family"] == 1
    assert frame.loc["family", "self_comparison"] == 0
    assert frame.loc["self", "self_comparison"] == 1
    assert frame.loc["other", "same_model_family"] == 0


def test_a_single_row_cannot_be_standardised():
    """Without spread the scale is NaN, which must raise rather than emit NaN columns."""
    with pytest.raises(ValueError, match="not finite"):
        build(verdicts(), CAPABILITIES, "gpqa", ["a_baseline"])


def test_pick_the_weaker_condition_is_rejected():
    with pytest.raises(ValueError, match="weaker model"):
        build(verdicts(condition="d_pick_weaker"), CAPABILITIES, "gpqa", ["d_pick_weaker"])


def test_reasoning_condition_is_rejected():
    with pytest.raises(ValueError, match="final-text-only"):
        build(verdicts(condition="e_with_reasoning"), CAPABILITIES, "gpqa", ["e_with_reasoning"])


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="cannot verify"):
        build(verdicts(), CAPABILITIES, "gpqa", ["not_a_condition"])
