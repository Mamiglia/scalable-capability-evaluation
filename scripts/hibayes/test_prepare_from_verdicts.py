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


@pytest.mark.parametrize(
    "order,left,right,left_chars",
    # order="1" is the case that discriminates: an uncoerced `"1" == 1` is False, so
    # the minor model silently goes left. order="2" agrees either way, so on its own
    # it cannot tell the fixed code from the broken code.
    [("1", "strong", "weak", 100), ("2", "weak", "strong", 400)],
)
def test_string_order_maps_the_same_as_integer_order(order, left, right, left_chars):
    """A re-export writing "1"/"2" must not put the minor model on the left."""
    row = built(verdicts(order=order, picked=1))[0].loc["i1"]
    assert (row["left_model"], row["right_model"]) == (left, right)
    assert row["left_chars"] == left_chars


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


def style_rates(**overrides):
    """Rates for both transcripts of every verdicts()/ballast() row."""
    rows = [
        dict(model="strong", item="i1", n_chars=100, hedging=8.0),
        dict(model="weak", item="i1", n_chars=400, hedging=2.0),
        dict(model="mid", item="ballast", n_chars=250, hedging=5.0),
        dict(model="weak", item="ballast", n_chars=150, hedging=1.0),
    ]
    return pd.DataFrame([row | overrides.get(row["model"] + row["item"], {}) for row in rows])


def built_with_style(rates, *frames):
    rows = pd.concat([*frames, ballast()], ignore_index=True)
    frame, _ = build(rows, CAPABILITIES, "gpqa", ["a_baseline"], style_rates=rates)
    return frame.set_index("item")


def test_style_difference_is_left_minus_right_and_is_standardised():
    frame = built_with_style(style_rates(), verdicts(order=1))
    assert frame.loc["i1", "hedging_difference"] == 6.0  # strong 8.0 left - weak 2.0 right
    assert frame.loc["ballast", "hedging_difference"] == 4.0
    # Standardised against the frame's own spread, like every other predictor.
    assert frame["hedging_difference_sd"].mean() == pytest.approx(0.0, abs=1e-12)
    assert frame["hedging_difference_sd"].std() == pytest.approx(1.0)


def test_style_difference_follows_the_order_flip():
    frame = built_with_style(style_rates(), verdicts(order=2))
    assert frame.loc["i1", "hedging_difference"] == -6.0  # weak now on the left


def test_style_rates_measured_on_different_text_are_rejected():
    """n_chars is the guard that the rates describe what the judge actually saw."""
    with pytest.raises(ValueError, match="differ in length"):
        built_with_style(style_rates(strongi1={"n_chars": 999}), verdicts())


def test_missing_style_rates_are_rejected():
    rates = style_rates()
    with pytest.raises(ValueError, match="No style rates"):
        built_with_style(rates[rates["model"] != "strong"], verdicts())


def test_style_rates_without_categories_are_rejected():
    rates = style_rates().drop(columns=["hedging"])
    with pytest.raises(ValueError, match="no category columns"):
        built_with_style(rates, verdicts())
