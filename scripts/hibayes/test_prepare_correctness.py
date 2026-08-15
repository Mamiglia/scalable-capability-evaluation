"""Orientation guard for the correctness frame: differences run stronger-minus-weaker."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prepare_correctness import difference_columns, to_correctness
from prepare_from_verdicts import build

CAPABILITIES = {"strong": 140.0, "mid": 130.0, "weak": 120.0}
# item i1: the stronger model wrote the shorter, less hedged transcript, so every
# re-oriented difference below is negative whichever side it was presented on.
LENGTHS = {("strong", "i1"): 100, ("weak", "i1"): 400, ("mid", "ballast"): 250, ("weak", "ballast"): 150}
HEDGING = {("strong", "i1"): 1.0, ("weak", "i1"): 5.0, ("mid", "ballast"): 3.0, ("weak", "ballast"): 2.0}


def verdicts(**overrides):
    row = dict(
        bench="gpqa", condition="a_baseline", judge="j", major="strong", minor="weak",
        item_id="i1", order=1, picked=1, len_major=100, len_minor=400, parse_ok=True,
    )
    row |= overrides
    row["len_major"] = LENGTHS[(row["major"], row["item_id"])]
    row["len_minor"] = LENGTHS[(row["minor"], row["item_id"])]
    return pd.DataFrame([row])


def ballast():
    """A row whose gaps differ from every case below, so standardising has spread."""
    return verdicts(item_id="ballast", major="mid", minor="weak")


def style_rates():
    return pd.DataFrame(
        [{"model": model, "item": item, "n_chars": LENGTHS[(model, item)], "hedging": rate}
         for (model, item), rate in HEDGING.items()]
    )


def built(*frames, capabilities=CAPABILITIES):
    rows = pd.concat([*frames, ballast()], ignore_index=True)
    rates = style_rates()
    frame, _ = build(rows, capabilities, "gpqa", ["a_baseline"], style_rates=rates)
    frame, predictors = to_correctness(frame, difference_columns(rates))
    return frame.set_index("item"), predictors


@pytest.mark.parametrize("order,picked,truth_side,is_correct", [
    (1, 1, 1.0, 1), (1, 2, 1.0, 0), (2, 2, -1.0, 1), (2, 1, -1.0, 0),
])
def test_truth_side_and_correctness_follow_the_presented_order(order, picked, truth_side, is_correct):
    row = built(verdicts(order=order, picked=picked))[0].loc["i1"]
    assert row["truth_side"] == truth_side
    assert row["is_correct"] == is_correct


def test_re_oriented_differences_do_not_move_when_the_order_flips():
    first = built(verdicts(order=1))[0].loc["i1"]
    second = built(verdicts(order=2))[0].loc["i1"]
    for column in ("length_difference_stronger", "hedging_difference_stronger",
                   "capability_gap", "total_length"):
        assert first[column] == pytest.approx(second[column]), column
    # and the left-minus-right originals do flip, which is why they are not used
    assert first["length_difference"] == pytest.approx(-second["length_difference"])
    assert first["hedging_difference"] == pytest.approx(-second["hedging_difference"])


def test_re_oriented_differences_are_stronger_minus_weaker():
    row = built(verdicts(order=2))[0].loc["i1"]
    # strong wrote 100 chars and hedged at 1.0; weak wrote 400 and hedged at 5.0
    assert row["length_difference_stronger"] < 0
    assert row["hedging_difference_stronger"] == pytest.approx(1.0 - 5.0)


def test_capability_gap_is_absolute_and_total_length_is_the_sum():
    row = built(verdicts(order=2))[0].loc["i1"]
    assert row["capability_difference"] == -20.0
    assert row["capability_gap"] == 20.0
    assert row["total_length"] == pytest.approx(math.log1p(100 + 400))


def test_the_interaction_is_truth_side_times_standardised_total_length():
    frame, predictors = built(verdicts(order=2))
    assert (frame["truth_side_x_total_length"] == frame["truth_side"] * frame["total_length_sd"]).all()
    assert "truth_side_x_total_length" in predictors


def test_left_minus_right_standardised_columns_are_dropped():
    frame, predictors = built(verdicts())
    for column in ("length_difference_sd", "hedging_difference_sd", "capability_difference_sd"):
        assert column not in frame.columns
    assert not any(predictor.startswith("capability_difference") for predictor in predictors)


def test_a_major_model_that_is_not_the_stronger_one_is_rejected():
    flipped = CAPABILITIES | {"strong": 100.0}  # major "strong" now scores below "weak"
    with pytest.raises(ValueError, match="not the higher-capability one"):
        built(verdicts(), capabilities=flipped)
