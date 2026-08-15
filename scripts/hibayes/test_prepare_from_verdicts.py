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


# --- reasoning filter ----------------------------------------------------


def filter_rows():
    """Five comparisons, enough spread that any one of them can be dropped."""
    return pd.concat(
        [
            verdicts(item_id="i1", major="strong", minor="weak", len_major=100, len_minor=400),
            verdicts(item_id="i2", major="strong", minor="mid", len_major=120, len_minor=300),
            verdicts(item_id="i3", major="mid", minor="strong", len_major=260, len_minor=140),
            verdicts(item_id="i4", major="weak", minor="strong", len_major=380, len_minor=110),
            verdicts(item_id="ballast", major="mid", minor="weak", len_major=250, len_minor=150),
        ],
        ignore_index=True,
    )


FAT = 500  # a transcript comfortably over any floor the tests use


def filter_rates(**overrides):
    """Style rates for every transcript filter_rows() names, all of them fat."""
    lengths = {
        ("strong", "i1"): 100, ("weak", "i1"): 400,
        ("strong", "i2"): 120, ("mid", "i2"): 300,
        ("mid", "i3"): 260, ("strong", "i3"): 140,
        ("weak", "i4"): 380, ("strong", "i4"): 110,
        ("mid", "ballast"): 250, ("weak", "ballast"): 150,
    }
    rows = [
        dict(model=model, item=item, n_chars=n, reasoning_chars=FAT, hedging=n / 50)
        for (model, item), n in lengths.items()
    ]
    return pd.DataFrame([row | overrides.get(row["model"] + row["item"], {}) for row in rows])


def built_filtered(rates, mode, min_chars=100, suffixes=(), rows=None, capabilities=None):
    frame, counts = build(
        filter_rows() if rows is None else rows,
        CAPABILITIES if capabilities is None else capabilities,
        "gpqa",
        ["a_baseline"],
        style_rates=rates,
        reasoning_mode=mode,
        min_reasoning_chars=min_chars,
        effort_suffixes=suffixes,
    )
    return frame.set_index("item"), counts


def test_no_filter_keeps_every_comparison():
    frame, _ = built_filtered(filter_rates(), "none")
    assert sorted(frame.index) == ["ballast", "i1", "i2", "i3", "i4"]


def test_per_transcript_drops_the_thin_comparison_only():
    """One thin transcript takes its own comparison, not its model's others."""
    frame, counts = built_filtered(filter_rates(weaki1={"reasoning_chars": 10}), "per_transcript")
    assert sorted(frame.index) == ["ballast", "i2", "i3", "i4"]
    assert counts["dropped_rows_thin_transcript"] == 1
    assert "weak" in set(frame["left_model"]) | set(frame["right_model"])


def test_per_model_keeps_a_model_whose_median_clears_the_floor():
    """The same thin transcript, judged by the model's median, changes nothing."""
    frame, counts = built_filtered(filter_rates(weaki1={"reasoning_chars": 10}), "per_model")
    assert sorted(frame.index) == ["ballast", "i1", "i2", "i3", "i4"]
    assert counts["dropped_models"] == 0


def test_per_model_drops_every_comparison_of_a_terse_model():
    thin = {"reasoning_chars": 10}
    frame, counts = built_filtered(
        filter_rates(weaki1=thin, weaki4=thin, weakballast=thin), "per_model"
    )
    assert sorted(frame.index) == ["i2", "i3"]
    assert counts["dropped_models"] == 1
    assert counts["dropped_model_names"] == ["weak"]


def test_effort_variants_go_under_any_mode():
    rows = filter_rows().replace({"major": {"mid": "mid-high"}, "minor": {"mid": "mid-high"}})
    rates = filter_rates().replace({"model": {"mid": "mid-high"}})
    frame, counts = built_filtered(
        rates, "none", suffixes=("low", "high"), rows=rows,
        capabilities=CAPABILITIES | {"mid-high": 130.0},
    )
    assert sorted(frame.index) == ["i1", "i4"]
    assert counts["dropped_effort_variants"] == 1


def test_standardising_uses_only_the_surviving_rows():
    """Filtering after standardising would leave the scales set by dropped rows."""
    frame, _ = built_filtered(filter_rates(weaki1={"reasoning_chars": 10}), "per_transcript")
    for column in ("length_difference_sd", "capability_difference_sd", "hedging_difference_sd"):
        assert frame[column].mean() == pytest.approx(0.0, abs=1e-6)
        assert frame[column].std() == pytest.approx(1.0)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown reasoning filter"):
        built_filtered(filter_rates(), "per_item")


def test_a_model_without_transcript_lengths_is_rejected():
    rates = filter_rates()
    with pytest.raises(ValueError, match="No transcript lengths"):
        built_filtered(rates[rates["model"] != "weak"], "per_model")
