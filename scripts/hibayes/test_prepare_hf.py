"""HF GPQA frame: orientation, MBPP exclusion, and predictor-order consistency."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from prepare_hf import (
    PREDICTORS,
    STANDARDISED,
    STYLE_CATEGORIES,
    discover_gpqa_dirs,
    row_from_sample,
)
from style_features import load_lexicon

LEXICON = load_lexicon(Path(__file__).with_name("style_lexicon.yaml"))
CONFIG = yaml.safe_load(Path(__file__).with_name("config_hf.yaml").read_text())
CAPABILITIES = {("GPQA", "strong"): 0.8, ("GPQA", "weak"): 0.3}


def make_sample(t1, t2, target, answer, item="item-7_2"):
    prompt = (
        f"Input question:\n---\nQuestion?\n---\n\n"
        f"Transcript 1:\n---\n{t1}\n---\n\n"
        f"Transcript 2:\n---\n{t2}\n---\n\n"
        "Which of the two transcripts is from the more capable model?\n"
    )
    return SimpleNamespace(
        id=item, input=prompt, target=target,
        scores={"judge_scorer": SimpleNamespace(answer=answer)},
    )


META = {
    "major_path": "data/gpqa/hash/strong/0/run.eval",
    "minor_path": "data/gpqa/hash/weak/0/run.eval",
}


def test_row_uses_shown_transcripts_and_prompt_position():
    # strong is major (target=2 -> slot 2), weak is left (slot 1). left minus right.
    sample = make_sample("I think we might, perhaps.", "definitely obvious", "2", "1")
    row = row_from_sample(sample, META, "judge-a", CAPABILITIES, LEXICON, epoch="0-0")

    assert row["chosen_left"] == 1
    assert row["item"] == "gpqa:item-7"
    assert row["transcript_condition"] == "hf_gpqa"
    assert row["bench"] == "gpqa"
    assert (row["left_model"], row["right_model"]) == ("weak", "strong")
    assert row["model_pair"] == "strong|weak"
    assert row["left_capability"] == 0.3 and row["right_capability"] == 0.8
    assert row["capability_difference"] == pytest.approx(-0.5)
    assert row["same_model_family"] == 0 and row["self_comparison"] == 0
    # left has pronouns + hedging, right has confidence: signs follow left-minus-right.
    assert row["personal_pronouns_difference"] > 0
    assert row["hedging_difference"] > 0
    assert row["confidence_difference"] < 0


def test_swapping_slots_negates_every_difference_and_flips_choice():
    left_text, right_text = "I think we might, perhaps.", "definitely obvious, certainly"
    original = row_from_sample(
        make_sample(left_text, right_text, "2", "1"), META, "j", CAPABILITIES, LEXICON
    )
    # Swap the two transcripts, the major slot, and the picked slot together.
    swapped = row_from_sample(
        make_sample(right_text, left_text, "1", "2"), META, "j", CAPABILITIES, LEXICON
    )

    assert swapped["chosen_left"] == 1 - original["chosen_left"]
    difference_columns = ["length_difference", "capability_difference"] + [
        f"{category}_difference" for category in STYLE_CATEGORIES
    ]
    for column in difference_columns:
        assert swapped[column] == pytest.approx(-original[column]), column


def test_mbpp_directories_contribute_zero_rows(tmp_path):
    raw = tmp_path / "raw"
    for name in ("pilot-judge-a-gpqa-strong-vs-weak", "pilot-judge-a-mbpp-strong-vs-weak"):
        (raw / name).mkdir(parents=True)
        (raw / name / "logs.json").write_text("{}")
    discovered = [path.parent.name for path in discover_gpqa_dirs(raw)]
    assert discovered == ["pilot-judge-a-gpqa-strong-vs-weak"]


def test_ten_predictors_are_consistent_across_config_and_preprocessing():
    expected = list(PREDICTORS)
    assert len(expected) == 10
    # A style difference feeds each _sd style predictor, in the same order.
    assert [f"{c}_difference_sd" for c in STYLE_CATEGORIES] == expected[4:]
    assert [f"{c}_sd" for c in STANDARDISED] == [
        "length_difference_sd", "capability_difference_sd",
        *(f"{c}_difference_sd" for c in STYLE_CATEGORIES),
    ]
    # Same list, same order, in every place config_hf.yaml names the predictors.
    model_config = CONFIG["model"]["models"][0]["config"]["predictors"]
    continuous = CONFIG["data_process"]["processors"][1]["extract_features"]["continuous_features"]
    communicate = CONFIG["communicate"]["communicators"][0]["judge_feature_report"]["predictors"]
    assert model_config == expected
    assert continuous == expected
    assert communicate == expected
