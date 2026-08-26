import numpy as np
import pandas as pd
import pytest

from build_design_matrices import (KEYS, block_columns, leave_item_out_accuracy, pair_differences,
                                   scale, transform)


def _transcripts():
    rows = []
    for bench in ("gpqa", "swe"):
        for model, correct in (("a", [1, 1, 0]), ("b", [1, 0, 0])):
            for item, c in zip("xyz", correct):
                rows.append(dict(bench=bench, model=model, item=item, epoch=0, log=f"{model}-{item}",
                                 sample_epoch=1, content_hash=f"{bench}{model}{item}", correct=c,
                                 is_canonical=True, n_words=10.0 if model == "a" else 100.0,
                                 surprisal_full_mean__llama=1.0 if model == "a" else 3.0,
                                 surprisal_question_gain_sum__llama=-2.0))
    return pd.DataFrame(rows)


def test_transform_logs_extensive_nonnegative_columns_only():
    frame = _transcripts()
    out, dictionary = transform(frame, ["n_words", "surprisal_full_mean__llama",
                                        "surprisal_question_gain_sum__llama"])
    assert np.allclose(out.n_words, np.log1p(frame.n_words))
    assert (out.surprisal_full_mean__llama == frame.surprisal_full_mean__llama).all()
    assert (out.surprisal_question_gain_sum__llama == frame.surprisal_question_gain_sum__llama).all()
    assert dictionary.set_index("feature")["transform"].to_dict() == {
        "n_words": "log1p", "surprisal_full_mean__llama": "none",
        "surprisal_question_gain_sum__llama": "none"}


def test_leave_item_out_accuracy_uses_development_items_excluding_own():
    identity = _transcripts()
    split = pd.DataFrame({"bench": ["gpqa"] * 3 + ["swe"] * 3, "item": list("xyz") * 2,
                          "split": ["representation_train", "representation_validation", "confirmation"] * 2})
    acc = leave_item_out_accuracy(identity, split)
    row = acc.set_index(["bench", "model", "item"])
    assert row.loc[("gpqa", "a", "x"), "model_accuracy"] == 1.0     # dev items x,y minus own x -> y correct
    assert row.loc[("gpqa", "a", "z"), "model_accuracy"] == 1.0     # z is confirmation: x,y both correct
    assert row.loc[("gpqa", "b", "x"), "model_accuracy"] == 0.0
    assert row.loc[("gpqa", "b", "z"), "model_accuracy"] == 0.5


def test_pair_differences_are_antisymmetric():
    frame = _transcripts()
    pairs = pd.DataFrame([
        dict(bench="gpqa", item="x", model_left="a", model_right="b", epoch_left=0, epoch_right=0,
             log_left="a-x", log_right="b-x"),
        dict(bench="gpqa", item="x", model_left="b", model_right="a", epoch_left=0, epoch_right=0,
             log_left="b-x", log_right="a-x"),
    ])
    diff = pair_differences(pairs, frame, ["n_words", "surprisal_full_mean__llama"])
    assert list(diff.columns) == ["d_n_words", "d_surprisal_full_mean__llama"]
    assert np.allclose(diff.iloc[0].to_numpy(), -diff.iloc[1].to_numpy())
    assert diff.iloc[0].d_n_words == -90.0


def test_scale_uses_training_rows_and_fills_missing_with_zero():
    frame = pd.DataFrame({"bench": ["gpqa"] * 4, "split": ["representation_train"] * 2 + ["confirmation"] * 2,
                          "f": [1.0, 3.0, 5.0, np.nan]})
    out, params = scale(frame, ["f"], centre=True)
    assert params.iloc[0][["mean", "sd"]].tolist() == [2.0, pytest.approx(np.sqrt(2))]
    assert out.f.tolist() == pytest.approx([-1 / np.sqrt(2), 1 / np.sqrt(2), 3 / np.sqrt(2), 0.0])
    out, params = scale(frame, ["f"], centre=False)
    assert params.iloc[0]["mean"] == 0.0


def test_block_columns_match_regex_and_drop_keys():
    frame = _transcripts()
    assert block_columns(frame, "__llama$") == ["surprisal_full_mean__llama",
                                                 "surprisal_question_gain_sum__llama"]
    assert KEYS[0] == "bench"


def test_pool_aime_merges_years_and_keeps_items_unique():
    from build_design_matrices import pool_aime
    frame = pd.DataFrame({"bench": ["aime2024", "aime2025", "gpqa"], "item": [0, 0, "x"]})
    pooled = pool_aime(frame)
    assert pooled.bench.tolist() == ["aime", "aime", "gpqa"]
    assert pooled.item.tolist() == ["aime2024/0", "aime2025/0", "x"]
