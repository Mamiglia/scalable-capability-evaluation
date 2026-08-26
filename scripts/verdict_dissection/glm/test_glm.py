import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

sys.path.insert(0, str(Path(__file__).parent))
import model as glm_models  # noqa: E402
from prepare import SPLITS, capability_from_transcript, held_out, judge_verdict_from_pair, stronger_model_from_pair  # noqa: E402

PREDICTORS = ["log_chars", "sae_shared_0"]
HELD_OUT_MODEL = SPLITS["model_designs"]["model-fold-0"][0]


def _pair():
    return pd.DataFrame({
        "bench": ["gpqa"] * 4, "item": ["x", "x", "x", "y"], "judge": ["j1", "j2", "j1", "j1"],
        "model_left": ["a", "a", "b", HELD_OUT_MODEL], "model_right": ["b", "b", "a", "c"],
        "epoch_left": 0, "epoch_right": 0, "log_left": "l", "log_right": "r",
        "pick_left": [1, 0, 0, 1], "accuracy_left": [0.8, 0.8, 0.6, 0.5], "accuracy_right": [0.6, 0.6, 0.8, 0.5],
        "d_log_chars": [1.0, 1.0, -1.0, 0.5], "d_sae_shared_0": [0.3, 0.3, -0.3, 0.1],
    })


def test_held_out_marks_model_folds_and_strict_folds():
    bench = pd.Series(["gpqa"] * 4)
    item = pd.Series(["i0", "i1", "i0", "i1"])                   # sorted items i0, i1 -> folds 0, 1
    left = pd.Series(["a", "a", HELD_OUT_MODEL, HELD_OUT_MODEL])
    right = pd.Series(["b"] * 4)
    assert held_out("model-fold-0", bench, item, left, right).tolist() == ["train", "train", "test", "test"]
    # strict fold 0: unseen model on unseen item = test, seen model on seen item = train, mixed rows dropped
    assert held_out("new-models-and-items-0", bench, item, left, right).tolist() == ["drop", "train", "test", "drop"]


def test_stronger_model_rows_drop_judges_and_ties():
    rows = stronger_model_from_pair(_pair(), PREDICTORS, "model-fold-0")
    assert len(rows) == 2 and rows.left_stronger.tolist() == [1, 0] and rows.role.tolist() == ["train", "train"]
    assert list(rows.columns[-2:]) == PREDICTORS


def test_judge_verdict_rows_keep_every_draw_and_the_gap():
    rows = judge_verdict_from_pair(_pair(), PREDICTORS, "model-fold-0")
    assert len(rows) == 4 and rows.role.tolist() == ["train", "train", "train", "test"]
    assert rows.accuracy_difference.round(2).tolist() == [0.2, 0.2, -0.2, 0.0]


def test_capability_rows_take_logit_of_clipped_accuracy():
    transcript = pd.DataFrame({"bench": ["gpqa"], "item": ["x"], "model": ["a"], "accuracy": [1.0],
                               "log_chars": [0.1], "sae_shared_0": [0.2]})
    rows = capability_from_transcript(transcript, PREDICTORS, "model-fold-0")
    assert rows.logit_model_accuracy.iloc[0] == pytest.approx(np.log(0.99 / 0.01)) and rows.role.iloc[0] == "train"


def test_models_trace_on_fake_features():
    n = 6
    features = {"obs": jnp.array([1, 0, 1, 0, 1, 0]), "bench_index": jnp.array([0, 0, 1, 1, 2, 2]),
                "num_bench": 3, "item_index": jnp.arange(n), "num_item": n, "judge_index": jnp.array([0, 1] * 3),
                "num_judge": 2, "accuracy_difference": jnp.linspace(-1, 1, n),
                "log_chars": jnp.ones(n), "sae_shared_0": jnp.zeros(n)}
    for build in (glm_models.stronger_model_from_pair, glm_models.judge_verdict_from_pair):
        trace = handlers.trace(handlers.seed(build(predictors=PREDICTORS), 0)).get_trace(features)
        assert trace["obs"]["value"].shape == (n,)
    trace = handlers.trace(handlers.seed(glm_models.judge_verdict_from_pair(predictors=PREDICTORS, use_accuracy_gap=True), 0)).get_trace(features)
    assert trace["judge_accuracy_gap_slopes"]["value"].shape == (2,)
    assert trace["judge_position_bias"]["value"].shape == (2, 3)          # judge x benchmark
    trace = handlers.trace(handlers.seed(glm_models.capability_from_transcript(predictors=PREDICTORS), 0)).get_trace(
        features | {"obs": jnp.linspace(-2, 2, n)})
    assert trace["benchmark_slopes"]["value"].shape == (3, 2)
