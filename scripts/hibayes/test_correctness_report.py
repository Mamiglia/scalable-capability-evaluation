"""Guard the position flip in the model-implied quartile table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from correctness_report import loo_accuracy, quartile_pattern

BETA, GAMMA = 0.4, 0.25
PREDICTORS = ["truth_side", "truth_side_x_total_length", "total_length_sd"]


def arm(draws=3):
    """One judge, one item, known coefficients repeated across draws."""
    effects = np.tile(np.array([[[BETA, GAMMA, 0.0]]]), (draws, 1, 1))
    return {
        "tag": "synthetic", "predictors": PREDICTORS, "judges": ["j"], "items": ["i"],
        "judge_intercepts": np.zeros((draws, 1)),
        "item_effects": np.zeros((draws, 1)),
        "judge_predictor_effects": effects,
    }


def frame(lengths):
    rows = []
    for index, length in enumerate(lengths):
        for side in (1.0, -1.0):
            rows.append({"judge": "j", "item": "i", "total_length": float(index),
                         "truth_side": side, "truth_side_x_total_length": side * length,
                         "total_length_sd": length})
    return pd.DataFrame(rows)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_flipping_position_reproduces_the_analytic_difference():
    lengths = [-1.0, -0.5, 0.5, 1.0]
    table = quartile_pattern(arm(), frame(lengths), draws=2, seed=0).set_index("quartile")
    assert list(table["n"]) == [2, 2, 2, 2]
    for label, length in zip(table.index, lengths):
        expected = sigmoid(-(BETA + GAMMA * length)) - sigmoid(BETA + GAMMA * length)
        assert abs(table.loc[label, "mean"] - expected) < 1e-9, label


def test_the_stored_row_orientation_does_not_change_the_answer():
    """Each cell holds both orders, so which one a row was stored as must not matter."""
    lengths = [-1.0, -0.5, 0.5, 1.0]
    upright = quartile_pattern(arm(), frame(lengths), draws=2, seed=0)
    reversed_rows = frame(lengths).iloc[::-1].reset_index(drop=True)
    flipped = quartile_pattern(arm(), reversed_rows, draws=2, seed=0)
    assert np.allclose(sorted(upright["mean"]), sorted(flipped["mean"]))


def test_an_arm_without_position_yields_no_table():
    without = arm()
    without["predictors"] = ["total_length_sd"]
    without["judge_predictor_effects"] = without["judge_predictor_effects"][:, :, 2:]
    assert quartile_pattern(without, frame([-1.0, 1.0]), draws=2, seed=0).empty


def test_loo_accuracy_counts_observations_the_model_got_right():
    # exp(elpd_i) is P(observed outcome); above a half means the observed outcome was favoured
    assert loo_accuracy({"elpd": np.log(np.array([0.9, 0.6, 0.4, 0.2]))}) == 0.5
