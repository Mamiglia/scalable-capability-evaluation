from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd


module = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/analyze.py")
calibrate = module["calibrate"]
merge_rollouts = module["merge_rollouts"]
prefix_verdicts = module["prefix_verdicts"]
consensus_verdicts = module["consensus_verdicts"]
predict_block = module["predict_block"]


def test_target_actual_does_not_enter_calibration():
    ability = pd.Series({"a": -2.0, "b": -0.5, "c": 0.5, "d": 2.0,
                         "frontier-a": 1.0, "frontier-b": 1.5})
    actual = pd.Series({"a": 0.1, "b": 0.3, "c": 0.6, "d": 0.9,
                        "frontier-a": 0.0, "frontier-b": 1.0})
    first, anchors = calibrate(ability, actual, ["a", "b", "c", "d"], ["frontier-a"])
    changed = actual.copy()
    changed[["frontier-a", "frontier-b"]] = [1.0, 0.0]
    second, _ = calibrate(ability, changed, ["a", "b", "c", "d"], ["frontier-a"])
    pd.testing.assert_series_equal(first, second)
    assert anchors == 4


def test_future_edges_are_absent_from_an_earlier_prefix():
    verdicts = pd.DataFrame({
        "model_left": ["past", "target"], "model_right": ["target", "future"],
        "pick_left": [0, 1], "bench": ["gpqa", "gpqa"],
        "judge": ["j", "j"], "item": ["x", "x"],
    })
    releases = {"past": "2024-01-01", "target": "2025-01-01", "future": "2026-01-01"}
    prefix = prefix_verdicts(verdicts, "target", releases)
    assert len(prefix) == 1
    assert set(prefix.iloc[0][["model_left", "model_right"]]) == {"past", "target"}


def test_consensus_weights_judges_not_uploaded_rounds():
    rows = pd.DataFrame({
        "bench": ["gpqa"] * 3, "item": ["x"] * 3,
        "model_left": ["a"] * 3, "model_right": ["b"] * 3,
        "epoch_left": [0] * 3, "epoch_right": [0] * 3,
        "judge": ["many", "many", "one"], "judge_epoch": [1, 2, 1],
        "pick_left": [0, 0, 1],
    })
    assert consensus_verdicts(rows).iloc[0].pick_left == 0.5


def test_empty_earliest_prefix_is_actual_only():
    row = predict_block(pd.DataFrame(), pd.Series({"first": 0.7}), "first",
                        {"first": "2024-01-01"}, 10, np.random.default_rng(0))
    assert row["actual"] == 0.7
    assert pd.isna(row["predicted"])
    assert row["n_prefix_verdicts"] == 0


def test_canonical_rollouts_win_when_model_is_already_present():
    canonical = pd.DataFrame({"bench": ["gpqa"], "model": ["a"], "item": ["x"],
                              "epoch": [0], "correct": [True], "is_canonical": [True]})
    frontier = pd.DataFrame({"bench": ["gpqa", "gpqa"], "model": ["a", "b"],
                             "item": ["x", "x"], "epoch": [0, 0],
                             "correct": [False, False], "is_canonical": [True, True]})
    merged = merge_rollouts(canonical, frontier)
    assert merged.set_index("model").correct.to_dict() == {"a": True, "b": False}
