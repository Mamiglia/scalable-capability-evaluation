from pathlib import Path
from runpy import run_path

import pandas as pd


def load_module():
    return run_path(Path(__file__).parents[2] / "scripts/reasoning_ablation/analyze.py")


def test_match_conditions_pairs_exact_display_orders_and_marks_flips():
    off = pd.DataFrame([
        {"bench": "gpqa", "judge": "j", "item": "i", "model_left": "a",
         "model_right": "b", "epoch_left": 0, "epoch_right": 0, "judge_epoch": 1,
         "pick_left": 1},
        {"bench": "gpqa", "judge": "j", "item": "i", "model_left": "b",
         "model_right": "a", "epoch_left": 0, "epoch_right": 0, "judge_epoch": 1,
         "pick_left": 0},
    ])
    cot = off.copy()
    cot.loc[0, "pick_left"] = 0
    matched = load_module()["match_conditions"](off, cot)
    assert len(matched) == 2
    assert matched.flipped.tolist() == [True, False]
    assert matched.pick_left_off.tolist() == [1, 0]
    assert matched.pick_left_cot.tolist() == [0, 0]


def test_non_reasoning_figure_rotates_model_labels(tmp_path):
    module = load_module()
    points = pd.DataFrame({
        "bench": ["gpqa"] * 4,
        "judge": ["gpt-5.2"] * 4,
        "model": ["gpt-4o-1120", "gpt-4.1", "claude-3.7-sonnet", "mistral-medium-3"],
        "actual": [0.4, 0.5, 0.6, 0.7],
        "selected_fitted": [0.45, 0.48, 0.58, 0.68],
        "fitted_low": [0.3, 0.4, 0.5, 0.6],
        "fitted_high": [0.6, 0.6, 0.7, 0.8],
    })
    figures = []
    module["write_non_reasoning_figures"].__globals__["save"] = (
        lambda figure, path: figures.append(figure))
    module["write_non_reasoning_figures"](points, tmp_path)
    labels = figures[0].axes[0].get_xticklabels()
    assert {label.get_rotation() for label in labels} == {45.0}
    assert {label.get_ha() for label in labels} == {"right"}
