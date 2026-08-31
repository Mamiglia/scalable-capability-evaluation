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
