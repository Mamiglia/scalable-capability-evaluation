from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "scripts/frontier_eci/rank_regression.py"


def load_module():
    return run_path(MODULE_PATH)


def hard_rows(successes_by_quartile):
    return pd.DataFrame([
        {"bench": "gpqa", "model": model, "target_quartile": quartile,
         "item": f"{quartile}-{item}", "correct": item < successes}
        for quartile, by_model in successes_by_quartile.items()
        for model, successes in by_model.items()
        for item in range(10)
    ])


def test_linear_candidate_is_exact_free_intercept_lstsq_and_uses_all_joined_models():
    module = load_module()
    ability = pd.Series({"a": -1.0, "b": 0.0, "c": 1.0, "frontier": 2.0})
    outcomes = hard_rows({2: {"a": 1, "b": 4, "c": 6, "frontier": 9}})
    fit, points = module["fit_accuracy_candidates"](ability, outcomes)
    x = ability.to_numpy()
    y = np.array([0.1, 0.4, 0.6, 0.9])
    expected = np.linalg.lstsq(np.column_stack([np.ones(4), x]), y, rcond=None)[0]
    assert fit["linear_intercept"] == pytest.approx(expected[0])
    assert fit["linear_slope"] == pytest.approx(expected[1])
    assert fit["n_training_models"] == 4
    assert fit["training_model_ids"] == "a|b|c|frontier"
    assert points.model.tolist() == ["a", "b", "c", "frontier"]


def test_in_sample_frontier_truth_intentionally_changes_fitted_curve():
    module = load_module()
    ability = pd.Series({"a": -1.0, "b": 0.0, "c": 1.0, "frontier": 2.0})
    outcomes = hard_rows({2: {"a": 1, "b": 4, "c": 6, "frontier": 0}})
    first, _ = module["fit_accuracy_candidates"](ability, outcomes)
    changed = outcomes.copy()
    changed.loc[changed.model.eq("frontier"), "correct"] = True
    second, _ = module["fit_accuracy_candidates"](ability, changed)
    assert first["linear_intercept"] != second["linear_intercept"]
    assert first["linear_slope"] != second["linear_slope"]
    assert first["fit_label"] == second["fit_label"] == "Rank–accuracy fit (in-sample)"


@pytest.mark.parametrize(
    ("linear", "linear_rmse", "sigmoid_rmse", "expected"),
    [
        (np.array([-0.01, 0.5]), 0.01, 0.02, "sigmoid"),
        (np.array([0.1, 0.9]), 0.10, 0.09, "sigmoid"),
        (np.array([0.1, 0.9]), 0.09, 0.09, "linear"),
        (np.array([0.1, 0.9]), 0.08, 0.09, "linear"),
    ],
)
def test_method_selection_rule(linear, linear_rmse, sigmoid_rmse, expected):
    select = load_module()["select_method"]
    assert select(linear, linear_rmse, sigmoid_rmse) == expected


def test_rank_regressions_emit_q2_q3_q4_without_eci_or_anchor_inputs():
    module = load_module()
    abilities = pd.DataFrame({
        "bench": ["gpqa"] * 4, "judge": ["judge"] * 4,
        "model": ["a", "b", "c", "frontier"],
        "ability": [-1.0, 0.0, 1.0, 2.0],
    })
    hard = hard_rows({quartile: {"a": 1, "b": 4, "c": 6, "frontier": 9}
                      for quartile in (2, 3, 4)})
    fits, points = module["fit_rank_regressions"](abilities, hard)
    assert fits.target_quartile.tolist() == [2, 3, 4]
    assert points.groupby("target_quartile").size().to_dict() == {2: 4, 3: 4, 4: 4}
    source = MODULE_PATH.read_text().lower()
    assert "epoch-eci" not in source and "eci_mapping" not in source


def test_one_bt_ranking_uses_all_protocol_rows_per_judge_benchmark():
    module = load_module()
    calls = []

    def fake_rank(block):
        calls.append((block.bench.iloc[0], block.judge.iloc[0], len(block)))
        models = sorted(set(block.model_left) | set(block.model_right))
        return pd.DataFrame({
            "model": models, "ability": np.arange(len(models), dtype=float),
            "rank": np.arange(1, len(models) + 1), "n_comparisons": 1,
            "n_wins": 1.0, "n_ties": 0,
        })

    module["rank_all_blocks"].__globals__["rank"] = fake_rank
    protocol = pd.DataFrame({
        "bench": ["gpqa"] * 3 + ["swe_bench_mini"] * 2,
        "judge": ["j1"] * 3 + ["j2"] * 2,
        "item": ["1", "2", "3", "1", "2"],
        "model_left": ["a", "b", "a", "a", "b"],
        "model_right": ["b", "c", "c", "b", "c"],
        "pick_left": [1, 1, 0, 1, 0],
    })
    _, blocks = module["rank_all_blocks"](protocol, pair_limit=None)
    assert calls == [("gpqa", "j1", 3), ("swe_bench_mini", "j2", 2)]
    assert blocks.n_protocol_verdicts.tolist() == [3, 2]


def test_frontier_grid_keeps_actual_when_ability_is_missing():
    module = load_module()
    points = pd.DataFrame({
        "bench": ["gpqa"], "judge": ["judge"], "target_quartile": [2],
        "model": ["frontier-a"], "actual": [0.7], "n_items": [10],
        "ability": [1.0], "selected_fitted": [0.6], "selected_method": ["linear"],
    })
    actual = pd.DataFrame({
        "bench": ["gpqa", "gpqa"], "target_quartile": [2, 2],
        "model": ["frontier-a", "frontier-b"], "actual": [0.7, 0.8],
        "n_items": [10, 10],
    })
    grid = module["frontier_grid"](
        points, actual, ["frontier-a", "frontier-b"], ["judge"],
        {"frontier-a": "2025-01-01", "frontier-b": "2025-02-01",
         "judge": "2024-12-01"}, benches=["gpqa"], quartiles=(2,),
    )
    missing = grid[grid.model.eq("frontier-b")].iloc[0]
    assert len(grid) == 2 and missing.actual == 0.8
    assert pd.isna(missing.fitted)
    assert missing.fit_status == "missing_ability"
