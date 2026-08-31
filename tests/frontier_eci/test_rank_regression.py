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


def test_combined_hard_fit_uses_q2_q3_q4_as_one_target():
    module = load_module()
    abilities = pd.DataFrame({
        "bench": ["gpqa"] * 4, "judge": ["judge"] * 4,
        "model": ["a", "b", "c", "frontier"],
        "ability": [-1.0, 0.0, 1.0, 2.0],
    })
    hard = hard_rows({
        1: {"a": 10, "b": 10, "c": 10, "frontier": 10},
        2: {"a": 1, "b": 4, "c": 6, "frontier": 9},
        3: {"a": 2, "b": 5, "c": 7, "frontier": 8},
        4: {"a": 3, "b": 6, "c": 8, "frontier": 7},
    })
    fits, points = module["fit_combined_rank_regressions"](abilities, hard)
    assert len(fits) == 1
    assert fits.iloc[0].target_slice == "Q2–Q4"
    assert set(points.model) == {"a", "b", "c", "frontier"}
    assert points.set_index("model").loc["a", "actual"] == pytest.approx(0.2)
    assert points.set_index("model").loc["frontier", "n_items"] == 30


def test_two_source_item_bootstrap_exports_nonzero_combined_intervals():
    module = load_module()
    rows = []
    for item in range(12):
        for left, right, pick in (("a", "b", item % 3 != 0),
                                  ("b", "a", item % 3 == 0),
                                  ("b", "frontier", item % 4 != 0),
                                  ("frontier", "b", item % 4 == 0),
                                  ("a", "frontier", item % 5 == 0),
                                  ("frontier", "a", item % 5 != 0)):
            rows.append({"bench": "gpqa", "judge": "judge", "item": str(item),
                         "model_left": left, "model_right": right,
                         "pick_left": int(pick)})
    protocol = pd.DataFrame(rows)
    hard = hard_rows({
        2: {"a": 2, "b": 5, "frontier": 9},
        3: {"a": 3, "b": 6, "frontier": 8},
        4: {"a": 1, "b": 7, "frontier": 7},
    })
    _, _, fits, points = module["fit_combined_with_bootstrap"](
        protocol, hard, resamples=30, seed=7, workers=1)
    frontier = points.set_index("model").loc["frontier"]
    assert fits.iloc[0].n_bootstrap == 30
    assert frontier.fitted_low <= frontier.selected_fitted <= frontier.fitted_high
    assert frontier.fitted_high > frontier.fitted_low


def test_main_manifest_declares_combined_q2_q4_item_bootstrap():
    manifest = load_module()["MAIN_OUTPUT_MANIFEST"]
    assert manifest["target"] == "combined Q2–Q4 accuracy"
    assert manifest["uncertainty"] == "95% two-source item bootstrap"


def test_time_trend_control_fits_only_strictly_pre_judge_frontier_scores():
    actual = pd.DataFrame({
        "bench": ["gpqa"] * 5,
        "model": ["old-a", "pool-old", "old-b", "at-cutoff", "later"],
        "actual": [0.2, 0.9, 0.3, 0.4, 0.5],
        "n_items": [100] * 5,
    })
    releases = {
        "old-a": "2024-01-01", "pool-old": "2024-01-06", "old-b": "2024-01-11",
        "judge": "2024-01-21", "at-cutoff": "2024-01-21",
        "later": "2024-01-31",
    }
    fits, points = load_module()["fit_time_trend_controls"](
        actual, ["old-a", "old-b", "at-cutoff", "later"], ["judge"], releases,
        resamples=40, seed=7,
    )
    fit = fits.iloc[0]
    assert fit.training_model_ids == "old-a|old-b"
    assert fit.n_training_models == 2
    indexed = points.set_index("model")
    assert not bool(indexed.loc["at-cutoff", "is_fit_point"])
    assert indexed.loc["at-cutoff", "predicted"] == pytest.approx(0.4)
    assert indexed.loc["later", "predicted"] == pytest.approx(0.5)
    assert indexed.loc["later", "predicted_low"] <= 0.5 <= indexed.loc["later", "predicted_high"]


def test_prediction_comparison_joins_same_frontier_cells():
    judge = pd.DataFrame({
        "bench": ["gpqa", "gpqa"], "judge": ["j", "j"],
        "model": ["a", "b"], "actual": [0.2, 0.4], "n_items": [10, 10],
        "target_release": ["2025-01-01", "2025-04-01"],
        "judge_release": ["2025-02-01", "2025-02-01"],
        "fitted": [0.25, 0.45], "fitted_low": [0.15, 0.35],
        "fitted_high": [0.35, 0.55],
    })
    trend = pd.DataFrame({
        "bench": ["gpqa", "gpqa"], "judge": ["j", "j"],
        "model": ["b", "a"], "actual": [0.4, 0.2],
        "target_release": ["2025-04-01", "2025-01-01"],
        "judge_release": ["2025-02-01", "2025-02-01"],
        "predicted": [0.5, 0.3], "predicted_low": [0.4, 0.2],
        "predicted_high": [0.6, 0.4],
    })
    comparison = load_module()["merge_prediction_comparison"](judge, trend)
    assert comparison.model.tolist() == ["a", "b"]
    assert comparison.judge_predicted.tolist() == [0.25, 0.45]
    assert comparison.trend_predicted.tolist() == [0.3, 0.5]


def test_prediction_comparison_rejects_different_frontier_cells():
    judge = pd.DataFrame({
        "bench": ["gpqa"], "judge": ["j"], "model": ["a"],
        "actual": [0.2], "n_items": [10], "target_release": ["2025-01-01"],
        "judge_release": ["2025-02-01"], "fitted": [0.25],
        "fitted_low": [0.15], "fitted_high": [0.35],
    })
    trend = pd.DataFrame(columns=[
        "bench", "judge", "model", "actual", "target_release", "judge_release",
        "predicted", "predicted_low", "predicted_high",
    ])
    with pytest.raises(ValueError, match="exact frontier cells"):
        load_module()["merge_prediction_comparison"](judge, trend)


def test_pooled_protocol_uses_every_strictly_earlier_judge():
    protocol = pd.DataFrame({
        "bench": ["gpqa"] * 4,
        "judge": ["old-a", "old-b", "same-day", "future"],
        "item": ["1"] * 4,
        "model_left": ["a"] * 4, "model_right": ["target"] * 4,
        "pick_left": [1, 0, 1, 0],
    })
    releases = {
        "old-a": "2025-01-01", "old-b": "2025-01-05",
        "target": "2025-01-10", "same-day": "2025-01-10",
        "future": "2025-01-11",
    }
    block, eligible, contributing = load_module()["pooled_protocol_for_target"](
        protocol, "gpqa", "target",
        ["old-a", "old-b", "same-day", "future"], releases, pair_limit=8)
    assert eligible == ["old-a", "old-b"]
    assert contributing == ["old-a", "old-b"]
    assert set(block.judge) == {"old-a", "old-b"}


def test_pooled_judges_emit_one_bootstrapped_prediction_per_target():
    rows = []
    for judge, shift in (("old-a", 0), ("old-b", 1), ("same-day", 2)):
        for item in range(12):
            for left, right, pick in (
                ("a", "b", (item + shift) % 3 == 0),
                ("b", "target", (item + shift) % 4 == 0),
                ("a", "target", (item + shift) % 5 == 0),
            ):
                rows.append({
                    "bench": "gpqa", "judge": judge, "item": str(item),
                    "model_left": left, "model_right": right,
                    "pick_left": int(pick),
                })
    protocol = pd.DataFrame(rows)
    hard = hard_rows({
        2: {"a": 2, "b": 5, "target": 8},
        3: {"a": 3, "b": 6, "target": 9},
        4: {"a": 1, "b": 7, "target": 7},
    })
    releases = {
        "old-a": "2025-01-01", "old-b": "2025-01-15",
        "target": "2025-02-01", "same-day": "2025-02-01",
    }
    abilities, fits, predictions = load_module()["fit_pooled_judges"](
        protocol, hard, ["target"], ["old-a", "old-b", "same-day"], releases,
        resamples=20, seed=3, workers=1, pair_limit=8)
    fit = fits.iloc[0]
    prediction = predictions.iloc[0]
    assert fit.fit_status == "ok"
    assert fit.eligible_judges == "old-a|old-b"
    assert fit.n_judges == 2
    assert prediction.predicted_low <= prediction.predicted <= prediction.predicted_high
    assert set(abilities.model) == {"a", "b", "target"}


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
