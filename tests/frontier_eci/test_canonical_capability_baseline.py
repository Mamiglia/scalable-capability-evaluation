from pathlib import Path
from runpy import run_path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.frontier_eci.canonical_capability_baseline import (  # noqa: E402
    _eci_tasks as eci_tasks,
    fit_eci_blocks_parallel,
)

module = run_path(
    Path(__file__).parents[2] / "scripts/frontier_eci/canonical_capability_baseline.py"
)
eci_mapping = module["eci_mapping"]
eci_anchor_subsets = module["eci_anchor_subsets"]
fit_eci_blocks = module["fit_eci_blocks"]
load_eci = module["load_eci"]


def test_eci_baseline_uses_exactly_the_judge_blocks_anchor_ids():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "e": 130.0, "target": 140.0})
    outcomes = pd.DataFrame({"model": ["a", "c", "d", "e", "target"],
                             "successes": [2, 7, 13, 18, 0], "total": [20] * 5})
    result = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    assert result.iloc[0].anchor_ids == "a|c|d|e"


def test_eci_baseline_excludes_target_truth():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "e": 130.0, "target": 140.0})
    outcomes = pd.DataFrame({"model": ["a", "c", "d", "e", "target"],
                             "successes": [2, 7, 13, 18, 0], "total": [20] * 5})
    first = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    outcomes.loc[outcomes.model.eq("target"), "successes"] = 20
    second = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    assert first.iloc[0].predicted == second.iloc[0].predicted


def test_eci_target_truth_and_item_availability_never_change_bootstrap():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "e": 130.0, "target": 140.0})
    successes = {"a": 2, "c": 7, "d": 13, "e": 18, "target": 0}
    outcomes = pd.DataFrame([
        {"model": model, "item": str(item), "correct": item < success}
        for model, success in successes.items() for item in range(20)
    ])
    first = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"},
                           resamples=30, seed=4).iloc[0]
    changed = pd.concat([
        outcomes[outcomes.model.ne("target")],
        pd.DataFrame({"model": ["target"] * 7,
                      "item": [f"target-only-{item}" for item in range(7)],
                      "correct": [1] * 7}),
    ], ignore_index=True)
    second = fit_eci_blocks(judge_fit, eci, changed, targets={"b": "target"},
                            resamples=30, seed=4).iloc[0]
    fit_fields = [
        "predicted", "predicted_low", "predicted_median", "predicted_high",
        "intercept", "slope", "n_bootstrap", "bootstrap_unstable",
    ]
    assert first.actual == 0.0 and second.actual == 1.0
    assert first.residual != second.residual
    assert first[fit_fields].to_dict() == second[fit_fields].to_dict()


def test_eci_selected_anchor_missing_from_outcomes_fails_explicitly():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|b|c|d|e"]})
    eci = pd.Series({"a": 100.0, "b": 110.0, "c": 120.0, "d": 130.0,
                     "e": 140.0, "target": 150.0})
    outcomes = pd.DataFrame({"model": ["a", "b", "c", "d", "target"],
                             "successes": [2, 7, 13, 18, 0], "total": [20] * 5})
    result = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"}).iloc[0]
    assert result.fit_status == "failed"
    assert result.failure_reason == "selected anchors unavailable: outcomes=e"
    assert result.anchor_table is None


def test_eci_baseline_reports_missing_anchor_without_changing_the_set():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "target": 140.0})
    outcomes = pd.DataFrame({"model": ["a", "c", "d", "e", "target"],
                             "successes": [2, 7, 13, 18, 0], "total": [20] * 5})
    result = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    assert judge_fit.iloc[0].anchor_ids == "a|c|d|e"
    assert result.iloc[0].headline_anchor_ids == "a|c|d|e"
    assert result.iloc[0].eci_anchor_ids == "a|c|d"
    assert result.iloc[0].dropped_missing_eci_ids == "e"
    assert result.iloc[0].fit_status == "failed"
    assert result.iloc[0].failure_reason == "only 3 mapped ECI anchors"


def test_eci_and_paired_judge_sensitivity_use_the_same_mapped_subset():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e|f"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0,
                     "f": 130.0, "target": 140.0})
    outcomes = pd.DataFrame({"model": ["a", "c", "d", "e", "f", "target"],
                             "successes": [2, 7, 13, 15, 18, 0], "total": [20] * 6})
    subsets = eci_anchor_subsets(judge_fit, eci)
    result = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    paired_judge_ids = subsets.set_index("block").eci_anchor_ids["b"]
    assert result.iloc[0].eci_anchor_ids == paired_judge_ids == "a|c|d|f"
    assert result.iloc[0].dropped_missing_eci_ids == "e"
    assert result.iloc[0].fit_status == "ok"
    assert judge_fit.iloc[0].anchor_ids == "a|c|d|e|f"


def test_eci_baseline_bootstraps_hard_items_for_figure_band():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "e": 130.0, "target": 140.0})
    successes = {"a": 2, "c": 7, "d": 13, "e": 18, "target": 20}
    outcomes = pd.DataFrame([
        {"model": model, "item": str(item), "correct": item < success}
        for model, success in successes.items() for item in range(20)
    ])
    result = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"},
                            resamples=20, seed=4)
    assert result.iloc[0].n_bootstrap > 0
    assert result.iloc[0].predicted_low < result.iloc[0].predicted_high


def test_eci_name_mapping_is_explicit_and_leaves_unknown_models_missing(tmp_path):
    path = tmp_path / "eci.csv"
    pd.DataFrame({"Model": ["DeepSeek-R1 (May 2025)"], "eci": [141.8]}).to_csv(path, index=False)
    eci = load_eci(path, ["deepseek-r1-0528", "qwen3-32b"])
    assert eci["deepseek-r1-0528"] == 141.8
    assert pd.isna(eci["qwen3-32b"])


def test_eci_mapping_requires_each_reviewed_name_exactly_once(tmp_path):
    path = tmp_path / "eci.csv"
    pd.DataFrame({"Model": ["DeepSeek-R1"], "eci": [139.52]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"DeepSeek-R1 \(May 2025\)"):
        eci_mapping(path, ["deepseek-r1-0528"])


def test_eci_tasks_are_single_rows_and_parallel_matches_serial():
    judge_fits = pd.DataFrame({"block": ["b1", "b2"],
                               "anchor_ids": ["a|b|c|d", "a|b|c|d"]})
    eci = pd.Series({"a": 100.0, "b": 110.0, "c": 120.0, "d": 130.0,
                     "target-1": 140.0, "target-2": 145.0})
    successes = {"a": 2, "b": 7, "c": 13, "d": 18,
                 "target-1": 0, "target-2": 20}
    outcomes = pd.DataFrame([
        {"model": model, "item": str(item), "correct": item < success}
        for model, success in successes.items() for item in range(20)
    ])
    assert [len(task) for task in eci_tasks(judge_fits)] == [1, 1]
    targets = {"b1": "target-1", "b2": "target-2"}
    serial = fit_eci_blocks(judge_fits, eci, outcomes, targets, resamples=5, seed=4)
    parallel = fit_eci_blocks_parallel(
        judge_fits, eci, outcomes, targets, resamples=5, seed=4, workers=2)
    pd.testing.assert_frame_equal(serial.drop(columns="anchor_table"),
                                  parallel.drop(columns="anchor_table"))
    for left, right in zip(serial.anchor_table, parallel.anchor_table):
        pd.testing.assert_frame_equal(left, right)
