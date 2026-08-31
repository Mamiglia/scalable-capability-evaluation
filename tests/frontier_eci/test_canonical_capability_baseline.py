from pathlib import Path
from runpy import run_path

import pandas as pd
import pytest


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
