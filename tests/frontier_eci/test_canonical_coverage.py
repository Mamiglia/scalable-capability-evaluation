from pathlib import Path
from runpy import run_path

import pandas as pd


module = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/canonical_coverage.py")
prediction_cutoff = module["prediction_cutoff"]
eligible_models = module["eligible_models"]
canonical_anchor_models = module["canonical_anchor_models"]
stable_items = module["stable_items"]
choose_pairs = module["choose_pairs"]


def test_anchor_is_eligible_when_older_than_target_or_judge():
    releases = {"target": "2024-01-01", "judge": "2026-01-01",
                "older": "2025-01-01", "future": "2027-01-01"}
    assert prediction_cutoff("target", "judge", releases) == pd.Timestamp("2026-01-01")
    assert eligible_models(["older", "future"], "target", "judge", releases) == ["older"]


def test_item_cap_is_stable_and_never_exceeds_eight():
    items = [f"item-{index}" for index in range(20)]
    assert stable_items(items) == stable_items(list(reversed(items)))
    assert len(stable_items(items)) == 8


def test_missing_direct_anchor_pairs_are_preferred_to_more_items():
    existing = pd.DataFrame({"target": ["t"] * 16, "opponent": ["a"] * 16,
                             "item": [f"i{x // 2}" for x in range(16)]})
    candidates = pd.DataFrame({"target": ["t", "t"], "opponent": ["a", "b"],
                               "hard_accuracy": [0.2, 0.8]})
    selected = choose_pairs(existing, candidates, desired_direct_anchors=2)
    assert selected[["target", "opponent"]].drop_duplicates().values.tolist() == [["t", "b"]]


def test_target_truth_is_not_an_anchor():
    releases = {"a": "2023-01-01", "b": "2023-02-01",
                "frontier": "2024-01-01", "judge": "2024-01-01"}
    anchors = canonical_anchor_models(["a", "b", "frontier"], "frontier", "judge", releases)
    assert anchors == ["a", "b"]
