from pathlib import Path
from runpy import run_path

import json

import pandas as pd


module = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/canonical_coverage.py")
prediction_cutoff = module["prediction_cutoff"]
eligible_models = module["eligible_models"]
canonical_anchor_models = module["canonical_anchor_models"]
stable_items = module["stable_items"]
choose_pairs = module["choose_pairs"]
coverage_blocks = module["coverage_blocks"]
pair_items = module["pair_items"]
historical_rollouts = module.get("historical_rollouts")


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


def test_coverage_uses_q1_verdicts_and_rejects_zero_item_topups():
    config = {"releases": {"target": "2024-01-01", "judge": "2025-01-01",
                           "q1-anchor": "2023-01-01", "q2-only": "2023-01-01",
                           "zero-item": "2023-01-01"},
              "candidates": ["target"], "judges": {"checkpoint": ["judge"]},
              "desired_direct_anchors": 3, "minimum_anchors": 3}
    quartiles = pd.DataFrame({"bench": ["b", "b"], "item": ["q1", "q2"],
                              "item_quartile": [1, 2]})
    rollouts = pd.DataFrame({
        "bench": ["b"] * 8,
        "model": ["target", "target", "q1-anchor", "q1-anchor", "q2-only", "q2-only", "zero-item", "zero-item"],
        "item": ["q1", "q2"] * 4,
        "correct": [True, True, True, True, True, True, False, True],
        "is_canonical": [True] * 8,
    })
    verdicts = pd.DataFrame({"bench": ["b", "b"], "judge": ["judge", "judge"],
                             "item": ["q1", "q2"], "model_left": ["target", "target"],
                             "model_right": ["q1-anchor", "q2-only"], "pick_left": [True, True]})
    block = coverage_blocks(config, rollouts, verdicts, quartiles).iloc[0]
    assert json.loads(block.direct_anchors) == ["q1-anchor"]
    assert "q2-only" not in json.loads(block.connected_anchors)
    assert "zero-item" not in json.loads(block.recommended_anchors)
    assert block.status == "needs_historical_backfill"


def test_pair_items_require_both_models_correct_on_q1():
    blocks = pd.DataFrame({"bench": ["b"], "judge": ["judge"], "target": ["target"],
                           "recommended_anchors": [json.dumps(["anchor"])]})
    rollouts = pd.DataFrame({"bench": ["b"] * 4, "model": ["target", "target", "anchor", "anchor"],
                             "item": ["q1", "q2", "q1", "q2"],
                             "correct": [True, True, True, False], "is_canonical": [True] * 4,
                             "item_quartile": [1, 2, 1, 2]})
    selected = pair_items(blocks, rollouts, pd.DataFrame(), limit=8)
    assert selected.item.tolist() == ["q1"]


def test_pair_items_reject_item_with_any_incorrect_duplicate_rollout():
    blocks = pd.DataFrame({"bench": ["b"], "judge": ["judge"], "target": ["target"],
                           "recommended_anchors": [json.dumps(["anchor"])]})
    rollouts = pd.DataFrame({"bench": ["b"] * 3, "model": ["target", "target", "anchor"],
                             "item": ["q1"] * 3, "correct": [True, False, True],
                             "is_canonical": [True] * 3, "item_quartile": [1] * 3})
    assert pair_items(blocks, rollouts, pd.DataFrame(), limit=8).empty


def test_coverage_skips_invalid_early_anchor_for_later_viable_topup():
    config = {"releases": {"target": "2024-01-01", "judge": "2025-01-01",
                           "a-invalid": "2023-01-01", "z-valid": "2023-01-01"},
              "candidates": ["target"], "judges": {"checkpoint": ["judge"]},
              "desired_direct_anchors": 1, "minimum_anchors": 1}
    quartiles = pd.DataFrame({"bench": ["b"], "item": ["q1"], "item_quartile": [1]})
    rollouts = pd.DataFrame({"bench": ["b"] * 3, "model": ["target", "a-invalid", "z-valid"],
                             "item": ["q1"] * 3, "correct": [True, False, True],
                             "is_canonical": [True] * 3})
    verdicts = pd.DataFrame(columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
    block = coverage_blocks(config, rollouts, verdicts, quartiles).iloc[0]
    assert json.loads(block.recommended_anchors) == ["z-valid"]
    assert block.status == "identifiable"


def test_historical_backfill_is_used_only_below_four_viable_canonical_anchors():
    config = {"releases": {"target": "2024-01-01", "judge": "2024-01-01",
                           **{name: "2023-01-01" for name in
                              ["a", "b", "c", "d", "old-1", "old-2", "old-3", "old-4"]}},
              "candidates": ["target"], "judges": {"checkpoint": ["judge"]},
              "desired_direct_anchors": 6, "minimum_anchors": 4}
    quartiles = pd.DataFrame({"bench": ["bench"], "item": ["q1"], "item_quartile": [1]})
    rollouts = pd.DataFrame({
        "bench": ["bench"] * 9,
        "model": ["target", "a", "b", "c", "d", "old-1", "old-2", "old-3", "old-4"],
        "item": ["q1"] * 9,
        "correct": [True] * 9,
        "is_canonical": [True] * 5 + [False] * 4,
        "is_historical_backfill": [False] * 5 + [True] * 4,
    })
    verdicts = pd.DataFrame(columns=[
        "bench", "judge", "item", "model_left", "model_right", "pick_left"])

    enough = coverage_blocks(config, rollouts, verdicts, quartiles).iloc[0]
    assert json.loads(enough.recommended_anchors) == ["a", "b", "c", "d"]
    assert json.loads(enough.historical_backfill_anchors) == []

    rollouts.loc[rollouts.model == "d", "correct"] = False
    short = coverage_blocks(config, rollouts, verdicts, quartiles).iloc[0]
    assert json.loads(short.recommended_anchors) == ["a", "b", "c", "old-1"]
    assert json.loads(short.historical_backfill_anchors) == ["old-1"]
    assert short.attainable_direct_anchor_count == 4
    selected = pair_items(pd.DataFrame([short]), rollouts.assign(item_quartile=1), verdicts, limit=8)
    assert selected[selected.opponent == "old-1"].reason.tolist() == ["historical_backfill"]

    rollouts.loc[rollouts.model.isin(["a", "b", "c", "d"]), "correct"] = False
    early = coverage_blocks(config, rollouts, verdicts, quartiles).iloc[0]
    assert json.loads(early.historical_backfill_anchors) == ["old-1", "old-2", "old-3", "old-4"]
    assert early.status == "identifiable"


def test_historical_rollouts_labels_only_configured_sources(tmp_path, monkeypatch):
    source = tmp_path / "source.eval"
    source.touch()
    sample = type("Sample", (), {"id": "item-1"})()
    globals_ = historical_rollouts.__globals__
    monkeypatch.setitem(globals_, "read_eval_log_samples", lambda *_args, **_kwargs: [sample])
    monkeypatch.setitem(globals_, "extract_score", lambda _: True)

    rows = historical_rollouts({"gpqa": {"approved": str(source)}})

    assert rows.to_dict("records") == [{
        "bench": "gpqa", "model": "approved", "item": "item-1", "correct": True,
        "is_canonical": False, "is_historical_backfill": True,
    }]
