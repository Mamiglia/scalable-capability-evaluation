from pathlib import Path
from runpy import run_path

import pandas as pd


MODULE = run_path(
    Path(__file__).parents[2] / "scripts/reasoning_ablation/prepare.py")


def test_select_pair_items_is_order_invariant_and_caps_each_pair():
    rows = pd.DataFrame([
        {"bench": "gpqa", "judge": "j", "target": "b", "opponent": "a", "item": f"i{i}"}
        for i in range(7)
    ])
    first = MODULE["select_pair_items"](rows, 4)
    second = MODULE["select_pair_items"](rows.sample(frac=1, random_state=3), 4)
    assert len(first) == 4
    assert first.item.tolist() == second.item.tolist()


def test_available_source_filter_drops_pairs_with_missing_logs(tmp_path):
    (tmp_path / "a.eval").write_text("a")
    (tmp_path / "c.eval").write_text("c")
    manifest = pd.DataFrame([
        {"bench": "gpqa", "judge": "j", "target": "b", "opponent": "a", "item": "i1"},
        {"bench": "gpqa", "judge": "j", "target": "c", "opponent": "a", "item": "i2"},
    ])
    bases = {("gpqa", "j"): {
        "transcripts": {"a": "a.eval", "b": "b.eval", "c": "c.eval"}
    }}
    kept = MODULE["filter_available_sources"](manifest, bases, tmp_path)
    assert kept[["target", "opponent", "item"]].to_dict("records") == [
        {"target": "c", "opponent": "a", "item": "i2"}
    ]


def test_topup_items_are_jointly_available_q1_only():
    selected = MODULE["select_topup_items"](
        {"q1-a", "q1-b", "hard"}, {"q1-a", "hard", "other"}, {"q1-a", "q1-b"})
    assert selected == ["q1-a"]
