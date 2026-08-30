import json
import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest
import yaml


budget = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/run_budgeted.py")
can_afford = budget["can_afford"]
remaining_budget = budget["remaining_budget"]


def write_config(path, *, log_root="logs", only_items=None):
    path.write_text(yaml.safe_dump({
        "log_root": log_root,
        "only_items": only_items or ["i1"],
        "comparisons": [["openrouter/openai/gpt-5.2", "model-a", "model-b"]],
    }))


def test_reserves_worst_case_pair_cost():
    assert can_afford(spent=190, reserve=4, limit=200)
    assert not can_afford(spent=197, reserve=4, limit=200)


def test_historical_spend_reduces_incremental_allowance():
    assert remaining_budget(total_limit=200, historical_spend=133.4) == 66.6


def test_pair_is_not_started_without_worst_case_reserve():
    assert not can_afford(spent=63.0, reserve=4.0, limit=66.6)


def test_resume_rejects_a_conflicting_baseline(tmp_path, monkeypatch):
    config, ledger = tmp_path / "pair.yaml", tmp_path / "ledger.jsonl"
    write_config(config)
    globals_ = budget["main"].__globals__
    monkeypatch.setitem(globals_, "pair_successful", lambda *_: False)
    monkeypatch.setitem(globals_, "key_usage", iter([100.0, 100.0]).__next__)
    monkeypatch.setitem(globals_, "subprocess", SimpleNamespace(run=lambda *_: SimpleNamespace(returncode=0)))
    monkeypatch.setattr(sys, "argv", ["run_budgeted.py", str(config), "--baseline", "100", "--ledger", str(ledger)])
    assert budget["main"]() == 0
    assert json.loads(ledger.read_text().splitlines()[0])["baseline"] == 100.0

    monkeypatch.setitem(globals_, "key_usage", lambda: pytest.fail("usage must not be queried"))
    monkeypatch.setattr(sys, "argv", ["run_budgeted.py", str(config), "--baseline", "101", "--ledger", str(ledger)])
    with pytest.raises(ValueError, match="baseline"):
        budget["main"]()


def test_preflight_reserve_normalizes_relative_path_and_pair_order(tmp_path):
    preflight = tmp_path / "preflight.csv"
    preflight.write_text("config,target,opponent,upper_cost\nconfigs/runs/pair.yaml,model-b,model-a,4\n")

    reserve = budget["reserve_for"](
        budget["reserves"](preflight), Path.cwd() / "configs/runs/pair.yaml", "model-a", "model-b",
        fallback=5, strict=True,
    )

    assert reserve == 4


def test_preflight_missing_reserve_fails(tmp_path):
    preflight = tmp_path / "preflight.csv"
    preflight.write_text("config,target,opponent,upper_cost\n")

    with pytest.raises(ValueError, match="preflight missing reserve"):
        budget["reserve_for"](budget["reserves"](preflight), Path("pair.yaml"), "a", "b", fallback=5, strict=True)


def test_completed_pair_skips_usage_and_reserve(tmp_path, monkeypatch):
    config, ledger = tmp_path / "pair.yaml", tmp_path / "ledger.jsonl"
    write_config(config, log_root=str(tmp_path / "logs"))
    globals_ = budget["main"].__globals__
    expected = tmp_path / "logs" / "gpt-5.2" / "model-a-vs-model-b" / "phase-1"
    monkeypatch.setitem(globals_, "has_success_log", lambda path: path == expected)
    monkeypatch.setitem(globals_, "key_usage", lambda: pytest.fail("completed pair must not query usage"))
    monkeypatch.setattr(sys, "argv", ["run_budgeted.py", str(config), "--baseline", "100", "--ledger", str(ledger)])

    assert budget["main"]() == 0
