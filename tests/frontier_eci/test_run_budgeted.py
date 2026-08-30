from pathlib import Path
from runpy import run_path


can_afford = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/run_budgeted.py")["can_afford"]
remaining_budget = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/run_budgeted.py")["remaining_budget"]


def test_reserves_worst_case_pair_cost():
    assert can_afford(spent=190, reserve=4, limit=200)
    assert not can_afford(spent=197, reserve=4, limit=200)


def test_historical_spend_reduces_incremental_allowance():
    assert remaining_budget(total_limit=200, historical_spend=133.4) == 66.6


def test_pair_is_not_started_without_worst_case_reserve():
    assert not can_afford(spent=63.0, reserve=4.0, limit=66.6)
