from pathlib import Path
from runpy import run_path


can_afford = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/run_budgeted.py")["can_afford"]


def test_reserves_worst_case_pair_cost():
    assert can_afford(spent=190, reserve=4, limit=200)
    assert not can_afford(spent=197, reserve=4, limit=200)
