from pathlib import Path
from runpy import run_path


module = run_path(
    Path(__file__).parents[2] / "scripts/frontier_eci/runpod/parallel_topup.py")
partition_configs = module["partition_configs"]
should_stop = module["should_stop"]


def test_partition_is_deterministic_disjoint_and_complete():
    paths = [
        Path("runs/swe_bench_mini/gpt-5.2/b.yaml"),
        Path("runs/gpqa/gpt-5.2/a.yaml"),
        Path("runs/gpqa/o4-mini/c.yaml"),
    ]

    groups = partition_configs(reversed(paths), judges=("gpt-5.2", "o4-mini"))

    assert groups == {
        "gpt-5.2": sorted(paths[:2]),
        "o4-mini": [paths[2]],
    }
    assigned = [path for group in groups.values() for path in group]
    assert len(assigned) == len(set(assigned)) == len(paths)


def test_watchdog_enforces_usage_and_inflight_thresholds():
    assert not should_stop(spent=100, active_workers=5, max_reserve=5.03076)
    assert should_stop(spent=125, active_workers=1, max_reserve=1)
    assert should_stop(spent=124.9, active_workers=5, max_reserve=5.03076)
