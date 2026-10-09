"""The frontier subset: 2026-I train (EASY) ∪ HARD items of the coding domain, agent_bench_os excluded.

    PYTHONPATH=src .venv/bin/python src/challenges/frontier/items.py            # print the lists
    PYTHONPATH=src .venv/bin/python src/challenges/frontier/items.py --check    # bench ymls hold exactly them

The lists are frozen into configs/bench/*_frontier.yml as task arguments (the hash dir identifies
the subset, see src/collect/tasks/mmlu_pro.py); `--check` is the one test that they still match
the frozen coding items (reference.tables).
"""

import sys
from pathlib import Path

import yaml

from reference import tables

SEMESTER = "2026-I"
BENCHES = {  # bench -> its subset yml (task_args.ids holds the sample ids)
    "terminal_bench_2_1": "configs/bench/terminal_bench_2_1_frontier.yml",
    "swe_bench_mini": "configs/bench/swe-bench-mini-frontier.yml",
}


def subset() -> dict[str, list[str]]:
    """bench -> sorted sample ids of the items a judge or a theta* fit needs."""
    items = tables.load("coding")[1]
    keep = items[f"{SEMESTER}_train"] | (items[f"{SEMESTER}_set"] == "hard")
    return {bench: sorted(items[keep & (items.bench == bench)].item) for bench in BENCHES}


def main() -> int:
    wanted = subset()
    if "--check" not in sys.argv:
        for bench, values in wanted.items():
            print(f"{bench}: {len(values)}")
            print(yaml.safe_dump(values, default_flow_style=False), end="")
        return 0
    bad = 0
    for bench, yml in BENCHES.items():
        frozen = sorted(yaml.safe_load(Path(yml).read_text())["task_args"]["ids"])
        if frozen != wanted[bench]:
            bad += 1
            print(f"MISMATCH {yml}: {len(frozen)} in yml vs {len(wanted[bench])} wanted; "
                  f"extra={sorted(set(frozen) - set(wanted[bench]))} missing={sorted(set(wanted[bench]) - set(frozen))}")
        else:
            print(f"OK {yml}: {len(frozen)} ids")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
