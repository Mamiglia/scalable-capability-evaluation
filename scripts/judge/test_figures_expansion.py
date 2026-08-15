"""The per-transcript filter re-tallies run stats from the item-level verdicts.

With the floor at zero it must reproduce the exported stats exactly, or the
filtered numbers would be measuring the re-tally rather than the filter.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import figures_expansion as expansion

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["data-expansion-gpqa", "data-expansion-aime2025", "data-expansion-aime2026"]
STATS = ("score", "n_items", "slot_a", "noanswer_rate")


@pytest.mark.parametrize("dataset", DATASETS)
def test_refit_reproduces_the_exported_run_stats(dataset, monkeypatch):
    d = ROOT / "interactive_viewer" / "data" / dataset
    if not (d / "index.json").exists():
        pytest.skip(f"{dataset} not exported here")
    monkeypatch.setattr(expansion, "MIN_REASONING_CHARS", 0)
    expansion.transcript_chars.cache_clear()
    expansion.refit_runs.cache_clear()

    stats = expansion.refit_runs(d)
    checked = 0
    for run in json.load(open(d / "index.json"))["runs"]:
        # self-pairs have no major to pick, so the refit leaves them alone
        if run.get("frac", "full") != "full" or run["is_self"]:
            continue
        refit = stats[(run["judge"], run["major"], run["minor"])]
        for name in STATS:
            if run[name] is not None:
                assert refit[name] == pytest.approx(run[name]), f"{run['run_id']} {name}"
        checked += 1
    assert checked > 0
