import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))
from judged_capability_over_time import frontier, load_frontier_roster  # noqa: E402


def test_frontier_uses_frozen_roster_not_highest_hard_accuracy():
    table = pd.DataFrame(
        {
            "actual": [0.6, 0.9],
            "release": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "n_opponents": [3, 3],
        },
        index=["frozen-model", "higher-hard-accuracy-model"],
    )

    selected = frontier(table, min_opponents=2, roster={"2025Q1": "frozen-model"})

    assert selected.index.tolist() == ["frozen-model"]


def test_frontier_reports_roster_models_without_enough_verdicts():
    table = pd.DataFrame(
        {
            "actual": [0.6],
            "release": pd.to_datetime(["2025-01-01"]),
            "n_opponents": [1],
        },
        index=["underjudged-model"],
    )

    with pytest.raises(ValueError, match="underjudged-model"):
        frontier(table, min_opponents=2, roster={"2025Q1": "underjudged-model"})


def test_frontier_rejects_model_assigned_to_wrong_quarter():
    table = pd.DataFrame(
        {
            "actual": [0.6],
            "release": pd.to_datetime(["2025-01-01"]),
            "n_opponents": [3],
        },
        index=["model"],
    )

    with pytest.raises(ValueError, match="wrong release quarter"):
        frontier(table, min_opponents=2, roster={"2024Q4": "model"})


def test_frontier_roster_keeps_medium_and_rejects_excluded_effort(tmp_path):
    path = tmp_path / "roster.yaml"
    path.write_text("gpqa:\n  2025Q3: gpt-5-medium\n")

    assert load_frontier_roster(path, "gpqa") == {"2025Q3": "gpt-5-medium"}

    path.write_text("gpqa:\n  2025Q3: gpt-5-high\n")
    with pytest.raises(ValueError, match="excluded effort variant"):
        load_frontier_roster(path, "gpqa")


def test_frontier_gpqa_uses_hf_gold_standard_prompt():
    for path in ("configs/frontier_pairs.yaml", "configs/pool_pairs.yaml"):
        config = yaml.safe_load(Path(path).read_text())
        assert config["system_prompts"]["gpqa"] == "assets/system_prompt/mcq.txt"
        assert config["system_prompts"]["swe_bench_mini"] == "assets/system_prompt/default.txt"


def test_judging_runner_reads_configured_system_prompt():
    from run_judging import configured_system_prompt

    assert configured_system_prompt({"system_prompt": "assets/system_prompt/mcq.txt"}) == "assets/system_prompt/mcq.txt"
    assert configured_system_prompt({}) == "assets/system_prompt/default.txt"
