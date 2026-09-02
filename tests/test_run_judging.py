import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml
from inspect_ai.model import get_model

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import run_judging
import export_viewer
from run_judging import (
    budget_settings,
    configured_comparisons,
    logged_cost,
    select_judge,
    validated_truncate_keep,
)


def test_expands_judges_across_pairs():
    config = {
        "judges": ["judge-a", "judge-b"],
        "pairs": [["model-1", "model-2"], ["model-3", "model-4"]],
    }

    assert configured_comparisons(config) == [
        ["judge-a", "model-1", "model-2"],
        ["judge-a", "model-3", "model-4"],
        ["judge-b", "model-1", "model-2"],
        ["judge-b", "model-3", "model-4"],
    ]


def test_selects_one_judge_queue():
    comparisons = [
        ["provider/judge-a", "model-1", "model-2"],
        ["provider/judge-b", "model-1", "model-2"],
    ]

    assert select_judge(comparisons, "judge-b") == [comparisons[1]]


def test_one_judge_gets_its_own_budget_and_log_directory(tmp_path):
    config = {
        "total_cost_limit": 20.0,
        "cost_limit_per_sample": 0.006,
        "judge_budgets": {
            "provider/judge-a": {
                "total_cost_limit": 2.0,
                "cost_limit_per_sample": 0.001,
            },
        },
    }

    assert budget_settings(config, "provider/judge-a", tmp_path) == (
        2.0,
        0.001,
        tmp_path / "judge-a",
        True,
    )
    assert budget_settings(config, None, tmp_path) == (
        20.0,
        0.006,
        tmp_path,
        False,
    )


def test_logged_cost_counts_failed_calls_but_ignores_unreadable_logs(tmp_path, monkeypatch):
    for name in ("ok.eval", "failed.eval", "broken.eval"):
        (tmp_path / name).touch()

    logs = {
        "ok.eval": SimpleNamespace(
            status="success",
            stats=SimpleNamespace(model_usage={
                "judge": SimpleNamespace(total_cost=1.25),
            }),
        ),
        "failed.eval": SimpleNamespace(
            status="error",
            stats=SimpleNamespace(model_usage={
                "judge": SimpleNamespace(total_cost=0.50),
            }),
        ),
    }

    def fake_read(path, header_only=True):
        if path.name == "broken.eval":
            raise ValueError("partial file")
        return logs[path.name]

    monkeypatch.setattr(run_judging, "read_eval_log", fake_read)

    assert logged_cost(tmp_path) == 1.75


def test_qwen_expansion_judge_disables_reasoning():
    name = "openrouter/qwen/qwen3.5-122b-a10b"
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "judge_expand_gpqa_models.yaml")
        .read_text()
    )

    model = get_model(name, api_key="test", **config["judge_model_args"][name])

    assert model.api.reasoning_enabled is False


def test_more_gpqa_pairs_config_pins_judges_and_uses_deepseek_only_as_judge():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "judge_more_gpqa_pairs.yaml")
        .read_text()
    )

    assert any("deepseek" in judge for judge in config["judges"])
    assert all("deepseek" not in model for pair in config["pairs"] for model in pair)
    assert config["total_cost_limit"] == 20.0
    assert set(config["judges"]) <= set(config["judge_budgets"])
    assert sum(b["total_cost_limit"] for b in config["judge_budgets"].values()) == 20.0
    for judge in config["judges"]:
        provider = config["judge_model_args"][judge]["provider"]
        assert len(provider["only"]) == 1
        assert provider["allow_fallbacks"] is False
        assert provider["require_parameters"] is True
        assert provider["data_collection"] == "deny"


def test_exporter_uses_only_current_pairs_and_preserves_canonical_transcripts(tmp_path):
    config = tmp_path / "judge.yaml"
    config.write_text(yaml.safe_dump({
        "pairs": [["kept-a", "kept-b"]],
        "transcripts": {
            "canonical": "unrescored.eval",
            "new-model": "new.eval",
        },
    }))

    assert export_viewer.configured_pair_names(config) == {"kept-a-vs-kept-b"}
    assert export_viewer.add_missing_transcripts(
        {"canonical": "rescored.eval"}, [config],
    ) == {
        "canonical": "rescored.eval",
        "new-model": "new.eval",
    }


def test_truncate_keep_defaults_to_first():
    assert validated_truncate_keep({}) == "first"


def test_truncate_keep_accepts_last():
    assert validated_truncate_keep({"truncate_keep": "last"}) == "last"


def test_truncate_keep_rejects_bad_value():
    with pytest.raises(ValueError, match="truncate_keep"):
        validated_truncate_keep({"truncate_keep": "Last"})
