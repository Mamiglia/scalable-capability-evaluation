"""Unit tests for draccus-based CLI modules (collect and judge)."""

from pathlib import Path
import pytest
import draccus
from collect.cli import CollectConfig
from judge.cli import ExperimentConfig, main as judge_main


def test_collect_config_defaults():
    cfg = CollectConfig()
    assert cfg.cache is not None
    assert cfg.max_tasks == 30
    assert cfg.bench.name == "gpqa"


def test_collect_config_parse():
    args = [
        "--bench.name", "core_bench",
        "--bench.task", "inspect_evals/core_bench",
        "--limit", "50",
    ]
    cfg = draccus.parse(CollectConfig, args=args)
    assert cfg.bench.name == "core_bench"
    assert cfg.bench.task == "inspect_evals/core_bench"
    assert cfg.limit == 50


def test_experiment_config_parse():
    args = [
        "--config_path", "configs/experiments/gemma-gpqa.yaml",
        "--limit", "20",
        "--include_reverse", "false",
        "--dry_run", "true",
    ]
    cfg = draccus.parse(ExperimentConfig, args=args)
    assert cfg.name == "gemma-31b-v-2b-gpqa"
    assert cfg.judge == "openrouter/inclusionai/ling-2.6-flash"
    assert cfg.dry_run is True
    assert cfg.limit == 20
    assert cfg.include_reverse is False


def test_judge_main_dry_run(monkeypatch, tmp_path):
    test_cfg = tmp_path / "test_exp.yaml"
    test_cfg.write_text("""
name: test-exp
judge: test-judge
major: test-major
minor: test-minor
benchmark_name: gpqa
dry_run: true
refresh_registry: false
""")
    monkeypatch.setattr("judge.cli.resolve_eval_path", lambda *a, **kw: Path("data/fake_eval.json"))
    monkeypatch.setattr("judge.cli.build_registry", lambda *a, **kw: None)
    monkeypatch.setattr("sys.argv", ["judge", "--config_path", str(test_cfg), "--dry_run", "true"])
    ret = judge_main()
    assert ret == 0
