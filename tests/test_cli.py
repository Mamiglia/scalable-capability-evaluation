"""Unit tests for draccus-based CLI modules (collect and judge)."""

from pathlib import Path
import pytest
import draccus
from collect.cli import BenchConfig
from judge.cli import ExperimentConfig, main as judge_main


def test_collect_config_defaults():
    cfg = BenchConfig()
    assert cfg.cache is not None
    assert cfg.max_tasks == 30
    assert cfg.name == "gpqa"


def test_collect_model_options(monkeypatch, tmp_path):
    from collect.cli import main

    model_config = tmp_path / "model.yaml"
    model_config.write_text("model_name: provider/model2\nslug: custom/model2\n")
    bench_config = tmp_path / "bench.yaml"
    bench_config.write_text("name: mbpp\ntask: inspect_evals/mbpp\nlog_dir: data/mbpp\n")
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect", "--bench", str(bench_config),
            "--model", str(model_config), "--model", "provider/model1",
            "--limit", "2",
        ],
    )
    monkeypatch.setattr(
        "collect.cli.eval_set", lambda **kwargs: (captured.update(kwargs) or True, None)
    )
    assert main() == 0
    assert captured["tasks"] == ["inspect_evals/mbpp"]
    assert captured["log_dir"] == "data/mbpp"
    assert captured["model"] == ["custom/model2", "openrouter/provider/model1"]
    assert captured["limit"] == 2


def test_collect_config_parse():
    args = [
        "--name", "core_bench",
        "--task", "inspect_evals/core_bench",
        "--limit", "50",
    ]
    cfg = draccus.parse(BenchConfig, args=args)
    assert cfg.name == "core_bench"
    assert cfg.task == "inspect_evals/core_bench"
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
