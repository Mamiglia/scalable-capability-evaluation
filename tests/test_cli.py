"""Unit tests for draccus-based CLI modules (collect and judge)."""

from pathlib import Path
import draccus
from collect.cli import BenchConfig
from judge.cli import ExperimentConfig, main as judge_main


def test_collect_model_options(monkeypatch, tmp_path):
    from collect.cli import main

    model_config = tmp_path / "model.yml"
    model_config.write_text("model_name: provider/model2\nslug: custom/model2\n")
    bench_config = tmp_path / "bench.yml"
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
    assert captured["log_dir_allow_dirty"] is True


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
        "--config_path", "configs/experiments/test.yml",
        "--limit", "20",
        "--include_reverse", "false",
        "--mode", "combinatorial",
    ]
    cfg = draccus.parse(ExperimentConfig, args=args)
    assert cfg.name == "test"
    assert cfg.limit == 20
    assert cfg.include_reverse is False
    assert cfg.mode == "combinatorial"


def test_judge_main_execution(monkeypatch, tmp_path):
    test_cfg = tmp_path / "test_exp.yml"
    test_cfg.write_text("""
name: test-exp
judge: test-judge
major: test-major
minor: test-minor
mode: paired
benchmark:
  name: gpqa
refresh_registry: false
""")
    captured = {}
    monkeypatch.setattr("judge.cli.build_judge_tasks", lambda **kw: ["dummy_task"])
    monkeypatch.setattr("judge.cli.build_registry", lambda *a, **kw: None)
    monkeypatch.setattr(
        "judge.cli.eval_set", lambda **kw: (captured.update(kw) or True, [])
    )
    monkeypatch.setattr("sys.argv", ["judge", "--config_path", str(test_cfg)])
    ret = judge_main()
    assert ret == 0
    assert captured["tasks"] == ["dummy_task"]
    assert captured["log_dir"] == "logs/judge/test-exp"
