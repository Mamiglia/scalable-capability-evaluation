"""Unit tests for CLI modules (collect and judge)."""

from pathlib import Path
from unittest.mock import MagicMock
from collect.cli import main as collect_main
from judge.cli import main as judge_main


def test_collect_single_eval(monkeypatch, tmp_path):
    model_config = tmp_path / "model.yml"
    model_config.write_text("name: custom/model2\nslug: custom/model2\n")
    bench_config = tmp_path / "bench.yml"
    bench_config.write_text("name: inspect_evals/mbpp\n")
    
    captured = {}
    mock_log = MagicMock()
    mock_log.location = str(tmp_path / "output.eval")
    mock_log.status = "success"
    (tmp_path / "output.eval").write_text("fake_log")

    def mock_eval(**kwargs):
        captured.update(kwargs)
        return [mock_log]

    monkeypatch.setattr("collect.cli.eval_func", mock_eval)
    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: tmp_path / "target.eval")
    monkeypatch.setattr(
        "sys.argv",
        ["collect", "--bench", str(bench_config), "--model", str(model_config), "-n", "1"],
    )

    ret = collect_main()
    assert ret == 0
    assert captured["tasks"] == "inspect_evals/mbpp"
    assert captured["model"] == "custom/model2"
    assert captured["seed"] == 0


def test_collect_multiple_evals(monkeypatch, tmp_path):
    m1 = tmp_path / "m1.yml"
    m1.write_text("name: custom/m1\nslug: custom/m1\n")
    m2 = tmp_path / "m2.yml"
    m2.write_text("name: custom/m2\nslug: custom/m2\n")
    b1 = tmp_path / "b1.yml"
    b1.write_text("name: inspect_evals/b1\n")
    b2 = tmp_path / "b2.yml"
    b2.write_text("name: inspect_evals/b2\n")

    runs = []
    mock_log = MagicMock()
    mock_log.status = "success"

    def mock_eval(**kwargs):
        runs.append((kwargs["tasks"], kwargs["model"]))
        return [mock_log]

    monkeypatch.setattr("collect.cli.eval_func", mock_eval)
    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: tmp_path / "target.eval")
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect",
            "-b", str(b1), "-b", str(b2),
            "-m", str(m1), "-m", str(m2),
            "-n", "1",
        ],
    )

    ret = collect_main()
    assert ret == 0
    assert len(runs) == 4
    assert runs == [
        ("inspect_evals/b1", "custom/m1"),
        ("inspect_evals/b1", "custom/m2"),
        ("inspect_evals/b2", "custom/m1"),
        ("inspect_evals/b2", "custom/m2"),
    ]


def test_collect_args_parsing(monkeypatch, tmp_path):
    m1 = tmp_path / "m1.yml"
    m1.write_text("name: custom/m1\nslug: custom/m1\n")
    b1 = tmp_path / "b1.yml"
    b1.write_text("name: inspect_evals/b1\n")

    captured = {}
    mock_log = MagicMock()
    mock_log.status = "success"

    def mock_eval(**kwargs):
        captured.update(kwargs)
        return [mock_log]

    monkeypatch.setattr("collect.cli.eval_func", mock_eval)
    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: tmp_path / "target.eval")
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect",
            "-b", str(b1),
            "-m", str(m1),
            "-T", "fewshot=5",
            "-M", "temp=0.7",
            "--limit", "10",
            "--max-connections", "4",
        ],
    )

    ret = collect_main()
    assert ret == 0
    assert captured["task_args"] == {"fewshot": 5}
    assert captured["model_args"]["temp"] == 0.7
    assert captured["limit"] == 10
    assert captured["max_connections"] == 4


def test_judge_main_execution(monkeypatch, tmp_path):
    exp_config = tmp_path / "exp.yml"
    exp_config.write_text("""name: test_exp
benchmark:
  name: inspect_evals/mbpp
major:
  name: p/maj
  slug: p/maj
minor:
  name: p/min
  slug: p/min
judge:
  name: p/jdg
  slug: p/jdg
""")

    captured = {}
    monkeypatch.setattr("judge.cli.build_judge_tasks", lambda *args, **kw: ["dummy_task"])
    monkeypatch.setattr(
        "judge.cli.eval_set", lambda **kw: (captured.update(kw) or (True, [True]))
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "judge",
            "--config_path", str(exp_config),
        ],
    )
    ret = judge_main()
    assert ret == 0
    assert captured["tasks"] == ["dummy_task"]
    assert captured["model"] == "p/jdg"
