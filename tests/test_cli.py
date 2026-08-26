"""Unit tests for CLI modules (collect and judge)."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inspect_ai import Epochs

from collect.cli import main as collect_main
from judge.cli import main as judge_main


class SynchronousExecutor:
    """Runs each job inline and hands back an already-completed real Future."""

    def __init__(self, max_workers=None, mp_context=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # mirror ProcessPoolExecutor error propagation
            future.set_exception(exc)
        return future


@pytest.fixture(autouse=True)
def mock_process_pool(monkeypatch):
    monkeypatch.setattr("collect.runner.ProcessPoolExecutor", SynchronousExecutor)
    # The bench configs here name tasks that do not exist; the manifest must not resolve them.
    monkeypatch.setattr("collect.cli.task_args_with_defaults", lambda task: dict(task.task_args))




def test_collect_single_eval(monkeypatch, tmp_path):
    model_config = tmp_path / "model.yml"
    model_config.write_text("name: custom/model2\nslug: custom/model2\n")
    bench_config = tmp_path / "bench.yml"
    bench_config.write_text("name: inspect_evals/mbpp\nepochs_reducer: mean\n")

    captured = {}
    mock_log = MagicMock()
    mock_log.location = str(tmp_path / "output.eval")
    mock_log.status = "success"
    (tmp_path / "output.eval").write_text("fake_log")

    def mock_eval_set(**kwargs):
        captured.update(kwargs)
        return True, [mock_log]

    monkeypatch.setattr("collect.cli.eval_set", mock_eval_set)
    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: tmp_path / "target_dir")
    monkeypatch.setattr(
        "sys.argv",
        ["collect", "--bench", str(bench_config), "--model", str(model_config), "-n", "1"],
    )

    assert collect_main() == 0
    assert captured["tasks"] == "inspect_evals/mbpp"
    assert captured["model"] == "custom/model2"
    assert "seed" not in captured
    assert captured["log_dir"] == str(tmp_path / "target_dir" / "0")
    assert captured["display"] == "plain"
    assert isinstance(captured["epochs"], Epochs)
    assert captured["epochs"].epochs == 1


def test_collect_skips_completed_epoch(monkeypatch, tmp_path):
    model_config = tmp_path / "model.yml"
    model_config.write_text("name: custom/model\n")
    bench_config = tmp_path / "bench.yml"
    bench_config.write_text("name: inspect_evals/mbpp\n")
    target = tmp_path / "target" / "0"
    target.mkdir(parents=True)
    (target / "existing.eval").touch()

    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: target.parent)
    monkeypatch.setattr("collect.cli.is_successful_eval", lambda path: True)
    eval_set = MagicMock()
    monkeypatch.setattr("collect.cli.eval_set", eval_set)
    monkeypatch.setattr(
        "sys.argv",
        ["collect", "-b", str(bench_config), "-m", str(model_config)],
    )

    assert collect_main() == 0
    eval_set.assert_not_called()


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

    def mock_eval_set(**kwargs):
        runs.append((kwargs["tasks"], kwargs["model"]))
        return True, [mock_log]

    monkeypatch.setattr("collect.cli.eval_set", mock_eval_set)
    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: tmp_path / "target_dir")
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect",
            "-b", str(b1), "-b", str(b2),
            "-m", str(m1), "-m", str(m2),
            "-n", "1",
        ],
    )

    assert collect_main() == 0
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

    def mock_eval_set(**kwargs):
        captured.update(kwargs)
        return True, [MagicMock()]

    monkeypatch.setattr("collect.cli.eval_set", mock_eval_set)
    monkeypatch.setattr("collect.cli.get_eval_path", lambda **kw: tmp_path / "target_dir")
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

    assert collect_main() == 0
    assert captured["task_args"] == {"fewshot": 5}
    assert captured["model_args"]["temp"] == 0.7
    assert captured["limit"] == 10
    assert captured["max_connections"] == 4
    assert captured["display"] == "plain"


def test_judge_pipeline(monkeypatch, tmp_path):
    (tmp_path / "dom.yml").write_text("name: dom\nbenches:\n  - bench: b1\n")
    (tmp_path / "exp.yml").write_text(f"""name: test_exp
domain: {tmp_path / "dom.yml"}
judge: p/jdg
models: [p/a, p/b]
q_src: 1
""")

    captured, built, ranked = {}, [], []
    monkeypatch.setattr("judge.cli.build_domain", lambda domain, *a: built.append(domain.name))
    monkeypatch.setattr("judge.cli.build_tasks", lambda exp: ["dummy_task"])
    monkeypatch.setattr("judge.cli.eval_set", lambda **kw: (captured.update(kw) or (True, [True])))
    monkeypatch.setattr("judge.cli.rank_logs", lambda log_dir, domain: ranked.append(log_dir))
    monkeypatch.setattr("sys.argv", ["judge", "--cfg", str(tmp_path / "exp.yml"), "--q_src", "2",
                                     "--limit", "10", "--max-connections", "4"])

    assert judge_main() == 0
    assert built == ["dom"]  # artefacts missing -> built first
    assert captured["tasks"] == ["dummy_task"]
    assert captured["model"] == "openrouter/p/jdg"
    assert captured["temperature"] == 0.0 and captured["cache"].expiry is None
    assert captured["limit"] == 10 and captured["max_connections"] == 4
    assert ranked == [Path(captured["log_dir"])] and "/dom/jdg/" in captured["log_dir"]
