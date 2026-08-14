from configs.model import ModelConfig
from configs.paths import get_eval_path
from configs.task import TaskConfig


def test_model_evals_orders_successful_repeats(tmp_path, monkeypatch):
    from configs.experiment import ExperimentConfig

    bench = TaskConfig(name="bench1")
    model = ModelConfig(name="model1", slug="model1")

    exp = ExperimentConfig(
        name="test_exp",
        benchmark=bench,
        major=model,
        minor=model,
        judge=model,
    )

    eval_dir = tmp_path / "bench1" / "hash" / "model1"
    files = {}
    for repeat, filename in enumerate(["run1.eval", "run2.eval", "failed_run3.eval"]):
        repeat_dir = eval_dir / str(repeat)
        repeat_dir.mkdir(parents=True)
        files[filename] = repeat_dir / filename
        files[filename].write_text("dummy")

    monkeypatch.setattr(
        "configs.experiment.is_successful_eval", lambda path: "failed" not in str(path)
    )
    monkeypatch.setattr("configs.experiment.get_eval_path", lambda **kw: eval_dir)

    assert exp.model_evals(model) == [files["run1.eval"], files["run2.eval"]]


def test_resolve_pairs_without_logs_raises(tmp_path, monkeypatch):
    import pytest

    from configs.experiment import ExperimentConfig

    model = ModelConfig(name="model1", slug="model1")
    exp = ExperimentConfig(
        name="test_exp",
        benchmark=TaskConfig(name="bench1"),
        major=model,
        minor=model,
        judge=model,
    )
    monkeypatch.setattr("configs.experiment.get_eval_path", lambda **kw: tmp_path / "empty")

    with pytest.raises(ValueError, match="No successful eval logs"):
        exp.resolve_pairs()


def test_eval_kwargs_does_not_mutate_config():
    from configs.experiment import ExperimentConfig

    model = ModelConfig(name="model1", slug="model1")
    exp = ExperimentConfig(
        name="test_exp",
        benchmark=TaskConfig(name="bench1"),
        major=model,
        minor=model,
        judge=model,
        limit=5,
    )
    assert exp.eval_kwargs() == {"limit": 5}
    assert exp.eval_kwargs() == {"limit": 5}
    assert exp._kwargs == {}


def test_task_eval_kwargs_does_not_mutate_config():
    task = TaskConfig(name="bench1", limit=3)
    first = task.eval_kwargs()
    assert first["limit"] == 3
    assert task.eval_kwargs() == first
    assert task._kwargs == {}


def test_get_eval_path_is_lowercased():
    task = TaskConfig(name="Bench1", task="mock_bench_no_args")
    model = ModelConfig(name="Org/Model-1", slug="org/model-1")
    assert get_eval_path(task, model, base_dir="data").parts[1] == "bench1"
    assert get_eval_path(task, model, base_dir="data").parts[-1] == "model-1"
