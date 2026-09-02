"""The stateless bash+python SWE-bench scaffold builds and wires up via the bench config."""

from configs.task import TaskConfig
from inspect_ai._eval.loader import solver_from_spec


def test_swe_bash_python_builds():
    from collect.swe_solver import swe_bash_python

    assert swe_bash_python() is not None


def test_module_drops_openai_incompatible_tools():
    # The scaffold exists to avoid bash_session and text_editor, whose schemas OpenAI rejects.
    import collect.swe_solver as m

    assert not hasattr(m, "bash_session")
    assert not hasattr(m, "text_editor")


def test_bench_config_resolves_solver():
    task = TaskConfig.from_cli("configs/bench/swe-bench-mini.yml")
    kwargs = task.eval_kwargs()
    assert "solver" in kwargs
    # file.py@name spec resolves to a solver (agent bridged via as_solver) without error.
    assert solver_from_spec(kwargs["solver"]) is not None
