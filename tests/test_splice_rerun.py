import sys
from pathlib import Path

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import match
from inspect_ai.solver import solver

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "collect"))

from splice_rerun import failure_kind, recompute_results, splice


@solver
def scripted(kinds):
    async def solve(state, generate):
        kind = kinds[str(state.sample_id)]
        if kind == "errored":
            raise RuntimeError("provider blew up")
        state.output = ModelOutput.from_content(
            model="mockllm/model",
            content="ANSWER: x" if kind == "correct" else "",
            stop_reason="max_tokens" if kind == "truncated" else "stop",
        )
        state.messages.append(state.output.message)
        return state

    return solve


def write_log(kinds, log_dir, sample_ids=None):
    task = Task(
        dataset=[Sample(input=f"q{id}", target="x", id=id) for id in ("a", "b", "c")],
        solver=scripted(kinds),
        scorer=match(),
        name="probe",
    )
    return Path(
        inspect_eval(
            task,
            model="mockllm/model",
            log_dir=str(log_dir),
            sample_id=sample_ids,
            fail_on_error=False,
            display="none",
        )[0].location
    )


def test_splice_uses_only_successful_reruns_and_recomputes_metrics(tmp_path):
    canonical_path = write_log(
        {"a": "correct", "b": "truncated", "c": "errored"}, tmp_path / "canonical"
    )
    rerun_path = write_log(
        {"a": "correct", "b": "correct", "c": "truncated"},
        tmp_path / "rerun",
        ["b", "c"],
    )
    log = read_eval_log(str(canonical_path))
    replaced = splice(log, [(str(rerun_path), read_eval_log(str(rerun_path)))])
    recompute_results(log)

    assert replaced == {("b", 1): str(rerun_path)}
    assert {s.id: failure_kind(s) for s in log.samples} == {
        "a": None,
        "b": None,
        "c": "errored",
    }
    assert log.results.completed_samples == 2
    assert log.results.scores[0].metrics["accuracy"].value == pytest.approx(1.0)
