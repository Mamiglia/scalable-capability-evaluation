import sys
from pathlib import Path

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import NOANSWER
from inspect_ai.solver import solver

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import rescore_judge
from judge.scorer import judge_scorer


@solver
def fixed_verdict():
    async def solve(state, generate):
        state.output = ModelOutput.from_content(
            model="mockllm/model", content='{"ANSWER": 1}', stop_reason="stop"
        )
        state.messages.append(state.output.message)
        return state

    return solve


def test_judge_rescore_copy_in_place_and_path_safety(monkeypatch, tmp_path):
    task = Task(
        dataset=[Sample(input="question", target="1", id="item_1")],
        solver=fixed_verdict(),
        scorer=judge_scorer(),
        name="judge-probe",
    )
    source = Path(
        inspect_eval(
            task, model="mockllm/model", log_dir=str(tmp_path / "source"), display="none"
        )[0].location
    )
    log = read_eval_log(str(source))
    log.samples[0].scores["judge_scorer"].value = NOANSWER
    log.samples[0].scores["judge_scorer"].answer = None
    write_eval_log(log, str(source))

    destination = tmp_path / "mirror" / source.name
    rescore_judge.rescore((source, destination))
    assert read_eval_log(str(source)).samples[0].scores["judge_scorer"].answer is None
    assert read_eval_log(str(destination)).samples[0].scores["judge_scorer"].answer == "1"

    rescore_judge.rescore((source, source))
    assert read_eval_log(str(source)).samples[0].scores["judge_scorer"].answer == "1"

    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setattr(
        sys, "argv", ["rescore_judge", str(root), "--out", str(root / "rescored")]
    )
    with pytest.raises(SystemExit):
        rescore_judge.main()
