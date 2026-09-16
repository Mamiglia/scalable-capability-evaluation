"""Exercise run.py (rendering, saved logs, rating reuse, calibration, incomplete status) without API calls."""
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from challenges.truncation import run

with tempfile.TemporaryDirectory() as directory:
    run.ROOT = Path(directory)
    condition = "cyber/mock/s/head-100"
    inputs = run.ROOT / "inputs" / condition
    inputs.mkdir(parents=True)
    payload = {"domain": "cyber", "semester": "s", "judge": "mock", "max_length": 100, "truncate": "head",
               "config": {"name": "mock", "judge": {"name": "mock", "slug": "mockllm/model"},
                          "include_reasoning": True, "max_length": 100, "truncate": "head"},
               "references": {"r1": 0., "r2": 2.}, "truth": {"t": 4.},
               "reused_ratings": [{"bench": "b", "judge": "mock", "item": "i", "model": "r1",
                                   "score": 2., "semester": "s", "epoch": 1}],
               "audit": [{"task": f"b/{model}", "item": "i"} for model in ("r1", "r2", "t")],
               "tasks": [{"name": f"b/{model}",
                          "metadata": {"protocol": "direct", "bench": "b", "model": model, "semester": "s",
                                       "include_reasoning": True, "max_length": 100, "truncate": "head"},
                          "samples": [{"id": "i", "input": "Correct response: ANSWER: A",
                                       "metadata": {"truncated": True, "chars": 100}}]} for model in ("r2", "t")]}

    def freeze():
        (inputs / "job.json").write_text(json.dumps(payload))
        (inputs / "prompt.txt").write_text("Rate capability from the visible correct response.")
        (inputs / "summary.json").write_text(json.dumps({
            "job_sha256": hashlib.sha256((inputs / "job.json").read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256((inputs / "prompt.txt").read_bytes()).hexdigest()}))

    freeze()
    requests = []

    def mock_requests(tasks, config, logs, **kwargs):
        assert config.include_reasoning and config.max_length == 100
        for task in tasks:
            requests.append(task.name)
            number = {"b/r2": 4, "b/t": 6}[task.name]
            output = ModelOutput.from_content("mockllm/model", f"ANSWER: {number}")
            output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
            result = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=[output]),
                                  log_dir=str(logs), display="none")
            assert result[0].status == "success"
        return True

    run.run_tasks = mock_requests
    assert run.run(condition) == 0
    forecasts = pd.read_csv(run.ROOT / "results" / condition / "forecasts.csv")
    assert forecasts.model.tolist() == ["t"] and abs(forecasts.theta_hat.iloc[0] - 4.) < 1e-10
    assert len(pd.read_csv(run.ROOT / "results" / condition / "ratings.csv")) == 3
    assert run.run(condition) == 0 and len(requests) == 2  # finished tasks are reused
    payload["tasks"][0]["samples"][0]["input"] = "Changed evidence: ANSWER: A"
    freeze()
    assert run.run(condition) == 0 and len(requests) == 4  # changed inputs get a new task version
    payload["tasks"][0]["samples"][0]["input"] = "Missing-rating case: ANSWER: A"
    freeze()
    run.run_tasks = lambda *args, **kwargs: True
    assert run.run(condition) == 1
    status = json.loads((run.ROOT / "results" / condition / "status.json").read_text())
    assert status["status"] == "incomplete" and len(status["missing_rating_keys"]) == 2

# A channel condition runs from its own root and checks the channel field on config and tasks.
with tempfile.TemporaryDirectory() as directory:
    run.CHANNELS_ROOT = Path(directory)
    condition = "cyber/mock/s/channels-drop-outputs"
    inputs = run.CHANNELS_ROOT / "inputs" / condition
    inputs.mkdir(parents=True)
    payload["config"]["channels"] = payload["channels"] = "drop-outputs"
    for record in payload["tasks"]:
        record["metadata"]["channels"] = "drop-outputs"
    freeze()
    run.run_tasks, requests = mock_requests, []
    assert run.run(condition) == 0 and len(requests) == 2
    payload["tasks"][0]["metadata"]["channels"] = "only-outputs"
    freeze()
    try:
        run.run(condition)
    except AssertionError:
        pass
    else:
        raise SystemExit("a task rendered under another channel condition was not refused")
print("Truncation collection self-check passed.")
