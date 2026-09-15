"""Run frozen truncated direct tasks and reference-only calibration."""
import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, system_message
from inspect_ai._util import appdirs

from capability.evaluate import score
from configs import DirectJudgeConfig
from configs.paths import is_successful_eval
from evals.transcripts import read_header
from judge.run import run_tasks
from judge.dataset import ENDS_SEPARATOR
from judge.scorer import direct_judge_scorer
from protocols.direct_judge import from_logs
from protocols.direct_judge.scoring import calibrate, mean_scores

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/direct-judging-truncation"  # frozen inputs, logs, results


def condition_summary(condition):
    return json.loads((ROOT / "inputs" / condition / "summary.json").read_text())


def tag(condition):
    """Flat file-name form of a condition path: cyber/qwen3-235b-a22b/2025-II/head-4000 -> cyber-qwen3-235b-a22b-2025-II-head-4000."""
    return condition.replace("/", "-")


def interrupt(*_):
    raise KeyboardInterrupt


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def run(condition, max_retries=None, max_connections=None):
    """Execution knobs (retries, connections) do not enter the condition or the judged request."""
    inputs, output = ROOT / "inputs" / condition, ROOT / "results" / condition
    output.mkdir(parents=True, exist_ok=True)
    expected = condition_summary(condition)
    assert hashlib.sha256((inputs / "job.json").read_bytes()).hexdigest() == expected["job_sha256"]
    assert hashlib.sha256((inputs / "prompt.txt").read_bytes()).hexdigest() == expected["prompt_sha256"]
    payload = json.loads((inputs / "job.json").read_text())
    config = DirectJudgeConfig.from_cli(payload["config"])
    assert config.include_reasoning and config.limit == 0 and config.max_length == payload["max_length"] and config.truncate == payload["truncate"]
    config.system_prompt = str(inputs / "prompt.txt")
    overrides = {k: v for k, v in {"max_retries": max_retries, "max_connections": max_connections}.items() if v}
    config._kwargs = config._kwargs | overrides
    logs = output / "logs" / f"{config.condition_hash}-{expected['job_sha256'][:8]}"
    appdirs.user_data_path = lambda name: output / "inspect-data"
    appdirs.user_cache_path = lambda name: output / "inspect-cache"
    done = {read_header(p).eval.task for p in logs.glob("*.eval") if is_successful_eval(p)}
    tasks = []
    for record in payload["tasks"]:
        if record["name"] in done:
            continue
        samples = [Sample.model_validate(s) for s in record["samples"]]
        slack = len(ENDS_SEPARATOR) if payload["truncate"] == "ends" else 0  # the separator is not transcript text
        assert all(s.metadata["chars"] <= payload["max_length"] + slack for s in samples)
        assert record["metadata"]["include_reasoning"] is True and record["metadata"]["max_length"] == payload["max_length"]
        assert record["metadata"]["truncate"] == payload["truncate"]
        tasks.append(Task(name=record["name"], dataset=MemoryDataset(samples),
                          version=int(expected["job_sha256"][:8], 16),
                          solver=[system_message(config.system_prompt), generate()], scorer=direct_judge_scorer(),
                          metadata=record["metadata"] | {"system_prompt": config.system_prompt,
                              "execution_host": os.uname().nodename,
                              "input_manifest_sha256": expected["job_sha256"]}))
    state = expected | {"status": "running", "pid": os.getpid(), "started_at": time.time(), "execution_overrides": overrides,
                        "log_dir": str(logs), "execution_host": os.uname().nodename}
    save(output / "status.json", state)
    print(json.dumps(state), flush=True)
    try:
        success = run_tasks(tasks, config, logs, display="none") if tasks else True
        fresh = from_logs(logs)
        frames = [frame for frame in (fresh, pd.DataFrame(payload["reused_ratings"])) if not frame.empty]
        ratings = pd.concat(frames, ignore_index=True) if frames else fresh
        allowed = {(row["task"], row["item"]) for row in payload["audit"]}
        assert all((f"{row.bench}/{row.model}", str(row.item)) in allowed for row in ratings.itertuples())
        assert not ratings.duplicated(["model", "bench", "item", "epoch"]).any()
        refs, truth = pd.Series(payload["references"]), pd.Series(payload["truth"])
        prediction = calibrate(ratings, refs).reindex(truth.index)
        forecasts = pd.DataFrame({"semester": payload["semester"], "model": truth.index,
                                  "theta_hat": prediction.to_numpy(), "theta": truth.to_numpy()})
        ratings.to_csv(output / "ratings.csv", index=False)
        forecasts.to_csv(output / "forecasts.csv", index=False)
        mean_scores(ratings).rename("mean_rating").to_frame().join(refs.rename("theta"), how="inner").to_csv(output / "reference-scores.csv")
        pd.DataFrame([score(prediction, truth)]).to_csv(output / "metrics.csv", index=False)
        present = {(f"{row.bench}/{row.model}", str(row.item)) for row in ratings.itertuples()}
        missing = sorted(allowed - present)
        complete = success and not missing and prediction.notna().all()
        state.update(status="completed" if complete else "incomplete", valid_ratings=len(ratings),
                     missing_rating_keys=missing, missing_forecasts=prediction.index[prediction.isna()].tolist(),
                     metrics=score(prediction, truth))
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        state["finished_at"] = time.time()
        save(output / "status.json", state)
    print(json.dumps(state), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    load_dotenv(Path.cwd() / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="<domain>/<judge>/<semester>/<head|tail|ends>-<max_length>")
    parser.add_argument("--max-retries", type=int, help="override the frozen job's API retry count")
    parser.add_argument("--max-connections", type=int, help="override the frozen job's concurrency")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(6 * 3600)
    raise SystemExit(run(args.condition, args.max_retries, args.max_connections))
