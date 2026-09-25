"""Retry only unusable judgments, preserving configured generation and provider settings."""
import argparse
import hashlib
import json
import signal
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.solver import generate, system_message
from inspect_ai._util import appdirs

from configs import DirectJudgeConfig
from judge.run import run_tasks
from judge.scorer import direct_judge_scorer, parse_direct_judge_output
from challenges.truncation.run import condition_summary, interrupt, root, save

REPO = Path(__file__).resolve().parents[3]


def unusable(sample):
    answer, _ = parse_direct_judge_output(sample.output.completion if sample.output and sample.output.choices else "")
    return sample.error is not None or answer is None or sample.output.stop_reason == "max_tokens"


def main(domain, attempt, execute, max_retries=None, max_connections=None):
    ROOT = root(domain)
    summary = condition_summary(domain)
    assert hashlib.sha256((ROOT / "inputs" / domain / "job.json").read_bytes()).hexdigest() == summary["job_sha256"]
    assert hashlib.sha256((ROOT / "inputs" / domain / "prompt.txt").read_bytes()).hexdigest() == summary["prompt_sha256"]
    payload = json.loads((ROOT / "inputs" / domain / "job.json").read_text())
    source = Path(json.loads((ROOT / "results" / domain / "status.json").read_text())["log_dir"])
    target = ROOT / "recollections" / domain / f"attempt-{attempt}"
    target.mkdir(parents=True, exist_ok=True)
    plan_path = target / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        bad, repaired = {}, set()
        for path in source.glob("*.eval"):
            header = read_eval_log(path, header_only=True)
            if header.status != "success":
                continue  # a failed task attempt that eval_set later retried; its samples are not the record
            for sample in read_eval_log_samples(path, all_samples_required=False, resolve_attachments=True):
                if unusable(sample):
                    bad[(header.eval.task, str(sample.id))] = str(path)
        for path in (ROOT / "recollections" / domain).rglob("*.eval"):
            header = read_eval_log(path, header_only=True)
            if header.status != "success":
                continue
            for sample in read_eval_log_samples(path, resolve_attachments=True):
                if not unusable(sample):
                    repaired.add((header.eval.task, str(sample.id)))
        plan = {"domain": domain, "attempt": attempt, "created_at": time.time(),
                "settings": "same configured provider, temperature and output cap; bypass cached unusable completions",
                "samples": [{"task": key[0], "item": key[1], "source": path}
                            for key, path in sorted(bad.items()) if key not in repaired]}
        save(plan_path, plan)
    print(json.dumps({"domain": domain, "attempt": attempt, "retry_judgments": len(plan["samples"]), "execute": execute}), flush=True)
    if not execute or not plan["samples"]:
        return 0
    wanted = {(row["task"], row["item"]) for row in plan["samples"]}
    config = DirectJudgeConfig.from_cli(payload["config"])
    config.system_prompt = str(ROOT / "inputs" / domain / "prompt.txt")
    config.cache = False
    overrides = {k: v for k, v in {"max_retries": max_retries, "max_connections": max_connections}.items() if v}
    config._kwargs = config._kwargs | overrides
    assert config.include_reasoning and config.max_length == payload["max_length"] and config.truncate == payload["truncate"]
    appdirs.user_data_path = lambda name: target / "inspect-data"
    appdirs.user_cache_path = lambda name: target / "inspect-cache"
    tasks = []
    for record in payload["tasks"]:
        samples = [Sample.model_validate(row) for row in record["samples"] if (record["name"], str(row["id"])) in wanted]
        if samples:
            tasks.append(Task(name=record["name"], dataset=MemoryDataset(samples),
                version=int(hashlib.sha256(plan_path.read_bytes()).hexdigest()[:8], 16),
                solver=[system_message(config.system_prompt), generate()], scorer=direct_judge_scorer(),
                metadata=record["metadata"] | {"system_prompt": config.system_prompt,
                    "recollection_attempt": attempt, "cache_bypassed": True,
                    "input_manifest_sha256": hashlib.sha256((ROOT / "inputs" / domain / "job.json").read_bytes()).hexdigest()}))
    assert sum(len(task.dataset) for task in tasks) == len(wanted)
    state = {"status": "running", "started_at": time.time(), "requested": len(wanted), "execution_overrides": overrides}
    save(target / "status.json", state)
    try:
        success = run_tasks(tasks, config, target / "logs", display="none")
        fixed = set()
        for path in (target / "logs").glob("*.eval"):
            log = read_eval_log(path, resolve_attachments=True)
            if log.status == "success":
                fixed.update((log.eval.task, str(s.id)) for s in log.samples if not unusable(s))
        assert fixed <= wanted
        state.update(status="completed" if success and fixed == wanted else "incomplete",
                     recovered=len(fixed), missing=sorted(wanted - fixed))
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        state["finished_at"] = time.time()
        save(target / "status.json", state)
    print(json.dumps(state), flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    load_dotenv(Path.cwd() / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="<domain>/<judge>/<semester>/<head|tail|ends>-<max_length>")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-retries", type=int, help="override the frozen job's API retry count")
    parser.add_argument("--max-connections", type=int, help="override the frozen job's concurrency")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(3 * 3600)
    raise SystemExit(main(args.domain, args.attempt, args.run, args.max_retries, args.max_connections))
