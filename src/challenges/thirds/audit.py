"""Audit the thirds runs of one domain x semester against their full-transcript baseline; no API calls.

    PYTHONPATH=src .venv/bin/python src/challenges/thirds/audit.py <domain> <semester> [--inputs]

Per thirds condition dir: every log succeeded with no sample limit and the right truncate metadata, every
logged sample id exists in the baseline dir (a pair the baseline left unscored on overflow may be scored
here, so ids are compared, not verdicts), and the unscored / no-answer counts. --inputs also rebuilds the
condition's tasks from data/ and checks every logged prompt byte for byte (reads every transcript).
One JSON line per condition; exit 1 on any mismatch.
"""
import argparse
import json
import sys

from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

from capability.task import Task
from challenges.thirds.runs import harnesses
from configs.paths import is_successful_eval


def sample_ids(run_dir):
    return {(read_eval_log(str(p), header_only=True).eval.task, str(s.id))
            for p in run_dir.glob("*.eval") if is_successful_eval(p) for s in read_eval_log_sample_summaries(str(p))}


def audit(domain, semester, inputs=False):
    dom, runs = harnesses(domain, semester)
    baseline = sample_ids(runs["full"].log_dir(semester))
    task = Task.from_semester(semester, dom) if inputs else None
    ok = True
    for condition, harness in runs.items():
        if condition == "full":
            continue
        run_dir = harness.log_dir(semester)
        logs = sorted(run_dir.glob("*.eval"))
        if not logs:
            continue
        row = {"domain": domain, "semester": semester, "condition": condition, "dir": str(run_dir), "logs": len(logs),
               "failed_logs": 0, "bad_metadata": 0, "samples": 0, "unscored": 0, "not_in_baseline": 0, "input_mismatch": 0}
        built = {}
        if inputs:
            built = {(t.name, str(s.id)): s.input for t in harness.tasks(task.references, task.test) for s in t.dataset}
        for p in logs:
            log = read_eval_log(str(p), header_only=True)
            if log.status != "success":
                row["failed_logs"] += 1
                continue
            md = log.eval.metadata or {}
            if md.get("truncate") != condition or log.eval.config.limit is not None or not md.get("include_reasoning"):
                row["bad_metadata"] += 1
            for s in read_eval_log_sample_summaries(str(p)):
                row["samples"] += 1
                score = next(iter((s.scores or {}).values()), None)
                row["unscored"] += (score.answer if score else None) not in ("1", "2")
                row["not_in_baseline"] += (log.eval.task, str(s.id)) not in baseline
            if inputs:
                for s in read_eval_log(str(p), resolve_attachments=True).samples:
                    row["input_mismatch"] += built.get((log.eval.task, str(s.id))) != s.input
        row["ok"] = not (row["failed_logs"] or row["bad_metadata"] or row["not_in_baseline"] or row["input_mismatch"])
        ok &= row["ok"]
        print(json.dumps(row), flush=True)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain")
    parser.add_argument("semester")
    parser.add_argument("--inputs", action="store_true", help="also rebuild every prompt from data/ and compare")
    args = parser.parse_args()
    sys.exit(0 if audit(args.domain, args.semester, args.inputs) else 1)
