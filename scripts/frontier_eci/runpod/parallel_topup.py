#!/usr/bin/env python3
"""Run one resumable budgeted queue per judge behind one aggregate watchdog."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from runpy import run_path


key_usage = run_path(Path(__file__).parents[1] / "run_budgeted.py")["key_usage"]


JUDGES = (
    "deepseek-v4-flash-0731",
    "gemini-2.5-pro",
    "gpt-4o-2024-11-20",
    "gpt-5.2",
    "o4-mini",
)


def partition_configs(paths, judges=JUDGES):
    paths = sorted(map(Path, paths))
    groups = {judge: [] for judge in judges}
    for path in paths:
        if path.parent.name not in groups:
            raise ValueError(f"unrecognized judge directory: {path}")
        groups[path.parent.name].append(path)
    assigned = [path for group in groups.values() for path in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(paths):
        raise ValueError("judge queue partition is not disjoint and complete")
    return groups


def shard_groups(groups, workers_per_judge):
    queues = {}
    for judge, configs in groups.items():
        count = min(workers_per_judge, len(configs))
        for shard in range(count):
            queues[f"{judge}-{shard}"] = (judge, configs[shard::count])
    return queues


def should_stop(*, spent, active_workers, max_reserve,
                usage_limit=125, aggregate_limit=150):
    return spent >= usage_limit or spent + active_workers * max_reserve >= aggregate_limit


def worker_command(judge, configs, *, baseline, limit, preflight, ledger):
    return [
        sys.executable, "scripts/frontier_eci/run_budgeted.py", *map(str, configs),
        "--baseline", str(baseline), "--limit", str(limit),
        "--preflight", str(preflight), "--ledger", str(ledger),
    ]


def terminate(workers):
    for worker in workers.values():
        process = worker["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def consolidate_ledgers(run_root, baseline, aggregate_limit):
    paths = sorted(run_root.glob("ledger-*.jsonl"))
    events = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    headers = [event for event in events if event.get("event") == "header"]
    if not headers or {event["baseline"] for event in headers} != {baseline}:
        raise ValueError("worker ledgers do not share the supplied baseline")
    header = dict(headers[0], incremental_allowance=aggregate_limit)
    pairs = sorted((event for event in events if event.get("event") != "header"),
                   key=lambda event: event["time"])
    (run_root / "ledger.jsonl").write_text(
        "\n".join(json.dumps(event) for event in [header, *pairs]) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=float)
    parser.add_argument("--config-root", type=Path,
                        default=Path("configs/frontier_eci_canonical_anchors/runs"))
    parser.add_argument("--preflight", type=Path,
                        default=Path("judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/preflight.csv"))
    parser.add_argument("--run-root", type=Path,
                        default=Path("judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod"))
    parser.add_argument("--log-root", type=Path,
                        default=Path("logs/frontier-eci-canonical-anchors"))
    parser.add_argument("--provenance", type=Path,
                        default=Path("judge_results/frontier/eci-semiannual-canonical-anchors/provenance"))
    parser.add_argument("--worker-limit", type=float, default=125)
    parser.add_argument("--aggregate-limit", type=float, default=150)
    parser.add_argument("--reserve-multiplier", type=float, default=1.5)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-resumes", type=int, default=2)
    parser.add_argument("--judges", nargs="+", default=list(JUDGES))
    parser.add_argument("--workers-per-judge", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(args.config_root.glob("*/*/*.yaml"))
    groups = partition_configs(paths, judges=args.judges)
    if not paths or any(not group for group in groups.values()):
        raise ValueError("every judge must have at least one config")
    queues = shard_groups(groups, args.workers_per_judge)
    with args.preflight.open(newline="") as handle:
        max_reserve = args.reserve_multiplier * max(
            float(row["upper_cost"]) for row in csv.DictReader(handle))

    args.run_root.mkdir(parents=True, exist_ok=True)
    workers = {}

    def start(worker_id, judge, configs, attempt):
        ledger = args.run_root / f"ledger-{worker_id}.jsonl"
        log = (args.run_root / f"worker-{worker_id}-attempt{attempt}.log").open("a")
        process = subprocess.Popen(
            worker_command(judge, configs, baseline=args.baseline,
                           limit=args.worker_limit, preflight=args.preflight, ledger=ledger),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        workers[worker_id] = {"process": process, "attempt": attempt, "log": log,
                              "handled": False, "judge": judge, "configs": configs}

    for worker_id, (judge, configs) in queues.items():
        start(worker_id, judge, configs, 0)

    pid_path = args.run_root / "worker-pids.json"
    watchdog_path = args.run_root / "watchdog.csv"
    if not watchdog_path.exists():
        watchdog_path.write_text("time,usage,spent,active_workers,projected_inflight_upper\n")

    try:
        while True:
            for worker_id, worker in list(workers.items()):
                status = worker["process"].poll()
                if status is None or worker["handled"]:
                    continue
                worker["log"].close()
                worker["handled"] = True
                if status == 0:
                    continue
                if status == 2 or worker["attempt"] >= args.max_resumes:
                    terminate(workers)
                    return status or 1
                start(worker_id, worker["judge"], worker["configs"], worker["attempt"] + 1)

            active = {worker_id: worker for worker_id, worker in workers.items()
                      if worker["process"].poll() is None}
            pid_path.write_text(json.dumps({
                worker_id: {"pid": worker["process"].pid, "attempt": worker["attempt"]}
                for worker_id, worker in workers.items()
            }, indent=2) + "\n")
            if not active:
                break

            usage = key_usage()
            spent = usage - args.baseline
            projected = spent + len(active) * max_reserve
            with watchdog_path.open("a") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()},{usage},{spent},"
                             f"{len(active)},{projected}\n")
            if should_stop(spent=spent, active_workers=len(active), max_reserve=max_reserve,
                           usage_limit=args.worker_limit,
                           aggregate_limit=args.aggregate_limit):
                terminate(workers)
                return 2

            time.sleep(args.poll_seconds)
    except BaseException:
        terminate(workers)
        raise
    finally:
        for worker in workers.values():
            if not worker["log"].closed:
                worker["log"].close()

    consolidate_ledgers(args.run_root, args.baseline, args.aggregate_limit)
    subprocess.run([
        sys.executable, "scripts/frontier_eci/audit_api.py", str(args.log_root),
        "--out", str(args.provenance),
    ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
