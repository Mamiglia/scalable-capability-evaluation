"""Top up missing direct ratings item by item, into the existing JudgeDirect run dirs.

JudgeDirect.judge() resumes per task (one per bench/model) and skips a task that already has a
log, so items it never rated stay missing. This builds item-level tasks for the UNCOLLECTED rows
of a coverage-gap table and runs them into the same run dir; from_logs keeps the latest rating
per item, so a top-up log wins. Same judge config (provider, temperature 0, output cap), generate
cache bypassed so a cached unusable completion is not replayed.

    PYTHONPATH=src .venv/bin/python -m challenges.direct_topup.collect analysis/direct-coverage-gaps.csv [--judge J ...] [--run]
"""
import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from configs.domain import DomainConfig
from configs.judge import DirectJudgeConfig
from judge.run import run_tasks
from judge.task import direct_judge_task
from protocols.direct_judge import JudgeDirect
from protocols.direct_judge.judge import from_logs
from reference.tables import DATA

# declared failure: 32k output cap exhausted three times in the hidden-reasoning sweep
SKIP = {("math", "direct_reasoning-hidden", "2026-I", "glm-4.7", "kimi-k2-thinking", "apex_shortlist", "24")}
KEY = ["domain", "condition", "semester", "judge"]


def main(gaps_csv: Path, judges: list[str], run: bool) -> int:
    gaps = pd.read_csv(gaps_csv, dtype=str).fillna("")
    gaps = gaps[gaps.status == "UNCOLLECTED"]
    if judges:
        gaps = gaps[gaps.judge.isin(judges)]
    failed = 0
    for (domain, condition, semester, judge), group in gaps.groupby(KEY):
        dom = DomainConfig.from_cli(Path(f"configs/domain/{domain}.yml"))
        jd = JudgeDirect(DirectJudgeConfig.from_cli(f"configs/judges/{judge}.yml"), dom,
                         include_reasoning=condition == "direct")
        cfg = replace(jd.config, cache=False)
        log_dir = jd.log_dir(semester)
        assert log_dir.is_dir(), log_dir
        rated = from_logs(log_dir)
        done = set(zip(rated.model, rated.bench, rated["item"]))
        models = pd.read_csv(f"data/meta/{domain}/models.csv", dtype=str).set_index("model")
        prompts = jd._prompts()
        tasks, wanted = [], set()
        for (model, bench), rows in group.groupby(["model", "bench"]):
            items = {i for i in rows["item"] if (model, bench, i) not in done
                     and (domain, condition, semester, judge, model, bench, i) not in SKIP}
            if not items:
                continue
            task = direct_judge_task(
                DATA / models.loc[model, f"path_{bench}"], items=items, system_prompt=prompts[bench],
                max_length=cfg.max_length, include_reasoning=cfg.include_reasoning, truncate=cfg.truncate,
                task_name=f"{bench}/{model}",
                metadata={"protocol": "direct", "domain": domain, "bench": bench, "model": model,
                          "semester": semester, "topup_items": sorted(items), "cache_bypassed": True})
            if task is None:
                print(f"{domain} {semester} {judge} {bench}/{model}: no eligible transcript for {sorted(items)}", flush=True)
                continue
            # a new task version keeps eval_set from pairing this with the bench/model log already in the dir
            task.version = int(hashlib.sha256(repr(("topup", sorted(items))).encode()).hexdigest()[:8], 16)
            tasks.append(task)
            wanted.update((model, bench, i) for i in items)
        print(json.dumps({"domain": domain, "condition": condition, "semester": semester, "judge": judge,
                          "log_dir": str(log_dir), "tasks": len(tasks), "items": len(wanted), "run": run}), flush=True)
        if not run or not tasks:
            continue
        started = time.time()
        ok = run_tasks(tasks, cfg, log_dir, display="none")
        rated = from_logs(log_dir)
        got = wanted & set(zip(rated.model, rated.bench, rated["item"]))
        state = {"judge": judge, "domain": domain, "semester": semester, "ok": ok, "recovered": len(got),
                 "still_missing": sorted(wanted - got), "minutes": round((time.time() - started) / 60, 1)}
        print(json.dumps(state), flush=True)
        failed += int(not ok or got != wanted)
    return int(failed > 0)


if __name__ == "__main__":
    load_dotenv(Path.cwd() / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gaps", type=Path)
    parser.add_argument("--judge", nargs="*", default=[])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.gaps, args.judge, args.run))
