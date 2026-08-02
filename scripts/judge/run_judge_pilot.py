"""Run the judge comparisons listed in a config file.

The config (default configs/judge_pilot.yaml) has two sections:
  transcripts:  model shortname -> eval log path
  comparisons:  list of [judge, major, minor] rows

Each comparison runs in two phases: first the first PHASE1_ITEMS items of
every comparison (so every analysis has low-n data early), then the rest.
Phases with a successful log are skipped, so re-running only does new work.

Usage:
  uv run python scripts/judge/run_judge_pilot.py --dry-run
  uv run python scripts/judge/run_judge_pilot.py
"""

import argparse
from pathlib import Path

import yaml
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log

from judge.task import judge_task

LOG_ROOT = Path("logs/judge-pilot")
PHASE1_ITEMS = 20


def has_success_log(directory: Path) -> bool:
    for p in directory.glob("*.eval"):
        try:
            if read_eval_log(p, header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/judge_pilot.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]

    tasks = {}

    def get_task(judge_short, major, minor):
        key = (judge_short, major, minor)
        if key not in tasks:
            tasks[key] = judge_task(
                major_path=Path(paths[major]),
                minor_path=Path(paths[minor]),
                include_reasoning=False,
                task_name=f"{judge_short}_{major}-vs-{minor}",
                experiment_name="judge-pilot",
            )
        return tasks[key]

    # (phase, judge, major, minor, sample_ids, log_dir), phase 1 rows first.
    pending = []
    for phase in (1, 2):
        for judge, major, minor in cfg["comparisons"]:
            judge_short = judge.split("/")[-1]
            pair_dir = LOG_ROOT / judge_short / f"{major}-vs-{minor}"
            if has_success_log(pair_dir):
                continue  # fully judged before phasing was introduced
            task = get_task(judge_short, major, minor)
            ids = [s.id for s in task.dataset]
            split = 2 * PHASE1_ITEMS
            block = ids[:split] if phase == 1 else ids[split:]
            if not block:
                continue
            phase_dir = pair_dir / f"phase-{phase}"
            if has_success_log(phase_dir):
                continue
            pending.append((phase, judge, judge_short, major, minor, block, phase_dir))

    n_total = 2 * len(cfg["comparisons"])
    print(f"{len(pending)} phase-runs pending (of {n_total} possible)")
    if args.dry_run:
        for phase, _, judge_short, major, minor, block, _ in pending:
            print(f"  phase{phase} {judge_short:28s} {major}-vs-{minor} ({len(block)//2} items)")
        return 0

    for phase, judge, judge_short, major, minor, block, phase_dir in pending:
        success, _ = eval_set(
            tasks=[get_task(judge_short, major, minor)],
            model=judge,
            log_dir=str(phase_dir),
            sample_id=block,
            max_connections=40,
        )
        if not success:
            print(f"FAILED: phase{phase} {judge_short} on {major}-vs-{minor}")
            return 1
    return 0


if __name__ == "__main__":
    main()
