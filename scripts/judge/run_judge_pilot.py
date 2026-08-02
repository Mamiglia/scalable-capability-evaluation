"""Run the judge comparisons listed in a config file.

The config (default configs/judge_pilot.yaml) has two sections:
  transcripts:  model shortname -> eval log path
  comparisons:  list of [judge, major, minor] rows

Rows whose (judge, major, minor) already have a log under logs/judge-pilot/
are skipped, so the script can be re-run at any time and only does new work.

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


def existing_runs() -> set:
    """(judge, major_path, minor_path) of runs already logged, complete or in flight."""
    seen = set()
    for p in LOG_ROOT.rglob("*.eval"):
        try:
            log = read_eval_log(p, header_only=True)
        except Exception:
            continue
        md = log.eval.metadata or {}
        if md.get("major_path") and md.get("minor_path"):
            seen.add((log.eval.model,
                      str(Path(md["major_path"]).resolve()),
                      str(Path(md["minor_path"]).resolve())))
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/judge_pilot.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]

    seen = existing_runs()
    pending = []
    for judge, major, minor in cfg["comparisons"]:
        key = (judge,
               str(Path(paths[major]).resolve()),
               str(Path(paths[minor]).resolve()))
        if key not in seen:
            pending.append((judge, major, minor))

    print(f"{len(cfg['comparisons'])} comparisons, "
          f"{len(cfg['comparisons']) - len(pending)} already logged, {len(pending)} to run")
    if args.dry_run:
        for judge, major, minor in pending:
            print(f"  {judge:45s} {major}-vs-{minor}")
        return 0

    for judge, major, minor in pending:
        judge_short = judge.split("/")[-1]
        task = judge_task(
            major_path=Path(paths[major]),
            minor_path=Path(paths[minor]),
            include_reasoning=False,
            task_name=f"{judge_short}_{major}-vs-{minor}",
            experiment_name="judge-pilot",
        )
        kwargs = {"limit": args.limit} if args.limit else {}
        success, _ = eval_set(
            tasks=[task],
            model=judge,
            log_dir=str(LOG_ROOT / judge_short / f"{major}-vs-{minor}"),
            **kwargs,
        )
        if not success:
            print(f"FAILED: {judge_short} on {major}-vs-{minor}")
            return 1
    return 0


if __name__ == "__main__":
    main()
