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

DEFAULT_LOG_ROOT = Path("logs/gpqa-pilot-rd2")  # used when the config omits log_root
DEFAULT_PHASE1_ITEMS = 20  # config key phase1_items overrides


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
    parser.add_argument("--phase", type=int, choices=(1, 2),
                        help="run only this phase; default runs both")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]
    log_root = Path(cfg.get("log_root") or DEFAULT_LOG_ROOT)
    phase1_items = cfg.get("phase1_items") or DEFAULT_PHASE1_ITEMS
    # epochs > 1 judges every sample that many times; used to measure how much
    # verdicts vary for a fixed input.
    epochs = cfg.get("epochs") or 1

    fracs = cfg.get("truncate_fracs") or [None]
    tasks = {}

    def get_task(judge_short, major, minor, frac):
        key = (judge_short, major, minor, frac)
        if key not in tasks:
            suffix = f"_t{int(frac * 100)}" if frac else ""
            tasks[key] = judge_task(
                major_path=Path(paths[major]),
                minor_path=Path(paths[minor]),
                include_reasoning=False,
                truncate_frac=frac,
                task_name=f"{judge_short}_{major}-vs-{minor}{suffix}",
                experiment_name="gpqa-pilot-rd2",
            )
        return tasks[key]

    # (phase, judge, major, minor, frac, sample_ids, log_dir), phase 1 rows first.
    pending = []
    for phase in ((args.phase,) if args.phase else (1, 2)):
        for frac in fracs:
            root = log_root if frac is None else log_root / f"trunc-{int(frac * 100)}"
            for judge, major, minor in cfg["comparisons"]:
                judge_short = judge.split("/")[-1]
                pair_dir = root / judge_short / f"{major}-vs-{minor}"
                if frac is None and has_success_log(pair_dir):
                    continue  # fully judged before phasing was introduced
                task = get_task(judge_short, major, minor, frac)
                ids = [s.id for s in task.dataset]
                split = 2 * phase1_items
                block = ids[:split] if phase == 1 else ids[split:]
                if not block:
                    continue
                phase_dir = pair_dir / f"phase-{phase}"
                if has_success_log(phase_dir):
                    continue
                pending.append((phase, judge, judge_short, major, minor, frac, block, phase_dir))

    n_total = 2 * len(fracs) * len(cfg["comparisons"])
    print(f"{len(pending)} phase-runs pending (of {n_total} possible)")
    if args.dry_run:
        for phase, _, judge_short, major, minor, frac, block, _ in pending:
            t = f" t={frac}" if frac else ""
            print(f"  phase{phase}{t} {judge_short:28s} {major}-vs-{minor} ({len(block)//2} items)")
        return 0

    # Optional per-judge model args from the config, e.g. OpenRouter provider
    # routing for a judge whose default fallback provider rejects the request.
    model_args = cfg.get("judge_model_args") or {}

    failures = []
    for phase, judge, judge_short, major, minor, frac, block, phase_dir in pending:
        success, _ = eval_set(
            tasks=[get_task(judge_short, major, minor, frac)],
            model=judge,
            model_args=model_args.get(judge, {}),
            log_dir=str(phase_dir),
            sample_id=block,
            epochs=epochs,
            max_connections=40,
        )
        if not success:
            failures.append(f"phase{phase} {judge_short} on {major}-vs-{minor}")
    if failures:
        print(f"{len(failures)} FAILED runs (rest of queue completed):")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
