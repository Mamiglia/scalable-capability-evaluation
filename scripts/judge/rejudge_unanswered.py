"""Re-judge the samples a judge left without an ANSWER line.

For every pair in a run_judging.py config, collect the sample ids whose score has no
answer in the finished phase logs (provider errors come back as empty verdicts), and run
just those again into <out_root>/<judge>/<pair>/phase-<k>/ with the same task and the
judge's model YAML. package_judging.py then overlays these verdicts where they parsed.

Usage: uv run python scripts/judge/rejudge_unanswered.py <config.yaml> <log_root> <out_root> [--dry-run]
"""

import sys
from pathlib import Path

import yaml
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log, read_eval_log_samples

from judge.task import judge_task
from model import Model
from run_judging import configured_system_prompt, judge_config


def latest_success(phase_dir: Path) -> Path | None:
    logs = [(read_eval_log(str(p), header_only=True), p) for p in phase_dir.glob("*.eval")]
    logs = [(h, p) for h, p in logs if h.status == "success"]
    return max(logs, key=lambda hp: hp[0].eval.created)[1] if logs else None


def main() -> int:
    config, log_root, out_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dry_run = "--dry-run" in sys.argv
    cfg = yaml.safe_load(config.read_text())
    paths = cfg["transcripts"]
    include_reasoning = bool(cfg.get("include_reasoning", False))
    system_prompt = configured_system_prompt(cfg)
    total, failures, models = 0, [], {}
    for judge, major, minor in cfg["comparisons"]:
        judge_short = judge.split("/")[-1]
        pair_dir = log_root / judge_short / f"{major}-vs-{minor}"
        for phase_dir in sorted(pair_dir.glob("phase-*")):
            log = latest_success(phase_dir)
            if log is None:
                continue
            ids = [s.id for s in read_eval_log_samples(str(log), all_samples_required=False)
                   if all(sc.answer is None for sc in (s.scores or {}).values())]
            if not ids:
                continue
            total += len(ids)
            print(f"{judge_short} {major}-vs-{minor} {phase_dir.name}: {len(ids)} unanswered")
            if dry_run:
                continue
            task = judge_task(major_path=Path(paths[major]), minor_path=Path(paths[minor]),
                              system_prompt=system_prompt,
                              include_reasoning=include_reasoning,
                              task_name=f"{judge_short}_{major}-vs-{minor}", experiment_name=config.stem)
            if judge not in models:
                models[judge] = Model(judge_config(judge)).__enter__()
            ok, _ = eval_set(tasks=[task], log_dir=str(out_root / judge_short / pair_dir.name / phase_dir.name),
                             sample_id=ids, max_connections=16, **models[judge].eval_kwargs())
            if not ok:
                failures.append(f"{judge_short} {pair_dir.name} {phase_dir.name}")
    for model in models.values():
        model.__exit__(None, None, None)
    print(f"{total} unanswered samples; {len(failures)} failed runs: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
