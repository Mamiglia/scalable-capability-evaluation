"""Re-judge samples whose judge verdict did not parse.

For every comparison in a run_judging.py config, find the samples in the finished
phase (and topup) logs whose verdict is NOANSWER with an empty completion, the shape a
provider error leaves behind, and run just those again with the same task and config.
--all-unparsed also retries nonempty completions that never supplied a parseable verdict.
package_judging.py overlays these verdicts on the originals. Samples already answered in
<out_root> are not re-run. Endpoint pins come from model YAMLs under ``configs/models``.

Usage:
  uv run python scripts/judge/rejudge_unanswered.py \
    --config <yaml> --out-root <dir> [--log-root <dir>] [--dry-run]
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalSample, read_eval_log

from judge.task import judge_task
from model import Model
from run_judging import (
    DEFAULT_MAX_CONNECTIONS,
    configured_comparisons,
    configured_system_prompt,
    judge_config,
    judged_sample_ids,
    success_logs,
    validated_truncate_keep,
)


def unanswered(sample: EvalSample, all_unparsed: bool = False) -> bool:
    """Whether a sample has no parsed verdict, optionally despite nonempty output."""
    scores = sample.scores or {}
    no_answer = not scores or any(score.answer is None for score in scores.values())
    empty = sample.output is None or not (sample.output.completion or "").strip()
    return no_answer and (all_unparsed or empty)


def unanswered_sample_ids(directory: Path, all_unparsed: bool = False) -> set[str]:
    """Unanswered ids after later finished attempts override earlier ones."""
    latest = {}
    for path in success_logs(directory):
        for sample in read_eval_log(str(path)).samples or []:
            latest[str(sample.id)] = sample
    return {
        sample_id for sample_id, sample in latest.items()
        if unanswered(sample, all_unparsed)
    }


def current_retry_ids(task, ids: set[str]) -> list[str]:
    """Retry ids still present in the current comparison task."""
    if task is None:
        return []
    current = {str(sample.id) for sample in task.dataset}
    return sorted(ids & current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="run_judging.py config")
    parser.add_argument("--out-root", type=Path, required=True, help="where re-judged logs go")
    parser.add_argument("--log-root", type=Path, help="judge logs to scan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-unparsed", action="store_true", help="also retry nonempty completions with no verdict")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if "judge_model_args" in cfg:
        raise ValueError("judge_model_args is no longer read: endpoint pins live in model YAMLs")
    paths = cfg["transcripts"]
    log_root = args.log_root or Path(cfg["log_root"])
    system_prompt = configured_system_prompt(cfg)
    include_reasoning = bool(cfg.get("include_reasoning", False))
    keep = validated_truncate_keep(cfg)
    epochs = cfg.get("epochs") or 1
    max_connections = cfg.get("max_connections") or DEFAULT_MAX_CONNECTIONS
    generate_args = cfg.get("generate_args") or {}
    fracs = cfg.get("truncate_fracs") or [None]

    pending = []
    for frac in fracs:
        if frac is None:
            root = log_root
        elif keep == "last":
            root = log_root / f"trunc-end-{int(frac * 100)}"
        else:
            root = log_root / f"trunc-{int(frac * 100)}"
        for judge, major, minor in configured_comparisons(cfg):
            judge_short = judge.split("/")[-1]
            pair_dir = root / judge_short / f"{major}-vs-{minor}"
            for run_dir in sorted(path for path in pair_dir.glob("*") if path.is_dir()):
                out_dir = args.out_root / run_dir.relative_to(log_root)
                sample_ids = sorted(
                    unanswered_sample_ids(run_dir, args.all_unparsed)
                    - judged_sample_ids(out_dir)
                )
                if sample_ids:
                    pending.append(
                        (judge, judge_short, major, minor, frac, sample_ids, out_dir)
                    )

    current_pending = []
    for judge, judge_short, major, minor, frac, sample_ids, out_dir in pending:
        if not frac:
            suffix = ""
        elif keep == "last":
            suffix = f"_tend{int(frac * 100)}"
        else:
            suffix = f"_t{int(frac * 100)}"
        try:
            task = judge_task(
                major_path=Path(paths[major]), minor_path=Path(paths[minor]),
                system_prompt=system_prompt, include_reasoning=include_reasoning,
                truncate_frac=frac, truncate_keep=keep,
                task_name=f"{major}-vs-{minor}{suffix}",
                experiment_name=args.config.stem,
            )
        except ValueError as error:
            print(f"SKIP {judge_short} {major}-vs-{minor}: {error}")
            continue
        retry_ids = current_retry_ids(task, set(sample_ids))
        stale = sorted(set(sample_ids) - set(retry_ids))
        if stale:
            print(f"SKIP stale ids {judge_short} {major}-vs-{minor}: {stale}")
        if retry_ids:
            current_pending.append(
                (judge, judge_short, major, minor, frac, retry_ids, out_dir, task)
            )
    pending = current_pending

    n_samples = sum(len(ids) for *_, ids, _, _ in pending)
    print(f"{n_samples} unanswered samples in {len(pending)} runs")
    for _, judge_short, major, minor, frac, ids, _, _ in pending:
        t = f" t={frac}" if frac else ""
        print(f"  {judge_short:28s} {major}-vs-{minor}{t}: {len(ids)} samples")
    if args.dry_run:
        return 0

    failures = []
    models = {}
    with ExitStack() as stack:
        for judge, judge_short, major, minor, _frac, sample_ids, out_dir, task in pending:
            if judge not in models:
                models[judge] = stack.enter_context(Model(judge_config(judge)))
            eval_kwargs = models[judge].eval_kwargs()
            eval_kwargs.update(generate_args)
            eval_kwargs["max_connections"] = max_connections
            if "cache" in cfg:
                eval_kwargs["cache"] = bool(cfg["cache"])
            logs = inspect_eval(
                tasks=[task], log_dir=str(out_dir), sample_id=sample_ids,
                epochs=epochs, **eval_kwargs,
            )
            if not all(log.status == "success" for log in logs):
                failures.append(f"{judge_short} {major}-vs-{minor} -> {out_dir}")
    if failures:
        print(f"{len(failures)} FAILED runs:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
