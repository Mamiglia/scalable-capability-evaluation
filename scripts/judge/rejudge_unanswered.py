"""Re-judge samples whose judge verdict did not parse.

For every comparison in a run_judging.py config, find the samples in the finished
phase (and topup) logs whose verdict is NOANSWER with an empty completion, the shape a
provider error leaves behind, and run just those again with the same task and config.
--all-unparsed also retries nonempty completions that never supplied a parseable verdict.
package_judging.py overlays these verdicts on the originals. Samples already answered in
<out_root> are not re-run.

Usage:
  uv run python scripts/judge/rejudge_unanswered.py --config <yaml> --out-root <dir> [--log-root <dir>] [--dry-run]
"""

import argparse
from pathlib import Path

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalSample, read_eval_log

from run_judging import DEFAULT_MAX_CONNECTIONS, judged_sample_ids, make_task, success_logs


def unanswered(sample: EvalSample, all_unparsed: bool = False) -> bool:
    """A NOANSWER verdict on an empty completion: the judge never said anything."""
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
    return {id for id, sample in latest.items() if unanswered(sample, all_unparsed)}


def current_retry_ids(task, ids: set[str]) -> list[str]:
    """Retry ids still present in the current comparison task."""
    if task is None:
        return []
    current = {str(sample.id) for sample in task.dataset}
    return sorted(ids & current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="run_judging.py config")
    parser.add_argument("--out-root", type=Path, required=True, help="where the re-judged logs go")
    parser.add_argument("--log-root", type=Path, help="judge logs to scan (default: the config's log_root)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-unparsed", action="store_true", help="also retry nonempty completions with no verdict")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]
    log_root = args.log_root or Path(cfg["log_root"])
    include_reasoning = bool(cfg.get("include_reasoning", False))
    judge_model_args = cfg.get("judge_model_args") or {}
    epochs = cfg.get("epochs") or 1
    max_connections = cfg.get("max_connections") or DEFAULT_MAX_CONNECTIONS
    generate_args = cfg.get("generate_args") or {}
    fracs = cfg.get("truncate_fracs") or [None]

    # (judge, judge_short, major, minor, frac, sample ids, out dir)
    pending = []
    for frac in fracs:
        root = log_root if frac is None else log_root / f"trunc-{int(frac * 100)}"
        for judge, major, minor in cfg["comparisons"]:
            judge_short = judge.split("/")[-1]
            pair_dir = root / judge_short / f"{major}-vs-{minor}"
            for run_dir in sorted(p for p in pair_dir.glob("*") if p.is_dir()):
                out_dir = args.out_root / run_dir.relative_to(log_root)
                ids = sorted(unanswered_sample_ids(run_dir, args.all_unparsed) - judged_sample_ids(out_dir))
                if ids:
                    pending.append((judge, judge_short, major, minor, frac, ids, out_dir))

    current_pending = []
    for judge, judge_short, major, minor, frac, ids, out_dir in pending:
        try:
            task = make_task(
                paths, judge_short, major, minor, frac, include_reasoning, args.config.stem
            )
        except ValueError as error:
            print(f"SKIP {judge_short} {major}-vs-{minor}: {error}")
            continue
        retry_ids = current_retry_ids(task, set(ids))
        stale = sorted(set(ids) - set(retry_ids))
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
    for judge, judge_short, major, minor, frac, ids, out_dir, task in pending:
        logs = inspect_eval(
            tasks=[task],
            model=judge,
            model_args=judge_model_args.get(judge, {}),
            log_dir=str(out_dir),
            sample_id=ids,
            epochs=epochs,
            max_connections=max_connections,
            **generate_args,
        )
        if not all(log.status == "success" for log in logs):
            failures.append(f"{judge_short} {major}-vs-{minor} -> {out_dir}")
    if failures:
        print(f"{len(failures)} FAILED runs:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
