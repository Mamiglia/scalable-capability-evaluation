"""Run the judge comparisons listed in a config file.

The config (see gen_judge_config.py) has these sections:
  log_root:          where judge logs go
  transcripts:       model shortname -> eval log path
  comparisons:       list of [judge, major, minor] rows
  judge_model_args:  per-judge model args (endpoint pins), keyed by judge slug
Optional knobs:
  phase1_items:      items judged in the early phase (default 20)
  epochs:            judge every sample this many times, to measure how much
                     verdicts vary for a fixed input (default 1)
  truncate_fracs:    list of fractions; each adds a length-control run with both
                     transcripts cut to that fraction of the shorter one
                     (logs under trunc-<pct>/)
  include_reasoning: judges see candidates' reasoning (protocol default: false)
  generate_args:     judge generation settings passed to eval_set (max_tokens,
                     extra_body, ...); a config holds one judge, so these are per judge
  min_correct:       drop comparisons where either model has fewer than this many
                     correct samples (default 0: keep all)
  max_connections:   concurrent judge requests (default 40; lower for a
                     self-served pod)
  only_items:        mapping <major>-vs-<minor>: [item ids]; those comparisons judge
                     just these items (both orders), with no phase split, into
                     <log_root>/<judge>/<pair>/topup/. Items already in a finished
                     topup log are not re-run.
  only_sample_ids:   like only_items, but values are exact order-specific ids
                     (<item>_1 / <item>_2); generated repair configs use this form.

Each comparison runs in two phases: first the first phase1_items items of
every comparison (so every analysis has low-n data early), then the rest.
Phases with a successful log are skipped, so re-running only does new work.

Usage:
  uv run python scripts/judge/run_judging.py --config configs/judge_cheap6_gpqa.yaml --dry-run
  uv run python scripts/judge/run_judging.py --config configs/judge_cheap6_gpqa.yaml
"""

import argparse
from pathlib import Path

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log, read_eval_log_samples

from judge.task import DEFAULT_SYSTEM_PROMPT, judge_task
from judge.util import extract_score

DEFAULT_PHASE1_ITEMS = 20
DEFAULT_MAX_CONNECTIONS = 40


def success_logs(directory: Path) -> list[Path]:
    """Finished logs in a run dir, oldest first.

    Phase logs always carry a sample_id block, so configs.paths.is_successful_eval
    (which rejects partial runs) would call every phase pending; the log status alone
    decides here."""
    finished = []
    for path in directory.glob("*.eval"):
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        if header.status == "success":
            finished.append((header.eval.created, path))
    return [path for _, path in sorted(finished)]


def has_success_log(directory: Path) -> bool:
    """Whether this phase already produced a finished log, so it can be skipped."""
    return bool(success_logs(directory))


def topup_sample_ids(items: list) -> list[str]:
    """Both orders of each item, as judge.dataset names the samples."""
    return [f"{item}_{order}" for item in items for order in (1, 2)]


def judged_sample_ids(directory: Path) -> set[str]:
    """Sample ids with a verdict in the finished logs of a run dir."""
    ids = set()
    for path in success_logs(directory):
        for sample in read_eval_log_samples(str(path), all_samples_required=False):
            if any(score.answer is not None for score in (sample.scores or {}).values()):
                ids.add(str(sample.id))
    return ids


def make_task(paths: dict, judge_short: str, major: str, minor: str, frac: float | None,
              include_reasoning: bool, experiment_name: str):
    """The judging task for one comparison, named as its log dir is."""
    suffix = f"_t{int(frac * 100)}" if frac else ""
    return judge_task(
        major_path=Path(paths[major]),
        minor_path=Path(paths[minor]),
        include_reasoning=include_reasoning,
        truncate_frac=frac,
        task_name=f"{judge_short}_{major}-vs-{minor}{suffix}",
        experiment_name=experiment_name,
    )


def correct_count(path: str | Path) -> int:
    """Number of correct samples in an eval log, counted over its samples.

    The header metrics are not reliable here: the scorer metric name varies by
    benchmark, and a log that did not finish cleanly has no results at all."""
    return sum(extract_score(s) for s in read_eval_log_samples(str(path), all_samples_required=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", type=int, choices=(1, 2),
                        help="run only this phase; default runs both")
    args = parser.parse_args()

    # inspect's system_message() falls back to the path string itself when the file is
    # missing, so the judges would silently run with no instructions.
    if not Path(DEFAULT_SYSTEM_PROMPT).is_file():
        raise SystemExit(f"system prompt file not found: {DEFAULT_SYSTEM_PROMPT} (run from the repo root)")

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]
    log_root = Path(cfg["log_root"])
    phase1_items = cfg.get("phase1_items") or DEFAULT_PHASE1_ITEMS
    epochs = cfg.get("epochs") or 1
    fracs = cfg.get("truncate_fracs") or [None]
    # Ablation switch: judges see candidates' reasoning. Protocol default stays False.
    include_reasoning = bool(cfg.get("include_reasoning", False))
    # Per-judge OpenRouter routing (endpoint pins); keyed by judge slug.
    judge_model_args = cfg.get("judge_model_args") or {}
    max_connections = cfg.get("max_connections") or DEFAULT_MAX_CONNECTIONS
    generate_args = cfg.get("generate_args") or {}
    # Models with fewer correct samples than this don't qualify for judging.
    min_correct = cfg.get("min_correct") or 0
    # Comparisons judged on a fixed item list only (top-ups after re-collected rollouts).
    only_items = cfg.get("only_items") or {}
    only_sample_ids = cfg.get("only_sample_ids") or {}
    targeted_pairs = set(only_items) | set(only_sample_ids)
    qualified = set(paths)
    if min_correct:
        counts = {model: correct_count(path) for model, path in paths.items()}
        for model in sorted(paths):
            if counts[model] < min_correct:
                qualified.discard(model)
                print(f"DROP {model}: {counts[model]} correct < min_correct={min_correct}")

    tasks = {}

    def get_task(judge_short, major, minor, frac):
        key = (judge_short, major, minor, frac)
        if key not in tasks:
            tasks[key] = make_task(paths, judge_short, major, minor, frac, include_reasoning, args.config.stem)
        return tasks[key]

    def task_or_none(judge_short, major, minor, frac):
        try:
            task = get_task(judge_short, major, minor, frac)
        except ValueError as e:
            print(f"SKIP {judge_short} {major}-vs-{minor}: {e}")
            return None
        if task is None:
            # judge_task returns None when no question survives filtering
            # (e.g. no both-correct overlap, or a transcript log with no samples).
            print(f"SKIP {judge_short} {major}-vs-{minor}: no samples after filtering")
        return task

    # (phase, judge, major, minor, frac, sample_ids, log_dir), phase 1 rows first.
    # phase is 1, 2 or "topup".
    pending = []
    for phase in ((args.phase,) if args.phase else (1, 2)):
        for frac in fracs:
            root = log_root if frac is None else log_root / f"trunc-{int(frac * 100)}"
            for judge, major, minor in cfg["comparisons"]:
                if major not in qualified or minor not in qualified:
                    continue
                if f"{major}-vs-{minor}" in targeted_pairs:
                    continue  # top-up rows below
                judge_short = judge.split("/")[-1]
                pair_dir = root / judge_short / f"{major}-vs-{minor}"
                if frac is None and has_success_log(pair_dir):
                    continue  # fully judged before phasing was introduced
                task = task_or_none(judge_short, major, minor, frac)
                if task is None:
                    continue
                ids = [s.id for s in task.dataset]
                split = 2 * phase1_items
                block = ids[:split] if phase == 1 else ids[split:]
                if not block:
                    continue
                phase_dir = pair_dir / f"phase-{phase}"
                if has_success_log(phase_dir):
                    continue
                pending.append((phase, judge, judge_short, major, minor, frac, block, phase_dir))

    # Top-ups: the listed items only, both orders, minus what a finished topup log has.
    for frac in fracs:
        root = log_root if frac is None else log_root / f"trunc-{int(frac * 100)}"
        for judge, major, minor in cfg["comparisons"]:
            pair = f"{major}-vs-{minor}"
            if pair not in targeted_pairs or major not in qualified or minor not in qualified:
                continue
            judge_short = judge.split("/")[-1]
            task = task_or_none(judge_short, major, minor, frac)
            if task is None:
                continue
            wanted = (
                [str(id) for id in only_sample_ids[pair]]
                if pair in only_sample_ids
                else topup_sample_ids(only_items[pair])
            )
            in_dataset = {s.id for s in task.dataset}
            missing = sorted({id.rsplit("_", 1)[0] for id in wanted if id not in in_dataset})
            if missing:
                print(f"SKIP items {judge_short} {pair}: not both-correct: {missing}")
            topup_dir = root / judge_short / pair / "topup"
            done = judged_sample_ids(topup_dir)
            block = [id for id in wanted if id in in_dataset and id not in done]
            if block:
                pending.append(("topup", judge, judge_short, major, minor, frac, block, topup_dir))

    n_total = 2 * len(fracs) * len(cfg["comparisons"])
    print(f"{len(pending)} runs pending (of {n_total} phase-runs possible)")
    if args.dry_run:
        for phase, _, judge_short, major, minor, frac, block, _ in pending:
            t = f" t={frac}" if frac else ""
            name = "topup" if phase == "topup" else f"phase{phase}"
            print(f"  {name}{t} {judge_short:28s} {major}-vs-{minor} ({len(block)//2} items)")
        return 0

    failures = []
    for phase, judge, judge_short, major, minor, frac, block, log_dir in pending:
        run_kwargs = dict(
            tasks=[get_task(judge_short, major, minor, frac)],
            model=judge,
            model_args=judge_model_args.get(judge, {}),
            log_dir=str(log_dir),
            sample_id=block,
            epochs=epochs,
            max_connections=max_connections,
            **generate_args,
        )
        if phase == "topup":
            # Plain eval: eval_set keys reuse on the task identity, not on sample_id, so a
            # finished topup log would make it skip a later top-up with other items.
            success = all(log.status == "success" for log in inspect_eval(**run_kwargs))
        else:
            success, _ = eval_set(**run_kwargs)
        if not success:
            failures.append(f"{phase} {judge_short} on {major}-vs-{minor}")
    if failures:
        print(f"{len(failures)} FAILED runs (rest of queue completed):")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
