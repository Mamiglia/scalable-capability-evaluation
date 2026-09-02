"""Run the judge comparisons listed in a config file.

The config (see gen_judge_config.py) has these sections:
  log_root:          where judge logs go
  transcripts:       model shortname -> eval log path
  comparisons:       list of [judge, major, minor] rows; the judge is an inspect
                     slug whose model YAML under configs/models supplies the
                     endpoint pin, generate args and runpod settings
Optional knobs:
  system_prompt:     prompt file shown to the judge (default assets/system_prompt/default.txt)
  phase1_items:      items judged in the early phase (default 20)
  epochs:            judge every sample this many times, to measure how much
                     verdicts vary for a fixed input (default 1)
  truncate_fracs:    list of fractions; each adds a length-control run with both
                     transcripts cut to that fraction of the shorter one
                     (logs under trunc-<pct>/, or trunc-end-<pct>/ if truncate_keep: last)
  truncate_keep:     "first" (default) or "last" - which end of each transcript
                     to keep. One config covers one keep direction.
  include_reasoning: judges see candidates' reasoning (protocol default: false)
  only_items:        optional item-id allowlist, applied before phase splitting
  cache:             Inspect model-response cache setting (omit for framework default)
  judge_budgets:     optional per-judge hard limits for parallel --judge runs
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
from contextlib import ExitStack
from pathlib import Path

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info

from configs import ModelConfig
from judge.task import DEFAULT_SYSTEM_PROMPT, judge_task
from judge.util import extract_score
from model import Model

DEFAULT_PHASE1_ITEMS = 20
DEFAULT_MAX_CONNECTIONS = 40
MODEL_CONFIGS = Path("configs/models")


def selected_sample_ids(task, only_items):
    ids = [str(sample.id) for sample in task.dataset]
    if only_items is None:
        return ids
    return [sample_id for sample_id in ids if sample_id.rsplit("_", 1)[0] in only_items]


def configured_system_prompt(cfg: dict) -> str:
    return cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)


def judge_config(slug: str) -> ModelConfig:
    """The model YAML under configs/models whose inspect slug is ``slug``."""
    found = []
    for path in sorted(MODEL_CONFIGS.rglob("*.yml")):
        config = ModelConfig.from_cli(yaml.safe_load(path.read_text()))
        if config.slug == slug:
            found.append((path, config))
    if len(found) != 1:
        paths = [str(path) for path, _ in found] or "none"
        raise ValueError(f"judge {slug}: need exactly one model YAML in {MODEL_CONFIGS}, found {paths}")
    return found[0][1]


def register_model_costs(costs: dict) -> None:
    """Teach Inspect live endpoint prices for its per-sample cost limiter."""
    for model, values in costs.items():
        info = get_model_info(model) or ModelInfo(model=model)
        set_model_info(
            model,
            info.model_copy(update={"cost": ModelCost(**values)}),
        )


def validated_truncate_keep(config: dict) -> str:
    """Read truncate_keep from config; only "first" or "last" is a valid value."""
    keep = config.get("truncate_keep", "first")
    if keep not in ("first", "last"):
        raise ValueError(f"truncate_keep must be 'first' or 'last', got {keep!r}")
    return keep


def configured_comparisons(config: dict) -> list[list[str]]:
    """Return explicit rows, or apply every configured judge to every pair."""
    if "comparisons" in config:
        return config["comparisons"]
    return [
        [judge, major, minor]
        for judge in config["judges"]
        for major, minor in config["pairs"]
    ]


def select_judge(comparisons: list[list[str]], judge: str | None) -> list[list[str]]:
    """Keep one judge's queue, accepting either its full slug or short name."""
    if not judge:
        return comparisons
    selected = [row for row in comparisons
                if row[0] == judge or row[0].split("/")[-1] == judge]
    if not selected:
        raise ValueError(f"judge {judge!r} is not present in the config")
    return selected


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


def correct_count(path: str | Path) -> int:
    """Number of correct samples in an eval log, counted over its samples.

    The header metrics are not reliable here: the scorer metric name varies by
    benchmark, and a log that did not finish cleanly has no results at all."""
    return sum(extract_score(s) for s in read_eval_log_samples(str(path), all_samples_required=False))


def _log_cost(log) -> float:
    usage = log.stats.model_usage if log.stats else {}
    return sum(model.total_cost or 0.0 for model in usage.values())


def logged_cost(directory: Path) -> float:
    """Sum provider-reported spend, including failed evaluation logs."""
    total = 0.0
    for path in directory.rglob("*.eval"):
        try:
            total += _log_cost(read_eval_log(path, header_only=True))
        except Exception:
            # A partial file may exist while Inspect is still writing it. The
            # next check will include it once its header is readable.
            continue
    return total


def budget_settings(config: dict, judge: str | None, log_root: Path):
    """Use an isolated hard budget when one judge runs in its own process."""
    total = config.get("total_cost_limit")
    per_sample = config.get("cost_limit_per_sample")
    budgets = config.get("judge_budgets") or {}
    if not judge or not budgets:
        return total, per_sample, log_root, False

    if judge not in budgets:
        raise ValueError(f"judge_budgets has no entry for {judge}")
    if total is not None:
        allocated = sum(b["total_cost_limit"] for b in budgets.values())
        if allocated > total + 1e-9:
            raise ValueError(f"judge budgets total ${allocated:.2f}, above the ${total:.2f} cap")

    budget = budgets[judge]
    return (
        budget["total_cost_limit"],
        budget["cost_limit_per_sample"],
        log_root / judge.split("/")[-1],
        True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", type=int, choices=(1, 2),
                        help="run only this phase; default runs both")
    parser.add_argument("--judge", help="run only this configured judge")
    parser.add_argument("--max-connections", type=int,
                        help="override the config for a throttled provider")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]
    log_root = Path(cfg["log_root"])
    phase1_items = cfg.get("phase1_items") or DEFAULT_PHASE1_ITEMS
    epochs = cfg.get("epochs") or 1
    fracs = cfg.get("truncate_fracs") or [None]
    # One config covers one keep direction - no first/last cross-product.
    keep = validated_truncate_keep(cfg)
    only_items = cfg.get("only_items") or {}
    topup_items = only_items if isinstance(only_items, dict) else {}
    sample_allowlist = None if topup_items else set(map(str, only_items)) or None
    only_sample_ids = cfg.get("only_sample_ids") or {}
    targeted_pairs = set(topup_items) | set(only_sample_ids)
    system_prompt = configured_system_prompt(cfg)
    # inspect's system_message() falls back to the path string itself when the file is
    # missing, so the judges would silently run with no instructions.
    if not Path(system_prompt).is_file():
        raise SystemExit(f"system prompt file not found: {system_prompt} (run from the repo root)")
    # Ablation switch: judges see candidates' reasoning. Protocol default stays False.
    include_reasoning = bool(cfg.get("include_reasoning", False))
    max_tokens = cfg.get("max_tokens")
    budget_reserve = cfg.get("stop_with_budget_remaining", 0.0)
    max_connections = args.max_connections or cfg.get("max_connections") or DEFAULT_MAX_CONNECTIONS
    timeout = cfg.get("timeout")
    attempt_timeout = cfg.get("attempt_timeout")
    max_retries = cfg.get("max_retries")
    register_model_costs(cfg.get("model_costs") or {})
    generate_args = cfg.get("generate_args") or {}
    if "judge_model_args" in cfg:
        raise ValueError("judge_model_args is no longer read: endpoint pins live in the model YAMLs")
    comparisons = select_judge(configured_comparisons(cfg), args.judge)
    # Models with fewer correct samples than this don't qualify for judging.
    min_correct = cfg.get("min_correct") or 0
    if min_correct:
        counts = {model: correct_count(path) for model, path in paths.items()}
        for model in sorted(m for m, c in counts.items() if c < min_correct):
            print(f"DROP {model}: {counts[model]} correct < min_correct={min_correct}")
        comparisons = [row for row in comparisons
                       if counts[row[1]] >= min_correct and counts[row[2]] >= min_correct]
    selected_judges = {row[0] for row in comparisons}
    selected_judge = next(iter(selected_judges)) if args.judge else None
    if args.judge and len(selected_judges) != 1:
        raise ValueError(f"--judge {args.judge!r} matched more than one judge")
    total_cost_limit, cost_limit, budget_root, strict_budget = budget_settings(
        cfg, selected_judge, log_root
    )

    tasks = {}

    def get_task(judge_short, major, minor, frac):
        # The candidate dataset is independent of judge identity. Reusing it
        # avoids rereading the same large response logs for every judge.
        key = (major, minor, frac)
        if key not in tasks:
            if not frac:
                suffix = ""
            elif keep == "last":
                suffix = f"_tend{int(frac * 100)}"
            else:
                suffix = f"_t{int(frac * 100)}"
            tasks[key] = judge_task(
                major_path=Path(paths[major]),
                minor_path=Path(paths[minor]),
                system_prompt=system_prompt,
                include_reasoning=include_reasoning,
                truncate_frac=frac,
                truncate_keep=keep,
                task_name=f"{major}-vs-{minor}{suffix}",
                experiment_name=args.config.stem,
            )
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
            if frac is None:
                root = log_root
            elif keep == "last":
                root = log_root / f"trunc-end-{int(frac * 100)}"
            else:
                root = log_root / f"trunc-{int(frac * 100)}"
            for judge, major, minor in comparisons:
                if f"{major}-vs-{minor}" in targeted_pairs:
                    continue  # top-up rows below
                judge_short = judge.split("/")[-1]
                pair_dir = root / judge_short / f"{major}-vs-{minor}"
                if frac is None and has_success_log(pair_dir):
                    continue  # fully judged before phasing was introduced
                task = task_or_none(judge_short, major, minor, frac)
                if task is None:
                    continue
                ids = selected_sample_ids(task, sample_allowlist)
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
        if frac is None:
            root = log_root
        elif keep == "last":
            root = log_root / f"trunc-end-{int(frac * 100)}"
        else:
            root = log_root / f"trunc-{int(frac * 100)}"
        for judge, major, minor in comparisons:
            pair = f"{major}-vs-{minor}"
            if pair not in targeted_pairs:
                continue
            judge_short = judge.split("/")[-1]
            task = task_or_none(judge_short, major, minor, frac)
            if task is None:
                continue
            wanted = (
                [str(sample_id) for sample_id in only_sample_ids[pair]]
                if pair in only_sample_ids else topup_sample_ids(topup_items[pair])
            )
            in_dataset = {str(sample.id) for sample in task.dataset}
            missing = sorted({
                sample_id.rsplit("_", 1)[0]
                for sample_id in wanted if sample_id not in in_dataset
            })
            if missing:
                print(f"SKIP items {judge_short} {pair}: not both-correct: {missing}")
            topup_dir = root / judge_short / pair / "topup"
            done = judged_sample_ids(topup_dir)
            block = [sample_id for sample_id in wanted
                     if sample_id in in_dataset and sample_id not in done]
            if block:
                pending.append(("topup", judge, judge_short, major, minor, frac, block, topup_dir))

    n_total = 2 * len(fracs) * len(comparisons)
    print(f"{len(pending)} runs pending (of {n_total} phase-runs possible)")
    if args.dry_run:
        for phase, _, judge_short, major, minor, frac, block, _ in pending:
            t = f" t={frac}" if frac else ""
            name = "topup" if phase == "topup" else f"phase{phase}"
            print(f"  {name}{t} {judge_short:28s} {major}-vs-{minor} ({len(block)//2} items)")
        return 0

    failures = []
    models = {}
    with ExitStack() as stack:  # a judge's pod, if any, stays up from first use to the end
        for phase, judge, judge_short, major, minor, frac, block, log_dir in pending:
            # Re-read the ledger so resumed runs include failed and interrupted logs.
            spent = logged_cost(budget_root)
            if total_cost_limit is not None:
                if strict_budget:
                    worst_case = len(block) * cost_limit
                    if spent + worst_case > total_cost_limit:
                        print(
                            f"BUDGET STOP: ${spent:.2f} spent; this batch could cost "
                            f"${worst_case:.2f}, above the ${total_cost_limit:.2f} judge limit"
                        )
                        return 2
                elif spent >= total_cost_limit - budget_reserve:
                    print(
                        f"BUDGET STOP: ${spent:.2f} spent; keeping at least "
                        f"${budget_reserve:.2f} below the ${total_cost_limit:.2f} limit"
                    )
                    return 2
            if judge not in models:
                models[judge] = stack.enter_context(Model(judge_config(judge)))
            eval_kwargs = models[judge].eval_kwargs()
            eval_kwargs.update(generate_args)
            if "cache" in cfg:
                eval_kwargs["cache"] = bool(cfg["cache"])
            eval_kwargs.update({
                name: value for name, value in {
                    "max_connections": max_connections,
                    "max_tokens": max_tokens,
                    "cost_limit": cost_limit,
                    "timeout": timeout,
                    "attempt_timeout": attempt_timeout,
                    "max_retries": max_retries,
                }.items() if value is not None
            })
            run_kwargs = dict(
                tasks=[get_task(judge_short, major, minor, frac)],
                log_dir=str(log_dir),
                sample_id=block,
                epochs=epochs,
                **eval_kwargs,
            )
            if phase == "topup":
                # eval_set keys reuse on task identity rather than sample_id, so a finished
                # top-up could otherwise suppress a later run for different items.
                success = all(log.status == "success" for log in inspect_eval(**run_kwargs))
            else:
                success, _ = eval_set(
                    log_dir_allow_dirty=True,  # older failed attempts remain as spend evidence
                    **run_kwargs,
                )
            spent = logged_cost(budget_root)
            if total_cost_limit is not None:
                print(f"Collection spend: ${spent:.2f} / ${total_cost_limit:.2f}")
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
