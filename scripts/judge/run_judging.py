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
                     (logs under trunc-<pct>/, or trunc-end-<pct>/ if truncate_keep: last)
  truncate_keep:     "first" (default) or "last" - which end of each transcript
                     to keep. One config covers one keep direction.
  include_reasoning: judges see candidates' reasoning (protocol default: false)
  judge_budgets:     optional per-judge hard limits for parallel --judge runs

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
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info

from judge.task import judge_task

DEFAULT_PHASE1_ITEMS = 20


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


def has_success_log(directory: Path) -> bool:
    for p in directory.glob("*.eval"):
        try:
            if read_eval_log(p, header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False


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
    # Ablation switch: judges see candidates' reasoning. Protocol default stays False.
    include_reasoning = bool(cfg.get("include_reasoning", False))
    max_tokens = cfg.get("max_tokens")
    budget_reserve = cfg.get("stop_with_budget_remaining", 0.0)
    max_connections = args.max_connections or cfg.get("max_connections") or 40
    timeout = cfg.get("timeout")
    attempt_timeout = cfg.get("attempt_timeout")
    max_retries = cfg.get("max_retries")
    register_model_costs(cfg.get("model_costs") or {})
    # Per-judge OpenRouter routing (endpoint pins); keyed by judge slug.
    judge_model_args = cfg.get("judge_model_args") or {}
    comparisons = select_judge(configured_comparisons(cfg), args.judge)
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
                include_reasoning=include_reasoning,
                truncate_frac=frac,
                truncate_keep=keep,
                task_name=f"{major}-vs-{minor}{suffix}",
                experiment_name=args.config.stem,
            )
        return tasks[key]

    # (phase, judge, major, minor, frac, sample_ids, log_dir), phase 1 rows first.
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
                judge_short = judge.split("/")[-1]
                pair_dir = root / judge_short / f"{major}-vs-{minor}"
                if frac is None and has_success_log(pair_dir):
                    continue  # fully judged before phasing was introduced
                try:
                    task = get_task(judge_short, major, minor, frac)
                except ValueError as e:
                    print(f"SKIP {judge_short} {major}-vs-{minor}: {e}")
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

    n_total = 2 * len(fracs) * len(comparisons)
    print(f"{len(pending)} phase-runs pending (of {n_total} possible)")
    if args.dry_run:
        for phase, _, judge_short, major, minor, frac, block, _ in pending:
            t = f" t={frac}" if frac else ""
            print(f"  phase{phase}{t} {judge_short:28s} {major}-vs-{minor} ({len(block)//2} items)")
        return 0

    failures = []
    spent = logged_cost(budget_root)
    for phase, judge, judge_short, major, minor, frac, block, phase_dir in pending:
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
        success, logs = eval_set(
            tasks=[get_task(judge_short, major, minor, frac)],
            model=judge,
            model_args=judge_model_args.get(judge, {}),
            log_dir=str(phase_dir),
            sample_id=block,
            epochs=epochs,
            max_connections=max_connections,
            max_tokens=max_tokens,
            cost_limit=cost_limit,
            timeout=timeout,
            attempt_timeout=attempt_timeout,
            max_retries=max_retries,
            # Interrupted logs are evidence of spend and stay in place. A new
            # attempt may share their phase directory; success logs are still
            # skipped above, so completed work is never repeated.
            log_dir_allow_dirty=True,
        )
        spent = logged_cost(budget_root)
        if total_cost_limit is not None:
            print(f"Collection spend: ${spent:.2f} / ${total_cost_limit:.2f}")
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
