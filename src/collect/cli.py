"""CLI interface for collecting benchmark transcripts."""

import argparse
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai import eval_set

from collect.runner import eval_runner
from configs import ModelConfig, TaskConfig, flag_parse, kv_pair
from configs.logs import log_to, logger
from configs.paths import get_eval_path, is_successful_eval, task_args_with_defaults
from model import Model



def _run_eval_worker(
    bench_path: Path,
    model_path: Path,
    repeat: int,
    repeats: int,
    use_runpod: bool,
    task_arg: list | None,
    model_arg: list | None,
    eval_extra_args: dict,
    sample_ids: list[str] | None = None,
) -> int:
    bench = TaskConfig.from_cli(bench_path)
    if task_arg:
        bench.task_args.update(dict(task_arg))

    model_config = ModelConfig.from_cli(model_path)
    if model_arg:
        model_config.model_args.update(dict(model_arg))

    target_dir = get_eval_path(
        task=bench,
        model=model_config,
        base_dir=eval_extra_args.get("log_dir", "data"),
    ) / f"{repeat}"
    log_to(target_dir)
    logger.info("Running repeat %d/%d for benchmark '%s' and model '%s'",
                repeat + 1, repeats, bench.name, model_config.slug)
    if sample_ids is None and any(is_successful_eval(path) for path in target_dir.glob("*.eval")):
        logger.info("Skipping completed repeat %d: %s", repeat + 1, target_dir)
        return 0

    # The hash dir is opaque; leave the args it was computed from next to it, so old data
    # stays identifiable when a task's defaults change and move new runs to another hash.
    manifest = target_dir.parent.parent / "task_args.json"
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(task_args_with_defaults(bench), indent=1, sort_keys=True, default=str))

    with Model(model_config, use_runpod=use_runpod) as model:
        kwargs = eval_extra_args.copy()
        kwargs.pop("log_dir", None)

        # No seed: several pinned endpoints reject it under require_parameters,
        # and a fixed seed can collapse epochs into identical generations.
        eval_kwargs = model.eval_kwargs() | bench.eval_kwargs() | kwargs
        if sample_ids is not None:
            # Plain eval, not eval_set: eval_set keys reuse on task+model identity, ignores
            # sample_id, and would treat the finished full log in this dir as done and run
            # nothing. The partial log it writes here is never canonical (is_successful_eval).
            logs = inspect_eval(
                tasks=bench.task,
                task_args=bench.task_args or {},
                log_dir=str(target_dir),
                display="plain",
                sample_id=sample_ids,
                **eval_kwargs,
            )
            return 0 if all(log.status == "success" for log in logs) else 1
        success, _ = eval_set(
            tasks=bench.task,
            task_args=bench.task_args or {},
            log_dir=str(target_dir),
            log_dir_allow_dirty=True,
            display="plain",
            embed_viewer=False,
            **eval_kwargs,
        )
        return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluations for model(s) and benchmark(s).",
        epilog="Any other --flag value goes straight to inspect's eval/eval_set and overrides "
        "the model config, e.g. --max-tokens 32768 or --max-connections 4.",
    )
    parser.add_argument(
        "-b",
        "--bench",
        "--benchmark",
        action="extend",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to benchmark YAML configuration file(s).",
    )
    parser.add_argument(
        "-m",
        "--model",
        action="extend",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to model YAML configuration file(s) or model slug(s).",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        "--repeats",
        type=int,
        default=1,
        dest="epochs",
        help="Number of repeat runs, each stored in its own <model>/<repeat>/ directory "
        "(default: 1). Distinct from the task's internal 'epochs' argument.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="Run only this repeat (0-based), writing into <model>/<repeat>/. Default: all repeats.",
    )
    parser.add_argument(
        "--sample-id",
        type=lambda ids: ids.split(","),
        dest="sample_ids",
        metavar="ID[,ID...]",
        help="Re-run only these sample ids, even if the repeat already has a full log. Writes a "
        "separate partial log into the same <model>/<repeat>/ dir; merge it with "
        "scripts/collect/splice_rerun.py.",
    )
    parser.add_argument(
        "--runpod",
        action="store_true",
        help="Force Runpod for selected model(s), overriding runpod.use=false.",
    )
    parser.add_argument(
        "-T",
        "--task-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Task argument in KEY=VALUE format (can be specified multiple times).",
        type=kv_pair,
    )
    parser.add_argument(
        "-M",
        "--model-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Model argument in KEY=VALUE format (can be specified multiple times).",
        type=kv_pair,
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Maximum parallel eval processes (default: 8).",
    )
    args, extra = parser.parse_known_args()
    eval_extra_args = flag_parse(extra) if extra else {}
    if args.sample_ids is not None and args.repeat is None:
        parser.error("--sample-id requires --repeat")
    if args.repeat is not None and not 0 <= args.repeat < args.epochs:
        parser.error(f"--repeat must be between 0 and {args.epochs - 1} for -n {args.epochs}")

    jobs = [
        (
            _run_eval_worker,
            (
                bench_path,
                model_path,
                repeat,
                args.epochs,
                args.runpod,
                args.task_arg,
                args.model_arg,
                eval_extra_args,
                args.sample_ids,
            ),
            f"{bench_path.stem}/{model_path.stem} (repeat {repeat})",
        )
        for bench_path in args.bench
        for model_path in args.model
        for repeat in ([args.repeat] if args.repeat is not None else range(args.epochs))
    ]

    return eval_runner(jobs, max_workers=args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
