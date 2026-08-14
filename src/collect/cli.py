"""CLI interface for collecting benchmark transcripts."""

import argparse
from pathlib import Path

from inspect_ai import eval_set

from collect.runner import eval_runner
from configs import ModelConfig, TaskConfig, flag_parse, kv_pair
from configs.paths import get_eval_path, is_successful_eval
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
) -> int:
    bench = TaskConfig.from_cli(bench_path)
    if task_arg:
        bench.task_args.update(dict(task_arg))

    model_config = ModelConfig.from_cli(model_path)
    if model_arg:
        model_config.model_args.update(dict(model_arg))

    print(
        f"Running repeat {repeat + 1}/{repeats} for benchmark "
        f"'{bench.name}' and model '{model_config.slug}'"
    )

    target_dir = get_eval_path(
        task=bench,
        model=model_config,
        base_dir=eval_extra_args.get("log_dir", "data"),
    ) / f"{repeat}"
    if any(is_successful_eval(path) for path in target_dir.glob("*.eval")):
        print(f"Skipping completed repeat {repeat + 1}: {target_dir}")
        return 0

    with Model(model_config, use_runpod=use_runpod) as model:
        kwargs = eval_extra_args.copy()
        kwargs.pop("log_dir", None)

        # No seed: several pinned endpoints reject it under require_parameters,
        # and a fixed seed can collapse epochs into identical generations.
        eval_kwargs = model.eval_kwargs() | bench.eval_kwargs() | kwargs
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
        description="Run benchmark evaluations for model(s) and benchmark(s)."
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
    args, extra = parser.parse_known_args()
    eval_extra_args = flag_parse(extra) if extra else {}

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
            ),
            f"{bench_path.stem}/{model_path.stem} (repeat {repeat})",
        )
        for bench_path in args.bench
        for model_path in args.model
        for repeat in range(args.epochs)
    ]

    return eval_runner(jobs)


if __name__ == "__main__":
    raise SystemExit(main())
