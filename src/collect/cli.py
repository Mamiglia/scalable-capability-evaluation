"""CLI interface for collecting benchmark transcripts for a single benchmark and model pair."""

import argparse
from pathlib import Path
from typing import Any

from inspect_ai import eval_set

from configs import TaskConfig, ModelConfig, kv_pair, flag_parse
from collect.runpod import main as runpod_main
from configs.utils import get_eval_path


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
        type=int,
        default=1,
        help="Number of epochs to run (default: 1).",
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

    benches = [TaskConfig.from_cli(path) for path in args.bench]
    models = [ModelConfig.from_cli(path) for path in args.model]
    for bench in benches:
        if args.task_arg:
            bench.task_args.update(dict(args.task_arg))
    for model in models:
        if args.model_arg:
            model.model_args.update(dict(args.model_arg))

    # TODO: consider adding a way to run in parallel with detached processes, and show progress in a single terminal. This would be useful for running multiple benchmarks or models at once.
    # TODO: consider adding a way to resume from a previous run
    result = runpod_main(
        models,
        lambda model: _collect(benches, [model], args.epochs, eval_extra_args),
        force=args.runpod,
    )
    if result is not None:
        return result
    return _collect(benches, models, args.epochs, eval_extra_args)


def _collect(
    benches: list[TaskConfig],
    models: list[ModelConfig],
    epochs: int,
    eval_extra_args: dict[str, Any],
) -> int:
    """Run the ordinary collection loop."""
    for bench in benches:
        for model in models:
            for repeat in range(epochs):
                print(
                    f"Running epoch {repeat}/{epochs} for benchmark "
                    f"'{bench.name}' and model '{model.slug}'"
                )
                exit_code = _run_eval(
                    bench_cfg=bench,
                    model_cfg=model,
                    repeat=repeat,
                    **eval_extra_args,
                )
                if exit_code:
                    print(
                        f"Epoch {repeat} failed with exit code {exit_code} for "
                        f"benchmark '{bench.name}' and model '{model.slug}'. "
                        "Stopping further epochs."
                    )
                    return exit_code
    return 0


def _run_eval(
    bench_cfg: TaskConfig,
    model_cfg: ModelConfig,
    repeat: int,
    **kwargs,
) -> int:
    """Execute a single evaluation run for a given benchmark and model config."""
    kwargs = kwargs.copy()
    target_dir = (
        get_eval_path(
            task=bench_cfg,
            model=model_cfg,
            base_dir=kwargs.pop("log_dir", "data"),
        )
        / f"{repeat}"
    )
    success, _ = eval_set(
        tasks=bench_cfg.task,
        model=model_cfg.slug,
        model_args=model_cfg.model_args or {},
        task_args=bench_cfg.task_args or {},
        seed=repeat,
        log_dir=str(target_dir),
        **bench_cfg.eval_kwargs() | kwargs,
    )
    return 0 if success else 1


if __name__ == "__main__":
    main()
