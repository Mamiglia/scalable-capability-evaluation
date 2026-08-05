"""CLI interface for collecting benchmark transcripts."""

import argparse
from pathlib import Path

from inspect_ai import eval_set

from configs import ModelConfig, TaskConfig, flag_parse, kv_pair
from configs.utils import get_eval_path
from model import Model


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

    for bench_path in args.bench:
        bench = TaskConfig.from_cli(bench_path)
        if args.task_arg:
            bench.task_args.update(dict(args.task_arg))

        for model_path in args.model:
            model_config = ModelConfig.from_cli(model_path)
            if args.model_arg:
                model_config.model_args.update(dict(args.model_arg))

            with Model(model_config, use_runpod=args.runpod) as model:
                for repeat in range(args.epochs):
                    print(
                        f"Running epoch {repeat}/{args.epochs} for benchmark "
                        f"'{bench.name}' and model '{model_config.slug}'"
                    )
                    exit_code = _run_eval(
                        bench_cfg=bench,
                        model=model,
                        repeat=repeat,
                        **eval_extra_args,
                    )
                    if exit_code:
                        print(
                            f"Epoch {repeat} failed with exit code {exit_code} for "
                            f"benchmark '{bench.name}' and model "
                            f"'{model_config.slug}'. Stopping further epochs."
                        )
                        return exit_code
    return 0


def _run_eval(
    bench_cfg: TaskConfig,
    model: Model,
    repeat: int,
    **kwargs,
) -> int:
    """Execute one evaluation run."""
    target_dir = get_eval_path(
        task=bench_cfg,
        model=model.config,
        base_dir=kwargs.pop("log_dir", "data"),
    ) / f"{repeat}"

    success, _ = eval_set(
        tasks=bench_cfg.task,
        task_args=bench_cfg.task_args or {},
        seed=repeat,
        log_dir=str(target_dir),
        **model.eval_kwargs(),
        **bench_cfg.eval_kwargs() | kwargs,
    )
    return 0 if success else 1


if __name__ == "__main__":
    main()
