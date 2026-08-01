"""CLI interface for collecting benchmark transcripts for a single benchmark and model pair."""

import argparse
from pathlib import Path

from inspect_ai import eval as eval_func

from configs import TaskConfig, ModelConfig, kv_pair, flag_parse
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
    
    # TODO: consider adding a way to run in parallel with detached processes, and show progress in a single terminal. This would be useful for running multiple benchmarks or models at once.
    # TODO: consider adding a way to resume from a previous run
    for bench_path in args.bench:
        bench = TaskConfig.from_cli(bench_path)
        if args.task_arg:
            bench.task_args.update({k: v for k, v in args.task_arg})

        for model_path in args.model:
            model = ModelConfig.from_cli(model_path)
            if args.model_arg:
                model.model_args.update({k: v for k, v in args.model_arg})

            for i in range(args.epochs):
                print(
                    f"Running epoch {i}/{args.epochs} for benchmark '{bench.name}' and model '{model.slug}'"
                )
                
                exit_code = _run_eval(
                    bench_cfg=bench,
                    model_cfg=model,
                    repeat=i,
                    **eval_extra_args,
                )
                if exit_code != 0:
                    print(
                        f"Epoch {i} failed with exit code {exit_code} for benchmark '{bench.name}' and model '{model.slug}'. Stopping further epochs."
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
    target_path = get_eval_path(
        task=bench_cfg,
        model=model_cfg,
        base_dir=kwargs.get("log_dir", "data"),
    ) / f"{repeat}.eval"
    
    if target_path.exists():
        print(f"Skipping {target_path} (already exists)")
        return 0

    logs = eval_func(
        tasks=bench_cfg.task,
        model=model_cfg.slug,
        model_args=model_cfg.model_args or {},
        task_args=bench_cfg.task_args or {},
        seed=repeat,
        log_dir=str(target_path.parent),
        **bench_cfg.eval_kwargs() | kwargs,
    )
    return 0 if logs[0].status == "success" else 1


if __name__ == "__main__":
    main()
