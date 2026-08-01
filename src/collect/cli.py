"""CLI interface for collecting benchmark transcripts for a single benchmark and model pair."""

import argparse
from pathlib import Path

from inspect_ai import eval as eval_func

from configs import BenchConfig, ModelConfig
from path_utils import get_eval_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single benchmark evaluation for a model."
    )
    parser.add_argument(
        "--bench",
        type=Path,
        required=True,
        help="Path to benchmark YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model YAML configuration file or a model slug.",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run (default: 1).",
    )
    parsed_args, extra = parser.parse_known_args()
    bench = BenchConfig.from_cli(parsed_args.bench, args=extra)
    model = ModelConfig.from_cli(parsed_args.model, args=extra)

    for i in range(parsed_args.epochs):
        print(f"Running epoch {i}/{parsed_args.epochs} for benchmark '{bench.name}' and model '{model.slug}'")
        exit_code = _run_eval(bench_cfg=bench, model_cfg=model, repeat=i, **extra)
        if exit_code != 0:
            print(f"Epoch {i} failed with exit code {exit_code}. Stopping further epochs.")
            return exit_code
    return 0

def _run_eval(
    bench_cfg: BenchConfig,
    model_cfg: ModelConfig,
    repeat: int = 1,
    **kwargs,
) -> int:
    """Execute a single evaluation run for a given benchmark and model config."""
    target_path = get_eval_path(
        benchmark=bench_cfg.name,
        task_args=bench_cfg.task_args,
        model=model_cfg.slug,
        n=repeat,
    )

    if target_path.exists():
        print(f"Skipping {target_path} (already exists)")
        return 0

    logs = eval_func(
        tasks=bench_cfg.task,
        model=model_cfg.slug,
        model_args=model_cfg.model_args or {},
        task_args=bench_cfg.task_args or {},
        seed=repeat,
        log_dir=target_path.parent,
        **kwargs,
    )
    return 0 if logs[0].status == "success" else 1


if __name__ == "__main__":
    main()
