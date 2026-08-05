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

    # Avoid provisioning paid model capacity when every requested repeat already
    # has a successful log, and fail locally for dirty incomplete directories.
    pending_models = _preflight(benches, models, args.epochs, eval_extra_args)
    if not pending_models:
        print("All requested evaluations are already complete.")
        return 0

    # TODO: consider adding a way to run in parallel with detached processes, and show progress in a single terminal. This would be useful for running multiple benchmarks or models at once.
    result = runpod_main(
        pending_models,
        lambda model: _collect(benches, [model], args.epochs, eval_extra_args),
        force=args.runpod,
    )
    if result is not None:
        return result
    return _collect(benches, pending_models, args.epochs, eval_extra_args)


def _target_dir(
    bench: TaskConfig,
    model: ModelConfig,
    repeat: int,
    eval_extra_args: dict[str, Any],
) -> Path:
    return (
        get_eval_path(
            task=bench,
            model=model,
            base_dir=eval_extra_args.get("log_dir", "data"),
        )
        / f"{repeat}"
    )


def _has_existing_eval(target_dir: Path) -> bool:
    return any(target_dir.glob("*.eval"))


def _preflight(
    benches: list[TaskConfig],
    models: list[ModelConfig],
    epochs: int,
    eval_extra_args: dict[str, Any],
) -> list[ModelConfig]:
    """Return models with pending work, skipping occupied repeat directories."""
    pending_models = []
    for model in models:
        model_pending = False
        for bench in benches:
            for repeat in range(epochs):
                target_dir = _target_dir(bench, model, repeat, eval_extra_args)
                if _has_existing_eval(target_dir):
                    print(
                        f"SKIPPING epoch {repeat + 1}/{epochs} for benchmark "
                        f"{bench.name} and model {model.slug}"
                    )
                    continue
                model_pending = True
        if model_pending:
            pending_models.append(model)
    return pending_models


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
                target_dir = _target_dir(bench, model, repeat, eval_extra_args)
                if _has_existing_eval(target_dir):
                    continue
                print(
                    f"Running epoch {repeat + 1}/{epochs} for benchmark "
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
                        f"Epoch {repeat + 1} failed with exit code {exit_code} for "
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
    model_args = (model_cfg.model_args or {}).copy()
    model_base_url = model_args.pop("base_url", None)
    if model_base_url is not None:
        kwargs["model_base_url"] = model_base_url
    target_dir = _target_dir(bench_cfg, model_cfg, repeat, kwargs)
    kwargs.pop("log_dir", None)
    success, _ = eval_set(
        tasks=bench_cfg.task,
        model=model_cfg.slug,
        model_args=model_args,
        task_args=bench_cfg.task_args or {},
        seed=repeat,
        log_dir=str(target_dir),
        **bench_cfg.eval_kwargs() | kwargs,
    )
    return 0 if success else 1


if __name__ == "__main__":
    main()
