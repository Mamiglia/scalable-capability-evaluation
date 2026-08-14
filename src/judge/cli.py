"""CLI interface for running judge experiments comparing model pairs across benchmarks."""

import argparse
from pathlib import Path

from inspect_ai import eval_set

from collect.runner import eval_runner
from configs import ExperimentConfig, flag_parse
from judge.task import build_judge_tasks
from model import Model


def _run_experiment_worker(exp: ExperimentConfig) -> int:
    tasks = build_judge_tasks(exp)
    kwargs = exp.eval_kwargs()

    with Model(exp.judge) as model:
        success, _ = eval_set(
            tasks=tasks,
            log_dir=str(exp.log_dir),
            display="plain",
            embed_viewer=False,
            **model.eval_kwargs(),
            **kwargs,
        )
    if not success:
        print(f"Judge experiment failed for configuration: {exp}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run judge experiments comparing model pairs across benchmarks."
    )
    parser.add_argument(
        "--config_path",
        "--config",
        '--cfg',
        type=Path,
        action="extend",
        nargs="+",
        default=None,
        help="Path(s) to experiment YAML configuration file.",
    )
    parser.add_argument(
        "-b",
        "--bench",
        "--benchmark",
        action="extend",
        nargs="+",
        type=Path,
        help="Path(s) to benchmark YAML configuration file(s) or benchmark name(s).",
    )
    parser.add_argument(
        "--major",
        "--major-model", '-M',
        action="extend",
        nargs="+",
        type=Path,
        help="Path(s) to major model YAML configuration file(s) or model slug(s).",
    )
    parser.add_argument(
        "--minor",
        "--minor-model", '-m',
        action="extend",
        nargs="+",
        type=Path,
        help="Path(s) to minor model YAML configuration file(s) or model slug(s).",
    )
    parser.add_argument(
        "-j",
        "--judge",
        "--judge-model",
        action="extend",
        nargs="+",
        type=Path,
        help="Path(s) to judge model YAML configuration file(s) or model slug(s).",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run (default: 1).",
    )

    args, extra = parser.parse_known_args()
    eval_extra_args = flag_parse(extra) if extra else {}

    ad_hoc = (args.bench, args.major, args.minor, args.judge)
    if args.config_path and any(ad_hoc):
        parser.error(
            "Cannot specify both --config_path and any of (-b/--bench, --major, --minor, -j/--judge)."
        )
    if not args.config_path and not all(ad_hoc):
        parser.error(
            "Specify either --config_path, or all of -b/--bench, --major, --minor and -j/--judge."
        )

    if args.config_path:
        experiments = [
            ExperimentConfig.from_cli(cfg_path, args=extra, **eval_extra_args)
            for cfg_path in args.config_path
        ]
    else:
        # Ad-hoc mode compares one <bench, major, minor, judge> tuple per position, so the
        # four lists have to line up; zip() would silently drop the extras.
        lengths = {len(values) for values in ad_hoc}
        if len(lengths) != 1:
            parser.error(
                "-b/--bench, --major, --minor and -j/--judge must have the same number of "
                f"values (got {[len(v) for v in ad_hoc]}); use --config_path for other combinations."
            )
        experiments = [
            ExperimentConfig.from_cli(
                {
                    "name": f"{b.stem}/{j.stem}:{M.stem}|{m.stem}",
                    "benchmark": b,
                    "major": M,
                    "minor": m,
                    "judge": j,
                    "epochs": args.epochs,
                },
                **eval_extra_args,
            )
            for b, M, m, j in zip(args.bench, args.major, args.minor, args.judge)
        ]

    jobs = [(_run_experiment_worker, (exp,), exp.name) for exp in experiments]

    return eval_runner(jobs)


if __name__ == "__main__":
    raise SystemExit(main())
