"""CLI interface for running judge experiments comparing model pairs across benchmarks."""

import argparse
from pathlib import Path

from inspect_ai import eval_set

from configs import ExperimentConfig, flag_parse
from judge.task import build_judge_tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run judge experiments comparing model pairs across benchmarks."
    )
    parser.add_argument(
        "--config_path",
        "--config",
        '--cfg',
        type=Path,
        action ="extend",
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
    
    if args.config_path and (args.bench or args.major or args.minor or args.judge):
        parser.error(
            "Cannot specify both --config_path and any of (-b/--bench, --major, --minor, -j/--judge)."
        )
        
    if args.config_path:
        experiments = [ExperimentConfig.from_cli(cfg_path, args=extra, **eval_extra_args) for cfg_path in args.config_path]
    else:
        experiments = [
            ExperimentConfig.from_cli(
                {
                    "name": f"{Path(b).stem}/{Path(j).stem}:{Path(M).stem}|{Path(m).stem}",
                    "benchmark": b,
                    "major": M,
                    "minor": m,
                    "judge": j,
                    "epochs": args.epochs
                },
                **eval_extra_args,
            ) for b, M, m, j in zip(args.bench, args.major, args.minor, args.judge)
        ]
        
    for exp in experiments:
        tasks = build_judge_tasks(exp)
        kwargs = exp.eval_kwargs()

        success, _ = eval_set(
            tasks=tasks,
            model=exp.judge.slug,
            log_dir=exp.log_dir,
            **kwargs,
        )
        if not success:
            print(f"Judge experiment failed for configuration: {exp}")
            return 1
        
    return 0


if __name__ == "__main__":
    main()
