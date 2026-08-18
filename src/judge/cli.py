"""CLI interface for running judge experiments comparing model pairs across benchmarks."""

import argparse
from dataclasses import fields
from itertools import groupby
from pathlib import Path

from inspect_ai import eval_set

from collect.runner import eval_runner
from configs import ExperimentConfig, flag_parse
from configs.logs import log_to, logger
from judge.task import build_judge_tasks
from model import Model

EXPERIMENT_FIELDS = {f.name for f in fields(ExperimentConfig)}


def _split_extra(extra: list[str]) -> tuple[dict, dict]:
    """Split trailing flags into ExperimentConfig overrides and ``eval_set`` kwargs.

    Draccus errors on flags it does not know, so anything that is not a config field has to
    be routed to ``eval_set`` instead.
    """
    parsed = flag_parse(extra) if extra else {}
    overrides = {k: v for k, v in parsed.items() if k in EXPERIMENT_FIELDS}
    eval_kwargs = {k: v for k, v in parsed.items() if k not in EXPERIMENT_FIELDS}
    return overrides, eval_kwargs


def _run_experiments(experiments: list[ExperimentConfig], display: str | None = None) -> int:
    """One ``eval_set`` per log dir (= bench x judge x condition), all its pairs as tasks.

    A single eval_set shares one connection pool, retries, resumes and displays on its own;
    the log dir may already hold other pairs of the same condition, hence allow_dirty.
    """
    exp = experiments[0]
    log_to(exp.log_dir)
    tasks = [task for exp in experiments for task in build_judge_tasks(exp)]
    # Judging is API-bound and every task is small: run them all at once, with a wide
    # connection pool. Both are overridable from the CLI (--max-tasks / --max-connections).
    concurrency = {"max_tasks": len(tasks), "max_connections": 64}
    with Model(exp.judge) as model:
        success, _ = eval_set(
            tasks=tasks,
            log_dir=str(exp.log_dir),
            log_dir_allow_dirty=True,
            display=display,
            embed_viewer=False,
            **concurrency | model.eval_kwargs() | exp.eval_kwargs(),
        )
    if not success:
        logger.error("Judge run failed in %s", exp.log_dir)
    return 0 if success else 1


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Maximum parallel processes, one per <bench, judge, condition> group (default: 8).",
    )

    args, extra = parser.parse_known_args()
    overrides, eval_extra_args = _split_extra(extra)

    ad_hoc = (args.bench, args.major, args.minor, args.judge)
    if not args.config_path and not all(ad_hoc):
        parser.error(
            "Specify either --config_path, or all of -b/--bench, --major, --minor and -j/--judge."
        )

    if args.config_path:
        # With --config_path the model/bench flags are per-run overrides applied to every
        # config, so a config only has to pin what actually varies between configs.
        for field_name, values in (
            ("benchmark", args.bench),
            ("judge", args.judge),
            ("major", args.major),
            ("minor", args.minor),
        ):
            if not values:
                continue
            if len(values) > 1:
                parser.error(f"--{field_name} takes a single value when used with --config_path")
            overrides[field_name] = values[0]

        draccus_args = [f"--{k}={v}" for k, v in overrides.items()]
        experiments = [
            ExperimentConfig.from_cli(cfg_path, args=draccus_args, **eval_extra_args)
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
                    **overrides,
                },
                **eval_extra_args,
            )
            for b, M, m, j in zip(args.bench, args.major, args.minor, args.judge)
        ]

    experiments.sort(key=lambda exp: str(exp.log_dir))
    groups = [list(group) for _, group in groupby(experiments, key=lambda exp: str(exp.log_dir))]
    if len(groups) == 1:
        # Inline, with inspect's own live display.
        return _run_experiments(groups[0])
    jobs = [(_run_experiments, (group, "plain"), str(group[0].log_dir)) for group in groups]
    return eval_runner(jobs, max_workers=args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
