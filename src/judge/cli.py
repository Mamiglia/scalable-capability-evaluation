"""Judge pipeline: build the domain (if needed) -> judge every pair on q_src -> rank and score.

    uv run judge --cfg configs/experiments/domain/science_q1.yml [--q_src 2 --limit 2 --max-connections 16]

Any ExperimentConfig field is a CLI override; unknown flags go to ``eval_set``. Several --cfg
run in parallel processes, one eval_set per log dir. Each log dir ends up with one .eval per
<bench, pair> plus ranking.csv / scores.csv / ranking.json (see ``rank``).
"""

import argparse
from dataclasses import fields
from itertools import groupby
from pathlib import Path

from inspect_ai import Task, eval_set
from inspect_ai.model import CachePolicy

from collect.runner import eval_runner
from configs import ExperimentConfig, ModelConfig, flag_parse
from configs.logs import log_to, logger
from domain.cli import build as build_domain
from judge.task import build_tasks
from model import Model
from ranking.cli import rank_logs

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


def run_tasks(tasks: list[Task], judge: ModelConfig, log_dir: Path, display: str | None = None,
              **eval_kwargs) -> bool:
    """One ``eval_set`` over all tasks of one <judge, condition> log dir.

    A single eval_set shares one connection pool, retries, resumes and displays on its own;
    the log dir may already hold other pairs of the same condition, hence allow_dirty.
    """
    log_to(log_dir)
    # Judging is API-bound and every task is small: run them all at once, with a wide
    # connection pool. Both are overridable from the CLI (--max-tasks / --max-connections).
    concurrency = {"max_tasks": len(tasks), "max_connections": 64}
    with Model(judge) as model:
        success, _ = eval_set(
            tasks=tasks,
            log_dir=str(log_dir),
            log_dir_allow_dirty=True,
            display=display,
            embed_viewer=False,
            **concurrency | model.eval_kwargs() | eval_kwargs,
        )
    if not success:
        logger.error("Judge run failed in %s", log_dir)
    return success


def run_experiments(experiments: list[ExperimentConfig], display: str | None = None) -> int:
    """All experiments of one log dir as tasks of one eval_set, then rank that dir."""
    exp = experiments[0]
    if not exp.domain.items_path.exists():
        build_domain(exp.domain)
    tasks = [task for exp in experiments for task in build_tasks(exp)]
    kwargs = exp.eval_kwargs() | {"temperature": exp.temperature}
    display = kwargs.pop("display", display)  # --display on the CLI wins over the runner's choice
    if exp.cache:
        kwargs["cache"] = CachePolicy(expiry=None)  # repeated pairs must never re-bill
    if not run_tasks(tasks, exp.judge, exp.log_dir, display, **kwargs):
        return 1
    rank_logs(exp.log_dir, exp.domain)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge every pair of a model set on one quartile, rank, score.")
    parser.add_argument("--cfg", "--config", "--config_path", type=Path, action="extend", nargs="+",
                        required=True, help="ExperimentConfig YAML(s)")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Maximum parallel processes, one per <domain, judge, condition> log dir (default: 8).")
    args, extra = parser.parse_known_args()
    overrides, eval_extra_args = _split_extra(extra)
    draccus_args = [f"--{k}={v}" for k, v in overrides.items()]
    experiments = [ExperimentConfig.from_cli(cfg, args=draccus_args, **eval_extra_args) for cfg in args.cfg]

    experiments.sort(key=lambda exp: str(exp.log_dir))
    groups = [list(group) for _, group in groupby(experiments, key=lambda exp: str(exp.log_dir))]
    if len(groups) == 1:
        return run_experiments(groups[0])  # inline, with inspect's own live display
    jobs = [(run_experiments, (group, "plain"), str(group[0].log_dir)) for group in groups]
    return eval_runner(jobs, max_workers=args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
