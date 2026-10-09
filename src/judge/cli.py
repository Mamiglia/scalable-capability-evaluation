"""Collect judgments (paid API calls): what a judge's run in logs/ still lacks on every semester task of a domain.

    uv run judge configs/domain/science.yml --judge configs/judges/gemma4-31b.yml [--direct-judge ...] [--semester 2026-I]

Pairwise (--judge, JudgeBT) and direct (--direct-judge, JudgeDirect) judging resume at sample level; a finished run is only
read. Then `uv run evaluate <domain.yml> --judge <same yml>` scores the run, and scripts/export_judgments.py adds it
to data/meta/judgments_*.csv (each task's top-3 references as judges) for the paper's protocols.
"""

import argparse
from dataclasses import replace
from pathlib import Path

from capability.evaluate import semester_names
from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import DirectJudgeConfig, JudgeConfig
from configs.logs import logger
from protocols.direct_judge import JudgeDirect
from protocols.pairwise import JudgeBT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--judge", nargs="+", default=[], type=Path, help="JudgeConfig yml(s): pairwise verdicts")
    parser.add_argument("--direct-judge", nargs="+", default=[], type=Path, help="judge yml(s): 1-10 ratings, reasoning shown")
    parser.add_argument("--semester", nargs="+", default=None, help="default: every semester task")
    parser.add_argument("--truncate", help="override the --judge configs' truncate, e.g. thirds-13 (src/challenges/thirds)")
    args = parser.parse_args()
    domain = DomainConfig.from_cli(args.domain)
    override = {"truncate": args.truncate} if args.truncate else {}
    harnesses = [JudgeBT(replace(JudgeConfig.from_cli(p), **override), domain) for p in args.judge]
    harnesses += [JudgeDirect(DirectJudgeConfig.from_cli(p), domain) for p in args.direct_judge]
    for semester in args.semester or semester_names(domain):
        task = Task.from_semester(semester, domain)
        for harness in harnesses:
            logger.info("%s %s -> %s", harness.name, semester, harness.judge(task.references, task.test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
