"""Loading: semester tasks from the frozen tables and the judge outputs already on disk.

Never launches judging. A judge run is one <domain, semester, judge, condition> directory
under logs/judge/ (pairwise verdicts) or logs/direct*/ (1-10 ratings); its rows are read
raw here and restricted to the frozen train items by analysis.matching.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from capability.evaluate import semester_names
from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import DirectJudgeConfig, JudgeConfig
from configs.logs import logger
from protocols.direct_judge import JudgeDirect
from protocols.direct_judge.judge import from_logs as direct_from_logs
from protocols.pairwise import JudgeBT
from protocols.pairwise.judge import from_logs as pairwise_from_logs
from reference import tables

JUDGES = Path("configs/judges")  # one YAML per judge; reasoning/ holds the reasoning-shown variants
LOGS = Path("logs")


@dataclass
class JudgeRun:
    """One judge's logs for one semester task.

    protocol   "pairwise" or "direct"
    judge      the judge's shortname (its data/ directory name, e.g. "gpt-4-0613")
    condition  "reasoning-shown", "reasoning-hidden", or "reasoning-shown-<k>pq" for a
               pairwise run on a smaller train set (k items per quartile)
    harness    the JudgeBT / JudgeDirect that wrote the logs; matching uses it to count
               the judgments the campaign still lacks
    table      the raw rows: pairwise -> bench, judge, item, model_left, model_right,
               pick_left, semester, imported; direct -> bench, model, item, score, semester
    missing    judgments still unjudged on the train items (0 until matching fills it)
    """

    protocol: str
    judge: str
    condition: str
    harness: JudgeBT | JudgeDirect
    table: pd.DataFrame
    missing: int = 0


def domain(name: str) -> DomainConfig:
    """The domain config for a short name ("science", "math", "coding", "cyber")."""
    return DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))


def tasks(dom: DomainConfig, semesters: list[str] | None = None) -> list[Task]:
    """The semester tasks of a domain from its frozen tables, in roster order.

    semesters: restrict to these (e.g. ["2026-I"]); None means every frozen task.
    Each Task holds references (theta = theta*), an empty-or-not historical cohort, the
    targets (theta NaN) and the targets' theta* in `truth`.
    """
    return [Task.from_semester(s, dom) for s in semesters or semester_names(dom)]


def fetched(run_dir: Path) -> bool:
    """True when every .eval in the directory is a real log; False (with a warning) when
    some are still git-lfs pointer stubs, which means `git -C logs lfs pull` is needed."""
    stubs = [p for p in run_dir.glob("*.eval") if not zipfile.is_zipfile(p)]
    if stubs:
        logger.warning("skip %s: %d unfetched lfs pointer(s), run `git -C logs lfs pull`", run_dir, len(stubs))
    return not stubs


def pairwise_configs() -> dict[tuple[str, str], tuple[Path, str]]:
    """Map (judge shortname, pairwise condition hash) -> (judge YAML path, condition label).

    A run directory is named by the judge shortname and the condition hash, so this is how
    a directory is traced back to the config that produced it. Reasoning-shown configs live
    in configs/judges/reasoning/; a `train_per_quartile` there marks a smaller train set.
    """
    out = {}
    for path in sorted(JUDGES.glob("*.yml")):
        cfg = JudgeConfig.from_cli(path)
        out[(cfg.judge.shortname.lower(), cfg.condition_hash)] = (path, "reasoning-hidden")
    for path in sorted((JUDGES / "reasoning").glob("*.yml")):
        cfg = JudgeConfig.from_cli(path)
        label = "reasoning-shown" + (f"-{cfg.train_per_quartile}pq" if cfg.train_per_quartile else "")
        out[(cfg.judge.shortname.lower(), cfg.condition_hash)] = (path, label)
    return out


def pairwise_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """Every fetched pairwise run for the task whose judge is in the semester's top-3 roster
    (data/meta/<domain>/fit.yml `top`) and whose condition hash matches a current config.
    Pilot judges outside the roster and stale conditions are skipped silently."""
    configs = pairwise_configs()
    roster = yaml.safe_load((tables.frozen_dir(dom) / "fit.yml").read_text())["top"][task.semester]
    runs = []
    for run_dir in sorted((LOGS / "judge" / dom.name / task.semester).glob("*/*/")):
        judge, condition = run_dir.parts[-2], run_dir.parts[-1]
        if judge not in roster or (judge, condition) not in configs or not fetched(run_dir):
            continue
        path, label = configs[(judge, condition)]
        harness = JudgeBT(JudgeConfig.from_cli(path), dom)
        runs.append(JudgeRun("pairwise", judge, label, harness, pairwise_from_logs(run_dir)))
    return runs


def direct_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """Every fetched direct run for the task, subject reasoning shown and hidden, for every
    judge YAML in configs/judges/. The run directory comes from JudgeDirect.log_dir, which
    hashes the prompts too, so a config whose directory does not exist was never run."""
    runs = []
    for path in sorted(JUDGES.glob("*.yml")):
        for hidden in (False, True):
            harness = JudgeDirect(DirectJudgeConfig.from_cli(path), dom, include_reasoning=not hidden)
            run_dir = harness.log_dir(task.semester)
            if not run_dir.is_dir() or not fetched(run_dir):
                continue
            label = "reasoning-hidden" if hidden else "reasoning-shown"
            runs.append(JudgeRun("direct", path.stem, label, harness, direct_from_logs(run_dir)))
    return runs


def judge_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """All pairwise then direct runs on disk for the task (see pairwise_runs, direct_runs)."""
    return pairwise_runs(dom, task) + direct_runs(dom, task)
