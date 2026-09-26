"""Loading: semester tasks from the frozen tables and the judge outputs already on disk.

Never launches judging. A judge run is one <domain, semester, judge, condition> directory
under logs/judge/ (pairwise verdicts) or logs/direct*/ (1-10 ratings); its rows are read
raw here and restricted to the frozen train items by analysis.matching.
"""

import os
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
               pick_left, semester, imported; direct -> bench, judge, item, model, score,
               semester, epoch (ensemble_runs relies on judge)
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

    semesters: restrict to these (e.g. ["2026-I"]); None means every frozen task, or those of them
    SMALL_JUDGE_SEMESTERS names (space-separated, e.g. "2025-I 2025-II 2026-I").
    Each Task holds references (theta = theta*), an empty-or-not historical cohort, the
    targets (theta NaN) and the targets' theta* in `truth`.
    """
    only = os.environ.get("SMALL_JUDGE_SEMESTERS", "").split()
    return [Task.from_semester(s, dom) for s in semesters or [s for s in semester_names(dom) if not only or s in only]]


def fetched(run_dir: Path) -> bool:
    """True when every .eval in the directory is a real log. Raises when some are still git-lfs
    pointer stubs: a skipped run would silently fall back to a worse condition (this happened
    with the 2026-I math and science reasoning-shown runs), so the caller must pull first."""
    stubs = [p for p in run_dir.glob("*.eval") if not zipfile.is_zipfile(p)]
    if stubs:
        rel = run_dir.relative_to(LOGS) if run_dir.is_relative_to(LOGS) else run_dir
        raise RuntimeError(f"{run_dir}: {len(stubs)} unfetched lfs pointer(s); run "
                           f"`git -C logs lfs pull --include=\"{rel}/**\"` (scoped: a bare pull fetches everything)")
    return True


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
    judge YAML in configs/judges/ whose judge is in the semester's top-3 roster
    (data/meta/<domain>/fit.yml `top`; file stems are the roster shortnames). The run
    directories come from JudgeDirect.log_dirs, which hashes the prompts too, so a config with
    no directory was never run; a run split across bench sets (coding: SWE-bench and
    Terminal-Bench before AgentBench joined) is read from every directory."""
    roster = yaml.safe_load((tables.frozen_dir(dom) / "fit.yml").read_text())["top"][task.semester]
    runs = []
    for path in sorted(JUDGES.glob("*.yml")):
        if path.stem not in roster:
            continue
        for hidden in (False, True):
            harness = JudgeDirect(DirectJudgeConfig.from_cli(path), dom, include_reasoning=not hidden)
            run_dirs = [d for d in harness.log_dirs(task.semester) if fetched(d)]
            if not run_dirs:
                continue
            label = "reasoning-hidden" if hidden else "reasoning-shown"
            table = pd.concat([direct_from_logs(d) for d in run_dirs], ignore_index=True)
            runs.append(JudgeRun("direct", path.stem, label, harness, table))
    return runs


def judge_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """All pairwise then direct runs on disk for the task (see pairwise_runs, direct_runs)."""
    return pairwise_runs(dom, task) + direct_runs(dom, task)
