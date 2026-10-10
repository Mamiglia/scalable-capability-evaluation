"""
Reads what the analysis needs from disk: the domain configurations, the semester tasks from the frozen tables, and the
judge runs already collected in logs/. It never launches judging. A judge run is one folder per domain, semester,
judge and condition, under logs/pairwise/<view>/ (Pairwise judge verdicts) or logs/absolute/<view>/ (Absolute scoring ratings, 1 to 10);
its rows are read here as logged and cut down to the frozen train items by analysis.matching.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    """
    One judge's logs for one task (a domain and semester), as read from disk, with the harness that wrote them.

    Attributes:
        protocol: "pairwise" (Pairwise judge) or "direct" (Absolute scoring).
        judge: the judge's short name, its folder name under logs/ and data/ (e.g. "gpt-4-0613"); "ensemble" for a run
            pooled by analysis.calibration.ensemble_runs.
        condition: "reasoning-shown", "reasoning-hidden", or "reasoning-shown-<k>pq" for a pairwise run on a smaller
            train set (k items per difficulty quartile); "core" for a pooled run.
        harness: the JudgeBT or JudgeDirect that wrote the logs; analysis.matching asks it which judgments are missing.
        table: the rows as logged. Pairwise, one row per verdict: bench, judge, item, model_left, model_right,
            pick_left, semester, imported, chars_left, chars_right. Direct, one row per numeric rating (latest retry
            per transcript and epoch): bench, judge, item, model, score, semester, epoch.
        missing: the number of judgments still absent on the train items; 0 until analysis.matching sets it.
    """

    protocol: str
    judge: str
    condition: str
    harness: JudgeBT | JudgeDirect
    table: pd.DataFrame
    missing: int = 0


def domain(name: str) -> DomainConfig:
    """
    Loads a domain's configuration from configs/domain/<name>.yml.

    Inputs:
        name: the domain's code key: "science", "math", "coding" or "cyber".
    Outputs:
        the DomainConfig.
    """
    return DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))


def tasks(dom: DomainConfig, semesters: list[str] | None = None) -> list[Task]:
    """
    Builds a domain's semester tasks from its frozen tables (data/meta/<domain>/): for each semester, the references
    (the previous semester's models, with theta*), the targets (theta hidden), the targets' theta* and the train items.

    Inputs:
        dom: the domain configuration.
        semesters: the semesters to build (e.g. ["2026-I"]); None builds every semester that has a `<semester>_role`
            column in the frozen models table.
    Outputs:
        list of Task, one per semester in the order given (or the frozen table's column order). Each Task has
        references (theta = theta*), historical (older models with this task's theta*, possibly empty), test (the
        targets, theta NaN) and truth (the targets' theta*).
    """
    return [Task.from_semester(s, dom) for s in semesters or semester_names(dom)]


def fetched(run_dir: Path) -> bool:
    """
    Checks that every .eval log in a run folder has been downloaded rather than left as a git-lfs pointer stub. It
    raises instead of returning False because a skipped run would silently fall back to a worse condition (this
    happened with the 2026-I Maths and Science reasoning-shown runs), so the caller must pull first.

    Inputs:
        run_dir: a judge run folder under logs/.
    Outputs:
        True when every .eval is a real (zip) log; otherwise raises RuntimeError naming the scoped `git lfs pull`
        command that fetches the folder.
    """
    stubs = [p for p in run_dir.glob("*.eval") if not zipfile.is_zipfile(p)]
    if stubs:
        rel = run_dir.relative_to(LOGS) if run_dir.is_relative_to(LOGS) else run_dir
        raise RuntimeError(f"{run_dir}: {len(stubs)} unfetched lfs pointer(s); run "
                           f"`git -C logs lfs pull --include=\"{rel}/**\"` (scoped: a bare pull fetches everything)")
    return True


def pairwise_configs() -> dict[tuple[str, str], tuple[Path, str]]:
    """
    Traces pairwise run folders back to the judge configurations that produced them: a run folder is named by the
    judge's short name and its condition hash, so every YAML in configs/judges/ (reasoning hidden) and
    configs/judges/reasoning/ (reasoning shown) is indexed by that pair.

    Inputs:
        none.
    Outputs:
        dict (judge short name, condition hash) -> (judge YAML path, condition label). The label is
        "reasoning-hidden", "reasoning-shown", or "reasoning-shown-<k>pq" when a reasoning config sets
        train_per_quartile = k (a smaller train set).
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


def pairwise_dirs(logs: Path, domain: str, semester: str, judge: str = "*", condition: str = "*") -> list[Path]:
    """The pairwise run folders logs/pairwise/<view>/<domain>/<semester>/<judge>/<condition hash>/ of any view
    (a condition hash names one view, so the view need not be known)."""
    return [d for d in (logs / "pairwise").glob(f"*/{domain}/{semester}/{judge}/{condition}") if d.is_dir()]


def pairwise_run(logs: Path, domain: str, semester: str, judge: str, condition: str) -> Path:
    """The one pairwise run folder of (domain, semester, judge, condition hash), whatever its view."""
    (run_dir,) = pairwise_dirs(logs, domain, semester, judge, condition)
    return run_dir


def pairwise_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """
    Reads every pairwise run on disk for one task whose judge is one of the semester's three candidate judges (the
    `top` roster in data/meta/<domain>/fit.yml) and whose condition hash matches a current config (pairwise_configs).
    Folders of pilot judges outside the roster and of stale conditions are skipped silently.

    Inputs:
        dom: the domain configuration.
        task: the semester task; its semester selects logs/pairwise/<view>/<domain>/<semester>/<judge>/<condition hash>/.
    Outputs:
        list of JudgeRun with protocol "pairwise", one per run folder, verdicts as logged (not yet matched). Raises
        RuntimeError if a candidate run folder holds unfetched git-lfs stubs (see fetched).
    """
    configs = pairwise_configs()
    roster = tables.fit_settings(dom)["top"][task.semester]
    runs = []
    for run_dir in sorted(pairwise_dirs(LOGS, dom.name, task.semester), key=lambda d: d.parts[-2:]):
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
    directories come from JudgeDirect.log_dirs, which hashes the prompts too, so a config whose
    directory does not exist was never run; a run split across bench sets (coding: SWE-bench and
    Terminal-Bench before agent_bench_os joined) is read from every directory, the full bench set's
    rating winning where two directories rate the same (model, bench, item)."""
    roster = tables.fit_settings(dom)["top"][task.semester]
    runs = []
    for path in sorted(JUDGES.glob("*.yml")):
        if path.stem not in roster:
            continue
        for hidden in (False, True):
            harness = JudgeDirect(DirectJudgeConfig.from_cli(path), dom, include_reasoning=not hidden)
            run_dirs = [d for d in harness.log_dirs(task.semester) if fetched(d)]
            if not run_dirs:
                continue
            table = pd.concat([direct_from_logs(d) for d in run_dirs], ignore_index=True)
            table = table.drop_duplicates(["model", "bench", "item", "epoch"], keep="first")
            label = "reasoning-hidden" if hidden else "reasoning-shown"
            runs.append(JudgeRun("direct", path.stem, label, harness, table))
    return runs


def judge_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """
    Reads every judge run on disk for one task: the pairwise runs, then the Absolute scoring runs.

    Inputs:
        dom: the domain configuration.
        task: the semester task.
    Outputs:
        list of JudgeRun: pairwise_runs(dom, task) followed by direct_runs(dom, task).
    """
    return pairwise_runs(dom, task) + direct_runs(dom, task)
