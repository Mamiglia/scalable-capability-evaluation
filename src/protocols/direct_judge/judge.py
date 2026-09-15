"""JudgeDirect: one LLM rating per correct train transcript, judged on demand.

The run uses one task per <bench, model> for references, historical models and targets
instead of model pairs. Logs use direct/ for visible subject reasoning and
direct_reasoning-hidden/ for hidden subject reasoning, with a _truncated-<head|tail>-<chars>
suffix when max_length cuts transcripts, followed by domain/semester/judge/hash.
Item IDs are preserved verbatim; no reverse order or shared-correct intersection applies.

Collection requires judge.task.direct_judge_task and a direct-rating JudgeConfig/prompt,
provided by the shared judging integration. The factory is imported only when needed,
so log reading and calibration are usable independently of collection.
"""

from dataclasses import replace
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from inspect_ai import Task
from inspect_ai.log import read_eval_log_sample_summaries

from capability.task import Cohort
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from configs.paths import is_successful_eval
from evals.transcripts import read_header
from judge.run import run_tasks
from protocols.direct_judge.protocol import DirectRegression
from reference.tables import DATA

SCORE_COLUMNS = ["bench", "judge", "item", "model", "score", "semester", "epoch"]


def from_logs(log_dir: str | Path) -> pd.DataFrame:
    """Read completed, unrestricted direct logs; latest retry wins, including missing ratings."""
    rows = []
    for path in sorted(Path(log_dir).glob("*.eval")):
        if not is_successful_eval(path):
            continue
        log = read_header(path)
        md = log.eval.metadata or {}
        if md.get("protocol") != "direct":
            continue
        for sample in read_eval_log_sample_summaries(str(path)):
            score = next(iter((sample.scores or {}).values()), None)
            answer = score.answer if score else None
            value = pd.to_numeric(answer, errors="coerce")
            value = value if 1 <= value <= 10 else np.nan
            rows.append(dict(bench=md["bench"], judge=log.eval.model, item=str(sample.id),
                             model=md["model"], score=value, semester=md["semester"],
                             epoch=sample.epoch, created=log.eval.created))
    scores = pd.DataFrame(rows, columns=[*SCORE_COLUMNS, "created"])
    scores = scores.sort_values("created", kind="stable").drop_duplicates(
        ["semester", "bench", "model", "item", "epoch"], keep="last")
    return scores.dropna(subset=["score"])[SCORE_COLUMNS].reset_index(drop=True)


class JudgeDirect(DirectRegression):
    def __init__(self, config: JudgeConfig, domain: DomainConfig, *, include_reasoning: bool = True):
        # Keep reasoning visible by default when reusing pairwise configs; experiments
        # can explicitly hide it. Copy so the effective setting enters the run hash.
        self.config, self.domain = replace(config, include_reasoning=include_reasoning), domain
        self.name = f"direct_judge_{config.judge.shortname.lower()}"

    def _prompts(self) -> dict[str, str]:
        # Aliased bench configs share a frozen path; the last entry wins, as in JudgeBT.
        return {self.domain.task(bench).name: bench.direct_system_prompt or self.config.system_prompt
                for bench in self.domain.benches}

    def log_dir(self, semester: str) -> Path:
        prompts = tuple((name, Path(prompt).read_text(encoding="utf-8"))
                        for name, prompt in sorted(self._prompts().items()))
        condition = (self.config.condition_hash, prompts)
        condition_hash = hashlib.sha256(repr(condition).encode()).hexdigest()[:8]
        directory = "direct" if self.config.include_reasoning else "direct_reasoning-hidden"
        if self.config.truncation_label:  # e.g. direct_truncated-tail-4000
            directory += f"_{self.config.truncation_label}"
        return (self.config.log_dir / directory / self.domain.name / semester
                / self.config.judge.shortname.lower() / condition_hash)

    def tasks(self, train: Cohort, test: Cohort) -> list[Task]:
        """One task per <bench, model>, on that model's correct first-attempt train items."""
        tasks = []
        for name, system_prompt in self._prompts().items():
            for cohort in (train, test):
                for model in cohort.models.index:
                    items = {item for b, item in cohort.transcripts[model] if b == name}
                    if not items:
                        continue
                    # The direct dataset, prompt and Inspect scorer belong to the shared
                    # judging layer. Defer this import until that collection path is used.
                    from judge.task import direct_judge_task

                    task = direct_judge_task(
                        DATA / cohort.models.loc[model, f"path_{name}"], items=items,
                        system_prompt=system_prompt,
                        max_length=self.config.max_length, include_reasoning=self.config.include_reasoning,
                        truncate=self.config.truncate,
                        task_name=f"{name}/{model}",
                        metadata={"protocol": "direct", "domain": self.domain.name, "bench": name,
                                  "model": model, "semester": test.semester},
                    )
                    if task is not None:
                        tasks.append(task)
        return tasks

    def judge(self, train: Cohort, test: Cohort) -> Path:
        """Run missing tasks with the same runner and completion check as JudgeBT."""
        log_dir = self.log_dir(test.semester)
        tasks = self.tasks(train, test)
        done = {read_header(p).eval.task for p in log_dir.glob("*.eval") if is_successful_eval(p)}
        todo = [task for task in tasks if task.name not in done]
        if todo:
            if not run_tasks(todo, self.config, log_dir):
                raise RuntimeError(f"direct judge run failed in {log_dir}")
        return log_dir

    def scores(self, train: Cohort, test: Cohort) -> pd.DataFrame:
        return from_logs(self.judge(train, test))
