"""Matching: keep the judge rows that belong to the frozen train items and the correct
transcripts, one row per judgment, and count what the campaign still lacks.

A pairwise run may hold verdicts on items outside the current train set (an earlier
freeze) or duplicate (item, pair, order) rows from retries and imports; a direct run may
hold ratings of transcripts no longer eligible. Both are cut down here so that calibration
sees exactly the protocol's legitimate input.
"""

import pandas as pd
from inspect_ai.log import read_eval_log_sample_summaries

from analysis.loading import JudgeRun
from capability.task import Cohort, Task
from configs.paths import is_successful_eval
from evals.transcripts import read_header


def eligible(cohorts: list[Cohort]) -> set[tuple[str, str, str]]:
    """The (model, bench, item) triples with a correct first-attempt transcript on the
    cohort's train items: what a judge is allowed to rate. Reading transcripts is cached
    on the cohort, so calling this repeatedly is cheap after the first time."""
    out = set()
    for cohort in cohorts:
        items = set(zip(cohort.items.bench, cohort.items.item))
        for model in cohort.models.index:
            out.update((model, b, i) for b, i in cohort.transcripts[model] if (b, i) in items)
    return out


def match_pairwise(run: JudgeRun, task: Task) -> JudgeRun:
    """Restrict verdicts to the task's train items, one per (bench, item, left, right).

    When a retry and an imported verdict cover the same (item, pair, order), the run's own
    verdict wins (`imported` False sorts first). `missing` becomes the number of judging
    samples JudgeBT.tasks would still build given what the logs already cover.
    """
    verdicts = run.table.merge(task.references.items[["bench", "item"]]).sort_values("imported", kind="stable")
    run.table = verdicts.drop_duplicates(["bench", "item", "model_left", "model_right"], keep="first")
    done: dict[str, set[str]] = {}  # task name ("bench/major|minor") -> judged item ids
    for p in run.harness.log_dir(task.semester).glob("*.eval"):
        if is_successful_eval(p):
            done.setdefault(read_header(p).eval.task, set()).update(
                str(s.id).rsplit("_", 1)[0] for s in read_eval_log_sample_summaries(str(p)))  # id = "<item>_<order>"
    run.missing = sum(len(t.dataset) for t in run.harness.tasks(task.references, task.test, done))
    return run


def match_direct(run: JudgeRun, task: Task) -> JudgeRun:
    """Restrict ratings to eligible transcripts of references and targets.

    from_logs already keeps the latest retry per transcript. `missing` becomes the number
    of eligible transcripts without any rating row.
    """
    keys = pd.DataFrame(sorted(eligible([task.train, task.test])), columns=["model", "bench", "item"])
    run.table = run.table.merge(keys)
    run.missing = len(keys) - len(run.table.drop_duplicates(["model", "bench", "item"]))
    return run


def match(run: JudgeRun, task: Task) -> JudgeRun:
    """match_pairwise or match_direct by run.protocol; returns the same (mutated) run."""
    return (match_pairwise if run.protocol == "pairwise" else match_direct)(run, task)
