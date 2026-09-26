"""
Cuts each judge run's rows down to the frozen train items and the correct transcripts, one row per judgment, and counts
the judgments the run still lacks. A pairwise run may hold verdicts on items outside the current train set (from an
earlier freeze) or duplicate (item, pair, shown order) rows from retries and imports; an Absolute scoring run may hold
ratings of transcripts that are no longer eligible. After matching, calibration sees exactly the protocol's input.
"""

import pandas as pd

from analysis.loading import JudgeRun
from capability.task import Cohort, Task


def eligible(cohorts: list[Cohort]) -> set[tuple[str, str, str]]:
    """
    Lists the transcripts a judge is allowed to rate: every (model, bench, item) with a correct first-attempt
    transcript on its cohort's train items. Transcripts are read once and cached on each cohort, so repeated calls are
    cheap after the first.

    Inputs:
        cohorts: the cohorts to cover (e.g. [task.train, task.test]); uses each cohort's items (bench, item), models
            and transcripts.
    Outputs:
        set of (model, bench, item) triples.
    """
    out = set()
    for cohort in cohorts:
        items = set(zip(cohort.items.bench, cohort.items.item))
        for model in cohort.models.index:
            out.update((model, b, i) for b, i in cohort.transcripts[model] if (b, i) in items)
    return out


def match_pairwise(run: JudgeRun, task: Task) -> JudgeRun:
    """
    Restricts a pairwise run's verdicts to the task's train items, one verdict per (bench, item, left model, right
    model), and counts the judgments the run still lacks. When the run's own verdict and an imported one cover the same
    item, pair and shown order, the run's own verdict is kept.

    Inputs:
        run: a pairwise JudgeRun; uses table (bench, item, model_left, model_right, imported) and harness.
        task: the semester task; uses references.items (the train items), references and test.
    Outputs:
        the same run, modified: table cut as above; missing set to the number of judging samples JudgeBT.tasks would
        still build for the references and targets, given the samples already in the run folder.
    """
    verdicts = run.table.merge(task.references.items[["bench", "item"]]).sort_values("imported", kind="stable")
    run.table = verdicts.drop_duplicates(["bench", "item", "model_left", "model_right"], keep="first")
    done = run.harness.judged_samples(run.harness.log_dir(task.semester))
    run.missing = sum(len(t.dataset) for t in run.harness.tasks(task.references, task.test, done))
    return run


def match_direct(run: JudgeRun, task: Task) -> JudgeRun:
    """
    Restricts an Absolute scoring run's ratings to eligible transcripts (correct first attempts on the train items) of
    the labelled models and the targets, and counts the eligible transcripts without any rating. The direct judge's
    from_logs has already kept the latest retry per transcript and epoch.

    Inputs:
        run: a direct JudgeRun; uses table (model, bench, item).
        task: the semester task; uses train (the references and any historical models) and test (the targets).
    Outputs:
        the same run, modified: table cut to the eligible transcripts; missing set to the number of eligible
        transcripts with no rating row.
    """
    keys = pd.DataFrame(sorted(eligible([task.train, task.test])), columns=["model", "bench", "item"])
    run.table = run.table.merge(keys)
    run.missing = len(keys) - len(run.table.drop_duplicates(["model", "bench", "item"]))
    return run


def match(run: JudgeRun, task: Task) -> JudgeRun:
    """
    Matches a run by its protocol: match_pairwise for "pairwise", match_direct otherwise.

    Inputs:
        run: a JudgeRun as read by analysis.loading.
        task: the semester task the run belongs to.
    Outputs:
        the same run object, with table and missing updated.
    """
    return (match_pairwise if run.protocol == "pairwise" else match_direct)(run, task)
