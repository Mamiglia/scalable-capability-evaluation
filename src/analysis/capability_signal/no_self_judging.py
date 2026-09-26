"""Figure 3 with each task's judge left out of its own references (Figure 21).

Every task's best judge on references is itself one of the task's references, so the Pairwise judge compares targets
against its own transcripts and Absolute scoring calibrates on its own ratings. Here the judge's forecasts are refitted
on the references without the judge; the judge choice, the transcript protocols and the Baseline are unchanged. The
test uses the log likelihood ratio rows whose dropped reference is the judge.
"""

import pandas as pd

from analysis import likelihood_ratio as lr
from analysis import loading, matching


def refit_without_judge(forecasts: pd.DataFrame, best: pd.DataFrame, domains: list[str],
                        semesters: list[str]) -> pd.DataFrame:
    """`forecasts` with each task's best judge's forecasts of its targets refitted on the references without the judge.

    best: stats.best_judges rows (domain, semester, protocol, variant, judge, condition) of the tasks to refit.
    """
    forecasts = forecasts.copy()
    for name in domains:
        dom = loading.domain(name)
        for task in loading.tasks(dom, semesters):
            runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
            for row in best[(best.domain == name) & (best.semester == task.semester)].itertuples(index=False):
                run = next(r for r in runs if (r.protocol, r.judge, r.condition) == (row.protocol, row.judge, row.condition))
                assert row.judge in task.references.theta.index, (name, task.semester, row.judge)
                refs = task.references.theta.drop(row.judge)
                theta_hat = lr.judge_rule(run.table, row.protocol, row.variant)(refs, list(task.truth.index))
                at = ((forecasts.domain == name) & (forecasts.semester == task.semester) & (forecasts.protocol == row.protocol)
                      & (forecasts.variant == row.variant) & (forecasts.judge == row.judge)
                      & (forecasts.condition == row.condition) & forecasts.model.isin(theta_hat.index))
                assert at.sum() == len(theta_hat), (name, task.semester, row.protocol, at.sum())
                forecasts.loc[at, "theta_hat"] = forecasts.loc[at, "model"].map(theta_hat).to_numpy()
            print(name, task.semester, "refitted", flush=True)
    return forecasts


def judge_dropped_llr(llr: pd.DataFrame) -> pd.DataFrame:
    """The log likelihood ratio rows of the refitted forecasts, as one reference set (dropped "-"): transcript
    protocols on the full reference set, judge protocols on the references without their judge.

    llr: by_protocol.chosen_llr rows (every reference set).
    """
    own = (llr.judge == "-") & (llr.dropped == "-") | (llr.judge != "-") & (llr.dropped == llr.judge)
    return llr[own].assign(dropped="-")
