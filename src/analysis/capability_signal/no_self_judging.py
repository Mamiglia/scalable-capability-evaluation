"""
The squared error by protocol with each task's judge left out of its own references. Every task's best judge on
references is itself one of the task's references, so the Pairwise judge compares targets against its own transcripts
and Absolute scoring calibrates on its own ratings. Here the judge's predictions are refitted on the references without
the judge; the judge choice, the transcript protocols and the Baseline are unchanged. The test uses the log likelihood
ratio rows whose dropped reference is the judge.
"""

import pandas as pd

from analysis import likelihood_ratio as lr
from analysis import loading, matching


def refit_without_judge(forecasts: pd.DataFrame, best: pd.DataFrame, domains: list[str],
                        semesters: list[str]) -> pd.DataFrame:
    """
    Recomputes each task's best judge's predictions of the task's targets from that judge run's verdicts or ratings,
    calibrated on the references without the judge, and writes them over theta_hat in a copy of `forecasts`. Every
    other row, and every judge_loo_mse, is left as it was, so the same judge stays selected.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); uses domain,
            semester, protocol, variant, judge, condition and model, and overwrites theta_hat.
        best: stats.best_judges rows of the tasks to refit; uses domain, semester, protocol, variant, judge and
            condition.
        domains: the domains to refit.
        semesters: the semesters to refit.
    Outputs:
        a copy of `forecasts`, same columns and rows, with the refitted theta_hat on the best judges' rows. Reads the
        frozen tables and the judge logs of every task.
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
    """
    Selects the log likelihood ratio rows that match the refitted predictions: for transcript protocols the full
    reference set, for judge protocols the reference set without the judge; all are relabelled as one reference set
    (dropped "-") so protocol_tests treats them as its full set.

    Inputs:
        llr: by_protocol.chosen_llr's rows, of every reference set; uses judge and dropped.
    Outputs:
        the selected rows, same columns, with dropped set to "-"; one row per target and protocol.
    """
    own = (llr.judge == "-") & (llr.dropped == "-") | (llr.judge != "-") & (llr.dropped == llr.judge)
    return llr[own].assign(dropped="-")
