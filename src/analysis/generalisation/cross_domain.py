"""
Transcripts from one domain scored on another domain's capability. For each semester and ordered pair of domains, the
source domain's task is rebuilt with every θ* replaced by the scored domain's for the same semester (the references'
and historical models' θ*, which each protocol fits to, and the targets' truth, which the prediction is scored
against), while items, transcripts, judge runs and their matching stay the source domain's, so calibration.forecasts
runs unchanged. Each judge protocol keeps, in every cell of a row, the judge that is best on the references within
the source domain itself, so a row compares like with like. The Baseline is the references' mean scored-domain θ*,
identical to that domain's own, so the diagonal reproduces forecasts.csv (check_diagonal).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis import calibration, loading, matching, stats
from analysis.stats import BASELINE, PROTOCOLS
from capability.task import Cohort, Task
from reference import tables


def scored_theta(domain: str, semester: str) -> pd.Series:
    """
    Every model's θ* on one task of a domain, read from the frozen model table (the <semester>_theta column); models
    without one are dropped.

    Inputs:
        domain: the domain key whose θ* is used.
        semester: the task's semester.
    Outputs:
        Series of θ*, indexed by model.
    """
    models, _ = tables.load(loading.domain(domain))
    return models[f"{semester}_theta"].dropna()


def relabel(cohort: Cohort, theta: pd.Series, truth: bool = True) -> Cohort:
    """
    A copy of `cohort` keeping only the models `theta` labels, with their θ* replaced by it (NaN for a target cohort,
    whose θ is unknown by construction). A transcript cache already read stays with the new cohort, so no log is
    reread.

    Inputs:
        cohort: a task's references, historical models or targets.
        theta: the scored domain's θ*, indexed by model.
        truth: True to take the θ* from `theta`, False to set it NaN (the targets).
    Outputs:
        the relabelled Cohort, with the same items.
    """
    models = cohort.models[cohort.models.index.isin(theta.index)].copy()
    models["theta"] = theta.reindex(models.index) if truth else np.nan
    swapped = Cohort(models, cohort.items, cohort.semester)
    if "transcripts" in cohort.__dict__:  # cached_property, keyed by model
        swapped.__dict__["transcripts"] = {m: t for m, t in cohort.transcripts.items() if m in models.index}
    return swapped


def swap(task: Task, theta: pd.Series) -> Task:
    """
    A copy of `task` with every θ* replaced by `theta`: the references', the historical models' and the targets'
    truth. Models `theta` does not label are dropped from each cohort and from the truth.

    Inputs:
        task: the source domain's task.
        theta: the scored domain's θ* for the same semester (scored_theta).
    Outputs:
        the relabelled Task.
    """
    return Task(task.semester, relabel(task.references, theta), relabel(task.historical, theta),
                relabel(task.test, theta, truth=False), task.truth.index.to_series().map(theta).dropna().rename("theta"))


def forecasts(domains: list[str], semesters: list[str]) -> pd.DataFrame:
    """
    Every protocol's predictions for every ordered pair of domains and every semester, with the source domain's
    transcripts and the scored domain's θ*; a pair with no shared labelled targets or references is skipped. Judge
    runs and transcripts are read once per source domain and semester; the four domains and two semesters take about
    25 minutes.

    Inputs:
        domains: the domain keys, used both as sources and as scored domains.
        semesters: the semesters.
    Outputs:
        DataFrame with the columns of forecasts.csv (domain, semester, protocol, variant, judge, condition,
        judge_loo_mse, model, theta_hat, theta, missing) plus scored_domain, domain being the source; one row per
        source domain, scored domain, semester, target, protocol, variant, judge and condition.
    """
    frames = []
    for source in domains:
        dom = loading.domain(source)
        protocols = calibration.transcript_protocols(dom)
        for task in loading.tasks(dom, list(semesters)):
            runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
            for scored in domains:
                swapped = swap(task, scored_theta(scored, task.semester))
                if swapped.truth.empty or swapped.references.models.empty:
                    print(f"cross_domain: {source} -> {scored} {task.semester}: no shared labels, skipped")
                    continue
                frame = calibration.forecasts(dom, swapped, protocols, runs)
                frames.append(frame.assign(scored_domain=scored))
                print(f"cross_domain: {source} -> {scored} {task.semester}: {len(frame)} rows, "
                      f"{len(swapped.truth)} targets, {len(swapped.references.models)} references", flush=True)
    return pd.concat(frames, ignore_index=True)


def check_diagonal(cross: pd.DataFrame, forecasts_: pd.DataFrame, variants: dict[str, str], semesters: list[str]) -> str:
    """
    Checks that the diagonal cells (a domain scored on itself) reproduce forecasts.csv: both tables must have the same
    rows and the same predicted θ within 1e-3 for every target and protocol, judge protocols as each task's best judge
    on the references (so a change in which judge wins is caught too); raises AssertionError otherwise.

    Inputs:
        cross: the output of forecasts.
        forecasts_: the forecasts table (analysis/forecasts/forecasts.csv).
        variants: protocol -> its reported variant (config.yaml protocols).
        semesters: the semesters of forecasts.csv compared.
    Outputs:
        str, a one-line summary with the number of rows compared and the largest gap.
    """
    diagonal = stats.target_errors(cross[cross.domain == cross.scored_domain], variants)
    within = stats.target_errors(forecasts_, variants, tuple(semesters))
    keys = ["domain", "semester", "protocol", "model"]
    merged = diagonal.merge(within[[*keys, "theta_hat"]], on=keys, how="outer", suffixes=("", "_within"), indicator=True)
    alone = merged[merged._merge != "both"]
    assert alone.empty, f"rows in one table only: {alone[keys].head().to_dict('records')}"
    gap = (merged.theta_hat - merged.theta_hat_within).abs()
    worst = merged.loc[gap.idxmax(), keys].to_dict()
    assert gap.max() < 1e-3, f"diagonal differs from forecasts.csv, worst {gap.max():.4f}: {worst}"
    return f"diagonal matches forecasts.csv on {len(merged)} best-judge rows, worst gap {gap.max():.5f}"


def fixed_judge_rows(cross: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """
    One prediction per source domain, scored domain, semester, target and protocol, in the core condition and the
    reported variant, with its squared error and the Baseline's on the same target. Judge protocols use, in every cell
    of a row, the source domain's own best judge on the references (the diagonal cell's); targets without a Baseline
    error are dropped.

    Inputs:
        cross: the output of forecasts.
        variants: protocol -> its reported variant (config.yaml protocols).
    Outputs:
        the kept rows of `cross` with err (the squared error) and base (the Baseline's squared error on the same
        target) columns.
    """
    core = stats.in_variants(stats.core_condition(cross), variants)
    core = core[core.protocol.isin(PROTOCOLS)]
    judged = core.protocol.isin(stats.JUDGED)
    key = ["domain", "semester", "protocol", "variant"]
    diagonal = core[judged & (core.domain == core.scored_domain)]
    best = diagonal[diagonal.groupby(key).judge_loo_mse.transform("min") == diagonal.judge_loo_mse]
    best = best.drop_duplicates(key)[[*key, "judge", "condition"]]
    rows = pd.concat([core[~judged], core[judged].merge(best, on=[*key, "judge", "condition"])], ignore_index=True)
    rows = rows.dropna(subset=["theta_hat", "theta"]).assign(err=lambda r: (r.theta_hat - r.theta) ** 2)
    cell = ["domain", "scored_domain", "semester", "model"]
    base = rows[rows.protocol == BASELINE].set_index(cell).err
    return rows.assign(base=rows.set_index(cell).index.map(base)).dropna(subset=["base"])


def cell_errors(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Per source domain, scored domain and protocol, the number of targets, the mean squared error and the Baseline's
    mean squared error over the cell's targets, every semester pooled.

    Inputs:
        rows: the output of fixed_judge_rows (domain, scored_domain, protocol, err, base).
    Outputs:
        DataFrame with columns source, scored, protocol, n, mse, baseline_mse; one row per cell and protocol.
    """
    pooled = rows.groupby(["domain", "scored_domain", "protocol"]).agg(n=("err", "size"), mse=("err", "mean"),
                                                                        baseline_mse=("base", "mean")).reset_index()
    return pooled.rename(columns={"domain": "source", "scored_domain": "scored"})


def pairwise_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """
    For the Pairwise judge in each (source, scored) cell: the mean squared error, the Baseline's, and that of the
    scored domain's own transcripts (over every target of the scored domain's diagonal cell), their ratio, and
    two-sided paired t-tests over the cell's targets of its squared errors against the own-transcript prediction's
    (NaN on the diagonal) and against the Baseline's.

    Inputs:
        rows: the output of fixed_judge_rows (domain, scored_domain, semester, model, protocol, err, base).
    Outputs:
        DataFrame with columns source, scored, mse, baseline, own, ratio_own (mse / own), p_own, p_baseline; one row
        per cell.
    """
    pairwise = rows[rows.protocol == "pairwise"]
    cells = []
    for scored, g in pairwise.groupby("scored_domain"):
        own = g[g.domain == scored].set_index(["semester", "model"]).err
        for source, h in g.groupby("domain"):
            h = h.set_index(["semester", "model"])
            cells.append(dict(source=source, scored=scored, mse=h.err.mean(), baseline=h.base.mean(), own=own.mean(),
                              ratio_own=h.err.mean() / own.mean(),
                              p_own=np.nan if source == scored else scipy_stats.ttest_rel(h.err, own.reindex(h.index)).pvalue,
                              p_baseline=scipy_stats.ttest_rel(h.err, h.base).pvalue))
    return pd.DataFrame(cells)
