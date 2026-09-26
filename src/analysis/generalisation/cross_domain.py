"""Transcripts from one domain scored on another domain's capability (Appendix D.5, the numbers behind Figure 18).

For each semester and ordered pair of domains, the source domain's task is rebuilt with every θ* replaced by the scored
domain's for the same semester: the references' and historical models' θ*, which each protocol fits to, and the
targets' truth, which the forecast is scored against. Items, transcripts, judge runs and their matching stay the source
domain's, so calibration.forecasts runs unchanged. Each judge protocol keeps, in every cell of a row, the judge that is
best on references within the source domain itself, so a row compares like with like. The baseline is the references'
mean scored-domain θ*, identical to that domain's own, so the diagonal reproduces forecasts.csv (check_diagonal).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis import calibration, loading, matching, stats
from analysis.stats import BASELINE, PROTOCOLS
from capability.task import Cohort, Task
from reference import tables


def scored_theta(domain: str, semester: str) -> pd.Series:
    """Every model's θ* on one task of `domain`, from the frozen table; models without one dropped."""
    models, _ = tables.load(loading.domain(domain))
    return models[f"{semester}_theta"].dropna()


def relabel(cohort: Cohort, theta: pd.Series, truth: bool = True) -> Cohort:
    """`cohort` keeping only the models `theta` labels, with their θ* replaced by it (NaN for a test cohort, whose θ is
    unknown by construction). A transcript cache already read stays with the new cohort, so no log is reread."""
    models = cohort.models[cohort.models.index.isin(theta.index)].copy()
    models["theta"] = theta.reindex(models.index) if truth else np.nan
    swapped = Cohort(models, cohort.items, cohort.semester)
    if "transcripts" in cohort.__dict__:  # cached_property, keyed by model
        swapped.__dict__["transcripts"] = {m: t for m, t in cohort.transcripts.items() if m in models.index}
    return swapped


def swap(task: Task, theta: pd.Series) -> Task:
    """`task` with every θ* replaced by `theta` (references, historical cohort and the targets' truth); models `theta`
    does not label are dropped from each cohort."""
    return Task(task.semester, relabel(task.references, theta), relabel(task.historical, theta),
                relabel(task.test, theta, truth=False), task.truth.index.to_series().map(theta).dropna().rename("theta"))


def forecasts(domains: list[str], semesters: list[str]) -> pd.DataFrame:
    """One row per (source domain, scored domain, semester, target, protocol, variant, judge): the columns of
    forecasts.csv plus `scored_domain`; `domain` is the source. Judge runs and transcripts are read once per source
    domain and semester. About 25 minutes for the four domains and two semesters."""
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
    """Raises unless the diagonal cells (a domain scored on itself) reproduce forecasts.csv's forecast of every target
    and protocol, judge protocols as each task's best judge on references (so a change in which judge wins is caught
    too). Returns a one-line summary."""
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
    """One forecast per (source domain, scored domain, semester, target, protocol) with its squared error `err` and the
    baseline's on the same target `base`; judge protocols use, in every cell of a row, the source domain's own best
    judge on references (the diagonal cell's)."""
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
    """Per (source, scored, protocol): n, mse and baseline_mse over the cell's targets, every semester pooled.

    rows: fixed_judge_rows.
    """
    pooled = rows.groupby(["domain", "scored_domain", "protocol"]).agg(n=("err", "size"), mse=("err", "mean"),
                                                                        baseline_mse=("base", "mean")).reset_index()
    return pooled.rename(columns={"domain": "source", "scored_domain": "scored"})


def pairwise_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """The Pairwise judge per (source, scored) cell: mse, the baseline's (`baseline`) and the scored domain's own
    transcripts' (`own`), ratio_own = mse / own, and two-sided paired t-tests over the cell's targets of its squared
    errors against the own-transcript forecast's (p_own; NaN on the diagonal) and the baseline's (p_baseline).

    rows: fixed_judge_rows.
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
