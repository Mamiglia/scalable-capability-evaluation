"""Transcripts from one domain, calibrated on another domain's theta*.

Can a protocol forecast capability in one domain from transcripts collected in another, when the
references' capability in the scored domain is known? For each semester and each ordered pair of
domains, the source domain's task is rebuilt with every theta* replaced by the scored domain's for
the same semester: the references' and historical models' theta*, which is what each protocol fits
to, and the targets' truth, which is what the forecast is scored against. Items, transcripts,
judge runs and their matching stay the source domain's, so `calibration.forecasts` runs unchanged:

- Pairwise judge: verdicts on the source domain's items, references pinned at their scored-domain
  theta* in the linear conversion.
- Absolute scoring: OLS of the references' scored-domain theta* on their mean source-domain rating.
- Embedding and Deterministic features: regressions from source-domain transcript features to
  scored-domain theta*, fitted on references and history.
- Each judge protocol keeps, in every cell of a row, the judge that is best on references within
  the source domain itself, figure one's choice, so a row compares like with like
  (`fixed_judge_rows`).
- The saturated-item prediction is the references' mean scored-domain theta*, identical to that
  domain's own baseline, so a column of the matrix shares its baseline and the diagonal reproduces
  the forecasts in analysis/results/forecasts.csv. The runner asserts that.

Figures: plots.cross_domain.
"""

import numpy as np
import pandas as pd

from analysis import calibration, loading, matching
from analysis.challenges.signal_glm import PROTOCOLS, score
from analysis import stats
from analysis.stats import chosen
from analysis.stats import BASELINE
from capability.task import Cohort, Task
from reference import tables


def scored_theta(domain: str, semester: str) -> pd.Series:
    """Every model's theta* on one task of `domain`, from the frozen table; models without one
    are dropped."""
    models, _ = tables.load(loading.domain(domain))
    return models[f"{semester}_theta"].dropna()


def relabel(cohort: Cohort, theta: pd.Series, truth: bool = True) -> Cohort:
    """`cohort` keeping only the models `theta` labels, with their theta* replaced by it (NaN for
    a test cohort, whose theta is unknown by construction). A transcript cache already read stays
    with the new cohort, so swapping theta* costs no rereading of the logs."""
    models = cohort.models[cohort.models.index.isin(theta.index)].copy()
    models["theta"] = theta.reindex(models.index) if truth else np.nan
    swapped = Cohort(models, cohort.items, cohort.semester)
    if "transcripts" in cohort.__dict__:  # cached_property, keyed by model
        swapped.__dict__["transcripts"] = {m: t for m, t in cohort.transcripts.items() if m in models.index}
    return swapped


def swap(task: Task, theta: pd.Series) -> Task:
    """`task` with every theta* replaced by `theta`: the references and the historical cohort that
    protocols fit to, and the targets' truth that the forecast is scored against. Models `theta`
    does not label are dropped from each cohort."""
    return Task(task.semester, relabel(task.references, theta), relabel(task.historical, theta),
                relabel(task.test, theta, truth=False), task.truth.index.to_series()
                .map(theta).dropna().rename("theta"))


def forecasts(domains: list[str], semesters: tuple[str, ...]) -> pd.DataFrame:
    """One row per (source domain, scored domain, semester, target, protocol, variant, judge),
    the columns of analysis/results/forecasts.csv plus `scored_domain`; `domain` is the source.
    Judge runs and transcripts are read once per (source domain, semester)."""
    frames = []
    for source in domains:
        dom = loading.domain(source)
        protocols = calibration.transcript_protocols(dom)
        for task in loading.tasks(dom, list(semesters)):
            runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
            for scored in domains:
                theta = scored_theta(scored, task.semester)
                swapped = swap(task, theta)
                if swapped.truth.empty or swapped.references.models.empty:
                    print(f"cross_domain: {source} -> {scored} {task.semester}: no shared labels, skipped")
                    continue
                frame = calibration.forecasts(dom, swapped, protocols, runs)
                frames.append(frame.assign(scored_domain=scored))
                print(f"cross_domain: {source} -> {scored} {task.semester}: {len(frame)} rows, "
                      f"{len(swapped.truth)} targets, {len(swapped.references.models)} references")
    return pd.concat(frames, ignore_index=True)


def fixed_judge_rows(table: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """One forecast per (source domain, scored domain, semester, target, protocol), with its squared
    error `err` and the saturated-item prediction's on the same target `base`. Transcript protocols
    are taken as they are; a judge protocol uses, in every cell of a row, the judge that is best on
    references within the source domain itself (the diagonal cell, which is figure one's choice), so
    a row compares like with like instead of re-picking a judge for each scored domain."""
    core = chosen(stats.core_condition(table), variants)
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


def cell_scores(table: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """Prediction score per (source domain, scored domain, protocol): the targets' squared errors
    pooled over every semester before scoring, as everywhere else in this analysis, with each judge
    protocol fixed to its source domain's best judge (`fixed_judge_rows`)."""
    rows = fixed_judge_rows(table, variants)
    pooled = rows.groupby(["domain", "scored_domain", "protocol"]).agg(n=("err", "size"), mse=("err", "mean"),
                                                                        baseline_mse=("base", "mean")).reset_index()
    return pooled.rename(columns={"domain": "source", "scored_domain": "scored"}).assign(
        score=lambda d: score(d.mse, d.baseline_mse))


def model_scores(table: pd.DataFrame, variants: dict[str, str], cells: list[tuple[str, str]]) -> pd.DataFrame:
    """One prediction score per target model and protocol over `cells` (source, scored) pairs, the
    squared errors pooled over the cells and semesters first, judges fixed as in `fixed_judge_rows`."""
    rows = fixed_judge_rows(table, variants)
    rows = rows[[(s, d) in cells for s, d in zip(rows.domain, rows.scored_domain)]]
    pooled = rows.groupby(["model", "protocol"])[["err", "base"]].mean().reset_index()
    return pooled.assign(score=score(pooled.err, pooled.base))


def relative_to_within(grid: np.ndarray) -> np.ndarray:
    """Each cell minus its column's diagonal: the score of transcripts from the row's domain against
    the within-domain forecast of the same scored domain. On the log scale that difference is
    itself a prediction score with the within-domain forecast as the baseline: 0 on the diagonal,
    -1 means twice the within-domain forecast's error, +1 half of it."""
    return grid - np.diag(grid)[None, :]
