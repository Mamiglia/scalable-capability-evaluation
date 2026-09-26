"""Saturation-boundary refit (manuscript Appendix D.11): tighten the EASY boundary from the 3 strongest references
to k = 4..8 or every reference and target, refit theta* on the enlarged HARD set and cut the observed items to
the frozen train items that stay EASY. Spec: docs/saturation-boundary-refit.md.
"""

import numpy as np
import pandas as pd

from capability.task import Cohort, Task
from configs.domain import DomainConfig
from reference import tables
from reference.freeze import easy_hard, local_fit

LEVELS = [3, 4, 5, 6, 7, 8, "all"]


def saturating_models(task: Task, models: pd.DataFrame, level) -> list[str]:
    """The models that must solve an item at `level`: the `level` references with the highest global theta,
    or every reference and target for "all"."""
    if level == "all":
        return [*task.references.models.index, *task.truth.index]
    return models.global_theta.loc[task.references.models.index].nlargest(level).index.tolist()


def level_split(domain: DomainConfig, task: Task, level) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(hard items, observed items) at `level`: rows of the frozen items table (bench, item, guess, a, b, ...).
    Observed = the task's frozen train items still EASY at this level."""
    models, items = tables.load(domain)
    status = easy_hard(tables.load_responses(domain), items, saturating_models(task, models, level))
    observed = items[items[f"{task.semester}_train"].astype(bool) & (status == "easy")]
    return items[status == "hard"], observed


def refit_theta(domain: DomainConfig, task: Task, hard: pd.DataFrame, seed: int = 0) -> pd.Series:
    """theta* on `hard` as reference.freeze.semester_task fits it: references + targets together, then every
    model of the semesters before the references with HARD responses in a second fit (the same model set as the
    freeze, so level 3 reproduces the frozen theta*). Index = model."""
    responses = tables.load_responses(domain)
    theta = local_fit(responses, hard, [*task.references.models.index, *task.truth.index], seed=seed).theta
    rosters, models = tables.load_semesters(), tables.load(domain)[0]
    earlier = list(rosters)[:list(rosters).index(task.semester) - 1]
    answered = set(responses.merge(hard[["bench", "item"]]).model)
    historical = [m for past in earlier for m in rosters[past] if m in models.index and m in answered]
    if historical:
        theta = pd.concat([theta, local_fit(responses, hard, historical, seed=seed).theta])
    return theta


def level_task(task: Task, theta: pd.Series, observed: pd.DataFrame) -> Task:
    """`task` with every theta* replaced by `theta` and every cohort's items cut to `observed`. A historical
    model without a refit theta* is dropped (it has no HARD response at this level)."""
    items = task.references.items.merge(observed[["bench", "item"]])

    def cohort(c: Cohort, labelled: bool) -> Cohort:
        models = c.models.assign(theta=theta.reindex(c.models.index)) if labelled else c.models
        if labelled:
            models = models[np.isfinite(models.theta)]
        out = Cohort(models, items, c.semester)
        if "transcripts" in c.__dict__:  # reuse the transcripts already read (cached_property), cut to the items
            keep = set(zip(items.bench, items.item))
            out.__dict__["transcripts"] = {m: {k: t for k, t in ts.items() if k in keep} for m, ts in c.transcripts.items()}
        return out

    return Task(task.semester, cohort(task.references, True), cohort(task.historical, True),
                cohort(task.test, False), theta.reindex(task.truth.index))


def ratio(g: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """MSE of the condition over MSE of the Baseline on the same targets (columns err, err_baseline, one row
    per target), with a percentile-bootstrap 95% interval over targets."""
    c, b = g.err.to_numpy(float), g.err_baseline.to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(c), size=(draws, len(c)))
    boot = c[idx].mean(1) / b[idx].mean(1)
    return dict(n=len(c), mse=c.mean(), baseline_mse=b.mean(), ratio=c.mean() / b.mean(),
                lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))
