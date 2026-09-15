"""Calibration: every signal to theta-hat on the theta* scale, one forecast row per target.

Transcript protocols (easy-item prediction, oracle, surface-feature ridge, embedding ridge) fit
themselves on the task. Judge runs are converted offline from their matched tables with
every variant (pairwise: linear and calibrated; direct: raw and item_z), and each judge run
carries its reference leave-one-out error so stats can pick the best judge without labels.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis import judges
from analysis.baselines import AccuracyOracle, EasyItemPrediction
from analysis.embeddings import EmbeddingRegression
from analysis.loading import JudgeRun
from capability.evaluate import forecast
from capability.task import Task
from configs.domain import DomainConfig
from protocols import PROTOCOLS, Protocol
from protocols.pairwise.calibrated import JudgeCalibrated

COLUMNS = ["domain", "semester", "protocol", "variant", "judge", "condition", "judge_loo_mse",
           "model", "theta_hat", "theta", "missing"]

ENSEMBLE = "ensemble"  # the judge label of a pooled run
CORE = ("reasoning-shown", "reasoning-hidden")


def core_runs(runs: list[JudgeRun]) -> list[JudgeRun]:
    """One run per (protocol, judge): its reasoning-shown run where judged, else its
    reasoning-hidden run; smaller-train-set runs ("-<k>pq") are never core."""
    out: dict[tuple[str, str], JudgeRun] = {}
    for run in runs:
        if run.condition not in CORE:
            continue
        key = (run.protocol, run.judge)
        if key not in out or run.condition == "reasoning-shown":
            out[key] = run
    return list(out.values())


def ensemble_runs(runs: list[JudgeRun]) -> list[JudgeRun]:
    """Per protocol, one pooled JudgeRun (judge ENSEMBLE, condition "core") over the core runs
    of at least two judges: tables concatenated (pairwise: every judge's verdict is one more
    observation of the pair; direct: a transcript's ratings from several judges average in
    valid_ratings), `missing` summed, `harness` the first member's. Empty when no protocol
    has two judges."""
    out = []
    core = core_runs(runs)
    for protocol in ("pairwise", "direct"):
        members = [r for r in core if r.protocol == protocol]
        if len(members) < 2:
            continue
        table = pd.concat([r.table for r in members], ignore_index=True)
        out.append(JudgeRun(protocol, ENSEMBLE, "core", members[0].harness, table, sum(r.missing for r in members)))
    return out


def transcript_protocols(dom: DomainConfig, embedding_variants: tuple[str, ...] = ("full", "answer")) -> list[tuple[Protocol, str]]:
    """(protocol, variant) pairs needing no judge: easy-item prediction, oracle, glm_regex, and one
    embedding ridge per variant. Fresh instances per domain (the oracle reads that domain's
    frozen tables)."""
    out = [(EasyItemPrediction(), "-"), (AccuracyOracle(dom), "-"), (PROTOCOLS["glm_regex"], "-")]
    return out + [(EmbeddingRegression(v), v) for v in embedding_variants]


def judge_forecast(run: JudgeRun, task: Task, variant: str) -> tuple[pd.Series, float]:
    """(theta-hat indexed by target, reference leave-one-out MSE) for one judge run under one
    variant. Pairwise "calibrated" shrinks toward JudgeCalibrated's frontier prior for the
    semester, as the protocol does (frontier_prior needs only a `.domain`); "linear" needs
    none. Direct variants are OLS on mean ratings, raw or item-z-scored."""
    refs, targets = task.references.theta, list(task.test.models.index)
    if run.protocol == "pairwise":
        prior = (JudgeCalibrated.frontier_prior(SimpleNamespace(domain=run.harness.domain), task.semester)
                 if variant == "calibrated" else None)
        theta_hat = judges.pairwise_theta(run.table, refs, targets, variant, prior)
        loo = judges.pairwise_reference_loo(run.table, refs, variant)
    else:
        theta_hat = judges.direct_theta(run.table, refs, targets, variant)
        loo = judges.direct_reference_loo(run.table, refs, variant)
    return theta_hat, float(((loo - refs) ** 2).mean())


def forecasts(dom: DomainConfig, task: Task, protocols: list[tuple[Protocol, str]], runs: list[JudgeRun]) -> pd.DataFrame:
    """One row per target x (protocol, variant) x judge x condition, columns COLUMNS.

    judge/condition are "-" and judge_loo_mse NaN for transcript protocols; theta_hat is NaN
    where a protocol has nothing to say; missing is the run's unjudged count on its rows.
    Pooled ensemble runs (ensemble_runs) are added with protocol `<protocol>_ensemble`, judge
    `ensemble`, condition `core`.
    """
    frames = []
    for p, variant in protocols:
        frames.append(forecast(task, p).assign(protocol=p.name, variant=variant, judge="-", condition="-",
                                               judge_loo_mse=np.nan, missing=0))
    for run in runs + ensemble_runs(runs):
        label = f"{run.protocol}_{ENSEMBLE}" if run.judge == ENSEMBLE else run.protocol
        for variant in judges.PAIRWISE if run.protocol == "pairwise" else judges.DIRECT:
            theta_hat, loo_mse = judge_forecast(run, task, variant)
            frames.append(pd.DataFrame({"semester": task.semester, "model": task.truth.index,
                                        "theta_hat": theta_hat.reindex(task.truth.index).to_numpy(float),
                                        "theta": task.truth.to_numpy()})
                          .assign(protocol=label, variant=variant, judge=run.judge, condition=run.condition,
                                  judge_loo_mse=loo_mse, missing=run.missing))
    return pd.concat(frames, ignore_index=True).assign(domain=dom.name)[COLUMNS]
