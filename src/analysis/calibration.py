"""
Turns every protocol's output into predicted capabilities (theta-hat, on the theta* scale), one row per target. The
protocols that need no judge (Baseline, the HARD-item accuracy oracle, Length, Deterministic features, Embedding) fit
themselves on the task. Judge runs are converted from their matched tables with every variant (Pairwise judge: linear,
calibrated and bt; Absolute scoring: raw and item_z), and each run carries its leave-one-model-out error on the
references, which analysis.stats uses to pick a judge without target labels.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from protocols import judges
from analysis.stats import condition_rank
from analysis.baselines import AccuracyOracle, EasyItemPrediction
from protocols.embeddings import EmbeddingRegression
from analysis.loading import JudgeRun
from capability.evaluate import forecast
from capability.task import Task
from configs.domain import DomainConfig
from protocols import PROTOCOLS, Protocol
from protocols.pairwise.calibrated import JudgeCalibrated

COLUMNS = ["domain", "semester", "protocol", "variant", "judge", "condition", "judge_loo_mse",
           "model", "theta_hat", "theta", "missing"]

ENSEMBLE = "ensemble"  # the judge label of a pooled run


def core_runs(runs: list[JudgeRun]) -> list[JudgeRun]:
    """
    Keeps one run per (protocol, judge): its reasoning-shown run on the full train set if there is one, else its
    reasoning-shown run on a smaller train set ("-<k>pq"), else its reasoning-hidden run (the order of
    analysis.stats.condition_rank). Runs of any other condition are dropped.

    Inputs:
        runs: JudgeRun objects of one task; uses protocol, judge and condition.
    Outputs:
        list of JudgeRun, at most one per (protocol, judge).
    """
    out: dict[tuple[str, str], JudgeRun] = {}
    for run in runs:
        if condition_rank(run.condition) > 2:
            continue
        key = (run.protocol, run.judge)
        if key not in out or condition_rank(run.condition) < condition_rank(out[key].condition):
            out[key] = run
    return list(out.values())


def ensemble_runs(runs: list[JudgeRun]) -> list[JudgeRun]:
    """
    Pools the judges of each protocol that has at least two: one JudgeRun over the core runs (core_runs) of all its
    judges. For pairwise judging every judge's verdict counts as one more observation of the pair; for Absolute scoring
    a transcript's ratings from several judges are averaged in judges.valid_ratings.

    Inputs:
        runs: matched JudgeRun objects of one task.
    Outputs:
        list of JudgeRun, at most one per protocol ("pairwise", "direct"), empty when no protocol has two judges. Each
        has judge ENSEMBLE, condition "core", the members' tables concatenated, missing summed over the members (a
        transcript unrated by several judges counts once per judge) and the first member's harness.
    """
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
    """
    Lists the protocols that need no judge, each with its variant label: the Baseline, the HARD-item accuracy oracle,
    Length (glm_length, the length features alone), Deterministic features (glm_regex: the length, regular-expression
    and tool-call features), and one Embedding ridge per embedding variant. The oracle (which reads the domain's frozen
    tables) and the Embedding ridges are new instances per domain; the two feature protocols are the shared
    PROTOCOLS entries, which is safe because TabularRegression.fit clones the estimator.

    Inputs:
        dom: the domain configuration, for the oracle.
        embedding_variants: the transcript embeddings to fit an Embedding ridge on, e.g. "full" and "answer".
    Outputs:
        list of (Protocol, variant) pairs; the variant is "-" except for Embedding, where it is the embedding variant.
    """
    out = [(EasyItemPrediction(), "-"), (AccuracyOracle(dom), "-"), (PROTOCOLS["glm_length"], "-"), (PROTOCOLS["glm_regex"], "-")]
    return out + [(EmbeddingRegression(v), v) for v in embedding_variants]


def judge_forecast(run: JudgeRun, task: Task, variant: str) -> tuple[pd.Series, float]:
    """
    Converts one judge run into theta-hat for the task's targets under one variant, and measures the judge's
    leave-one-model-out error on the references. The pairwise "calibrated" variant shrinks toward JudgeCalibrated's
    frontier prior for the semester, as that protocol does; "linear" and "bt" take no outside prior. The Absolute
    scoring variants are a least-squares line from mean rating to theta*, on raw or item-z-scored ratings. The
    leave-one-out error ranks judges but is not the prediction's own error: under "calibrated" it shrinks each held-out
    reference toward the other references' mean and variance, not toward the frontier prior.

    Inputs:
        run: a matched JudgeRun; uses protocol, table and harness.domain (for the frontier prior).
        task: the semester task; uses semester, references.theta and test.models.
        variant: the conversion, one of judges.PAIRWISE for a pairwise run or judges.DIRECT for a direct run.
    Outputs:
        (theta-hat, a Series indexed by target model; the mean squared error of the references' leave-one-out theta-hat
        against their theta*, over the references that have one). A target without evidence gets the prior mean under
        "linear" and "calibrated" and NaN under "bt" and the Absolute scoring variants; the run's missing count says
        how many judgments are absent.
    """
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
    """
    Collects every prediction of one task: each judge-free protocol's theta-hat, and each judge run's, plus the pooled
    runs of ensemble_runs, under every variant of its protocol.

    Inputs:
        dom: the domain configuration; its name fills the domain column.
        task: the semester task; uses semester, truth (the targets' theta*) and whatever each protocol reads.
        protocols: (Protocol, variant) pairs, as from transcript_protocols.
        runs: the task's matched JudgeRun objects.
    Outputs:
        DataFrame with columns COLUMNS (domain, semester, protocol, variant, judge, condition, judge_loo_mse, model,
        theta_hat, theta, missing); one row per target, protocol, variant, judge and condition. For judge-free
        protocols judge and condition are "-", judge_loo_mse is NaN and missing is 0; pooled runs have protocol
        "<protocol>_ensemble", judge "ensemble" and condition "core"; missing is the run's count of absent judgments,
        repeated on each of its rows; theta_hat is NaN where a protocol gives no prediction.
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
