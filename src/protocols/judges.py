"""The judge protocols the paper reports, computed offline from the recorded judgments (data/meta/judgments_*.csv),
and the conversions they share with analysis/ (verdicts or ratings -> theta-hat, reference leave-one-out error).

JudgeTable("pairwise", "linear") is the paper's Pairwise judge and JudgeTable("direct", "raw") its Absolute scoring:
per task, every judge's core run is converted, and the judge with the lowest leave-one-out error on the references
(no target label) gives the forecast. Registered in protocols.PROTOCOLS as pairwise[_calibrated|_bt] and
direct[_item_z]. Collecting new judgments is `uv run judge`, never a protocol's job.

Pairwise: "linear" is JudgeLinear's per-bench weighted least squares on pair logits
(protocols.pairwise.linear); "calibrated" is JudgeCalibrated's calibrated Bradley-Terry with
shrinkage (protocols.pairwise.calibrated); "bt" is plain Bradley-Terry with the references
pinned at theta* (protocols.pairwise.bradley_terry.rank), no calibration and no prior. Direct: OLS of theta* on a model's mean rating,
"raw" or "item_z" (each rating z-scored per (bench, item) with the references' mean and sd).
Every statistic comes from the references passed in; targets never enter a fit.
"""

from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from capability.task import Task
from protocols import Protocol

from protocols.pairwise import calibrated
from protocols.pairwise.bradley_terry import rank
from protocols.pairwise.linear import weighted_theta
from reference.tables import META

PAIRWISE = ("linear", "calibrated", "bt")
DIRECT = ("raw", "item_z")


def pairwise_theta(verdicts: pd.DataFrame, refs: pd.Series, models: list[str], conversion: str,
                   prior: tuple[float, float] | None = None) -> pd.Series:
    """
    Places `models` on the theta* scale from pairwise verdicts, with the references fixed at their theta*. "linear" is
    weighted_theta: per-bench weighted least squares on win-rate logits under the references' own normal prior. "bt"
    is Bradley-Terry over the verdicts that involve a reference, the references pinned. "calibrated" fits a slope per
    bench and a shared bias on the reference pairs, finds each model's maximum-likelihood theta, then shrinks it toward
    `prior` with a weight set by the judge's leave-one-out error on the references.

    Inputs:
        verdicts: a matched verdict table; uses bench, model_left, model_right and pick_left ("bt" also item and
            judge).
        refs: the references' theta*, indexed by model.
        models: the models to place (the targets, or one held-out reference).
        conversion: "linear", "calibrated" or "bt".
        prior: (mean, variance) that "calibrated" shrinks toward; None uses the references' mean and variance. Unused
            by "linear" and "bt".
    Outputs:
        Series of theta-hat indexed by `models`. A model without verdicts against a reference gets the prior mean
        under "linear" and "calibrated" and NaN under "bt"; under "calibrated" every model is NaN when no reference
        pair was judged.
    """
    if conversion == "linear":
        return weighted_theta(verdicts, refs, list(models)).reindex(models)
    if conversion == "bt":  # pinned Bradley-Terry: a model never compared with a pinned reference stays NaN
        keep = verdicts[(verdicts.model_left.isin(refs.index) | verdicts.model_right.isin(refs.index))
                        & (verdicts.model_left.isin(list(models) + list(refs.index)) & verdicts.model_right.isin(list(models) + list(refs.index)))]
        if keep.empty:
            return pd.Series(np.nan, index=models)
        return rank(keep, anchors=refs).set_index("model").ability.reindex(models)
    assert conversion == "calibrated", conversion
    alpha, beta = calibrated.calibrate(verdicts, refs)
    if alpha.empty:
        return pd.Series(np.nan, index=models)
    sigma2 = float(((calibrated.loo(verdicts, refs) - refs) ** 2).mean())
    est = pd.Series({m: calibrated.estimate(verdicts, m, refs, alpha, beta) for m in models}, dtype=float)
    return calibrated.shrink(est, sigma2, prior or (float(refs.mean()), float(refs.var()))).reindex(models)


def pairwise_reference_loo(verdicts: pd.DataFrame, refs: pd.Series, conversion: str) -> pd.Series:
    """
    Places each reference from the reference-against-reference verdicts only, with the conversion fitted on the other
    references (the held-out reference's theta* is never used) and, under "calibrated", shrunk toward the other
    references' mean and variance.

    Inputs:
        verdicts: a matched verdict table; only rows with a reference on both sides are used.
        refs: the references' theta*, indexed by model.
        conversion: "linear", "calibrated" or "bt".
    Outputs:
        Series of leave-one-out theta-hat indexed by refs.index (NaN where the conversion gives none).
    """
    among = verdicts[verdicts.model_left.isin(refs.index) & verdicts.model_right.isin(refs.index)]
    out = {}
    for r in refs.index:
        others = refs.drop(r)
        out[r] = pairwise_theta(among, others, [r], conversion, prior=(float(others.mean()), float(others.var())))[r]
    return pd.Series(out, dtype=float).reindex(refs.index)


def valid_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the ratings with a numeric score between 1 and 10 and averages the repeated judging epochs (and, in a pooled
    run, the several judges) of each transcript, so every transcript weighs the same.

    Inputs:
        ratings: a rating table; uses model, bench, item and score (non-numeric scores are dropped).
    Outputs:
        DataFrame with columns model, bench, item, score; one row per transcript.
    """
    values = pd.to_numeric(ratings.score, errors="coerce")
    kept = ratings.assign(score=values)[values.between(1, 10)]
    return kept.groupby(["model", "bench", "item"], as_index=False).score.mean()


def normalise(ratings: pd.DataFrame, refs: pd.Series, normalisation: str) -> pd.DataFrame:
    """
    Applies a rating normalisation. "raw" leaves the scores unchanged; "item_z" replaces each score by
    (score - mean) / sd, with mean and sd over the references' scores of the same (bench, item). An item on which the
    references do not vary (sd 0, or a single reference) gives z = 0; an item no reference rated is dropped.

    Inputs:
        ratings: per-transcript ratings with columns model, bench, item, score (valid_ratings output).
        refs: the references' theta*, indexed by model; only the index is used.
        normalisation: "raw" or "item_z".
    Outputs:
        under "raw", `ratings` itself; under "item_z", a DataFrame with columns model, bench, item, score, one row per
        rated transcript of an item some reference rated.
    """
    if normalisation == "raw":
        return ratings
    assert normalisation == "item_z", normalisation
    stats = ratings[ratings.model.isin(refs.index)].groupby(["bench", "item"]).score.agg(["mean", "std"])
    merged = ratings.merge(stats, left_on=["bench", "item"], right_index=True)
    z = ((merged.score - merged["mean"]) / merged["std"]).where(merged["std"] > 0, 0.0).fillna(0.0)
    return merged.assign(score=z)[["model", "bench", "item", "score"]]


def direct_theta(ratings: pd.DataFrame, refs: pd.Series, models: list[str], normalisation: str) -> pd.Series:
    """
    Places `models` on the theta* scale from Absolute scoring ratings: each model's mean (normalised) rating over its
    transcripts, then a least-squares line of the references' theta* on their mean rating, applied to each model's mean.

    Inputs:
        ratings: a matched rating table; uses model, bench, item and score.
        refs: the references' theta*, indexed by model.
        models: the models to place (the targets, or one held-out reference).
        normalisation: "raw" or "item_z" (see normalise).
    Outputs:
        Series of theta-hat indexed by `models`; NaN for a model without valid ratings, and for every model when fewer
        than two references have both a mean rating and a finite theta*.
    """
    means = normalise(valid_ratings(ratings), refs, normalisation).groupby("model").score.mean()
    x, y = means.reindex(refs.index), refs
    known = np.isfinite(x) & np.isfinite(y)
    out = pd.Series(np.nan, index=models)
    if known.sum() < 2:
        return out
    line = LinearRegression().fit(x[known].to_frame(), y[known])
    target = means.reindex(models)
    if (has := np.isfinite(target)).any():
        out[has] = line.predict(target[has].to_frame())
    return out


def direct_reference_loo(ratings: pd.DataFrame, refs: pd.Series, normalisation: str) -> pd.Series:
    """
    Places each reference from its own ratings with the line (and, under "item_z", the per-item mean and sd) fitted on
    the other references only.

    Inputs:
        ratings: a matched rating table; only the references' rows are used.
        refs: the references' theta*, indexed by model.
        normalisation: "raw" or "item_z".
    Outputs:
        Series of leave-one-out theta-hat indexed by refs.index (NaN where direct_theta gives none).
    """
    among = ratings[ratings.model.isin(refs.index)]
    return pd.Series({r: direct_theta(among, refs.drop(r), [r], normalisation)[r] for r in refs.index}, dtype=float)


@cache
def judgments(kind: str) -> pd.DataFrame:
    """data/meta/judgments_<kind>.csv (scripts/export_judgments.py), kind "pairwise" or "direct"."""
    return pd.read_csv(META / f"judgments_{kind}.csv", dtype={"item": str, "semester": str})


class JudgeTable(Protocol):
    """theta-hat from the recorded judgments: each judge's core run on this task converted by `variant`, then the
    judge with the lowest reference leave-one-out MSE (`judge` fixes one instead). See the module docstring."""

    def __init__(self, kind: str, variant: str, judge: str | None = None):
        assert variant in (PAIRWISE if kind == "pairwise" else DIRECT), (kind, variant)
        self.kind, self.variant, self.judge = kind, variant, judge
        default = {"pairwise": "linear", "direct": "raw"}[kind]
        self.name = kind if variant == default else f"{kind}_{variant}"

    def runs(self, task: Task) -> dict[str, pd.DataFrame]:
        """judge -> its core run on the task (the export already kept only train items and eligible transcripts)."""
        t = judgments(self.kind)
        t = t[(t.domain == task.domain) & (t.semester == task.semester) & t.core]
        return {judge: run for judge, run in t.groupby("judge") if self.judge in (None, judge)}

    def per_judge(self, task: Task) -> pd.DataFrame:
        """One column per judge: theta-hat of the targets, plus the row "loo_mse" (reference leave-one-out MSE)."""
        refs, targets = task.references.theta, list(task.test.models.index)
        out = {}
        for judge, run in self.runs(task).items():
            if self.kind == "pairwise":
                from protocols.pairwise.calibrated import JudgeCalibrated  # its prior needs only a .domain
                prior = (JudgeCalibrated.frontier_prior(SimpleNamespace(domain=task.domain), task.semester)
                         if self.variant == "calibrated" else None)
                theta, loo = pairwise_theta(run, refs, targets, self.variant, prior), pairwise_reference_loo(run, refs, self.variant)
            else:
                theta, loo = direct_theta(run, refs, targets, self.variant), direct_reference_loo(run, refs, self.variant)
            out[judge] = pd.concat([theta, pd.Series({"loo_mse": float(((loo - refs) ** 2).mean())})])
        return pd.DataFrame(out)

    def __call__(self, task: Task) -> pd.Series:
        table = self.per_judge(task)
        if table.empty or table.loc["loo_mse"].isna().all():
            return pd.Series(np.nan, index=task.test.models.index)
        return table[table.loc["loo_mse"].idxmin()].drop("loo_mse").reindex(task.test.models.index)


PROTOCOLS = {p.name: p for p in [JudgeTable("pairwise", v) for v in PAIRWISE] + [JudgeTable("direct", v) for v in DIRECT]}
