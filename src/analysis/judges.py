"""Judge conversions applied offline to matched verdict and rating tables, and the reference
leave-one-out error that selects a judge without target labels.

Pairwise: "linear" is JudgeLinear's per-bench weighted least squares on pair logits
(protocols.pairwise.linear); "calibrated" is JudgeCalibrated's calibrated Bradley-Terry with
shrinkage (protocols.pairwise.calibrated); "bt" is plain Bradley-Terry with the references
pinned at theta* (protocols.pairwise.bradley_terry.rank), no calibration and no prior. Direct: OLS of theta* on a model's mean rating,
"raw" or "item_z" (each rating z-scored per (bench, item) with the references' mean and sd).
Every statistic comes from the references passed in; targets never enter a fit.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from protocols.pairwise import calibrated
from protocols.pairwise.bradley_terry import rank
from protocols.pairwise.linear import weighted_theta

PAIRWISE = ("linear", "calibrated", "bt")
DIRECT = ("raw", "item_z")


def pairwise_theta(verdicts: pd.DataFrame, refs: pd.Series, models: list[str], conversion: str,
                   prior: tuple[float, float] | None = None) -> pd.Series:
    """theta-hat of `models` from `verdicts` (bench, item, model_left, model_right, pick_left, ...)
    with the references `refs` (theta* indexed by model) fixed; index = models. A target with no
    verdicts against the references gets the prior mean, not NaN; the run's `missing` count says
    how many judgments are absent.

    conversion "linear": weighted_theta. "bt": plain pinned Bradley-Terry over the verdicts that
    involve a reference (`prior` unused). "calibrated": calibrate on the reference pairs, 1-D ML
    per model, shrunk by the judge's leave-one-out error on the references toward `prior`
    (mean, variance); prior None means the references' own mean and variance.
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
    """Each reference's theta-hat from the reference-vs-reference verdicts only, with the
    conversion calibrated on the other references; index = refs.index."""
    among = verdicts[verdicts.model_left.isin(refs.index) & verdicts.model_right.isin(refs.index)]
    out = {}
    for r in refs.index:
        others = refs.drop(r)
        out[r] = pairwise_theta(among, others, [r], conversion, prior=(float(others.mean()), float(others.var())))[r]
    return pd.Series(out, dtype=float).reindex(refs.index)


def valid_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Rows with a numeric score in [1, 10], one per (model, bench, item): repeated judging
    epochs of one transcript are averaged first, so every transcript weighs the same."""
    values = pd.to_numeric(ratings.score, errors="coerce")
    kept = ratings.assign(score=values)[values.between(1, 10)]
    return kept.groupby(["model", "bench", "item"], as_index=False).score.mean()


def normalise(ratings: pd.DataFrame, refs: pd.Series, normalisation: str) -> pd.DataFrame:
    """Ratings (model, bench, item, score) after the normalisation: "raw" unchanged; "item_z"
    replaces each score by (score - mean) / sd with mean and sd over the references' scores
    of that (bench, item); an item with no spread among the references (sd 0 or a single
    reference) gives z = 0, an item no reference rated is dropped."""
    if normalisation == "raw":
        return ratings
    assert normalisation == "item_z", normalisation
    stats = ratings[ratings.model.isin(refs.index)].groupby(["bench", "item"]).score.agg(["mean", "std"])
    merged = ratings.merge(stats, left_on=["bench", "item"], right_index=True)
    z = ((merged.score - merged["mean"]) / merged["std"]).where(merged["std"] > 0, 0.0).fillna(0.0)
    return merged.assign(score=z)[["model", "bench", "item", "score"]]


def direct_theta(ratings: pd.DataFrame, refs: pd.Series, models: list[str], normalisation: str) -> pd.Series:
    """theta-hat of `models` from `ratings` (bench, model, item, score): OLS of the references'
    theta* on their mean (normalised) rating, applied to each model's mean; index = models,
    NaN for a model without valid ratings or when fewer than two references have both."""
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
    """Each reference's theta-hat from a calibration (and normalisation) fit on the other
    references only; index = refs.index."""
    among = ratings[ratings.model.isin(refs.index)]
    return pd.Series({r: direct_theta(among, refs.drop(r), [r], normalisation)[r] for r in refs.index}, dtype=float)
