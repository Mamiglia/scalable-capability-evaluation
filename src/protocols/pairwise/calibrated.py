"""JudgeCalibrated: Bradley-Terry with the judge calibrated on the reference pairs, then trusted only as far as its
held-out error on the references allows.

    P(left picked) = sigmoid(alpha_bench * (theta_left - theta_right) + beta)

Stage 1 (calibrate): on the reference-vs-reference verdicts, theta* fixed on both sides, fit the per-bench discrimination
alpha_b (log-parametrised, shrunk toward their mean) and the position bias beta by maximum likelihood.
Stage 2 (estimate): per target, alpha, beta and the references fixed, the concave 1-D maximum-likelihood theta.
Stage 3 (shrink): the same estimator applied to each reference from the other references (loo) gives the judge's
squared error sigma_m^2 on this task; with 500+ verdicts per target the Fisher variance is negligible next to it (a judge's
taste for a model is shared across all its pairs and never averages out), so sigma_m^2 is the precision the estimate
carries into a N(frontier, old-to-new variance) prior: the mean theta of the TOP_K references, and how far the references
themselves sat from the frontier of the semester before them. Nothing is tuned per judge, semester or domain, and the
references are never refitted: they are the anchor that makes theta-hat comparable with theta*.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from capability.task import Cohort
from configs.logs import logger
from protocols.pairwise.judge import JudgeBT
from reference import tables
from reference.freeze import TOP_K

SHRINK = 2.0  # penalty on (log alpha_b - mean)^2, i.e. a N(., 0.5^2) prior; only bites on benches with few verdicts
MARGIN = 2.0  # theta-hat is searched within [min ref - MARGIN, max ref + MARGIN]: keeps the LOO error finite


def bernoulli_nll(eta: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(y * np.logaddexp(0, -eta) + (1 - y) * np.logaddexp(0, eta)))


def calibrate(verdicts: pd.DataFrame, theta: pd.Series) -> tuple[pd.Series, float]:
    """alpha per bench and beta, from the verdicts whose two models both have a theta."""
    known = verdicts[verdicts.model_left.isin(theta.index) & verdicts.model_right.isin(theta.index)]
    d = theta[known.model_left].to_numpy() - theta[known.model_right].to_numpy()
    y = known.pick_left.to_numpy(float)
    benches, b = np.unique(known.bench, return_inverse=True)
    if known.empty:  # no reference pairs judged: nothing to calibrate on
        return pd.Series(dtype=float, name="alpha"), float("nan")

    def nll(params):
        log_a, beta = params[:-1], params[-1]
        return bernoulli_nll(np.exp(log_a)[b] * d + beta, y) + SHRINK * np.sum((log_a - log_a.mean()) ** 2)

    fit = minimize(nll, np.zeros(len(benches) + 1), method="L-BFGS-B").x
    return pd.Series(np.exp(fit[:-1]), index=benches, name="alpha"), float(fit[-1])


def estimate(verdicts: pd.DataFrame, model: str, theta: pd.Series, alpha: pd.Series, beta: float) -> float:
    """Maximum-likelihood theta of one model from its verdicts against models of known theta (NaN without evidence)."""
    v = verdicts[(verdicts.model_left.eq(model) ^ verdicts.model_right.eq(model)) & verdicts.bench.isin(alpha.index)]
    is_left = v.model_left.eq(model).to_numpy()
    opponent = pd.Index(np.where(is_left, v.model_right, v.model_left))
    v, is_left, opponent = v[opponent.isin(theta.index)], is_left[opponent.isin(theta.index)], opponent[opponent.isin(theta.index)]
    if v.empty:
        return float("nan")
    sign = np.where(is_left, 1.0, -1.0)
    a, theta_o, y = alpha[v.bench].to_numpy(), theta[opponent].to_numpy(), v.pick_left.to_numpy(float)
    lo, hi = float(theta.min()) - MARGIN, float(theta.max()) + MARGIN  # unbeaten (or never-winning) models have an infinite ML
    return float(minimize_scalar(lambda t: bernoulli_nll(a * sign * (t - theta_o) + beta, y), bounds=(lo, hi), method="bounded").x)


def estimate_all(verdicts: pd.DataFrame, refs: pd.Series, models: list[str]) -> pd.Series:
    alpha, beta = calibrate(verdicts, refs)
    return pd.Series({m: estimate(verdicts, m, refs, alpha, beta) for m in models}, dtype=float)


def loo(verdicts: pd.DataFrame, refs: pd.Series) -> pd.Series:
    """theta-hat of each reference from the other references only (calibration excluding it)."""
    among = verdicts[verdicts.model_left.isin(refs.index) & verdicts.model_right.isin(refs.index)]
    return pd.Series({r: estimate_all(among, refs.drop(r), [r])[r] for r in refs.index}, dtype=float)


def shrink(est: pd.Series, sigma2: float, prior: tuple[float, float]) -> pd.Series:
    mean, var = prior
    return ((est / sigma2 + mean / var) / (1 / sigma2 + 1 / var)).fillna(mean)


class JudgeCalibrated(JudgeBT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.name.replace("judge_", "judgecal_", 1)

    def frontier_prior(self, semester: str) -> tuple[float, float]:
        """N(frontier of the references, old-to-new variance: the references' squared gap to the frontier before them)."""
        frozen, _ = tables.load(self.domain)
        theta, rosters = frozen.global_theta, tables.load_semesters()
        semesters = list(rosters)
        i = semesters.index(semester)
        refs = theta.reindex(rosters[semesters[i - 1]]).dropna()
        previous = theta.reindex(rosters[semesters[i - 2]]).dropna() if i >= 2 else pd.Series(dtype=float)
        if previous.empty:  # no models of the semester before the references (first task of a domain)
            return float(refs.nlargest(TOP_K).mean()), float(refs.var())
        return float(refs.nlargest(TOP_K).mean()), float(((refs - previous.nlargest(TOP_K).mean()) ** 2).mean())

    def predict(self, test: Cohort) -> pd.Series:
        verdicts = self.verdicts(self.train, test).merge(test.items[["bench", "item"]])
        refs = self.train.theta
        alpha, beta = calibrate(verdicts, refs)
        if alpha.empty:  # no reference pairs judged: nothing to say
            return pd.Series(np.nan, index=test.models.index)
        sigma2 = float(((loo(verdicts, refs) - refs) ** 2).mean())
        prior = self.frontier_prior(test.semester)
        logger.info("%s %s: alpha %s beta %.2f | reference LOO mse %.2f | prior N(%.2f, %.2f)", self.name, test.semester,
                    alpha.round(2).to_dict(), beta, sigma2, *prior)
        est = pd.Series({m: estimate(verdicts, m, refs, alpha, beta) for m in test.models.index}, dtype=float)
        return shrink(est, sigma2, prior).reindex(test.models.index)
