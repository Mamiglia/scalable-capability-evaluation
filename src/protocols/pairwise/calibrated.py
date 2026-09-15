"""JudgeCalibrated: the judge as a noisy instrument on the theta* scale, calibrated on the reference pairs.

    P(left picked) = sigmoid(alpha_bench * (theta_left - theta_right) + beta)

Stage 1 (calibrate): on the reference-vs-reference verdicts, theta* fixed on both sides, fit the per-bench
discrimination alpha_b (log-parametrised, shrunk toward their mean) and the position bias beta by ML.
Stage 2 (place): per target, all else fixed, maximise the concave 1-D log-likelihood of its verdicts against
the references, plus a N(mean, var) prior from the references' theta* so an unbeaten target stays finite.
The references are never refitted: they are the anchor that makes theta-hat comparable with theta*.
loo() is the free check: each reference placed from its other 7 as if it were a target.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from capability.task import Cohort
from configs.logs import logger
from protocols.pairwise.judge import JudgeBT

SHRINK = 2.0  # penalty on (log alpha_b - mean)^2, i.e. a N(., 0.5^2) prior; only bites on benches with few verdicts


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


def place(verdicts: pd.DataFrame, model: str, theta: pd.Series, alpha: pd.Series, beta: float,
          prior: tuple[float, float]) -> float:
    """MAP theta of one model from its verdicts against models of known theta."""
    mean, var = prior
    v = verdicts[(verdicts.model_left.eq(model) ^ verdicts.model_right.eq(model)) & verdicts.bench.isin(alpha.index)]
    is_left = v.model_left.eq(model).to_numpy()
    opponent = pd.Index(np.where(is_left, v.model_right, v.model_left))
    v, is_left, opponent = v[opponent.isin(theta.index)], is_left[opponent.isin(theta.index)], opponent[opponent.isin(theta.index)]
    if v.empty or alpha.empty:
        return float("nan") if alpha.empty else mean  # uncalibrated: nothing to say; calibrated but unjudged: the prior
    sign = np.where(is_left, 1.0, -1.0)
    a, theta_o, y = alpha[v.bench].to_numpy(), theta[opponent].to_numpy(), v.pick_left.to_numpy(float)

    def nll(t):
        return bernoulli_nll(a * sign * (t - theta_o) + beta, y) + (t - mean) ** 2 / (2 * var)

    return float(minimize_scalar(nll, bracket=(mean - 1, mean + 1)).x)


def place_all(verdicts: pd.DataFrame, refs: pd.Series, targets: list[str]) -> pd.Series:
    alpha, beta = calibrate(verdicts, refs)
    prior = (float(refs.mean()), float(refs.var()))
    return pd.Series({t: place(verdicts, t, refs, alpha, beta, prior) for t in targets}, dtype=float)


def loo(verdicts: pd.DataFrame, refs: pd.Series) -> pd.Series:
    """theta-hat of each reference from the others (calibration and prior excluding it)."""
    return pd.Series({r: place_all(verdicts, refs.drop(r), [r])[r] for r in refs.index}, dtype=float)


class JudgeCalibrated(JudgeBT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.name.replace("judge_", "judgecal_", 1)

    def predict(self, test: Cohort) -> pd.Series:
        verdicts = self.verdicts(self.train, test).merge(test.items[["bench", "item"]])
        refs = self.train.theta
        alpha, beta = calibrate(verdicts, refs)
        held_out = loo(verdicts, refs)
        logger.info("%s %s: alpha %s beta %.2f | reference LOO mse %.2f", self.name, test.semester,
                    alpha.round(2).to_dict(), beta, ((held_out - refs) ** 2).mean())
        return place_all(verdicts, refs, list(test.models.index)).reindex(test.models.index)
