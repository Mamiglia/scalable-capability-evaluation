"""JudgeLinear (formerly JudgeWLS): theta-hat from pair-level win rates, the judge calibrated per bench on the reference pairs.

A judge's verdicts are not the 2PL's item responses: on the reference-vs-reference pairs (theta*
known on both sides) the logit of a pair's win rate follows dtheta* with a slope and a noise
that differ per bench (gemma on science/2024-I: gpqa tracks theta*, arc_challenge is mostly style).
So each bench is treated as its own instrument,

    logit p(t beats r | bench b) = k_b (theta_t - theta_r) + e,   e ~ N(0, tau_b^2),

with k_b, tau_b fitted on the reference pairs (a least-squares line through logit vs dtheta*),
and a target's theta-hat is the posterior mean over its (bench, reference) win rates under the
references' own N(mean, variance) prior: inverse-variance weighted least squares that falls back
to the reference mean when no bench tracks theta* (k_b near 0 would otherwise divide by zero). Win rates are Laplace-smoothed, so a pair the target never lost
stays finite; the same smoothing is in the calibration, so the two are on one footing.
"""

import numpy as np
import pandas as pd

from capability.task import Cohort
from protocols.pairwise.judge import JudgeBT


def pair_logits(verdicts: pd.DataFrame, known: pd.Series) -> pd.DataFrame:
    """One row per (bench, model, opponent with known theta): smoothed logit of the model's win rate and theta_o."""
    side = pd.concat([
        pd.DataFrame({"bench": verdicts.bench, "model": verdicts.model_left, "o": verdicts.model_right, "win": verdicts.pick_left}),
        pd.DataFrame({"bench": verdicts.bench, "model": verdicts.model_right, "o": verdicts.model_left, "win": 1 - verdicts.pick_left}),
    ])
    side = side[side.o.isin(known.index)]
    table = side.groupby(["bench", "model", "o"]).win.agg(["sum", "size"]).reset_index()
    p = (table["sum"] + 1) / (table["size"] + 2)  # Laplace
    return table.assign(logit=np.log(p / (1 - p)), theta_o=known[table.o].to_numpy())


def bench_calibration(logits: pd.DataFrame, refs: pd.Series) -> pd.DataFrame:
    """Per bench, slope k and residual sd tau of logit on dtheta* over the reference pairs (each pair once)."""
    pairs = logits[logits.model.isin(refs.index) & (logits.model < logits.o)]
    dtheta = refs[pairs.model].to_numpy() - pairs.theta_o.to_numpy()
    rows = {}
    for bench, group in pairs.assign(dtheta=dtheta).groupby("bench"):
        if len(group) < 3:  # a line through fewer points has no residual to measure
            continue
        k = np.polyfit(group.dtheta, group.logit, 1)[0]
        if k <= 0:  # the judge does not prefer the more capable reference: not an instrument (a 2PL item needs a > 0)
            continue
        # residual noise, floored at the binomial sampling noise of the smoothed logits (delta method)
        sampling = np.mean(1 / (group["sum"] + 1) + 1 / (group["size"] - group["sum"] + 1))
        rows[bench] = {"k": k, "tau": float(np.sqrt(max(np.var(group.logit - k * group.dtheta), sampling)))}
    return pd.DataFrame(rows, index=["k", "tau"]).T


def weighted_theta(verdicts: pd.DataFrame, refs: pd.Series, targets: list[str]) -> pd.Series:
    """theta-hat per target: WLS over its (bench, reference) logits with the bench calibration from the references."""
    logits = pair_logits(verdicts, refs)
    calibration = bench_calibration(logits, refs)
    out = pd.Series(refs.mean(), index=targets, dtype=float)  # no evidence: the prior mean
    logits = logits[logits.bench.isin(calibration.index)]  # an uncalibrated bench says nothing
    for target, rows in logits[logits.model.isin(targets)].groupby("model"):
        k = calibration.k[rows.bench].to_numpy()
        w = k / calibration.tau[rows.bench].to_numpy() ** 2
        # each row says theta_t = theta_o + logit / k; minimise sum_w (logit - k (theta_t - theta_o))^2
        # plus the prior's (theta_t - mean)^2 / var
        numerator = np.sum(w * (rows.logit.to_numpy() + k * rows.theta_o.to_numpy())) + refs.mean() / refs.var()
        out[target] = numerator / (np.sum(w * k) + 1 / refs.var())
    return out


class JudgeLinear(JudgeBT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.name.replace("judge_", "judgelin_", 1)

    def predict(self, test: Cohort) -> pd.Series:
        verdicts = self.verdicts(self.train, test).merge(test.items[["bench", "item"]])
        return weighted_theta(verdicts, self.train.theta, list(test.models.index))
