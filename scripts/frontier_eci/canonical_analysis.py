"""Balanced pair ranking and hard-outcome calibration for canonical anchors."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from ranking import rank  # noqa: E402
from ranking.scoring import resample_items  # noqa: E402
from scripts.frontier_eci.canonical_coverage import stable_items  # noqa: E402


@dataclass
class Fit:
    intercept: float
    slope: float
    converged: bool
    gradient_intercept: float
    gradient_slope: float
    anchor_table: pd.DataFrame


@dataclass
class Prediction:
    predicted: float
    intercept: float
    slope: float
    converged: bool
    gradient_intercept: float
    gradient_slope: float
    anchor_table: pd.DataFrame


def cap_pair_items(verdicts: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Keep every row for the same stable item IDs within each unordered pair."""
    columns = list(verdicts.columns)
    table = verdicts.copy()
    pair = np.sort(table[["model_left", "model_right"]].astype(str).to_numpy(), axis=1)
    table["_pair_a"], table["_pair_b"] = pair[:, 0], pair[:, 1]
    table["_item"] = table.item.astype(str)
    groups = [column for column in ("bench", "judge") if column in table] + ["_pair_a", "_pair_b"]
    keep = table.groupby(groups, sort=False)["_item"].transform(
        lambda values: values.isin(stable_items(values.tolist(), limit)))
    sort = groups + ["_item", "model_left", "model_right"]
    sort += [column for column in columns if column not in sort and column != "item"]
    return (table[keep].sort_values(sort, key=lambda values: values.astype(str), kind="stable")
            [columns].reset_index(drop=True))


def _aggregate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if {"successes", "total"}.issubset(outcomes.columns):
        return outcomes.groupby("model", as_index=False)[["successes", "total"]].sum()
    if "correct" not in outcomes:
        raise ValueError("hard outcomes need successes/total or item-level correct rows")
    return (outcomes.groupby("model", as_index=False).correct
            .agg(successes="sum", total="size"))


def fit_binomial_logit(ability: pd.Series, outcomes: pd.DataFrame) -> Fit:
    """Fit an unregularized intercept and slope to grouped binomial outcomes."""
    table = _aggregate_outcomes(outcomes)
    table["ability"] = table.model.map(ability)
    table = table.dropna(subset=["ability", "successes", "total"]).copy()
    if len(table) < 4:
        raise ValueError(f"only {len(table)} calibration anchors")
    if (table.total <= 0).any() or (table.successes < 0).any() or (table.successes > table.total).any():
        raise ValueError("invalid binomial outcomes")

    x = table.ability.to_numpy(float)
    successes = table.successes.to_numpy(float)
    total = table.total.to_numpy(float)
    if np.ptp(x) == 0:
        raise RuntimeError("binomial logit is unidentifiable: anchor abilities are constant")
    positive = x[successes > 0]
    negative = x[successes < total]
    if (not len(positive) or not len(negative)
            or negative.max() <= positive.min() or positive.max() <= negative.min()):
        raise RuntimeError("binomial logit outcomes are separated")

    def loss(beta):
        eta = beta[0] + beta[1] * x
        return np.sum(total * np.logaddexp(0, eta) - successes * eta)

    def gradient(beta):
        residual = total * expit(beta[0] + beta[1] * x) - successes
        return np.array([residual.sum(), residual @ x])

    def hessian(beta):
        fitted = expit(beta[0] + beta[1] * x)
        weights = total * fitted * (1 - fitted)
        design = np.column_stack([np.ones(len(x)), x])
        return design.T @ (weights[:, None] * design)

    result = minimize(loss, np.zeros(2), jac=gradient, hess=hessian,
                      method="Newton-CG", options={"maxiter": 1_000})
    grad = gradient(result.x)
    if (not result.success or not np.isfinite(result.x).all()
            or np.abs(result.x).max() > 50 or np.abs(grad).max() >= 1e-6):
        raise RuntimeError(f"binomial logit did not converge: {result.message}")

    table["observed"] = table.successes / table.total
    table["fitted"] = expit(result.x[0] + result.x[1] * x)
    table["residual"] = table.observed - table.fitted
    return Fit(float(result.x[0]), float(result.x[1]), True,
               float(grad[0]), float(grad[1]), table.reset_index(drop=True))


def predict_from_ability(ability: pd.Series, hard_outcomes: pd.DataFrame,
                         target: str, anchors: list[str]) -> Prediction:
    """Fit only anchor outcomes, then predict the target from its judged ability."""
    if target not in ability:
        raise ValueError(f"target {target!r} has no fitted ability")
    anchor_ids = [anchor for anchor in anchors if anchor != target]
    anchor_outcomes = hard_outcomes[hard_outcomes.model.isin(anchor_ids)].copy()
    fit = fit_binomial_logit(ability.reindex(anchor_ids), anchor_outcomes)
    predicted = float(expit(fit.intercept + fit.slope * ability[target]))
    return Prediction(predicted, fit.intercept, fit.slope, fit.converged,
                      fit.gradient_intercept, fit.gradient_slope, fit.anchor_table)


def _resample_hard_items(outcomes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if "item" not in outcomes:
        raise ValueError("item-level hard outcomes are required for bootstrap")
    keys = [column for column in ("bench", "item") if column in outcomes]
    items = outcomes[keys].drop_duplicates()
    draws = items.iloc[rng.integers(len(items), size=len(items))].copy()
    draws["_draw"] = np.arange(len(draws)).astype(str)
    return (outcomes.merge(draws, on=keys).drop(columns="item")
            .rename(columns={"_draw": "item"}))


def predict_block(verdicts: pd.DataFrame, hard_outcomes: pd.DataFrame, target: str,
                  anchors: list[str], resamples: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Fit one capped block and independently bootstrap judged and hard item IDs."""
    capped = cap_pair_items(verdicts)
    ability = rank(capped).set_index("model").ability
    point = predict_from_ability(ability, hard_outcomes, target, anchors)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        try:
            draw_ability = rank(resample_items(capped, rng)).set_index("model").ability
            draw_outcomes = _resample_hard_items(hard_outcomes, rng)
            draws.append(predict_from_ability(
                draw_ability, draw_outcomes, target, anchors).predicted)
        except (RuntimeError, ValueError):
            continue
    if not draws:
        raise RuntimeError("every two-source item-bootstrap calibration failed")

    low, median, high = np.quantile(draws, [0.025, 0.5, 0.975])
    anchor_ids = point.anchor_table.model.astype(str).tolist()
    summary = {
        "model": target,
        "predicted": point.predicted,
        "predicted_low": float(low),
        "predicted_median": float(median),
        "predicted_high": float(high),
        "n_anchors": len(anchor_ids),
        "n_bootstrap": len(draws),
        "bootstrap_unstable": len(draws) < 0.8 * resamples,
        "anchor_ids": "|".join(anchor_ids),
        "intercept": point.intercept,
        "slope": point.slope,
        "converged": point.converged,
        "gradient_intercept": point.gradient_intercept,
        "gradient_slope": point.gradient_slope,
    }
    return summary, point.anchor_table
