"""Compact observed/fallback inputs for the upload-driven cost calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _summary(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> dict:
    result = {"n": int(len(frame)), "source": source}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        name = column.removesuffix("_tokens")
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def _group_summaries(
    frame: pd.DataFrame, group: str, columns: tuple[str, ...], source: str,
) -> dict[str, dict]:
    return {
        str(name): _summary(rows, columns, source)
        for name, rows in frame.groupby(group, sort=True)
    }


def _class_defaults(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict]:
    result = _group_summaries(
        frame.dropna(subset=["reasoning_class"]), "reasoning_class", columns,
        "reasoning_class_fallback",
    )
    result["_all"] = _summary(frame, columns, "overall_fallback")
    return result


def _correctness_fit(catalogue: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    theta = catalogue.set_index("model_id")["theta"]
    frame = candidates.assign(theta=candidates["model"].map(theta)).dropna(subset=["theta", "correct"])
    y = pd.to_numeric(frame["correct"], errors="coerce").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        mean = float(np.clip(y.mean() if len(y) else 0.5, 1e-6, 1 - 1e-6))
        return {
            "intercept": float(np.log(mean / (1 - mean))), "theta": 0.0,
            "covariance": (np.eye(2) * 1e6).tolist(),
        }
    fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2_000).fit(
        frame[["theta"]].to_numpy(dtype=float), y,
    )
    x = np.column_stack((np.ones(len(frame)), frame["theta"].to_numpy(dtype=float)))
    p = fit.predict_proba(frame[["theta"]].to_numpy(dtype=float))[:, 1]
    covariance = np.linalg.pinv(x.T @ ((p * (1 - p))[:, None] * x))
    return {
        "intercept": float(fit.intercept_[0]), "theta": float(fit.coef_[0, 0]),
        "covariance": covariance.tolist(),
    }


def _fit_accuracy(frame: pd.DataFrame) -> dict:
    predictors = ("capability_gap", "length_difference")
    x = frame[list(predictors)].to_numpy(dtype=float)
    y = frame["correct"].to_numpy(dtype=int)
    if not len(frame) or len(np.unique(y)) < 2:
        coefficients = np.zeros(2)
        covariance = np.eye(2) * 1e6
    else:
        fit = LogisticRegression(
            C=np.inf, fit_intercept=False, solver="lbfgs", max_iter=2_000,
        ).fit(x, y)
        coefficients = fit.coef_[0]
        p = fit.predict_proba(x)[:, 1]
        covariance = np.linalg.pinv(x.T @ ((p * (1 - p))[:, None] * x))
    judge_gap = []
    if "judge" in frame:
        for _, rows in frame.groupby("judge"):
            if len(np.unique(rows["correct"])) < 2:
                continue
            fit = LogisticRegression(
                C=np.inf, fit_intercept=False, solver="lbfgs", max_iter=2_000,
            ).fit(rows[list(predictors)].to_numpy(dtype=float), rows["correct"])
            judge_gap.append(float(fit.coef_[0, 0]))
    judge_gap_sd = float(np.std(judge_gap, ddof=1)) if len(judge_gap) > 1 else 0.0
    pair = (
        frame["model_pair"].astype(str)
        if "model_pair" in frame
        else frame[["left_model", "right_model"]].astype(str).agg("::".join, axis=1)
    )
    predicted = 1 / (1 + np.exp(-np.clip(x @ coefficients, -700, 700)))
    pair_residuals = pd.Series(y - predicted, index=frame.index).groupby(pair).mean()
    pair_residual_sd = float(pair_residuals.std(ddof=1)) if len(pair_residuals) > 1 else 0.0
    eta = float(coefficients[0]) * 0.1
    se = 0.1 * float(np.sqrt(max(0, covariance[0, 0] + judge_gap_sd ** 2)))
    probability = lambda value: float(1 / (1 + np.exp(-np.clip(value, -700, 700))))
    return {
        "coefficients": dict(zip(
            predictors, map(float, coefficients), strict=True,
        )),
        "covariance": covariance.tolist(),
        "n": int(len(frame)),
        "judge_gap_sd": judge_gap_sd,
        "pair_residual_sd": pair_residual_sd,
        "accuracy_at_10pct_eci": {
            "mean": probability(eta),
            "low": probability(eta - 1.959963984540054 * se),
            "high": probability(eta + 1.959963984540054 * se),
        },
    }


def _accuracy_glm(judgments: pd.DataFrame, catalogue: pd.DataFrame) -> dict:
    eci = catalogue.set_index("model_id")["eci"]
    frame = judgments.assign(
        left_eci=judgments["left_model"].map(eci),
        right_eci=judgments["right_model"].map(eci),
    ).dropna(subset=["chosen_left", "left_eci", "right_eci", "left_chars", "right_chars"])
    frame = frame.loc[
        frame["left_eci"].ne(frame["right_eci"])
        & frame["left_eci"].gt(0) & frame["right_eci"].gt(0)
    ]
    stronger_left = frame["left_eci"].gt(frame["right_eci"]).to_numpy()
    gap = (
        (frame["left_eci"] - frame["right_eci"]).abs()
        / frame[["left_eci", "right_eci"]].min(axis=1)
    ).to_numpy(dtype=float)
    length = np.where(stronger_left, 1.0, -1.0) * (
        np.log1p(frame["left_chars"].to_numpy(dtype=float))
        - np.log1p(frame["right_chars"].to_numpy(dtype=float))
    )
    correct = np.where(stronger_left, frame["chosen_left"], 1 - frame["chosen_left"]).astype(int)
    frame = frame.assign(
        capability_gap=gap,
        length_difference=length, correct=correct,
    )
    return _fit_accuracy(frame)


def _candidate_accuracy_default(
    catalogue: pd.DataFrame, candidates: pd.DataFrame, fit: dict,
) -> dict[str, float]:
    observed = candidates.groupby("model", sort=False)["correct"].mean()
    values = []
    for row in catalogue.dropna(subset=["theta"]).itertuples():
        value = observed.get(row.model_id, np.nan)
        if not np.isfinite(value):
            value = 1 / (1 + np.exp(-(fit["intercept"] + fit["theta"] * row.theta)))
        values.append(float(value))
    if not values:
        values = [0.5]
    return {
        "mean": float(np.mean(values)),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def build_calculator_payload(
    catalogue: pd.DataFrame, candidates: pd.DataFrame,
    judges: pd.DataFrame, judgments: pd.DataFrame,
) -> dict:
    candidate_frame = candidates.copy()
    candidate_frame["visible_tokens"] = (
        pd.to_numeric(candidate_frame["response_chars"], errors="coerce") / 4
        if "response_chars" in candidate_frame
        else pd.to_numeric(candidate_frame["output_tokens"], errors="coerce")
    )
    candidate_columns = ("input_tokens", "output_tokens", "visible_tokens")
    candidate_models = _group_summaries(
        candidate_frame, "model", candidate_columns, "observed_gpqa",
    )
    accuracy = candidate_frame.groupby("model", sort=True)["correct"].mean()
    for model, value in accuracy.items():
        candidate_models[str(model)]["accuracy"] = float(value)

    judge_frame = judges.copy()
    judge_frame["input_overhead"] = (
        pd.to_numeric(judge_frame["input_tokens"], errors="coerce")
        - pd.to_numeric(judge_frame["transcript_chars"], errors="coerce") / 4
    ).clip(lower=0)
    judge_columns = ("input_overhead", "output_tokens")
    correctness_fit = _correctness_fit(catalogue, candidates)
    catalogue_columns = [
        column for column in (
            "model_id", "display_name", "theta", "eci", "release", "reasoning_class", "price_in", "price_out",
        ) if column in catalogue
    ]
    return {
        "schema_version": 1,
        "catalogue": catalogue[catalogue_columns].astype(object).where(
            pd.notna(catalogue[catalogue_columns]), None,
        ).to_dict("records"),
        "candidate_models": candidate_models,
        "candidate_defaults": _class_defaults(candidate_frame, candidate_columns),
        "judge_models": _group_summaries(
            judge_frame, "model", judge_columns, "observed_gpqa_judging",
        ),
        "judge_defaults": _class_defaults(judge_frame, judge_columns),
        "correctness_fit": correctness_fit,
        "candidate_accuracy_default": _candidate_accuracy_default(
            catalogue, candidates, correctness_fit,
        ),
        "accuracy_glm": _accuracy_glm(judgments, catalogue),
    }
