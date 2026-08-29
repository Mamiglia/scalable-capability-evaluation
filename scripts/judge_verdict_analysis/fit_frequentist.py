"""Grouped elastic-net screening and hierarchical ridge refitting."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from itertools import product
from typing import TypeVar
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import csr_matrix

from .config import AnalysisConfig
from .fit_common import FitResult
from .hierarchy import HierarchyLevels, effective_coefficients, hierarchical_design


Input = TypeVar("Input")
Output = TypeVar("Output")
_PENALTY_DEFAULT = signature(LogisticRegression).parameters["penalty"].default


def _penalties(config: AnalysisConfig) -> dict[str, float]:
    return {
        name: getattr(config, name)
        for name in (
            "judge_deviation_penalty",
            "benchmark_deviation_penalty",
            "cell_deviation_penalty",
            "item_intercept_penalty",
        )
    }


def _splits(frame: pd.DataFrame, config: AnalysisConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = pd.MultiIndex.from_frame(frame[["bench", "item"]]).factorize()[0]
    if len(np.unique(groups)) < config.inner_folds:
        raise ValueError("inner_folds exceeds the number of bench/item groups")
    return list(GroupKFold(
        config.inner_folds, shuffle=True, random_state=config.random_seed,
    ).split(frame, frame["pick_left"], groups))


def _penalty_options(l1_ratio: float | None, penalty_default: object) -> dict[str, object]:
    """Return compatible elastic-net options for old and new sklearn APIs."""
    if l1_ratio is None:
        return {}
    options: dict[str, object] = {"l1_ratio": l1_ratio}
    if penalty_default != "deprecated":
        options["penalty"] = "elasticnet"
    return options


def _model(config: AnalysisConfig, *, c: float, l1_ratio: float | None) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="saga" if l1_ratio is not None else "lbfgs",
        fit_intercept=False,
        max_iter=config.glm_max_iter,
        random_state=config.random_seed,
        **_penalty_options(l1_ratio, _PENALTY_DEFAULT),
    )


def _fit_model(
    model: LogisticRegression, design: csr_matrix, outcome: np.ndarray
) -> tuple[LogisticRegression, int, bool]:
    """Fit one model and return explicit convergence diagnostics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(design, outcome)
    iterations = int(model.n_iter_[0])
    return model, iterations, iterations < model.max_iter


def _designs(
    frame: pd.DataFrame,
    fit_rows: np.ndarray,
    check_rows: np.ndarray,
    features: list[str],
    levels: HierarchyLevels,
    config: AnalysisConfig,
):
    penalties = _penalties(config)
    fit_x, column_map = hierarchical_design(
        frame.iloc[fit_rows], features, levels, penalties)
    check_x, _ = hierarchical_design(
        frame.iloc[check_rows], features, levels, penalties)
    return fit_x, check_x, column_map


def _parallel_map(
    function: Callable[[Input], Output], values: Iterable[Input], n_jobs: int
) -> list[Output]:
    """Apply independent fit tasks concurrently while preserving input order."""
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(function, values))


def select_features(
    train: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
    levels: HierarchyLevels,
) -> tuple[list[str], pd.DataFrame]:
    """Tune a grouped elastic-net screen and retain stable base features."""
    splits = _splits(train, config)
    y = train["pick_left"].to_numpy(dtype=int)
    designs = [
        (fit_rows, check_rows, *_designs(
            train, fit_rows, check_rows, features, levels, config))
        for fit_rows, check_rows in splits
    ]
    rows: list[dict[str, object]] = []

    def score(parameters: tuple[float, float]) -> tuple[float, float, float, int, bool]:
        c, l1_ratio = parameters
        losses = []
        iterations = []
        for fit_rows, check_rows, fit_x, check_x, _ in designs:
            model, n_iter, _ = _fit_model(
                _model(config, c=c, l1_ratio=l1_ratio), fit_x, y[fit_rows])
            iterations.append(n_iter)
            losses.append(log_loss(
                y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
        return c, l1_ratio, float(np.mean(losses)), max(iterations), all(
            n_iter < config.glm_max_iter for n_iter in iterations)

    scores = _parallel_map(
        score, list(product(config.elastic_net_cs, config.elastic_net_l1_ratios)),
        config.fit_n_jobs,
    )
    best_c, best_l1_ratio, _, _, _ = min(scores, key=lambda result: result[2])
    best = (best_c, best_l1_ratio)
    for c, l1_ratio, mean_loss, max_n_iter, converged in scores:
        rows.append({
            "stage": "elastic_net", "record_type": "tuning", "C": c,
            "l1_ratio": l1_ratio, "mean_log_loss": mean_loss,
            "max_n_iter": max_n_iter, "converged": converged,
        })

    counts = dict.fromkeys(features, 0)

    def active_features(design) -> tuple[set[str], int, bool]:
        fit_rows, _, fit_x, _, column_map = design
        model, n_iter, converged = _fit_model(
            _model(config, c=best[0], l1_ratio=best[1]), fit_x, y[fit_rows])
        coefficients = model.coef_[0]
        active = column_map.loc[
            column_map.block.isin((
                "population_slope", "judge_slope", "benchmark_slope", "cell_slope"))
            & (coefficients != 0),
            "feature",
        ]
        return set(active.dropna()), n_iter, converged

    stability = _parallel_map(active_features, designs, config.fit_n_jobs)
    for active, _, _ in stability:
        for feature in active:
            counts[feature] += 1
    stability_max_n_iter = max(n_iter for _, n_iter, _ in stability)
    stability_converged = all(converged for _, _, converged in stability)

    selected = [
        feature for feature in features
        if counts[feature] / len(splits) >= config.selection_min_fold_share
    ]
    for feature in features:
        share = counts[feature] / len(splits)
        rows.append({
            "stage": "elastic_net", "record_type": "stability", "C": best[0],
            "l1_ratio": best[1], "feature": feature, "fold_share": share,
            "retained": feature in selected,
            "max_n_iter": stability_max_n_iter, "converged": stability_converged,
        })
    tuning = pd.DataFrame(rows)
    tuning["selected"] = (
        (tuning.record_type == "tuning")
        & (tuning.C == best[0])
        & (tuning.l1_ratio == best[1])
    )
    return selected, tuning


def tune_ridge(
    train: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
    levels: HierarchyLevels | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose the hierarchical-ridge penalty without feature selection."""
    levels = levels or HierarchyLevels.from_training(train)
    splits = _splits(train, config)
    y = train["pick_left"].to_numpy(dtype=int)
    designs = [
        (fit_rows, check_rows, *_designs(
            train, fit_rows, check_rows, features, levels, config))
        for fit_rows, check_rows in splits
    ]

    def score(c: float) -> tuple[float, float, int, bool]:
        losses = []
        iterations = []
        for fit_rows, check_rows, fit_x, check_x, _ in designs:
            model, n_iter, _ = _fit_model(
                _model(config, c=c, l1_ratio=None), fit_x, y[fit_rows])
            iterations.append(n_iter)
            losses.append(log_loss(
                y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
        return c, float(np.mean(losses)), max(iterations), all(
            n_iter < config.glm_max_iter for n_iter in iterations)

    scores = _parallel_map(score, list(config.ridge_cs), config.fit_n_jobs)
    best_c, _, _, _ = min(scores, key=lambda result: result[1])
    rows = [
        {
            "stage": "ridge", "record_type": "tuning", "C": c,
            "mean_log_loss": loss, "max_n_iter": max_n_iter, "converged": converged,
        }
        for c, loss, max_n_iter, converged in scores
    ]
    tuning = pd.DataFrame(rows)
    tuning["selected"] = tuning.C == best_c
    return best_c, tuning


def fit_fixed_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
    ridge_c: float,
) -> FitResult:
    """Fit one hierarchical ridge model with fixed predictors and penalty."""
    folds = pd.concat([train["fold"], test["fold"]]).drop_duplicates().tolist()
    if len(folds) != 1:
        raise ValueError("train and test must contain exactly one shared fold")
    levels = HierarchyLevels.from_training(train)
    penalties = _penalties(config)
    train_x, column_map = hierarchical_design(train, features, levels, penalties)
    test_x, _ = hierarchical_design(test, features, levels, penalties)
    model, n_iter, converged = _fit_model(
        _model(config, c=ridge_c, l1_ratio=None), train_x,
        train.pick_left.to_numpy(dtype=int))
    probability = model.predict_proba(test_x)[:, 1]
    observed = test["pick_left"].to_numpy(dtype=int)
    predictions = test.drop(columns=features + ["pick_left"], errors="ignore").copy()
    predictions["observed"] = observed
    predictions["probability"] = probability
    return FitResult(
        backend="frequentist",
        fold=folds[0],
        selected_features=tuple(features),
        predictions=predictions,
        coefficients=effective_coefficients(model.coef_[0], column_map),
        tuning=pd.DataFrame(),
        metrics=pd.DataFrame({
            "metric": ["accuracy", "log_loss"],
            "value": [
                accuracy_score(observed, probability >= 0.5),
                log_loss(observed, probability, labels=[0, 1]),
            ],
        }),
        diagnostics={
            "ridge_C": ridge_c, "ridge_n_iter": n_iter, "ridge_converged": converged,
        },
    )


def fit_frequentist(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
) -> FitResult:
    """Fit one outer-fold elastic-net screen and hierarchical ridge model."""
    folds = pd.concat([train["fold"], test["fold"]]).drop_duplicates().tolist()
    if len(folds) != 1:
        raise ValueError("train and test must contain exactly one shared fold")
    levels = HierarchyLevels.from_training(train)
    selected, tuning = select_features(train, features, config, levels)
    y = train["pick_left"].to_numpy(dtype=int)
    best_c, ridge_tuning = tune_ridge(train, selected, config, levels)
    tuning = pd.concat([tuning, ridge_tuning], ignore_index=True, sort=False)

    penalties = _penalties(config)
    train_x, column_map = hierarchical_design(train, selected, levels, penalties)
    test_x, _ = hierarchical_design(test, selected, levels, penalties)
    model, ridge_n_iter, ridge_converged = _fit_model(
        _model(config, c=best_c, l1_ratio=None), train_x, y)
    probability = model.predict_proba(test_x)[:, 1]
    observed = test["pick_left"].to_numpy(dtype=int)
    predictions = test.drop(columns=features + ["pick_left"], errors="ignore").copy()
    predictions["observed"] = observed
    predictions["probability"] = probability
    metrics = pd.DataFrame({
        "metric": ["accuracy", "log_loss"],
        "value": [
            accuracy_score(observed, probability >= 0.5),
            log_loss(observed, probability, labels=[0, 1]),
        ],
    })
    stability = tuning[tuning.record_type == "stability"]
    screen_choice = tuning[
        (tuning.stage == "elastic_net")
        & (tuning.record_type == "tuning")
        & tuning.selected
    ].iloc[0]
    return FitResult(
        backend="frequentist",
        fold=folds[0],
        selected_features=tuple(selected),
        predictions=predictions,
        coefficients=effective_coefficients(model.coef_[0], column_map),
        tuning=tuning,
        metrics=metrics,
        diagnostics={
            "elastic_net_C": float(screen_choice.C),
            "elastic_net_l1_ratio": float(screen_choice.l1_ratio),
            "elastic_net_max_n_iter": int(screen_choice.max_n_iter),
            "elastic_net_converged": bool(screen_choice.converged),
            "screen_refits_max_n_iter": int(stability.max_n_iter.max()),
            "screen_refits_converged": bool(stability.converged.all()),
            "ridge_C": best_c,
            "ridge_n_iter": ridge_n_iter,
            "ridge_converged": ridge_converged,
            "selected_feature_stability": dict(zip(stability.feature, stability.fold_share)),
        },
    )
