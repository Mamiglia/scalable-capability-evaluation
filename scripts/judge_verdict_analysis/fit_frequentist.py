"""Shared sparse fitting plus historical ridge-sensitivity helpers."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from inspect import signature
from itertools import product
from typing import Literal, TypeVar
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
from .pooling import POOLING_PARAMETERS, estimate_pooling_ratios


Input = TypeVar("Input")
Output = TypeVar("Output")
SparsePenalty = Literal["l1", "ridge"]
_PENALTY_DEPRECATED = signature(LogisticRegression).parameters["penalty"].default == "deprecated"


@dataclass(frozen=True)
class SparseTask:
    outcome: str
    group_columns: tuple[str, ...]
    levels: Callable[[pd.DataFrame], object]
    design: Callable[[pd.DataFrame, list[str], object, AnalysisConfig], tuple[csr_matrix, pd.DataFrame]]
    coefficients: Callable[[np.ndarray, pd.DataFrame], pd.DataFrame]


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


def _judge_design(frame, features, levels, config):
    return hierarchical_design(
        frame, features, levels, _penalties(config), penalty=config.penalty,
    )


JUDGE_TASK = SparseTask(
    outcome="pick_left",
    group_columns=("bench", "item"),
    levels=HierarchyLevels.from_training,
    design=_judge_design,
    coefficients=effective_coefficients,
)


def _splits(
    frame: pd.DataFrame, config: AnalysisConfig, task: SparseTask = JUDGE_TASK,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = frame[task.outcome].to_numpy(dtype=int)
    groups = pd.MultiIndex.from_frame(frame[list(task.group_columns)]).factorize()[0]
    if len(np.unique(groups)) < config.inner_folds:
        raise ValueError("inner_folds exceeds the number of bench/item groups")
    return list(GroupKFold(
        config.inner_folds, shuffle=True, random_state=config.random_seed,
    ).split(frame, y, groups))


def _model(config: AnalysisConfig, *, c: float, l1_ratio: float | None) -> LogisticRegression:
    if l1_ratio is None:
        solver = "lbfgs"
        options = {"l1_ratio": 0.0} if _PENALTY_DEPRECATED else {"penalty": "l2"}
    elif l1_ratio == 1:
        solver = "saga"
        options = {"l1_ratio": 1.0} if _PENALTY_DEPRECATED else {"penalty": "l1"}
    else:  # Historical elastic-net sensitivity helpers only.
        solver = "saga"
        options = ({"l1_ratio": l1_ratio} if _PENALTY_DEPRECATED
                   else {"penalty": "elasticnet", "l1_ratio": l1_ratio})
    return LogisticRegression(
        C=c,
        solver=solver,
        fit_intercept=False,
        max_iter=config.glm_max_iter,
        random_state=config.random_seed,
        **options,
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
    levels: object,
    config: AnalysisConfig,
    task: SparseTask = JUDGE_TASK,
):
    fit_x, column_map = task.design(frame.iloc[fit_rows], features, levels, config)
    check_x, _ = task.design(frame.iloc[check_rows], features, levels, config)
    return fit_x, check_x, column_map


def _parallel_map(
    function: Callable[[Input], Output], values: Iterable[Input], n_jobs: int
) -> list[Output]:
    """Apply independent fit tasks concurrently while preserving input order."""
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(function, values))


def tune_sparse(
    train: pd.DataFrame,
    features: list[str],
    penalty: SparsePenalty,
    config: AnalysisConfig,
    task: SparseTask = JUDGE_TASK,
) -> tuple[dict[str, float | None], pd.DataFrame]:
    """Tune the sparse penalty once on an outer-training fold."""
    if penalty not in {"l1", "ridge"}:
        raise ValueError("penalty must be 'l1' or 'ridge'")
    levels = task.levels(train)
    y = train[task.outcome].to_numpy(dtype=int)
    splits = _splits(train, config, task)
    ratios = (1.0,) if penalty == "l1" else (None,)
    cs = config.elastic_net_cs if penalty == "l1" else config.ridge_cs

    if task is JUDGE_TASK:
        pooling, variance_tuning = estimate_pooling_ratios(
            train, features, penalty, config,
        )
        fit_config = replace(config, penalty=penalty, **pooling)
        designs = [
            (fit_rows, check_rows, *_designs(
                train, fit_rows, check_rows, features, levels, fit_config, task))
            for fit_rows, check_rows in splits
        ]

        def score(c: float) -> tuple[float, float, int, bool]:
            losses, iterations, convergence = [], [], []
            for fit_rows, check_rows, fit_x, check_x, _ in designs:
                model, n_iter, converged = _fit_model(
                    _model(config, c=c, l1_ratio=ratios[0]), fit_x, y[fit_rows])
                losses.append(log_loss(
                    y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
                iterations.append(n_iter)
                convergence.append(converged)
            return c, float(np.mean(losses)), max(iterations), all(convergence)

        scores = _parallel_map(score, cs, config.fit_n_jobs)
        best = min(scores, key=lambda result: (not result[3], result[1], result[0]))
        l1_ratio = ratios[0]
        c_tuning = pd.DataFrame([{
            "stage": penalty, "record_type": "tuning", "C": c,
            "l1_ratio": l1_ratio, "mean_log_loss": loss,
            "max_n_iter": iterations, "converged": converged,
            "selected": c == best[0], **pooling,
        } for c, loss, iterations, converged in scores])
        return {
            "C": float(best[0]), "l1_ratio": l1_ratio, **pooling,
        }, pd.concat([variance_tuning, c_tuning], ignore_index=True, sort=False)

    designs = [
        (fit_rows, check_rows, *_designs(
            train, fit_rows, check_rows, features, levels, config, task))
        for fit_rows, check_rows in splits
    ]
    candidates = list(product(cs, ratios))

    def score(parameters: tuple[float, float | None]) -> tuple[float, float | None, float, int, bool]:
        c, l1_ratio = parameters
        losses, iterations, convergence = [], [], []
        for fit_rows, check_rows, fit_x, check_x, _ in designs:
            model, n_iter, converged = _fit_model(
                _model(config, c=c, l1_ratio=l1_ratio), fit_x, y[fit_rows])
            losses.append(log_loss(
                y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
            iterations.append(n_iter)
            convergence.append(converged)
        return c, l1_ratio, float(np.mean(losses)), max(iterations), all(convergence)

    scores = _parallel_map(score, candidates, config.fit_n_jobs)
    best = min(scores, key=lambda result: (result[2], result[0], result[1]))
    tuning = pd.DataFrame([
        {
            "stage": penalty, "record_type": "tuning", "C": c,
            "l1_ratio": l1_ratio, "mean_log_loss": loss,
            "max_n_iter": max_n_iter, "converged": converged,
            "selected": (c, l1_ratio) == best[:2],
        }
        for c, l1_ratio, loss, max_n_iter, converged in scores
    ])
    return {
        "C": float(best[0]),
        "l1_ratio": None if best[1] is None else float(best[1]),
    }, tuning


def fit_sparse(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    penalty: SparsePenalty,
    frozen_hyperparameters: dict[str, float | None],
    config: AnalysisConfig,
    tuning: pd.DataFrame | None = None,
    task: SparseTask = JUDGE_TASK,
) -> FitResult:
    """Fit one hierarchical sparse model without changing its hyperparameters."""
    if penalty not in {"l1", "ridge"}:
        raise ValueError("penalty must be 'l1' or 'ridge'")
    try:
        c = float(frozen_hyperparameters["C"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen_hyperparameters require numeric C and l1_ratio") from error
    ratio = frozen_hyperparameters.get("l1_ratio")
    l1_ratio = None if ratio is None else float(ratio)
    if not np.isfinite(c) or c <= 0 or (l1_ratio is not None and not 0 < l1_ratio <= 1):
        raise ValueError("frozen sparse hyperparameters are out of range")
    if penalty == "l1" and l1_ratio != 1:
        raise ValueError("pure L1 requires l1_ratio=1")
    if penalty == "ridge" and l1_ratio is not None:
        raise ValueError("ridge requires l1_ratio=None")
    fit_config = replace(config, penalty=penalty)
    if task is JUDGE_TASK:
        try:
            pooling = {
                name: float(frozen_hyperparameters[name])
                for name in POOLING_PARAMETERS
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("frozen_hyperparameters require four pooling ratios") from error
        if any(not np.isfinite(value) or value <= 0 for value in pooling.values()):
            raise ValueError("frozen pooling ratios must be finite and positive")
        fit_config = replace(fit_config, **pooling)
    else:
        pooling = {}
    folds = pd.concat([train.fold, test.fold]).drop_duplicates().tolist()
    if len(folds) != 1:
        raise ValueError("train and test must contain exactly one shared fold")

    levels = task.levels(train)
    train_x, column_map = task.design(train, features, levels, fit_config)
    test_x, _ = task.design(test, features, levels, fit_config)
    model, n_iter, converged = _fit_model(
        _model(config, c=c, l1_ratio=l1_ratio), train_x,
        train[task.outcome].to_numpy(dtype=int),
    )
    probability = model.predict_proba(test_x)[:, 1]
    observed = test[task.outcome].to_numpy(dtype=int)
    predictions = test.drop(columns=features + [task.outcome], errors="ignore").copy()
    predictions["observed"] = observed
    predictions["probability"] = probability
    coefficients = task.coefficients(model.coef_[0], column_map)
    active = set(coefficients.loc[coefficients.effective != 0, "feature"])
    return FitResult(
        backend=f"{config.feature_space}-{penalty}",
        fold=str(folds[0]),
        selected_features=tuple(feature for feature in features if feature in active),
        predictions=predictions,
        coefficients=coefficients,
        tuning=pd.DataFrame() if tuning is None else tuning.copy(),
        metrics=pd.DataFrame({
            "metric": ["accuracy", "log_loss"],
            "value": [
                accuracy_score(observed, probability >= 0.5),
                log_loss(observed, probability, labels=[0, 1]),
            ],
        }),
        diagnostics={
            "penalty": penalty, "C": c, "l1_ratio": l1_ratio,
            **pooling,
            "sklearn_penalty": "l1" if penalty == "l1" else "l2",
            "solver": model.solver,
            "n_iter": n_iter, "converged": converged,
        },
    )


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
