"""Grouped elastic-net screening and hierarchical ridge refitting."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

from .config import AnalysisConfig
from .fit_common import FitResult
from .hierarchy import HierarchyLevels, effective_coefficients, hierarchical_design


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


def _model(config: AnalysisConfig, *, c: float, l1_ratio: float | None) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="saga" if l1_ratio is not None else "lbfgs",
        l1_ratio=0.0 if l1_ratio is None else l1_ratio,
        fit_intercept=False,
        max_iter=config.glm_max_iter,
        random_state=config.random_seed,
    )


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


def select_features(
    train: pd.DataFrame,
    features: list[str],
    config: AnalysisConfig,
    levels: HierarchyLevels,
) -> tuple[list[str], pd.DataFrame]:
    """Tune a grouped elastic-net screen and retain stable base features."""
    splits = _splits(train, config)
    y = train["pick_left"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    best: tuple[float, float] | None = None
    best_loss = np.inf
    for c, l1_ratio in product(config.elastic_net_cs, config.elastic_net_l1_ratios):
        losses = []
        for fit_rows, check_rows in splits:
            fit_x, check_x, _ = _designs(
                train, fit_rows, check_rows, features, levels, config)
            model = _model(config, c=c, l1_ratio=l1_ratio).fit(fit_x, y[fit_rows])
            losses.append(log_loss(
                y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
        mean_loss = float(np.mean(losses))
        rows.append({
            "stage": "elastic_net", "record_type": "tuning", "C": c,
            "l1_ratio": l1_ratio, "mean_log_loss": mean_loss,
        })
        if mean_loss < best_loss:
            best, best_loss = (c, l1_ratio), mean_loss

    assert best is not None
    counts = dict.fromkeys(features, 0)
    for fit_rows, check_rows in splits:
        fit_x, _, column_map = _designs(
            train, fit_rows, check_rows, features, levels, config)
        coefficients = _model(config, c=best[0], l1_ratio=best[1]).fit(
            fit_x, y[fit_rows]).coef_[0]
        active = column_map.loc[
            column_map.block.isin((
                "population_slope", "judge_slope", "benchmark_slope", "cell_slope"))
            & (coefficients != 0),
            "feature",
        ]
        for feature in set(active.dropna()):
            counts[feature] += 1

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
        })
    tuning = pd.DataFrame(rows)
    tuning["selected"] = (
        (tuning.record_type == "tuning")
        & (tuning.C == best[0])
        & (tuning.l1_ratio == best[1])
    )
    return selected, tuning


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
    splits = _splits(train, config)
    y = train["pick_left"].to_numpy(dtype=int)

    ridge_rows = []
    best_c = None
    best_loss = np.inf
    for c in config.ridge_cs:
        losses = []
        for fit_rows, check_rows in splits:
            fit_x, check_x, _ = _designs(
                train, fit_rows, check_rows, selected, levels, config)
            model = _model(config, c=c, l1_ratio=None).fit(fit_x, y[fit_rows])
            losses.append(log_loss(
                y[check_rows], model.predict_proba(check_x)[:, 1], labels=[0, 1]))
        mean_loss = float(np.mean(losses))
        ridge_rows.append({
            "stage": "ridge", "record_type": "tuning", "C": c,
            "mean_log_loss": mean_loss,
        })
        if mean_loss < best_loss:
            best_c, best_loss = c, mean_loss
    assert best_c is not None
    ridge_tuning = pd.DataFrame(ridge_rows)
    ridge_tuning["selected"] = ridge_tuning.C == best_c
    tuning = pd.concat([tuning, ridge_tuning], ignore_index=True, sort=False)

    penalties = _penalties(config)
    train_x, column_map = hierarchical_design(train, selected, levels, penalties)
    test_x, _ = hierarchical_design(test, selected, levels, penalties)
    model = _model(config, c=best_c, l1_ratio=None).fit(train_x, y)
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
            "ridge_C": best_c,
            "ridge_n_iter": int(model.n_iter_[0]),
            "selected_feature_stability": dict(zip(stability.feature, stability.fold_share)),
        },
    )
