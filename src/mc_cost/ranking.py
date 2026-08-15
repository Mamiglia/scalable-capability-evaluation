"""Packaged Bradley-Terry graph and leave-one-model-out helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def fit_bt(pair_wins, l2=1e-3):
    """Bradley-Terry strengths from ``{(a, b): [wins_a, n]}``."""
    models = sorted({model for pair in pair_wins for model in pair})
    index = {model: position for position, model in enumerate(models)}
    first = np.array([index[a] for a, _ in pair_wins])
    second = np.array([index[b] for _, b in pair_wins])
    wins = np.array([value[0] for value in pair_wins.values()], dtype=float)
    totals = np.array([value[1] for value in pair_wins.values()], dtype=float)

    def objective(theta):
        difference = theta[first] - theta[second]
        loss = (
            wins * np.logaddexp(0, -difference)
            + (totals - wins) * np.logaddexp(0, difference)
        ).sum() + l2 * (theta ** 2).sum()
        residual = wins - totals * expit(difference)
        gradient = 2 * l2 * theta
        np.subtract.at(gradient, first, residual)
        np.add.at(gradient, second, residual)
        return loss, gradient

    result = minimize(objective, np.zeros(len(models)), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {result.message}")
    strengths = result.x - result.x.mean()
    return dict(zip(models, strengths))


def components(pairs):
    """Connected components of a comparison graph, largest first."""
    adjacency = defaultdict(set)
    for first, second in pairs:
        adjacency[first].add(second)
        adjacency[second].add(first)
    seen, result = set(), []
    for start in adjacency:
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            model = stack.pop()
            if model in seen:
                continue
            seen.add(model)
            component.add(model)
            stack.extend(adjacency[model] - seen)
        result.append(component)
    return sorted(result, key=len, reverse=True)


def collect_pair_wins(rows, group_cols, ties="half"):
    """Counterbalance item orders into grouped pair-win counts."""
    as_win = {"True": 1.0, "False": 0.0}
    per_item = defaultdict(lambda: [0.0, 0])
    for row in rows:
        if row["major"] == row["minor"]:
            raise ValueError(f"self-comparison row: {row}")
        value = row["picked_major"]
        value = as_win[value] if value in as_win else float(value)
        if value not in (0.0, 0.5, 1.0):
            raise ValueError(f"bad picked_major in row: {row}")
        group = tuple(row[column] for column in group_cols)
        item = per_item[(group, row["major"], row["minor"], row["item_id"])]
        item[0] += value
        item[1] += 1
    groups = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for (group, major, minor, _item), (total, count) in per_item.items():
        value = total / count
        if value == 0.5 and ties == "drop":
            continue
        result = groups[group][(major, minor)]
        result[0] += value
        result[1] += 1
    return groups


def _metric_keys(table: pd.DataFrame) -> list[str]:
    candidates = [
        "analysis", "irt_model", "fit", "bank", "partition", "predictor_bin",
        "target_bin", "direction", "target_quartile", "judge", "component",
    ]
    return [column for column in candidates if column in table]


def _coverage_fields(group: pd.DataFrame) -> dict:
    candidates = [
        "n_response_models", "n_graph_models", "n_component_models",
        "component_models", "n_rounds", "n_calls", "n_unique_items", "n_judges",
        "n_matched_models",
    ]
    return {column: group[column].iloc[0] for column in candidates if column in group}


def _linear_predict(x_train, y_train, x_test) -> float:
    design = np.column_stack([np.ones(len(x_train)), np.asarray(x_train, float)])
    intercept, slope = np.linalg.lstsq(design, np.asarray(y_train, float), rcond=None)[0]
    return round(float(np.clip(intercept + slope * x_test, 0.0, 1.0)), 12)


def lomo_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate Elo and easy accuracy without seeing the held-out hard target."""
    rows = []
    keys = _metric_keys(frame)
    methods = ("elo", "easy_accuracy", "mean", "identity")
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        matched = group.dropna(
            subset=["elo", "easy_accuracy", "hard_accuracy"],
        ).drop_duplicates("model").reset_index(drop=True)
        coverage = _coverage_fields(group)
        count = len(matched)
        exploratory = identity.get("fit") == "aime"
        if count < 4:
            for method in methods:
                rows.append({
                    **identity, **coverage, "model": "", "method": method,
                    "predictor": np.nan, "prediction": np.nan, "observed": np.nan,
                    "n_models": count, "exploratory": exploratory,
                    "reason": "fewer than 4 matched models",
                })
            continue
        for held_index, held_out in matched.iterrows():
            training = matched.drop(index=held_index)
            predictions = {
                "elo": _linear_predict(training["elo"], training["hard_accuracy"], held_out["elo"]),
                "easy_accuracy": _linear_predict(
                    training["easy_accuracy"], training["hard_accuracy"], held_out["easy_accuracy"],
                ),
                "mean": round(float(training["hard_accuracy"].mean()), 12),
                "identity": float(np.clip(held_out["easy_accuracy"], 0.0, 1.0)),
            }
            predictors = {
                "elo": held_out["elo"], "easy_accuracy": held_out["easy_accuracy"],
                "mean": np.nan, "identity": held_out["easy_accuracy"],
            }
            for method, prediction in predictions.items():
                rows.append({
                    **identity, **coverage, "model": held_out["model"], "method": method,
                    "predictor": predictors[method], "prediction": prediction,
                    "observed": held_out["hard_accuracy"], "n_models": count,
                    "exploratory": exploratory, "reason": "",
                })
    return pd.DataFrame(rows)
