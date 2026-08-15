"""Held-out diagnostics and extrapolation gates for the cost simulator."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from mc_cost.model import JUDGMENT_SPECS, JudgmentSpec
from mc_cost.ranking import collect_pair_wins, components, fit_bt, lomo_predictions


_SPEC_ORDER = tuple(JUDGMENT_SPECS)
_TOKEN_OUTCOMES = {
    "candidate": ("input_tokens", "output_tokens", "reasoning_tokens", "response_chars"),
    "judge": ("input_tokens", "output_tokens", "reasoning_tokens"),
}


def _standard_error(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _paired(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "log_loss"}
    if required - set(first) or required - set(second):
        raise ValueError("paired results require fold and log_loss columns")
    if (
        first["fold"].duplicated().any()
        or second["fold"].duplicated().any()
        or set(first["fold"]) != set(second["fold"])
    ):
        raise ValueError("paired results must contain the same unique folds")
    paired = first[["fold", "log_loss"]].merge(
        second[["fold", "log_loss"]], on="fold", suffixes=("_first", "_second"),
        validate="one_to_one", sort=True,
    )
    losses = paired[["log_loss_first", "log_loss_second"]].apply(
        pd.to_numeric, errors="raise",
    ).to_numpy(float)
    if not np.isfinite(losses).all():
        raise ValueError("paired log losses must be finite")
    paired[["log_loss_first", "log_loss_second"]] = losses
    return paired


def choose_judge_gate(conditional: pd.DataFrame, pooled: pd.DataFrame) -> dict[str, Any]:
    """Enable judge covariates only beyond the paired-fold standard error."""
    paired = _paired(conditional, pooled)
    improvement = paired["log_loss_second"] - paired["log_loss_first"]
    mean = float(improvement.mean())
    standard_error = _standard_error(improvement)
    return {
        "enabled": bool(mean > standard_error),
        "mean_improvement": mean,
        "standard_error": standard_error,
        "n_folds": len(paired),
    }


def choose_judgment_spec(folds: pd.DataFrame) -> dict[str, Any]:
    """Apply the paired one-SE rule in baseline-to-release complexity order."""
    required = {"spec", "fold", "log_loss"}
    if missing := required - set(folds):
        raise ValueError(f"judgment folds lack columns: {', '.join(sorted(missing))}")
    analysis = folds.loc[folds["population"].eq("conditional")] if "population" in folds else folds
    if set(analysis["spec"]) != set(_SPEC_ORDER):
        raise ValueError("judgment folds must contain exactly the three sanctioned specifications")
    by_spec = {name: analysis.loc[analysis["spec"].eq(name)] for name in _SPEC_ORDER}
    means = {name: float(pd.to_numeric(rows["log_loss"], errors="raise").mean()) for name, rows in by_spec.items()}
    best = min(_SPEC_ORDER, key=lambda name: (means[name], _SPEC_ORDER.index(name)))
    candidates, comparisons = [], []
    for name in _SPEC_ORDER:
        paired = _paired(by_spec[name], by_spec[best])
        difference = paired["log_loss_first"] - paired["log_loss_second"]
        mean_difference = float(difference.mean())
        standard_error = _standard_error(difference)
        within_one_se = mean_difference <= standard_error
        candidates.append(name) if within_one_se else None
        comparisons.append({
            "spec": name, "mean_log_loss": means[name],
            "paired_difference_from_best": mean_difference,
            "standard_error": standard_error, "within_one_se": bool(within_one_se),
        })
    return {"selected": candidates[0], "best": best, "comparisons": comparisons}


def _probabilities(values: Any, size: int) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if "probability" not in values:
            raise ValueError("judgment fit callable must return a probability column")
        values = values["probability"]
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape == (size,):
        probabilities = probabilities[None, :]
    if (
        probabilities.ndim != 2 or probabilities.shape[1] != size
        or not len(probabilities) or not np.isfinite(probabilities).all()
    ):
        raise ValueError("judgment probabilities must be finite draw-by-held-out-row values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("judgment probabilities must be between zero and one")
    return probabilities


def _binary(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not set(values).issubset({0.0, 1.0}):
        raise ValueError(f"{column} must contain finite 0/1 values")
    return values


def _judgment_fold(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    validation: str,
    held_out: str,
    spec_name: str,
    spec: JudgmentSpec,
    population: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, JudgmentSpec, bool], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train, test = frame.loc[train_mask].copy(), frame.loc[~train_mask].copy()
    draw_probabilities = _probabilities(
        fit_predict(
            train, test.drop(columns="chosen_left"), spec, population == "conditional",
        ),
        len(test),
    )
    observed = _binary(test, "chosen_left")
    probabilities = draw_probabilities.mean(axis=0)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    fold = f"{validation}:{held_out}"
    predictions = []
    for draw, draw_values in enumerate(draw_probabilities):
        for (source_row, held_row), actual, probability in zip(
            test.iterrows(), observed, draw_values,
        ):
            context = {
                column: held_row[column]
                for column in ("length_difference", "difficulty", "correct", "visible_reasoning")
                if column in test
            }
            predictions.append({
                "spec": spec_name, "validation": validation, "population": population,
                "fold": fold, "held_out": held_out, "source_row": source_row,
                "draw": draw, "observed": int(actual),
                "probability": float(probability), **context,
            })
    metrics = {
        "spec": spec_name, "validation": validation, "population": population,
        "fold": fold, "held_out": held_out, "n_rows": len(test),
        "log_loss": float(np.mean(-(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped)))),
        "brier": float(np.mean((observed - probabilities) ** 2)),
    }
    return predictions, metrics


def cross_validate_judgments(
    frame: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, JudgmentSpec, bool], Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the sanctioned specs over leave-one-judge/model-configuration folds.

    ``fit_predict`` receives ``(training_rows, held_out_rows, spec,
    conditional_judges)`` and returns one displayed-left probability per held row.
    """
    required = {"judge", "left_model", "right_model", "chosen_left"}
    if missing := required - set(frame):
        raise ValueError(f"judgments lack columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("judgments must contain at least one row")
    _binary(frame, "chosen_left")
    predictions, folds = [], []
    for spec_name, spec in JUDGMENT_SPECS.items():
        for judge in sorted(frame["judge"].astype(str).unique()):
            train = frame["judge"].astype(str).ne(judge).to_numpy()
            for population in ("conditional", "pooled"):
                rows, metrics = _judgment_fold(
                    frame, train, "judge", judge, spec_name, spec, population, fit_predict,
                )
                predictions.extend(rows)
                folds.append(metrics)
        models = sorted(set(frame["left_model"].astype(str)) | set(frame["right_model"].astype(str)))
        for model in models:
            train = (
                frame["left_model"].astype(str).ne(model)
                & frame["right_model"].astype(str).ne(model)
            ).to_numpy()
            rows, metrics = _judgment_fold(
                frame, train, "candidate", model, spec_name, spec, "conditional", fit_predict,
            )
            predictions.extend(rows)
            folds.append(metrics)
    return pd.DataFrame(predictions), pd.DataFrame(folds)


def _prediction_frame(values: Any, test: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame) or len(values) != len(test):
        raise ValueError("token fit callable must return one prediction row per held-out row")
    required = {f"{outcome}_{suffix}" for outcome in outcomes for suffix in ("mean", "p5", "p95")}
    if missing := required - set(values):
        raise ValueError(f"token predictions lack columns: {', '.join(sorted(missing))}")
    result = values.loc[:, sorted(required)].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("token predictions must be finite")
    for outcome in outcomes:
        if (result[f"{outcome}_p5"] > result[f"{outcome}_p95"]).any():
            raise ValueError(f"{outcome} predictive interval is reversed")
    return result


def _bands(values: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() != len(values) or numeric.nunique() < 2:
        return None
    count = min(4, numeric.nunique())
    return pd.qcut(numeric, count, labels=False, duplicates="drop").map(lambda value: f"band_{int(value)}")


def validate_tokens(
    frame: pd.DataFrame,
    role: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate token/length prediction by exact model-role-class configuration."""
    if role not in _TOKEN_OUTCOMES:
        raise ValueError("role must be candidate or judge")
    required = {"model", "reasoning_class", *_TOKEN_OUTCOMES[role]}
    if missing := required - set(frame):
        raise ValueError(f"token rows lack columns: {', '.join(sorted(missing))}")
    rows = frame.copy()
    if "role" not in rows:
        rows["role"] = role
    if rows.empty or not rows["role"].astype(str).eq(role).all():
        raise ValueError(f"token rows must contain only the {role} role")
    predictions = []
    configurations = rows[["model", "role", "reasoning_class"]].drop_duplicates().sort_values(
        ["model", "role", "reasoning_class"], kind="stable",
    )
    for model, row_role, reasoning_class in configurations.itertuples(index=False, name=None):
        held_mask = (
            rows["model"].eq(model) & rows["role"].eq(row_role)
            & rows["reasoning_class"].eq(reasoning_class)
        )
        train, test = rows.loc[~held_mask].copy(), rows.loc[held_mask].copy()
        predictors = test.drop(columns=list(_TOKEN_OUTCOMES[role]))
        predicted = _prediction_frame(
            fit_predict(train, predictors, role), predictors, _TOKEN_OUTCOMES[role],
        )
        for position, (source_row, record) in enumerate(test.iterrows()):
            context = {
                "role": role, "reasoning_class": reasoning_class, "model": model,
                "configuration": f"{model}::{role}:{reasoning_class}", "source_row": source_row,
            }
            context.update({
                column: record[column]
                for column in ("release_day", "difficulty", "correct", "visible_reasoning")
                if column in test
            })
            for outcome in _TOKEN_OUTCOMES[role]:
                actual_value = float(pd.to_numeric(pd.Series([record[outcome]]), errors="raise").iloc[0])
                if not np.isfinite(actual_value) or actual_value < 0:
                    raise ValueError(f"{outcome} must contain finite non-negative values")
                actual = float(np.log1p(actual_value))
                mean = float(predicted.loc[position, f"{outcome}_mean"])
                lower = float(predicted.loc[position, f"{outcome}_p5"])
                upper = float(predicted.loc[position, f"{outcome}_p95"])
                predictions.append({
                    **context, "outcome": outcome, "actual_log": actual,
                    "predicted_log": mean, "p5_log": lower, "p95_log": upper,
                    "absolute_log_error": abs(mean - actual),
                    "covered_90": bool(lower <= actual <= upper),
                })
    prediction_frame = pd.DataFrame(predictions)
    summary = prediction_frame.groupby(
        ["role", "reasoning_class", "outcome"], as_index=False, sort=True,
    ).agg(
        n=("actual_log", "size"),
        log_mae=("absolute_log_error", "mean"),
        coverage_90=("covered_90", "mean"),
    )
    length_rows = []
    if role == "candidate":
        length = prediction_frame.loc[prediction_frame["outcome"].eq("response_chars")].copy()
        strata = (
            ("release_date_band", "release_day", True),
            ("item_difficulty", "difficulty", True),
            ("correctness", "correct", False),
            ("visible_reasoning", "visible_reasoning", False),
        )
        for name, column, banded in strata:
            if column not in length:
                length_rows.append({"stratum": name, "level": "", "valid": False, "reason": f"missing column: {column}"})
                continue
            levels = _bands(length[column]) if banded else length[column].astype(str)
            if levels is None:
                length_rows.append({"stratum": name, "level": "", "valid": False, "reason": "fewer than 2 distinct values"})
                continue
            for level, group in length.assign(_level=levels).groupby("_level", sort=True, dropna=False):
                length_rows.append({
                    "stratum": name, "level": str(level), "n": len(group),
                    "log_mae": float(group["absolute_log_error"].mean()),
                    "coverage_90": float(group["covered_90"].mean()),
                    "valid": True, "reason": "",
                })
    return prediction_frame, summary, pd.DataFrame(length_rows)


def support_distance(training: Any, query: Any) -> pd.DataFrame:
    """Nearest-neighbour distance with training-only scaling and LOO threshold."""
    training, query = np.asarray(training, dtype=float), np.asarray(query, dtype=float)
    if training.ndim != 2 or query.ndim != 2 or training.shape[1] != query.shape[1]:
        raise ValueError("training and query must be two-dimensional with matching features")
    if len(training) < 2 or not np.isfinite(training).all() or not np.isfinite(query).all():
        raise ValueError("support distance requires finite queries and at least two finite training rows")
    mean, scale = training.mean(axis=0), training.std(axis=0)
    scale[scale == 0] = 1.0
    standardized_training = (training - mean) / scale
    training_distances = cdist(standardized_training, standardized_training)
    np.fill_diagonal(training_distances, np.inf)
    loo = training_distances.min(axis=1)
    threshold = float(np.quantile(loo, 0.95))
    distances = cdist((query - mean) / scale, standardized_training).min(axis=1)
    return pd.DataFrame({
        "distance": distances,
        "threshold": threshold,
        "outside_support": distances > threshold,
    })


def _reference_scores(values: Any) -> dict[str, float]:
    if isinstance(values, pd.DataFrame):
        score_column = "theta" if "theta" in values else "score"
        if {"model", score_column} - set(values):
            raise ValueError("generating theta requires model and theta/score columns")
        values = dict(zip(values["model"].astype(str), values[score_column]))
    elif isinstance(values, pd.Series):
        values = values.to_dict()
    if not isinstance(values, Mapping):
        raise TypeError("generating theta must map model IDs to scores")
    result = {str(model): float(score) for model, score in values.items()}
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError("generating theta scores must be finite and nonempty")
    return result


def fitted_ranking_components(pair_wins: Mapping[tuple[str, str], Sequence[float]]) -> list[dict[str, Any]]:
    """Fit disconnected Bradley-Terry components with stable membership keys."""
    output = []
    graph_components = sorted(
        components(pair_wins), key=lambda members: (-len(members), tuple(sorted(members))),
    )
    for members in graph_components:
        ordered = tuple(sorted(members))
        component_wins = {
            pair: values for pair, values in pair_wins.items()
            if pair[0] in members and pair[1] in members
        }
        output.append({
            "component_models": "|".join(ordered),
            "members": ordered,
            "pair_wins": component_wins,
            "strengths": fit_bt(component_wins),
        })
    return output


def ranking_recovery(
    votes: pd.DataFrame,
    reference: pd.DataFrame,
    generating_theta: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score every comparison component without joining Elo scales."""
    required_votes = {"fit", "major", "minor", "item_id", "picked_major"}
    if missing := required_votes - set(votes):
        raise ValueError(f"votes lack columns: {', '.join(sorted(missing))}")
    if "model" not in reference or not ({"score"} <= set(reference) or {"hard_accuracy"} <= set(reference)):
        raise ValueError("reference requires model and score or hard_accuracy")
    if reference["model"].astype(str).duplicated().any():
        raise ValueError("reference model IDs must be unique")
    ref = reference.copy()
    ref["model"] = ref["model"].astype(str)
    ref = ref.set_index("model")
    direct_scores = (
        _reference_scores(generating_theta)
        if generating_theta is not None
        else ref["score" if "score" in ref else "hard_accuracy"].astype(float).to_dict()
    )
    recovery_type = (
        "generating_theta_consistency" if generating_theta is not None
        else "observed_full_data_recovery"
    )
    grouped = collect_pair_wins(votes.to_dict("records"), ["fit"])
    results, warning_rows = [], []
    for (fit_name,), pair_wins in sorted(grouped.items(), key=lambda entry: str(entry[0])):
        fitted_components = fitted_ranking_components(pair_wins)
        if len(fitted_components) > 1:
            membership = "; ".join(
                f"{fit_name}:{part['component_models']}"
                for part in fitted_components
            )
            warning_rows.append({
                "code": "disconnected_graph", "fit": fit_name,
                "components": membership,
                "message": f"ranking graph has {len(fitted_components)} components: {membership}",
            })
        for part in fitted_components:
            members = set(part["members"])
            component_wins = part["pair_wins"]
            strengths = part["strengths"]
            component_id = f"{fit_name}:{part['component_models']}"
            matched = sorted(set(strengths) & set(direct_scores))
            tau = (
                float(kendalltau(
                    [strengths[model] for model in matched],
                    [direct_scores[model] for model in matched],
                ).statistic)
                if len(matched) >= 2 else np.nan
            )
            analysis_rows = []
            for model in sorted(members & set(ref.index)):
                score = float(ref.at[model, "score"] if "score" in ref else ref.at[model, "hard_accuracy"])
                analysis_rows.append({
                    "irt_model": "fixed_3pl", "fit": fit_name, "bank": "simulated",
                    "judge": "consensus", "component": component_id, "model": model,
                    "elo": strengths[model],
                    "easy_accuracy": float(ref.at[model, "easy_accuracy"]) if "easy_accuracy" in ref else score,
                    "hard_accuracy": float(ref.at[model, "hard_accuracy"]) if "hard_accuracy" in ref else score,
                })
            lomo = (
                lomo_predictions(pd.DataFrame(analysis_rows))
                if analysis_rows
                else pd.DataFrame(columns=["method", "prediction", "observed", "reason"])
            )
            elo_lomo = lomo.loc[lomo["method"].eq("elo")]
            valid_lomo = elo_lomo.dropna(subset=["prediction", "observed"])
            lomo_reason = "" if not valid_lomo.empty else (
                str(elo_lomo["reason"].iloc[0]) if not elo_lomo.empty else "no matched reference models"
            )
            results.append({
                "fit": fit_name, "component": component_id,
                "component_models": part["component_models"],
                "matched_models": "|".join(matched), "n_models": len(members),
                "n_matched_models": len(matched),
                "n_votes": int(sum(value[1] for value in component_wins.values())),
                "kendall_tau": tau, "recovery_type": recovery_type,
                "reference": (
                    "fitted_posterior" if generating_theta is not None else "observed_full_data"
                ),
                "valid": bool(np.isfinite(tau)),
                "reason": "" if np.isfinite(tau) else "fewer than 2 matched reference models",
                "lomo_mae": (
                    float(np.mean(np.abs(valid_lomo["prediction"] - valid_lomo["observed"])))
                    if not valid_lomo.empty else np.nan
                ),
                "lomo_reason": lomo_reason,
            })
        graph_models = {model for pair in pair_wins for model in pair}
        for model in sorted(set(ref.index) - graph_models):
            component_id = f"{fit_name}:isolated:{model}"
            reason = "reference model has no comparison edge"
            results.append({
                "fit": fit_name, "component": component_id,
                "component_models": model, "matched_models": "",
                "n_models": 1, "n_matched_models": 0, "n_votes": 0,
                "kendall_tau": np.nan, "recovery_type": recovery_type,
                "reference": (
                    "fitted_posterior" if generating_theta is not None else "observed_full_data"
                ),
                "valid": False, "reason": reason, "lomo_mae": np.nan,
                "lomo_reason": reason,
            })
            warning_rows.append({
                "code": "isolated_reference_model", "fit": fit_name,
                "components": component_id,
                "message": f"reference model {model} has no comparison edge",
            })
    result_columns = [
        "fit", "component", "component_models", "matched_models", "n_models",
        "n_matched_models", "n_votes", "kendall_tau", "recovery_type", "reference",
        "valid", "reason", "lomo_mae", "lomo_reason",
    ]
    warning_columns = ["code", "fit", "components", "message"]
    return (
        pd.DataFrame(results, columns=result_columns),
        pd.DataFrame(warning_rows, columns=warning_columns),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def decision_payload(
    *, selected_spec: str, judge_gate: Mapping[str, Any], token_coverage: pd.DataFrame,
    predictors: Any, predictor_names: Sequence[str], support_threshold: float,
    data_hashes: Mapping[str, str], validation_timestamp: str,
) -> dict[str, Any]:
    """Build the deterministic, data-bound validation decision payload."""
    if selected_spec not in JUDGMENT_SPECS:
        raise ValueError("selected_spec must be one of the three sanctioned specifications")
    hashes = {str(key): str(value) for key, value in data_hashes.items()}
    if not hashes or any(not key.strip() or not value.strip() for key, value in hashes.items()):
        raise ValueError("data hashes must be a nonempty string mapping")
    try:
        timestamp = datetime.fromisoformat(validation_timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError("validation timestamp must be ISO-8601 with a timezone") from None
    if timestamp.tzinfo is None:
        raise ValueError("validation timestamp must be ISO-8601 with a timezone")
    timestamp = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    matrix = np.asarray(predictors, dtype=float)
    names = [str(name) for name in predictor_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("predictors must align to unique predictor names")
    if not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("predictors must be finite and nonempty")
    threshold = float(support_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("support threshold must be finite and non-negative")
    required_coverage = {"role", "reasoning_class", "outcome", "coverage_90"}
    if missing := required_coverage - set(token_coverage):
        raise ValueError(f"token coverage lacks columns: {', '.join(sorted(missing))}")
    coverage = token_coverage.loc[:, sorted(required_coverage)].sort_values(
        ["role", "reasoning_class", "outcome"], kind="stable",
    ).to_dict("records")
    correlations = np.corrcoef(matrix, rowvar=False) if len(names) > 1 else np.ones((1, 1))
    correlation_rows = {
        row_name: {
            column_name: (float(correlations[row, column]) if np.isfinite(correlations[row, column]) else None)
            for column, column_name in enumerate(names)
        }
        for row, row_name in enumerate(names)
    }
    spec = JUDGMENT_SPECS[selected_spec]
    return _jsonable({
        "schema_version": 1,
        "selected_judgment_spec": selected_spec,
        "judgment_spec": {
            "global_features": list(spec.global_features), "include_length": spec.include_length,
        },
        "judge_covariate_gate": dict(judge_gate),
        "token_coverage_by_role_class": coverage,
        "predictor_diagnostics": {
            "condition_number": float(np.linalg.cond(matrix)),
            "correlations": correlation_rows,
        },
        "support_threshold": threshold,
        "data_hashes": hashes,
        "validation_timestamp": timestamp,
    })


def write_decision(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a byte-stable ``decision.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def require_decision(path: str | Path, data_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Load a decision or refuse absent/stale validation data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"decision.json is required before simulation: {path}")
    payload = json.loads(path.read_text())
    expected = {str(key): str(value) for key, value in data_hashes.items()}
    if payload.get("data_hashes") != expected:
        raise ValueError("decision data hash mismatch; rerun validation")
    return payload


_REPLAY_COLUMNS = [
    "check", "slice", "observed_choice_rate", "predicted_mean", "p5", "p50", "p95",
    "calibration_error", "n_observations", "n_draws", "valid", "passed", "reason",
    "reference",
]


def _failed_replay(check: str, reason: str) -> dict[str, Any]:
    return {
        "check": check, "slice": "", "observed_choice_rate": np.nan,
        "predicted_mean": np.nan, "p5": np.nan, "p50": np.nan, "p95": np.nan,
        "calibration_error": np.nan, "n_observations": 0, "n_draws": 0,
        "valid": False, "passed": False,
        "reason": reason, "reference": "fitted_posterior",
    }


def _replay_groups(frame: pd.DataFrame, check: str, grouping: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    grouped = [((), frame)] if not grouping else frame.groupby(list(grouping), sort=True, dropna=False)
    for values, group in grouped:
        values = values if isinstance(values, tuple) else (values,)
        observed = group.drop_duplicates("row_id")["observed_choice"]
        by_draw = group.groupby("draw", sort=True)["predicted_choice"].mean()
        lower, median, upper = by_draw.quantile([0.05, 0.5, 0.95])
        observed_rate = float(observed.mean())
        passed = bool(lower <= observed_rate <= upper)
        rows.append({
            "check": check,
            "slice": "|".join(f"{name}={value}" for name, value in zip(grouping, values)) or "all",
            "observed_choice_rate": observed_rate, "predicted_mean": float(by_draw.mean()),
            "p5": float(lower), "p50": float(median), "p95": float(upper),
            "calibration_error": np.nan, "n_observations": len(observed), "n_draws": len(by_draw),
            "valid": True, "passed": passed,
            "reason": "" if passed else "observed rate outside 90% replay interval",
            "reference": "fitted_posterior",
        })
    return rows


def _calibration_groups(
    frame: pd.DataFrame, check: str, grouping: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for values, group in frame.groupby(list(grouping), sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        observed = float(group.drop_duplicates("row_id")["observed_choice"].mean())
        predicted = float(group["probability"].mean())
        rows.append({
            "check": check,
            "slice": "|".join(f"{name}={value}" for name, value in zip(grouping, values)),
            "observed_choice_rate": observed, "predicted_mean": predicted,
            "p5": np.nan, "p50": np.nan, "p95": np.nan,
            "calibration_error": observed - predicted,
            "n_observations": group["row_id"].nunique(), "n_draws": group["draw"].nunique(),
            "valid": True, "passed": True, "reason": "",
            "reference": "fitted_posterior",
        })
    return rows


def posterior_predictive_replay(frame: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate and slice replay checks, retaining failed-slice reasons."""
    required = {"row_id", "draw", "observed_choice", "predicted_choice"}
    if missing := required - set(frame):
        raise ValueError(f"replay rows lack columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("replay rows must be nonempty")
    rows = frame.copy()
    _binary(rows, "observed_choice")
    _binary(rows, "predicted_choice")
    if rows.duplicated(["row_id", "draw"]).any():
        raise ValueError("replay rows must be unique by row_id and draw")
    if rows.groupby("row_id")["observed_choice"].nunique().gt(1).any():
        raise ValueError("observed_choice must be constant within row_id")
    if "probability" in rows:
        probability = pd.to_numeric(rows["probability"], errors="raise")
        if not probability.between(0, 1).all():
            raise ValueError("probability must be finite and in [0, 1] for every row_id/draw")
        rows["probability"] = probability
    output = _replay_groups(rows, "aggregate_choice_rate", ())

    def add_raw(check: str, column: str) -> None:
        if column not in rows:
            output.append(_failed_replay(check, f"missing column: {column}"))
        else:
            output.extend(_replay_groups(rows, check, (column,)))

    def add_binned(check: str, column: str) -> bool:
        if column not in rows:
            output.append(_failed_replay(check, f"missing column: {column}"))
            return False
        unique = rows.drop_duplicates("row_id").set_index("row_id")
        bands = _bands(unique[column])
        if bands is None:
            output.append(_failed_replay(check, f"fewer than 2 distinct values: {column}"))
            return False
        rows[f"_{check}"] = rows["row_id"].map(bands)
        output.extend(_replay_groups(rows, check, (f"_{check}",)))
        return True

    add_raw("per_judge", "judge")
    add_raw("position_effect", "m1_is_left")
    for check, column in (
        ("capability_gap", "theta_gap"),
        ("oversight", "oversight"),
        ("item_difficulty", "difficulty"),
        ("release_date_difference", "release_gap_days"),
        ("transcript_length_difference", "length_difference"),
    ):
        add_binned(check, column)

    length_key = "_transcript_length_difference"
    for check, column, binned in (
        ("length_by_item_difficulty", "difficulty", True),
        ("length_by_correctness", "correct", False),
        ("length_by_visible_reasoning", "visible_reasoning", False),
    ):
        if length_key not in rows:
            output.append(_failed_replay(check, "transcript length difference slice unavailable"))
            continue
        if column not in rows:
            output.append(_failed_replay(check, f"missing column: {column}"))
            continue
        if "probability" not in rows:
            output.append(_failed_replay(check, "missing column: probability"))
            continue
        group_column = column
        if binned:
            unique = rows.drop_duplicates("row_id").set_index("row_id")
            bands = _bands(unique[column])
            if bands is None:
                output.append(_failed_replay(check, f"fewer than 2 distinct values: {column}"))
                continue
            group_column = f"_{check}"
            rows[group_column] = rows["row_id"].map(bands)
        output.extend(_calibration_groups(rows, check, (length_key, group_column)))
    return pd.DataFrame(output, columns=_REPLAY_COLUMNS)


def _grid_values(grid: Mapping[str, Sequence[int]]) -> list[tuple[int, int, int, int]]:
    aliases = {
        "J": "judge_counts", "M": "anchor_counts", "N": "comparator_counts",
        "attempted_items": "attempted_item_counts",
    }
    values = []
    for short, configured in aliases.items():
        source = grid.get(short, grid.get(configured))
        if not source or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in source):
            raise ValueError(f"configured grid requires positive integer {configured}")
        values.append(tuple(source))
    return list(product(*values))


def compare_downsamples(
    grid: Mapping[str, Sequence[int]],
    empirical_resample: Callable[[dict[str, int], np.random.Generator], pd.DataFrame],
    posterior_simulate: Callable[[dict[str, int], np.random.Generator], pd.DataFrame],
    *, seed: int = 0,
) -> pd.DataFrame:
    """Evaluate empirical and fitted-posterior recovery on every configured cell."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    random = np.random.default_rng(seed)
    output = []
    for j, m, n, attempted_items in _grid_values(grid):
        cell = {"J": j, "M": m, "N": n, "attempted_items": attempted_items}
        for reference, callback in (
            ("observed_full_data", empirical_resample),
            ("fitted_posterior", posterior_simulate),
        ):
            child = np.random.default_rng(random.integers(0, np.iinfo(np.int64).max))
            try:
                values = callback(cell.copy(), child)
                if not isinstance(values, pd.DataFrame):
                    raise ValueError("downsample callback must return a DataFrame")
                required = {"interval_width", "ranking_recovery"}
                if missing := required - set(values):
                    raise ValueError(f"downsample results lack columns: {', '.join(sorted(missing))}")
                if values.empty:
                    raise ValueError("downsample callback returned no rows")
                for position, row in values.reset_index(drop=True).iterrows():
                    replicate = row.get("replicate", position)
                    for metric in ("interval_width", "ranking_recovery"):
                        value = float(row[metric])
                        output.append({
                            **cell, "reference": reference, "replicate": replicate,
                            "metric": metric, "value": value,
                            "valid": bool(np.isfinite(value)),
                            "reason": "" if np.isfinite(value) else "non-finite metric",
                        })
            except ValueError as error:
                for metric in ("interval_width", "ranking_recovery"):
                    output.append({
                        **cell, "reference": reference, "replicate": pd.NA,
                        "metric": metric, "value": np.nan, "valid": False,
                        "reason": str(error),
                    })
    return pd.DataFrame(output)
