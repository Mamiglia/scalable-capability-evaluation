"""Posterior-predictive judgment, usage, and cost simulation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kendalltau

from mc_cost.data import RELEASE_EPOCH, normalize_family
from mc_cost.model import FEATURE_COLUMNS, JudgmentData, PosteriorArtifact, left_logits
from mc_cost.tokens import draw_token_usage, draw_transcript_lengths
from mc_cost.validation import build_support_index, fitted_ranking_components, require_decision


TRIAL_METRICS = (
    "usable_items", "judgment_calls", "candidate_calls", "choice_accuracy", "m1_wins",
    "m1_theta", "m2_theta",
    "candidate_input_tokens", "candidate_output_tokens",
    "candidate_ordinary_output_tokens", "candidate_reasoning_tokens",
    "judge_input_tokens", "judge_output_tokens",
    "judge_ordinary_output_tokens", "judge_reasoning_tokens", "cost",
)
_TRIAL_COLUMNS = [
    "trial", "configuration", "mode", "comparison_id", "judge", "m1", "m2",
    "model", "reasoning_class", "role", "metric", "value", "warning_state", "reason",
]
_WARNING_COLUMNS = ["code", "trial", "row", "source_row", "model", "message"]
_DASHBOARD_KEYS = (
    "schema_version", "generated_at", "source_hashes", "quantiles", "catalogue",
    "automatic", "manual", "scaling", "ranking_recovery", "warnings",
)
_DASHBOARD_ARRAYS = (
    "catalogue", "automatic", "manual", "scaling", "ranking_recovery", "warnings",
)
_MANUAL_IDENTITY = (
    "judge", "m1", "m2", "judge_reasoning_class", "m1_reasoning_class", "m2_reasoning_class",
)
_WARNING_STATES = {"ok", "partial", "unavailable", "missing_price"}
_UNAVAILABLE_STATES = {"unavailable", "missing_price"}


@dataclass(frozen=True)
class SimulationConfig:
    trials: int
    attempted_items: int
    presentations: int = 2
    reuse_responses: bool = True
    mode: str = "automatic"
    accuracy_target: str = "higher_theta"
    configuration: str = "default"

    def __post_init__(self) -> None:
        for name in ("trials", "attempted_items", "presentations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.reuse_responses) is not bool:
            raise ValueError("reuse_responses must be a bool")
        if self.mode not in {"automatic", "manual"}:
            raise ValueError("mode must be automatic or manual")
        if self.accuracy_target != "higher_theta":
            raise ValueError("accuracy_target must be higher_theta")
        if not isinstance(self.configuration, str) or not self.configuration:
            raise ValueError("configuration must be a nonempty string")


@dataclass(frozen=True)
class SimulationResult:
    trials: pd.DataFrame
    summary: pd.DataFrame
    warnings: pd.DataFrame


def _dashboard_sort(rows) -> list[dict[str, Any]]:
    records = [dict(row) for row in rows]
    return sorted(records, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"dashboard {path} must contain only finite numbers or null")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"dashboard {path} object keys must be strings")
        for key, item in value.items():
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"dashboard {path} contains a non-JSON value")


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))


def validate_dashboard_payload(payload: Mapping[str, Any]) -> None:
    """Validate the complete static-dashboard boundary before publication."""
    if not isinstance(payload, Mapping) or set(payload) != set(_DASHBOARD_KEYS):
        raise ValueError(f"dashboard schema fields must be exactly: {', '.join(_DASHBOARD_KEYS)}")
    if payload["schema_version"] != 1:
        raise ValueError("dashboard schema_version must be 1")
    try:
        generated_at = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("dashboard generated_at must be an ISO-8601 UTC timestamp") from None
    if generated_at.tzinfo is None or generated_at.utcoffset() != timezone.utc.utcoffset(generated_at):
        raise ValueError("dashboard generated_at must be an ISO-8601 UTC timestamp")
    hashes = payload["source_hashes"]
    if not isinstance(hashes, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in hashes.items()
    ):
        raise ValueError("dashboard source_hashes must map names to nonempty hashes")
    if payload["quantiles"] != [0.05, 0.5, 0.95]:
        raise ValueError("dashboard quantiles must be [0.05, 0.5, 0.95]")

    for name in _DASHBOARD_ARRAYS:
        rows = payload[name]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"dashboard {name} must be a list of objects")
        if rows != _dashboard_sort(rows):
            raise ValueError(f"dashboard {name} rows must be sorted")
    for row in payload["manual"]:
        if any(not isinstance(row.get(key), str) or not row[key] for key in _MANUAL_IDENTITY):
            raise ValueError("dashboard manual rows require an exact six-field scenario identity")
        if any(row[key] not in {"reasoning", "non_reasoning"} for key in _MANUAL_IDENTITY[3:]):
            raise ValueError("dashboard manual reasoning classes must be reasoning or non_reasoning")
    for name in ("automatic", "manual", "scaling", "ranking_recovery"):
        for row in payload[name]:
            if "warning_state" in row and row["warning_state"] not in _WARNING_STATES:
                raise ValueError(
                    "dashboard warning_state must be ok, partial, unavailable, or missing_price"
                )
            if row.get("exact") is True:
                if not _finite_number(row.get("value")):
                    raise ValueError(f"dashboard {name} exact rows require a declared exact numeric value")
                continue
            if not all(key in row for key in ("p5", "p50", "p95")):
                raise ValueError(f"dashboard {name} quantile rows require p5, p50, and p95")
            quantiles = [row[key] for key in ("p5", "p50", "p95")]
            if all(value is None for value in quantiles):
                warning_state = row.get("warning_state")
                reason = row.get("reason")
                has_reason = isinstance(reason, str) and bool(reason.strip())
                warned = warning_state in _UNAVAILABLE_STATES and has_reason
                invalid = (
                    name == "ranking_recovery" and row.get("valid") is False and has_reason
                )
                if not warned and not invalid:
                    raise ValueError(
                        "dashboard all-null quantiles require a recognized unavailable state and reason"
                    )
                continue
            if not all(_finite_number(value) for value in quantiles):
                raise ValueError(f"dashboard {name} quantiles must be finite numeric values")
            if not quantiles[0] <= quantiles[1] <= quantiles[2]:
                raise ValueError(f"dashboard {name} quantiles must be ordered p5 <= p50 <= p95")
    _validate_json_value(dict(payload), "payload")


def dashboard_payload(
    *, generated_at: str, source_hashes: Mapping[str, str], catalogue,
    automatic, manual, ranking_recovery, warnings,
) -> dict[str, Any]:
    """Build one canonical, formula-free payload for the static viewer."""
    automatic_rows = _dashboard_sort(automatic)
    scaling_rows = _dashboard_sort(
        row for row in automatic_rows
        if (
            row.get("metric") in {"choice_accuracy", "judgment_calls", "usable_items"}
            and row.get("comparison_id") is None
        ) or (
            row.get("metric") == "cost" and row.get("model") == "__total__"
            and row.get("role") == "all"
        )
    )
    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_hashes": dict(sorted(source_hashes.items())),
        "quantiles": [0.05, 0.5, 0.95],
        "catalogue": _dashboard_sort(catalogue),
        "automatic": automatic_rows,
        "manual": _dashboard_sort(manual),
        "scaling": scaling_rows,
        "ranking_recovery": _dashboard_sort(ranking_recovery),
        "warnings": _dashboard_sort(warnings),
    }
    validate_dashboard_payload(payload)
    return payload


def dashboard_ranking_recovery(
    result: SimulationResult, catalogue: pd.DataFrame, context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize call-weighted, trial-local Bradley-Terry component recovery."""
    presentations = context.get("presentations")
    if not isinstance(presentations, int) or isinstance(presentations, bool) or presentations < 1:
        raise ValueError("ranking recovery context presentations must be a positive integer")
    required = {"trial", "comparison_id", "m1", "m2", "metric", "value"}
    if missing := required - set(result.trials):
        raise ValueError(f"simulation trials lack ranking columns: {', '.join(sorted(missing))}")
    if "model_id" not in catalogue:
        raise ValueError("catalogue requires model_id for ranking recovery")
    comparison = result.trials.loc[
        result.trials["comparison_id"].notna()
        & result.trials["metric"].isin(["judgment_calls", "m1_wins", "m1_theta", "m2_theta"]),
        ["trial", "comparison_id", "m1", "m2", "metric", "value"],
    ]
    raw = []
    disconnected_trials = 0
    incomplete_reasons = set()
    for trial, trial_rows in comparison.groupby("trial", sort=True):
        pair_wins = defaultdict(lambda: [0.0, 0])
        theta: dict[str, float] = {}
        for _, rows in trial_rows.groupby(["comparison_id", "m1", "m2"], sort=True, dropna=False):
            values = rows.set_index("metric")["value"].to_dict()
            m1, m2 = str(rows["m1"].iloc[0]), str(rows["m2"].iloc[0])
            if not {"judgment_calls", "m1_wins", "m1_theta", "m2_theta"} <= set(values):
                incomplete_reasons.add("trial comparison lacks generating theta or vote totals")
                continue
            numeric = [values[key] for key in ("judgment_calls", "m1_wins", "m1_theta", "m2_theta")]
            if not all(_finite_number(value) for value in numeric):
                incomplete_reasons.add("trial comparison has non-finite generating theta or vote totals")
                continue
            call_count = float(values["judgment_calls"]) / presentations
            m1_wins = float(values["m1_wins"]) / presentations
            if call_count <= 0:
                incomplete_reasons.add("no usable judgment calls")
                continue
            if m1 == m2:
                incomplete_reasons.add("self comparison has no ranking edge")
                continue
            if not 0 <= m1_wins <= call_count:
                raise ValueError("simulation m1_wins must be between zero and judgment_calls")
            for model, value in ((m1, float(values["m1_theta"])), (m2, float(values["m2_theta"]))):
                if model in theta and not np.isclose(theta[model], value):
                    raise ValueError("simulation trial has inconsistent generating theta for a model")
                theta[model] = value
            if m1 < m2:
                pair, first_wins = (m1, m2), m1_wins
            else:
                pair, first_wins = (m2, m1), call_count - m1_wins
            pair_wins[pair][0] += first_wins
            pair_wins[pair][1] += call_count
        fitted_components = fitted_ranking_components(pair_wins)
        disconnected_trials += int(len(fitted_components) > 1)
        trial_components = []
        trial_reasons = set()
        for part in fitted_components:
            strengths = part["strengths"]
            matched = sorted(set(strengths) & set(theta))
            tau = (
                float(kendalltau(
                    [strengths[model] for model in matched],
                    [theta[model] for model in matched],
                ).statistic)
                if len(matched) >= 2 else np.nan
            )
            if np.isfinite(tau):
                weight = sum(float(values[1]) for values in part["pair_wins"].values())
                trial_components.append((tau, weight))
            else:
                trial_reasons.add("fewer than 2 matched generating-theta models")
        valid = [(tau, weight) for tau, weight in trial_components if weight > 0]
        raw.append({
            "trial": int(trial),
            "value": float(np.average(
                [tau for tau, _ in valid], weights=[weight for _, weight in valid],
            )) if valid else np.nan,
            "valid": bool(valid), "reason": "; ".join(sorted(trial_reasons)),
        })
    if not raw:
        reason = "; ".join(sorted(incomplete_reasons)) or "no per-comparison judgment calls"
        return ([{
            **dict(context), "metric": "kendall_tau",
            "component_models": "trial-local components",
            "p5": None, "p50": None, "p95": None,
            "n_trials": 0, "valid": False, "reason": reason,
            "partial_reasons": sorted(incomplete_reasons) or [reason],
            "reference": "trial_capability_draw", "recovery_type": "generating_theta_consistency",
        }], [
            {**dict(context), "code": "ranking_recovery_partial", "message": item}
            for item in (sorted(incomplete_reasons) or [reason])
        ])

    frame = pd.DataFrame(raw)
    finite = frame.loc[frame["valid"], "value"]
    reasons = sorted(set(frame.loc[frame["reason"].ne(""), "reason"]) | incomplete_reasons)
    recovery = [{
        **dict(context), "metric": "kendall_tau", "component_models": "trial-local components",
        "p5": float(finite.quantile(.05)) if len(finite) else None,
        "p50": float(finite.quantile(.5)) if len(finite) else None,
        "p95": float(finite.quantile(.95)) if len(finite) else None,
        "n_trials": int(frame["trial"].nunique()), "n_finite_trials": int(frame["valid"].sum()),
        "valid": bool(len(finite)), "reason": "; ".join(reasons), "partial_reasons": reasons,
        "reference": "trial_capability_draw", "recovery_type": "generating_theta_consistency",
    }]
    warnings = []
    if disconnected_trials:
        warnings.append({
            **dict(context), "code": "disconnected_graph",
            "message": f"ranking graph was disconnected in {disconnected_trials} trials; components were fit separately",
        })
    if incomplete_reasons:
        warnings.extend({
            **dict(context), "code": "ranking_recovery_partial", "message": reason,
        } for reason in sorted(incomplete_reasons))
    return recovery, warnings


class LatentCache(dict):
    """One cache shared by every latent unit in a simulation run."""

    def __init__(self, rng: np.random.Generator):
        super().__init__()
        self.rng = rng

    def normal(self, key: tuple[Any, ...], mean: float = 0.0, scale: float = 1.0) -> float:
        try:
            hash(key)
        except TypeError:
            raise ValueError("latent cache keys must be hashable") from None
        if not np.isfinite([mean, scale]).all() or scale < 0:
            raise ValueError("latent normal parameters must be finite with non-negative scale")
        if key not in self:
            self[key] = float(self.rng.normal(mean, scale))
        return float(self[key])


def cost_rows(usage: pd.DataFrame, catalogue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach exact per-million-token prices without rebilling reasoning tokens."""
    required_usage = {"model", "input_tokens", "output_tokens"}
    required_prices = {"model_id", "price_in", "price_out"}
    if missing := required_usage - set(usage):
        raise ValueError(f"usage lacks required columns: {', '.join(sorted(missing))}")
    if missing := required_prices - set(catalogue):
        raise ValueError(f"catalogue lacks required price columns: {', '.join(sorted(missing))}")
    if catalogue["model_id"].duplicated().any():
        raise ValueError("duplicate model_id in catalogue")

    result = usage.copy()
    for column in ("input_tokens", "output_tokens", "reasoning_tokens", "calls"):
        if column not in result:
            result[column] = 0 if column != "calls" else 1
        result[column] = pd.to_numeric(result[column], errors="raise")
        if result[column].isna().any() or (result[column] < 0).any():
            raise ValueError(f"{column} must contain non-negative numbers")
    if "ordinary_output_tokens" not in result:
        result["ordinary_output_tokens"] = result["output_tokens"] - result["reasoning_tokens"]
    result["ordinary_output_tokens"] = pd.to_numeric(result["ordinary_output_tokens"], errors="raise")
    if result["ordinary_output_tokens"].isna().any() or (result["ordinary_output_tokens"] < 0).any():
        raise ValueError("ordinary_output_tokens must contain non-negative numbers")
    if not np.allclose(
        result["ordinary_output_tokens"] + result["reasoning_tokens"],
        result["output_tokens"],
    ):
        raise ValueError("ordinary_output_tokens plus reasoning_tokens must equal output_tokens")

    prices = catalogue[["model_id", "price_in", "price_out"]].rename(columns={"model_id": "model"})
    result = result.merge(prices, on="model", how="left", validate="many_to_one", sort=False)
    for column in ("price_in", "price_out"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    missing_price = (
        (result["price_in"].isna() & result["input_tokens"].gt(0))
        | (result["price_out"].isna() & result["output_tokens"].gt(0))
    )
    result["cost"] = (
        result["input_tokens"] * result["price_in"].fillna(0)
        + result["output_tokens"] * result["price_out"].fillna(0)
    ) / 1_000_000
    result.loc[missing_price, "cost"] = np.nan
    models = result.loc[missing_price, "model"].astype(str).drop_duplicates().tolist()
    warnings = pd.DataFrame([
        {"code": "missing_price", "model": model, "message": f"missing input or output price for {model}"}
        for model in models
    ], columns=["code", "model", "message"])
    return result, warnings


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Return empirical 5th, median, and 95th percentiles for each tidy metric."""
    required = {"trial", "metric", "value"}
    if missing := required - set(trials):
        raise ValueError(f"trials lack required columns: {', '.join(sorted(missing))}")
    values = trials.copy()
    if "warning_state" not in values:
        values["warning_state"] = "ok"
    if "reason" not in values:
        values["reason"] = ""
    groups = [
        column for column in values
        if column not in {"trial", "value", "warning_state", "reason"}
    ]
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    rows = []
    for labels, group in values.groupby(groups, dropna=False, sort=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        finite = group.loc[np.isfinite(group["value"])]
        n_trials = int(group["trial"].nunique())
        n_finite = int(finite["trial"].nunique())
        states = sorted(set(group.loc[group["warning_state"].ne("ok"), "warning_state"].astype(str)))
        reasons = sorted(set(
            reason.strip() for reason in group["reason"]
            if isinstance(reason, str) and reason.strip()
        ))
        if n_finite == n_trials and not states:
            state, reason = "ok", ""
        elif n_finite:
            state, reason = "partial", "; ".join(reasons) or "some trials unavailable"
        else:
            state = states[0] if len(states) == 1 and states[0] in _UNAVAILABLE_STATES else "unavailable"
            reason = "; ".join(reasons) or "no finite trials"
        rows.append({
            **dict(zip(groups, labels)),
            "p5": finite["value"].quantile(.05) if len(finite) else np.nan,
            "p50": finite["value"].quantile(.5) if len(finite) else np.nan,
            "p95": finite["value"].quantile(.95) if len(finite) else np.nan,
            "n_trials": n_trials, "n_finite_trials": n_finite,
            "warning_state": state, "reason": reason,
        })
    return pd.DataFrame(rows)


def counterbalance(
    rows: pd.DataFrame, presentations: int, rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Repeat each usable cell with alternating M1-left/M1-right presentation."""
    if not isinstance(presentations, int) or isinstance(presentations, bool) or presentations < 1:
        raise ValueError("presentations must be a positive integer")
    if missing := {"m1", "m2"} - set(rows):
        raise ValueError(f"rows lack required columns: {', '.join(sorted(missing))}")
    if rows.empty:
        return rows.assign(
            presentation=pd.Series(dtype=int), left_model=pd.Series(dtype=str),
            right_model=pd.Series(dtype=str), m1_is_left=pd.Series(dtype=bool),
        )
    result = rows.loc[rows.index.repeat(presentations)].reset_index(drop=True)
    result["presentation"] = np.tile(np.arange(presentations), len(rows))
    starts_left = (
        np.ones(len(rows), dtype=bool) if rng is None
        else rng.integers(0, 2, len(rows)).astype(bool)
    )
    result["m1_is_left"] = result["presentation"].mod(2).eq(0).eq(
        np.repeat(starts_left, presentations),
    )
    result["left_model"] = np.where(result["m1_is_left"], result["m1"], result["m2"])
    result["right_model"] = np.where(result["m1_is_left"], result["m2"], result["m1"])
    return result


def _artifact_parts(artifact: PosteriorArtifact | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(artifact, PosteriorArtifact):
        return artifact.posterior, artifact.metadata
    if not isinstance(artifact, Mapping) or "posterior" not in artifact:
        raise TypeError("artifact must contain posterior arrays")
    return dict(artifact["posterior"]), {key: value for key, value in artifact.items() if key != "posterior"}


def _posterior_count(artifact: PosteriorArtifact | Mapping[str, Any], label: str) -> int:
    posterior, _ = _artifact_parts(artifact)
    counts = [np.asarray(value).shape[0] for value in posterior.values() if np.asarray(value).ndim]
    if not counts or min(counts) < 1:
        raise ValueError(f"{label} artifact has no posterior draws")
    if len(set(counts)) != 1:
        raise ValueError(f"{label} artifact posterior arrays must have aligned draw counts")
    return counts[0]


def _token_prediction_rows(
    artifact: PosteriorArtifact | Mapping[str, Any], rows: pd.DataFrame,
) -> pd.DataFrame:
    """Use fitted token effects only for an exact role/class configuration."""
    _, metadata = _artifact_parts(artifact)
    role = metadata.get("role")
    known = dict(zip(metadata.get("models", ()), metadata.get("model_reasoning_classes", ())))
    result = rows.copy()
    result["model"] = [
        model if known.get(model) == reasoning_class else f"{model}::{role}:{reasoning_class}"
        for model, reasoning_class in zip(result["model"].astype(str), result["reasoning_class"])
    ]
    return result


def _irt_parts(irt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if all(name in irt for name in ("a", "b", "floor", "items")):
        return {name: np.asarray(irt[name]) for name in ("a", "b", "floor", "items")}
    prefix = "fixed_3pl__gpqa__"
    parts = {name: np.asarray(irt[prefix + name]) for name in ("a", "b", "floor", "items") if prefix + name in irt}
    if len(parts) != 4:
        raise ValueError("irt lacks fixed_3pl GPQA a/b/floor/item arrays")
    return parts


def _item_lookup(labels: Sequence[Any]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, label in enumerate(labels):
        full = str(label)
        for key in {full, full.split(":", 1)[-1]}:
            if key in lookup and lookup[key] != index:
                raise ValueError(f"ambiguous IRT item label: {key}")
            lookup[key] = index
    return lookup


def _canonical_item(value: Any) -> str:
    return str(value).split(":", 1)[-1]


def _draw_value(values: Any, posterior_index: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not array.ndim or posterior_index >= array.shape[0]:
        raise IndexError(f"posterior_index {posterior_index} is out of range for {name}")
    return np.asarray(array[posterior_index])


def _price_metadata(catalogue: pd.DataFrame) -> pd.DataFrame:
    required = {"model_id", "release", "family", "reasoning_class"}
    if missing := required - set(catalogue):
        raise ValueError(f"catalogue lacks required columns: {', '.join(sorted(missing))}")
    if catalogue["model_id"].isna().any() or catalogue["model_id"].astype(str).duplicated().any():
        raise ValueError("catalogue model_id values must be unique and non-null")
    result = catalogue.copy()
    result["model_id"] = result["model_id"].astype(str)
    result["release_day"] = (pd.to_datetime(result["release"], errors="coerce") - RELEASE_EPOCH).dt.days
    result["family"] = result["family"].map(normalize_family)
    return result.set_index("model_id", drop=False)


def _comparison_frame(source: Any, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = source(rng) if callable(source) else source
    if isinstance(selected, tuple):
        selected, warnings = selected
    else:
        warnings = pd.DataFrame()
    if not isinstance(selected, pd.DataFrame):
        raise TypeError("comparisons must be a DataFrame or a callable returning one")
    if missing := {"judge", "m1", "m2"} - set(selected):
        raise ValueError(f"comparisons lack required columns: {', '.join(sorted(missing))}")
    frame = selected.copy().reset_index(drop=True)
    if frame[["judge", "m1", "m2"]].isna().any(axis=None):
        raise ValueError("comparison identities must be non-null")
    frame[["judge", "m1", "m2"]] = frame[["judge", "m1", "m2"]].astype(str)
    frame.insert(0, "comparison_id", np.arange(len(frame)))
    return frame, pd.DataFrame(warnings).copy()


def _resolve_classes(comparisons: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    result = comparisons.copy()
    predictable = result.get("predictable", pd.Series(True, index=result.index)).fillna(False).astype(bool)
    for role in ("judge", "m1", "m2"):
        unknown = sorted(set(result[role]) - set(metadata.index))
        if unknown:
            raise ValueError(f"unknown model identity: {', '.join(unknown)}")
        fallback = result[role].map(metadata["reasoning_class"])
        column = f"{role}_reasoning_class"
        result[column] = result[column].where(result[column].notna(), fallback) if column in result else fallback
        valid = result[column].isin({"reasoning", "non_reasoning"})
        if not valid[predictable].all():
            raise ValueError(f"{column} must be reasoning or non_reasoning")
    return result


def _observed_groups(
    observed_cells: pd.DataFrame, metadata: pd.DataFrame,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    required = {"model", "item", "correct"}
    if missing := required - set(observed_cells):
        raise ValueError(f"observed cells lack required columns: {', '.join(sorted(missing))}")
    cells = observed_cells.copy()
    if cells[["model", "item", "correct"]].isna().any(axis=None):
        raise ValueError("observed model, item, and correctness values must be non-null")
    cells["model"] = cells["model"].astype(str)
    cells["item"] = cells["item"].map(_canonical_item)
    fallback = cells["model"].map(metadata["reasoning_class"])
    cells["reasoning_class"] = (
        cells["reasoning_class"].where(cells["reasoning_class"].notna(), fallback)
        if "reasoning_class" in cells else fallback
    )
    if not cells["reasoning_class"].isin({"reasoning", "non_reasoning"}).all():
        raise ValueError("observed reasoning_class must be reasoning or non_reasoning")
    correct = pd.to_numeric(cells["correct"], errors="raise")
    if not correct.isin([0, 1]).all():
        raise ValueError("observed correctness must contain only 0/1 values")
    cells["correct"] = correct.astype(int)
    return {
        (model, reasoning_class, item): rows
        for (model, reasoning_class, item), rows in cells.groupby(
            ["model", "reasoning_class", "item"], sort=False,
        )
    }


def _candidate_cells(
    comparisons: pd.DataFrame, attempted: Sequence[str], metadata: pd.DataFrame,
    observed: dict[tuple[str, str, str], pd.DataFrame], capability: dict[str, float],
    irt: dict[str, np.ndarray], irt_index: int, rng: np.random.Generator,
    candidate_artifact: Any, candidate_index: int, cache: LatentCache, trial: int,
) -> pd.DataFrame:
    refs = pd.concat([
        comparisons[["m1", "m1_reasoning_class"]].rename(
            columns={"m1": "model", "m1_reasoning_class": "reasoning_class"},
        ),
        comparisons[["m2", "m2_reasoning_class"]].rename(
            columns={"m2": "model", "m2_reasoning_class": "reasoning_class"},
        ),
    ], ignore_index=True).drop_duplicates()
    rows = []
    lookup = _item_lookup(irt["items"])
    a = _draw_value(irt["a"], irt_index, "IRT discrimination")
    b = _draw_value(irt["b"], irt_index, "IRT difficulty")
    floor_values = np.asarray(irt["floor"])
    floor = _draw_value(floor_values, irt_index, "IRT floor") if floor_values.ndim == 2 else floor_values
    for ref in refs.itertuples(index=False):
        if ref.model not in capability:
            raise ValueError(f"capability draws lack model: {ref.model}")
        for item in attempted:
            if item not in lookup:
                raise ValueError(f"IRT posterior lacks attempted item: {item}")
            item_index = lookup[item]
            source = observed.get((ref.model, ref.reasoning_class, item))
            sampled = None if source is None else source.iloc[int(rng.integers(len(source)))]
            if sampled is None:
                probability = floor[item_index] + (1 - floor[item_index]) * expit(
                    a[item_index] * (capability[ref.model] - b[item_index]),
                )
                correct = int(rng.binomial(1, probability))
                response_chars = np.nan
            else:
                correct = int(sampled["correct"])
                response_chars = pd.to_numeric(pd.Series([sampled.get("response_chars", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "model": ref.model, "item": item, "reasoning_class": ref.reasoning_class,
                "correct": correct, "response_chars": response_chars, "visible_reasoning": 0,
                "theta": capability[ref.model], "release_day": metadata.at[ref.model, "release_day"],
                "difficulty": float(b[item_index]),
            })
    columns = [
        "model", "item", "reasoning_class", "correct", "response_chars",
        "visible_reasoning", "theta", "release_day", "difficulty",
    ]
    cells = pd.DataFrame(rows, columns=columns).drop_duplicates(
        ["model", "item", "reasoning_class"], keep="first",
    )
    missing_length = cells["response_chars"].isna()
    if missing_length.any():
        prediction_rows = _token_prediction_rows(candidate_artifact, cells.loc[missing_length])
        cells.loc[missing_length, "response_chars"] = draw_transcript_lengths(
            candidate_artifact, prediction_rows, rng, candidate_index,
            latent_cache=cache, trial_id=trial,
        )
    cells["response_chars"] = pd.to_numeric(cells["response_chars"], errors="raise")
    if cells["response_chars"].isna().any() or (cells["response_chars"] < 0).any():
        raise ValueError("candidate response lengths must be non-negative")
    return cells


def _usable_rows(comparisons: pd.DataFrame, attempted: Sequence[str], cells: pd.DataFrame) -> pd.DataFrame:
    lookup = cells.set_index(["model", "item", "reasoning_class"])
    rows = []
    for comparison in comparisons.itertuples(index=False):
        for item in attempted:
            m1 = lookup.loc[(comparison.m1, item, comparison.m1_reasoning_class)]
            m2 = lookup.loc[(comparison.m2, item, comparison.m2_reasoning_class)]
            if bool(m1["correct"]) and bool(m2["correct"]):
                rows.append({**comparison._asdict(), "item": item})
    return pd.DataFrame(rows, columns=[*comparisons.columns, "item"])


def _standardized(metadata: Mapping[str, Any], name: str, values: np.ndarray) -> np.ndarray:
    scales = metadata.get("training_scales", {})
    if name not in scales:
        raise ValueError(f"judgment artifact lacks training scale: {name}")
    try:
        mean, scale = float(scales[name]["mean"]), float(scales[name]["scale"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"judgment artifact has invalid training scale: {name}") from None
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError(f"judgment artifact has invalid training scale: {name}")
    return (np.asarray(values, dtype=float) - mean) / scale


def _raw_judgment_features(
    presented: pd.DataFrame, metadata: pd.DataFrame, capability: Mapping[str, float],
    names: Sequence[str],
) -> pd.DataFrame:
    left_theta = presented["left_model"].map(capability).to_numpy(dtype=float)
    right_theta = presented["right_model"].map(capability).to_numpy(dtype=float)
    judge_theta = presented["judge"].map(capability).to_numpy(dtype=float)
    left_release = presented["left_model"].map(metadata["release_day"]).to_numpy(dtype=float)
    right_release = presented["right_model"].map(metadata["release_day"]).to_numpy(dtype=float)
    judge_release = presented["judge"].map(metadata["release_day"]).to_numpy(dtype=float)
    theta_gap = left_theta - right_theta
    oversight = (left_theta + right_theta) / 2 - judge_theta
    difficulty = presented["difficulty"].to_numpy(dtype=float)
    left_family = presented["left_model"].map(metadata["family"])
    right_family = presented["right_model"].map(metadata["family"])
    raw = {
        "theta_gap": theta_gap,
        "oversight": oversight,
        "release_gap_days": left_release - right_release,
        "relative_release_days": (left_release + right_release) / 2 - judge_release,
        "difficulty": difficulty,
        "theta_gap_x_oversight": theta_gap * oversight,
        "theta_gap_x_difficulty": theta_gap * difficulty,
        "same_family_x_theta_gap": (
            left_family.eq(right_family) & presented["left_model"].ne(presented["right_model"])
        ).astype(float).to_numpy() * theta_gap,
        "length_difference": (
            np.log1p(presented["left_chars"].to_numpy(dtype=float))
            - np.log1p(presented["right_chars"].to_numpy(dtype=float))
        ),
        "judge_theta": judge_theta,
        "judge_release_days": judge_release,
    }
    if missing := [name for name in names if name not in raw]:
        raise ValueError(f"judgment support has unsupported predictor names: {', '.join(missing)}")
    values = pd.DataFrame({name: raw[name] for name in names}, index=presented.index)
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("simulated judgment support predictors must be finite")
    return values


def _decision_support(
    decision: Mapping[str, Any], artifact: PosteriorArtifact | Mapping[str, Any],
):
    posterior, metadata = _artifact_parts(artifact)
    support = decision.get("judgment_support")
    required = {
        "coordinate_system", "predictor_names", "training_matrix", "training_ranges",
        "nearest_neighbor_threshold",
    }
    if not isinstance(support, Mapping) or set(support) != required:
        raise ValueError("decision.json lacks complete judgment support")
    if support["coordinate_system"] != "raw":
        raise ValueError("judgment support coordinate_system must be raw")
    names = support["predictor_names"]
    if (
        not isinstance(names, list) or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("judgment support predictor names must be unique nonempty strings")
    unsupported = [
        feature for feature in metadata.get("feature_order", ()) if feature not in FEATURE_COLUMNS
    ]
    if unsupported:
        raise ValueError(f"unsupported judgment feature: {unsupported[0]}")
    expected = [
        "theta_gap",
        *(FEATURE_COLUMNS.get(feature, feature) for feature in metadata.get("feature_order", ())),
        *(["length_difference"] if "length_mean" in posterior else []),
    ]
    gate = decision.get("judge_covariate_gate", {}).get("enabled")
    if type(gate) is not bool:
        raise ValueError("decision.json lacks a boolean judge-covariate gate")
    if gate:
        expected.extend(["judge_theta", "judge_release_days"])
    expected = list(dict.fromkeys(expected))
    if names != expected:
        raise ValueError("judgment support predictor names do not match the selected artifact and judge gate")
    matrix = np.asarray(support["training_matrix"], dtype=float)
    ranges = np.asarray(support["training_ranges"], dtype=float)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(names) or len(matrix) < 2
        or ranges.shape != (len(names), 2)
        or not np.isfinite(matrix).all() or not np.isfinite(ranges).all()
        or not np.allclose(ranges[:, 0], matrix.min(axis=0))
        or not np.allclose(ranges[:, 1], matrix.max(axis=0))
    ):
        raise ValueError("judgment support matrix and training ranges are malformed")
    return names, ranges, build_support_index(
        matrix, support["nearest_neighbor_threshold"],
    )


def _judgment_presentation(metadata: Mapping[str, Any]) -> int:
    if metadata.get("presentation_include_reasoning") is not False:
        raise ValueError(
            "judgment artifact must declare the supported final-only include_reasoning=false presentation"
        )
    max_chars = metadata.get("presentation_max_chars")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("judgment artifact must declare positive integer presentation max chars")
    if metadata.get("presentation_truncate_frac") is not None:
        raise ValueError("judgment artifact uses unsupported paired truncation")
    if metadata.get("presentation_truncate_keep") != "first":
        raise ValueError("judgment artifact must declare supported truncate keep=first")
    return max_chars


def _scenario_left_probabilities(
    artifact: PosteriorArtifact | Mapping[str, Any], presented: pd.DataFrame,
    metadata: pd.DataFrame, capability: dict[str, float], posterior_index: int,
    cache: LatentCache, trial: int,
) -> np.ndarray:
    posterior, artifact_metadata = _artifact_parts(artifact)
    _judgment_presentation(artifact_metadata)
    if presented.empty:
        return np.empty(0)
    draw = {name: _draw_value(values, posterior_index, name) for name, values in posterior.items()}
    maps = artifact_metadata.get("category_maps", {})
    observed_judges = maps.get("judges", {})
    observed_models = maps.get("models", {})
    observed_items = {_canonical_item(item): index for item, index in maps.get("items", {}).items()}

    judges = list(dict.fromkeys(zip(presented["judge"], presented["judge_reasoning_class"])))
    models = list(dict.fromkeys([
        *zip(presented["left_model"], presented["left_reasoning_class"]),
        *zip(presented["right_model"], presented["right_reasoning_class"]),
    ]))
    items = list(pd.unique(presented["item"]))
    judge_effects = []
    length_deviations = []
    include_length = "length_mean" in draw
    for judge, reasoning_class in judges:
        if judge in observed_judges and metadata.at[judge, "reasoning_class"] == reasoning_class:
            judge_effects.append(np.asarray(draw["judge_effects"])[observed_judges[judge]])
            if include_length:
                length_deviations.append(float(np.asarray(draw["length_deviation"])[observed_judges[judge]]))
            continue
        population = np.asarray(draw.get("judge_population"))
        scales = np.asarray(draw.get("judge_scales"))
        corr = np.asarray(draw.get("judge_corr", np.eye(2)))
        if population.ndim != 2 or population.shape[0] != 2 or scales.shape != (2,) or corr.shape != (2, 2):
            raise ValueError("judgment artifact lacks unseen-judge population arrays")
        design = [1.0]
        if population.shape[1] == 3:
            release = metadata.at[judge, "release_day"]
            design.extend([
                float(_standardized(artifact_metadata, "judge_theta", np.array([capability[judge]]))[0]),
                float(_standardized(artifact_metadata, "judge_release_days", np.array([release]))[0]),
            ])
        if population.shape[1] != len(design):
            raise ValueError("judge population shape does not match conditional metadata")
        raw = np.array([
            cache.normal((trial, "judge_raw", judge, reasoning_class, axis))
            for axis in range(2)
        ])
        judge_effects.append(population @ np.asarray(design) + (raw @ corr.T) * scales)
        if include_length:
            length_deviations.append(cache.normal(
                (trial, "judge_length", judge, reasoning_class),
                0.0, float(draw.get("length_scale", 0.0)),
            ))

    candidate_effects = []
    for model, reasoning_class in models:
        if model in observed_models and metadata.at[model, "reasoning_class"] == reasoning_class:
            candidate_effects.append(float(np.asarray(draw["candidate_effects"])[observed_models[model]]))
        else:
            candidate_effects.append(cache.normal(
                (trial, "candidate", model, reasoning_class),
                0.0, float(draw.get("candidate_scale", 0.0)),
            ))
    item_effects = []
    for item in items:
        if item in observed_items:
            item_effects.append(float(np.asarray(draw["item_effects"])[observed_items[item]]))
        else:
            item_effects.append(cache.normal(
                (trial, "item", item), 0.0, float(draw.get("item_scale", 0.0)),
            ))

    left_theta = presented["left_model"].map(capability).to_numpy(dtype=float)
    right_theta = presented["right_model"].map(capability).to_numpy(dtype=float)
    judge_theta = presented["judge"].map(capability).to_numpy(dtype=float)
    left_release = presented["left_model"].map(metadata["release_day"]).to_numpy(dtype=float)
    right_release = presented["right_model"].map(metadata["release_day"]).to_numpy(dtype=float)
    judge_release = presented["judge"].map(metadata["release_day"]).to_numpy(dtype=float)
    theta_gap = left_theta - right_theta
    difficulty = presented["difficulty"].to_numpy(dtype=float)
    oversight = (left_theta + right_theta) / 2 - judge_theta
    feature_order = tuple(artifact_metadata.get("feature_order", ()))
    left_family = presented["left_model"].map(metadata["family"])
    right_family = presented["right_model"].map(metadata["family"])
    if "same_family_x_theta_gap_sd" in feature_order and (left_family.isna() | right_family.isna()).any():
        missing = sorted(set(
            presented.loc[left_family.isna(), "left_model"]
        ) | set(presented.loc[right_family.isna(), "right_model"]))
        raise ValueError(f"models lack family metadata: {', '.join(missing)}")
    raw_features = {
        "oversight": oversight,
        "release_gap_days": left_release - right_release,
        "relative_release_days": (left_release + right_release) / 2 - judge_release,
        "difficulty": difficulty,
        "theta_gap_x_oversight": theta_gap * oversight,
        "theta_gap_x_difficulty": theta_gap * difficulty,
        "same_family_x_theta_gap": (
            left_family.eq(right_family) & presented["left_model"].ne(presented["right_model"])
        ).astype(float) * theta_gap,
    }
    columns = []
    for feature in feature_order:
        source = FEATURE_COLUMNS.get(feature, feature)
        if source not in raw_features:
            raise ValueError(f"unsupported judgment feature: {feature}")
        values = raw_features[source]
        columns.append(_standardized(artifact_metadata, source, values) if feature.endswith("_sd") else values)
    global_x = np.column_stack(columns) if columns else np.empty((len(presented), 0))
    length_difference = (
        np.log1p(presented["left_chars"].to_numpy(dtype=float))
        - np.log1p(presented["right_chars"].to_numpy(dtype=float))
    )
    if include_length:
        length_difference = _standardized(artifact_metadata, "length_difference", length_difference)

    judge_map = {name: index for index, name in enumerate(judges)}
    model_map = {name: index for index, name in enumerate(models)}
    item_map = {name: index for index, name in enumerate(items)}
    data = JudgmentData(
        y=np.zeros(len(presented), dtype=int),
        judge=np.array([
            judge_map[key] for key in zip(presented["judge"], presented["judge_reasoning_class"])
        ], dtype=int),
        item=presented["item"].map(item_map).to_numpy(dtype=int),
        left_model=np.array([
            model_map[key]
            for key in zip(presented["left_model"], presented["left_reasoning_class"])
        ], dtype=int),
        right_model=np.array([
            model_map[key]
            for key in zip(presented["right_model"], presented["right_reasoning_class"])
        ], dtype=int),
        theta_gap=_standardized(artifact_metadata, "theta_gap", theta_gap),
        global_x=global_x, length_difference=length_difference,
        judge_covariates=np.zeros((len(judges), 2)), include_length=include_length,
        num_judges=len(judges), num_items=len(items), num_models=len(models),
        global_features=feature_order,
    )
    scenario_posterior = {
        "judge_effects": np.asarray(judge_effects),
        "candidate_effects": np.asarray(candidate_effects),
        "item_effects": np.asarray(item_effects),
    }
    if feature_order:
        scenario_posterior["global_beta"] = draw["global_beta"]
    if include_length:
        scenario_posterior["length_mean"] = draw["length_mean"]
        scenario_posterior["length_deviation"] = np.asarray(length_deviations)
    return expit(left_logits(scenario_posterior, data)[0])


def _presentation_metadata(presented: pd.DataFrame, cells: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    result = presented.copy()
    indexed = cells.set_index(["model", "item", "reasoning_class"])
    result["left_reasoning_class"] = np.where(
        result["m1_is_left"], result["m1_reasoning_class"], result["m2_reasoning_class"],
    )
    result["right_reasoning_class"] = np.where(
        result["m1_is_left"], result["m2_reasoning_class"], result["m1_reasoning_class"],
    )
    for side in ("left", "right"):
        keys = list(zip(result[f"{side}_model"], result["item"], result[f"{side}_reasoning_class"]))
        result[f"{side}_chars"] = [indexed.at[key, "response_chars"] for key in keys]
        result[f"{side}_visible_reasoning"] = [indexed.at[key, "visible_reasoning"] for key in keys]
    result["difficulty"] = [
        indexed.at[(model, item, reasoning_class), "difficulty"]
        for model, item, reasoning_class in zip(result["m1"], result["item"], result["m1_reasoning_class"])
    ]
    return result


def _usage_rows(
    comparisons: pd.DataFrame, cells: pd.DataFrame, presented: pd.DataFrame, candidate_artifact: Any,
    judge_artifact: Any, rng: np.random.Generator, candidate_index: int, judge_index: int,
    cache: LatentCache, trial: int, reuse_responses: bool,
) -> pd.DataFrame:
    parts = []
    if not reuse_responses and not cells.empty:
        requests = cells.drop_duplicates(["model", "item", "reasoning_class"]).reset_index(drop=True)
        prediction_rows = _token_prediction_rows(candidate_artifact, requests)
        usage = draw_token_usage(
            candidate_artifact, prediction_rows, rng, candidate_index,
            latent_cache=cache, trial_id=trial,
        )
        parts.append(pd.concat([
            requests[["model", "reasoning_class"]].assign(
                trial=trial, role="candidate", calls=1,
            ),
            usage.reset_index(drop=True),
        ], axis=1))
    if not presented.empty:
        requests = pd.DataFrame({
            "model": presented["judge"].to_numpy(),
            "reasoning_class": presented["judge_reasoning_class"].to_numpy(),
            "difficulty": presented["difficulty"].to_numpy(),
            "transcript_chars": (presented["left_chars"] + presented["right_chars"]).to_numpy(),
        })
        prediction_rows = _token_prediction_rows(judge_artifact, requests)
        usage = draw_token_usage(
            judge_artifact, prediction_rows, rng, judge_index,
            latent_cache=cache, trial_id=trial,
        )
        parts.append(pd.concat([
            requests[["model", "reasoning_class"]].assign(
                trial=trial, role="judge", calls=1,
            ),
            usage.reset_index(drop=True),
        ], axis=1))
    planned = pd.concat([
        cells[["model", "reasoning_class"]].drop_duplicates().assign(role="candidate"),
        comparisons[["judge", "judge_reasoning_class"]].rename(columns={
            "judge": "model", "judge_reasoning_class": "reasoning_class",
        }).drop_duplicates().assign(role="judge"),
    ], ignore_index=True).drop_duplicates()
    if not planned.empty:
        parts.append(planned.assign(
            trial=trial, calls=0, input_tokens=0, output_tokens=0,
            reasoning_tokens=0, ordinary_output_tokens=0,
        ))
    columns = [
        "trial", "role", "model", "reasoning_class", "calls", "input_tokens", "output_tokens",
        "reasoning_tokens", "ordinary_output_tokens",
    ]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).groupby(
        ["trial", "role", "model", "reasoning_class"], as_index=False, sort=False,
    )[["calls", "input_tokens", "output_tokens", "reasoning_tokens", "ordinary_output_tokens"]].sum()


def _trial_rows(
    config: SimulationConfig, trial: int, comparisons: pd.DataFrame, presented: pd.DataFrame,
    usage: pd.DataFrame, priced: pd.DataFrame, missing_prices: set[str],
) -> list[dict[str, Any]]:
    base = {
        "trial": trial, "configuration": config.configuration, "mode": config.mode,
        "comparison_id": pd.NA, "judge": pd.NA, "m1": pd.NA, "m2": pd.NA,
        "model": pd.NA, "reasoning_class": pd.NA, "role": pd.NA,
        "warning_state": "ok", "reason": "",
    }
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, **context: Any) -> None:
        rows.append({**base, "metric": metric, "value": value, **context})

    usable_items = len(presented) / config.presentations
    candidate_calls = int(usage.loc[usage["role"].eq("candidate"), "calls"].sum()) if not usage.empty else 0
    add("usable_items", usable_items)
    add("judgment_calls", len(presented))
    add("candidate_calls", candidate_calls)
    for metric in ("support_distance", "support_threshold", "outside_training_range", "outside_joint_support"):
        if metric in presented and not presented.empty:
            add(metric, float(pd.to_numeric(presented[metric], errors="raise").max()))
    def outcomes(frame: pd.DataFrame) -> tuple[float, int, str]:
        if frame.empty:
            return np.nan, 0, "no usable judgments"
        tied = frame["m1_theta"].eq(frame["m2_theta"])
        correct = np.where(
            frame["m1_theta"].gt(frame["m2_theta"]), frame["chose_m1"],
            1 - frame["chose_m1"],
        )
        if not (~tied).any():
            return np.nan, int(tied.sum()), "all usable judgments tied"
        return float(np.mean(correct[~tied])), int(tied.sum()), ""

    accuracy, ties, reason = outcomes(presented)
    add(
        "choice_accuracy", accuracy,
        warning_state="ok" if np.isfinite(accuracy) else "unavailable", reason=reason,
    )
    add("ties", ties)
    for source in comparisons.itertuples(index=False):
        group = presented.loc[presented["comparison_id"].eq(source.comparison_id)]
        accuracy, ties, reason = outcomes(group)
        source_reason = getattr(source, "unavailable_reason", "")
        if group.empty and isinstance(source_reason, str) and source_reason:
            reason = source_reason
        context = {
            "comparison_id": source.comparison_id, "judge": source.judge,
            "m1": source.m1, "m2": source.m2,
        }
        add("usable_items", len(group) / config.presentations, **context)
        add("judgment_calls", len(group), **context)
        for metric in ("support_distance", "support_threshold", "outside_training_range", "outside_joint_support"):
            if metric in group and not group.empty:
                add(metric, float(pd.to_numeric(group[metric], errors="raise").max()), **context)
        add(
            "choice_accuracy", accuracy, **context,
            warning_state="ok" if np.isfinite(accuracy) else "unavailable", reason=reason,
        )
        add("m1_wins", int(group["chose_m1"].sum()), **context)
        if not group.empty:
            add("m1_theta", float(group["m1_theta"].iloc[0]), **context)
            add("m2_theta", float(group["m2_theta"].iloc[0]), **context)
        add("ties", ties, **context)
        if config.mode == "manual":
            value = float(group["p_choose_m1"].mean()) if not group.empty else np.nan
            add(
                "p_choose_m1", value, **context,
                warning_state="ok" if np.isfinite(value) else "unavailable",
                reason="" if np.isfinite(value) else reason,
            )

    token_columns = {
        "input_tokens": "{role}_input_tokens",
        "output_tokens": "{role}_output_tokens",
        "ordinary_output_tokens": "{role}_ordinary_output_tokens",
        "reasoning_tokens": "{role}_reasoning_tokens",
    }
    for role in ("candidate", "judge"):
        role_usage = usage.loc[usage["role"].eq(role)]
        for record in role_usage.to_dict("records"):
            for column, template in token_columns.items():
                add(
                    template.format(role=role), record[column], model=record["model"],
                    reasoning_class=record["reasoning_class"], role=role,
                )
        for column, template in token_columns.items():
            add(template.format(role=role), role_usage[column].sum() if column in role_usage else 0,
                model="__total__", role=role)
    for record in priced.to_dict("records"):
        missing = record["model"] in missing_prices and pd.isna(record["cost"])
        add(
            "cost", record["cost"], model=record["model"], role=record["role"],
            reasoning_class=record["reasoning_class"],
            warning_state="missing_price" if missing else "ok",
            reason=f"missing input or output price for {record['model']}" if missing else "",
        )
    for role in ("candidate", "judge"):
        role_costs = priced.loc[priced["role"].eq(role), "cost"] if not priced.empty else pd.Series(dtype=float)
        role_cost = (
            role_costs.sum() if len(role_costs) and not role_costs.isna().any()
            else (0.0 if role_costs.empty else np.nan)
        )
        add(
            "cost", role_cost, model="__total__", role=role,
            warning_state="missing_price" if role_costs.isna().any() else "ok",
            reason=(
                "missing input or output price for "
                + ", ".join(sorted(priced.loc[
                    priced["role"].eq(role) & priced["cost"].isna(), "model",
                ].astype(str).unique()))
                if role_costs.isna().any() else ""
            ),
        )
    total_cost = priced["cost"].sum() if len(priced) and not priced["cost"].isna().any() else (0.0 if priced.empty else np.nan)
    add(
        "cost", total_cost, model="__total__", role="all",
        warning_state="missing_price" if priced["cost"].isna().any() else "ok",
        reason=(
            "missing input or output price for "
            + ", ".join(sorted(priced.loc[priced["cost"].isna(), "model"].astype(str).unique()))
            if priced["cost"].isna().any() else ""
        ),
    )
    return rows


def simulate(
    comparisons: pd.DataFrame | Callable[[np.random.Generator], Any],
    config: SimulationConfig, *, item_ids: Sequence[str], observed_cells: pd.DataFrame,
    capability_models: Sequence[str], capability_draws: np.ndarray, irt: Mapping[str, Any],
    judgment_artifact: PosteriorArtifact | Mapping[str, Any],
    candidate_token_artifact: PosteriorArtifact | Mapping[str, Any],
    judge_token_artifact: PosteriorArtifact | Mapping[str, Any], catalogue: pd.DataFrame,
    decision_path: str | Path, data_hashes: Mapping[str, str], seed: int = 0,
) -> SimulationResult:
    """Run fixed-budget posterior-predictive trials with shared latent effects."""
    decision = require_decision(decision_path, data_hashes)
    _, judgment_metadata = _artifact_parts(judgment_artifact)
    presentation_max_chars = _judgment_presentation(judgment_metadata)
    support_names, support_ranges, support_index = _decision_support(decision, judgment_artifact)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    items = [_canonical_item(item) for item in item_ids]
    if len(items) != len(set(items)) or any(not item for item in items):
        raise ValueError("item_ids must contain unique nonempty strings")
    if config.attempted_items > len(items):
        raise ValueError(
            f"requested attempted_items={config.attempted_items}, eligible item count={len(items)}"
        )
    models = [str(model) for model in capability_models]
    draws = np.asarray(capability_draws, dtype=float)
    if len(models) != len(set(models)) or draws.ndim != 2 or draws.shape[1] != len(models):
        raise ValueError("capability draws must align to unique capability model labels")
    if not np.isfinite(draws).all() or not len(draws):
        raise ValueError("capability draws must be finite and nonempty")

    irt_parts = _irt_parts(irt)
    metadata = _price_metadata(catalogue)
    observed = _observed_groups(observed_cells, metadata)
    aligned_counts = [len(draws), len(np.asarray(irt_parts["a"])), len(np.asarray(irt_parts["b"]))]
    if np.asarray(irt_parts["floor"]).ndim == 2:
        aligned_counts.append(len(np.asarray(irt_parts["floor"])))
    if len(set(aligned_counts)) != 1:
        raise ValueError("capability and IRT posterior draw counts must be aligned")
    schedule_rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    capability_schedule = schedule_rngs[0].integers(len(draws), size=config.trials)
    judgment_schedule = schedule_rngs[1].integers(
        _posterior_count(judgment_artifact, "judgment"), size=config.trials,
    )
    candidate_schedule = schedule_rngs[2].integers(
        _posterior_count(candidate_token_artifact, "candidate token"), size=config.trials,
    )
    judge_schedule = schedule_rngs[3].integers(
        _posterior_count(judge_token_artifact, "judge token"), size=config.trials,
    )
    rng = np.random.default_rng(seed)
    cache = LatentCache(rng)
    trial_records: list[dict[str, Any]] = []
    warning_frames: list[pd.DataFrame] = []
    comparison_count: int | None = None

    for trial in range(config.trials):
        capability_index = int(capability_schedule[trial])
        judgment_index = int(judgment_schedule[trial])
        candidate_index = int(candidate_schedule[trial])
        judge_index = int(judge_schedule[trial])
        selected, design_warnings = _comparison_frame(comparisons, rng)
        selected = _resolve_classes(selected, metadata)
        if comparison_count is None:
            comparison_count = len(selected)
        elif len(selected) != comparison_count:
            raise ValueError("comparison factory must return a fixed comparison count")
        if not design_warnings.empty:
            design_warnings = design_warnings.copy()
            design_warnings["trial"] = trial
            warning_frames.append(design_warnings)
        predictable_mask = selected.get(
            "predictable", pd.Series(True, index=selected.index),
        ).fillna(False).astype(bool)
        if "unavailable_reason" not in selected:
            selected["unavailable_reason"] = ""
        capability_warnings = []
        for row in selected.index[predictable_mask]:
            missing = sorted(set(selected.loc[row, ["judge", "m1", "m2"]]) - set(models))
            if not missing:
                continue
            selected.at[row, "predictable"] = False
            selected.at[row, "unavailable_reason"] = f"missing capability draw: {', '.join(missing)}"
            capability_warnings.extend({
                "code": "missing_capability_draw", "trial": trial, "row": int(row),
                "source_row": pd.NA, "model": model,
                "message": f"missing capability draw: {model}",
            } for model in missing)
        if capability_warnings:
            warning_frames.append(pd.DataFrame(capability_warnings))
        predictable_mask = selected.get(
            "predictable", pd.Series(True, index=selected.index),
        ).fillna(False).astype(bool)
        predictable = selected.loc[predictable_mask]
        capability = dict(zip(models, draws[capability_index]))
        attempted = rng.choice(np.asarray(items), config.attempted_items, replace=False).tolist()
        cells = _candidate_cells(
            predictable, attempted, metadata, observed, capability, irt_parts,
            capability_index, rng, candidate_token_artifact, candidate_index, cache, trial,
        )
        cells["response_chars"] = cells["response_chars"].clip(upper=presentation_max_chars)
        usable = _usable_rows(predictable, attempted, cells)
        presented = _presentation_metadata(counterbalance(usable, config.presentations, rng), cells, metadata)
        if not presented.empty:
            raw_support = _raw_judgment_features(
                presented, metadata, capability, support_names,
            )
            support = support_index.query(raw_support.to_numpy(float))
            outside_range = (
                raw_support.to_numpy(float) < support_ranges[:, 0]
            ) | (
                raw_support.to_numpy(float) > support_ranges[:, 1]
            )
            presented["support_distance"] = support["distance"].to_numpy(float)
            presented["support_threshold"] = support_index.threshold
            presented["outside_training_range"] = outside_range.any(axis=1).astype(float)
            presented["outside_joint_support"] = support["outside_support"].to_numpy(float)
            for code, message in (
                ("outside_training_range", "judgment predictors are outside their training ranges"),
                ("outside_joint_support", "judgment predictors are outside joint training support"),
            ):
                comparison_ids = sorted(
                    presented.loc[presented[code].astype(bool), "comparison_id"].unique()
                )
                if comparison_ids:
                    warning_frames.append(pd.DataFrame([{
                        "code": code, "trial": trial, "row": int(comparison_id),
                        "source_row": pd.NA, "model": pd.NA, "message": message,
                    } for comparison_id in comparison_ids]))
        probabilities = _scenario_left_probabilities(
            judgment_artifact, presented, metadata, capability, judgment_index, cache, trial,
        )
        presented["p_choose_left"] = probabilities
        presented["chose_left"] = rng.binomial(1, probabilities).astype(int)
        presented["p_choose_m1"] = np.where(
            presented["m1_is_left"], probabilities, 1 - probabilities,
        )
        presented["chose_m1"] = np.where(
            presented["m1_is_left"], presented["chose_left"], 1 - presented["chose_left"],
        ).astype(int)
        presented["m1_theta"] = presented["m1"].map(capability)
        presented["m2_theta"] = presented["m2"].map(capability)
        usage = _usage_rows(
            predictable, cells, presented, candidate_token_artifact, judge_token_artifact, rng,
            candidate_index, judge_index, cache, trial, config.reuse_responses,
        )
        priced, price_warnings = cost_rows(usage, catalogue)
        if not price_warnings.empty:
            price_warnings = price_warnings.copy()
            price_warnings["trial"] = trial
            warning_frames.append(price_warnings)
        missing_prices = set(price_warnings.get("model", []))
        trial_records.extend(_trial_rows(
            config, trial, selected, presented, usage, priced, missing_prices,
        ))

    trial_frame = pd.DataFrame(trial_records).reindex(columns=_TRIAL_COLUMNS)
    upper_bound = int((comparison_count or 0) * config.attempted_items * config.presentations)
    calls = trial_frame.loc[trial_frame["metric"].eq("judgment_calls"), "value"]
    if (calls > upper_bound).any():
        raise AssertionError("simulated judgment calls exceed the pre-filter upper bound")
    summary = summarize_trials(trial_frame)
    summary["exact"] = False
    exact = {column: pd.NA for column in summary.columns}
    exact.update({
        "configuration": config.configuration, "mode": config.mode,
        "metric": "judgment_call_upper_bound", "value": upper_bound,
        "p5": np.nan, "p50": np.nan, "p95": np.nan,
        "n_trials": config.trials, "exact": True, "warning_state": "ok", "reason": "",
    })
    summary = pd.concat([summary, pd.DataFrame([exact])], ignore_index=True)
    warnings = pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame(columns=_WARNING_COLUMNS)
    warnings = warnings.reindex(columns=_WARNING_COLUMNS)
    warnings = warnings.drop_duplicates(
        ["code", "row", "source_row", "model", "message"], keep="first", ignore_index=True,
    )
    return SimulationResult(trial_frame, summary, warnings)
