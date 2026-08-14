"""Editable model metadata and uncertainty-backed capability draws."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

CATALOGUE_COLUMNS = [
    "model_id", "display_name", "release", "theta", "theta_p5", "theta_p95",
    "theta_source", "gpqa_correct", "gpqa_total", "family", "reasoning_class", "judge_pool",
    "candidate_pool", "observed_judge", "observed_candidate", "price_in", "price_out",
]


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _records(source: str | Path | pd.DataFrame | list[dict]) -> list[dict]:
    if isinstance(source, pd.DataFrame):
        return source.to_dict("records")
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path).to_dict("records")
        with path.open() as handle:
            loaded = json.load(handle)
        if isinstance(loaded, Mapping):
            loaded = loaded.get("models", loaded.get("pool", loaded))
        return list(loaded)
    return list(source)


def _canonical_models(path: str | Path) -> list[str]:
    result = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if match and match.group(1) not in result:
            result.append(match.group(1))
    return result


def _first(record: Mapping, *names: str):
    for name in names:
        if name in record:
            return record[name]
    return pd.NA


def _reasoning_class(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"reasoning", "non_reasoning"}:
            return value
        if value in {"true", "yes", "1"}:
            return "reasoning"
        if value in {"false", "no", "0"}:
            return "non_reasoning"
        return value
    return "reasoning" if bool(value) else "non_reasoning"


def build_catalogue(
    canonical_models: str | Path,
    model_pool: str | Path,
    capabilities: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Join canonical IDs to editable metadata and fitted capabilities."""
    pool = _records(model_pool)
    capability_rows = _records(capabilities)
    pool_by_name = {normalized_name(_first(row, "name", "model", "model_id")): row for row in pool}

    fitted = {}
    for row in capability_rows:
        if str(row.get("irt_model", "")) == "fixed_3pl" and str(row.get("fit", "")) == "gpqa":
            fitted[normalized_name(_first(row, "model", "model_id", "name"))] = row

    rows = []
    for model_id in _canonical_models(canonical_models):
        metadata = pool_by_name.get(normalized_name(model_id), {})
        capability = fitted.get(normalized_name(model_id), {})
        row = {column: pd.NA for column in CATALOGUE_COLUMNS}
        row.update({
            "model_id": model_id,
            "display_name": _first(metadata, "display_name", "name") if metadata else model_id,
            "release": _first(metadata, "release"),
            "family": _first(metadata, "family", "dev", "organization"),
            "reasoning_class": _reasoning_class(_first(metadata, "reasoning_class", "reasoning")),
            "judge_pool": _first(metadata, "judge_pool") if metadata else pd.NA,
            "candidate_pool": _first(metadata, "candidate_pool") if metadata else pd.NA,
            "observed_judge": _first(metadata, "observed_judge"),
            "observed_candidate": _first(metadata, "observed_candidate"),
            "price_in": _first(metadata, "price_in"),
            "price_out": _first(metadata, "price_out"),
        })
        for column in ("theta", "theta_p5", "theta_p95", "gpqa_correct", "gpqa_total"):
            if column in capability:
                row[column] = capability[column]
        if capability:
            row["theta_source"] = "fixed_3pl"
        for column in ("theta_source", "judge_pool", "candidate_pool", "observed_judge", "observed_candidate"):
            if column in capability:
                row[column] = capability[column]
        rows.append(row)

    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)


def load_catalogue(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text())
        return pd.DataFrame(loaded, columns=CATALOGUE_COLUMNS)
    return pd.read_csv(path).reindex(columns=CATALOGUE_COLUMNS)


def _present(value) -> bool:
    return not pd.isna(value)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _enabled(value) -> bool:
    if not _present(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def _valid_counts(correct, total) -> bool:
    correct, total = _number(correct), _number(total)
    return (
        np.isfinite(correct) and np.isfinite(total)
        and correct.is_integer() and total.is_integer()
        and 0 <= correct <= total and total > 0
    )


def _uncertainty_source(row: pd.Series) -> bool:
    source = str(row.get("theta_source", "")).lower()
    correct, total = row.get("gpqa_correct", pd.NA), row.get("gpqa_total", pd.NA)
    if _present(correct) or _present(total):
        return _valid_counts(correct, total)
    if source in {"fixed_3pl", "fixed_3pl_posterior"}:
        return True
    p5, p95 = _number(row.get("theta_p5")), _number(row.get("theta_p95"))
    return np.isfinite(p5) and np.isfinite(p95) and p5 <= p95


def validate_catalogue(frame: pd.DataFrame) -> None:
    if "model_id" in frame and frame["model_id"].duplicated().any():
        raise ValueError("duplicate model_id")
    classes = frame.get("reasoning_class", pd.Series(dtype=str)).dropna().astype(str)
    if not classes.isin({"reasoning", "non_reasoning"}).all():
        raise ValueError("reasoning_class must be reasoning or non_reasoning")
    for index, row in frame.iterrows():
        release = row.get("release", pd.NA)
        if _present(release):
            try:
                pd.to_datetime(release, errors="raise")
            except (TypeError, ValueError):
                raise ValueError(f"unparsable release at row {index}") from None
        theta = row.get("theta", pd.NA)
        if _present(theta) and not np.isfinite(_number(theta)):
            raise ValueError(f"non-finite theta at row {index}")
        for column in ("price_in", "price_out"):
            raw_value = row.get(column, pd.NA)
            value = _number(raw_value)
            if _present(raw_value) and (not np.isfinite(value) or value < 0):
                raise ValueError(f"negative {column} at row {index}")
        enabled = any(_enabled(row[column]) for column in ("judge_pool", "candidate_pool") if column in row)
        if enabled and (not _present(release) or not _uncertainty_source(row)):
            raise ValueError(f"pool-enabled row {row.get('model_id', index)} lacks release or capability uncertainty")


def theta_draws_from_accuracy(correct, total, irt_draws, rng) -> np.ndarray:
    if not _valid_counts(correct, total):
        raise ValueError("GPQA correct/total counts must be finite integer values satisfying 0 <= correct <= total")
    a = np.asarray(irt_draws["a"], dtype=float)
    b = np.asarray(irt_draws["b"], dtype=float)
    floor = np.asarray(irt_draws["floor"], dtype=float)
    accuracy = rng.beta(correct + 0.5, total - correct + 0.5, size=len(a))
    result = np.empty_like(accuracy)
    for draw, target in enumerate(accuracy):
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2
            p = floor + (1 - floor) * expit(a[draw] * (mid - b[draw]))
            if p.mean() < target:
                lo = mid
            else:
                hi = mid
        result[draw] = (lo + hi) / 2
    return result


def _posterior(irt_draws):
    if all(key in irt_draws for key in ("theta", "models", "a", "b", "floor")):
        return {key: irt_draws[key] for key in ("theta", "models", "a", "b", "floor")}
    prefix = "fixed_3pl__gpqa"
    keys = {key.rsplit("__", 1)[-1]: irt_draws[key] for key in irt_draws if key.startswith(prefix + "__")}
    required = {"theta", "models", "a", "b", "floor"}
    if not required <= keys.keys():
        raise ValueError("irt_draws lacks fixed_3pl GPQA posterior arrays")
    return {key: keys[key] for key in required}


def build_capability_draws(catalogue: pd.DataFrame, irt_draws, rng) -> tuple[list[str], np.ndarray]:
    posterior = _posterior(irt_draws)
    theta = np.asarray(posterior["theta"], dtype=float)
    models = [str(value) for value in np.asarray(posterior["models"]).tolist()]
    n_draws = theta.shape[0]
    posterior_by_name = {normalized_name(model): column for column, model in enumerate(models)}
    rows, labels = [], []
    for _, row in catalogue.iterrows():
        name = normalized_name(row["model_id"])
        if name in posterior_by_name:
            values = theta[:, posterior_by_name[name]]
        elif _valid_counts(row.get("gpqa_correct", pd.NA), row.get("gpqa_total", pd.NA)):
            values = theta_draws_from_accuracy(row["gpqa_correct"], row["gpqa_total"], posterior, rng)
        else:
            p5, p95 = _number(row.get("theta_p5")), _number(row.get("theta_p95"))
            if not (np.isfinite(p5) and np.isfinite(p95) and p5 <= p95):
                continue
            mean = (p5 + p95) / 2
            sd = (p95 - p5) / (2 * 1.6448536269514722)
            values = rng.normal(mean, sd, size=n_draws)
        labels.append(str(row["model_id"]))
        rows.append(np.asarray(values, dtype=float))
    matrix = np.column_stack(rows) if rows else np.empty((n_draws, 0), dtype=float)
    return labels, matrix
