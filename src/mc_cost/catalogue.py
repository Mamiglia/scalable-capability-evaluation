"""Editable model metadata and uncertainty-backed capability draws."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

CATALOGUE_COLUMNS = [
    "model_id", "display_name", "aliases", "release", "theta", "theta_p5", "theta_p95",
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


def _aliases(value) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [value]
        else:
            return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(alias) for alias in value if str(alias)]
    return []


def _pool_names(record: Mapping) -> list[str]:
    names = []
    for field in ("name", "model", "model_id", "or_id"):
        value = record.get(field)
        if isinstance(value, str) and value:
            names.extend((value, value.rsplit("/", 1)[-1]))
    return names


def catalogue_aliases(catalogue: pd.DataFrame) -> dict[str, str]:
    """Return normalized raw/canonical names mapped to catalogue IDs."""
    result = {}
    for _, row in catalogue.iterrows():
        model_id = str(row["model_id"])
        for name in (model_id, *_aliases(row.get("aliases", pd.NA))):
            key = normalized_name(name)
            if key in result and result[key] != model_id:
                raise ValueError(f"catalogue alias {name!r} refers to multiple model IDs")
            result[key] = model_id
    return result


def resolve_model_id(value: str, catalogue: pd.DataFrame) -> str | None:
    """Resolve case/punctuation variants and declared aliases to a canonical ID."""
    return catalogue_aliases(catalogue).get(normalized_name(value))


def build_catalogue(
    canonical_models: str | Path,
    model_pool: str | Path,
    capabilities: str | Path | pd.DataFrame,
    overrides: str | Path | pd.DataFrame | list[dict] | None = None,
) -> pd.DataFrame:
    """Union canonical seeds and fitted models, then apply explicit MC metadata."""
    pool = _records(model_pool)
    capability_rows = _records(capabilities)
    override_rows = _records(overrides) if overrides is not None else []
    pool_by_name = {}
    for record in pool:
        for name in _pool_names(record):
            pool_by_name.setdefault(normalized_name(name), record)

    model_ids = _canonical_models(canonical_models)
    by_name = {normalized_name(model_id): model_id for model_id in model_ids}
    aliases = defaultdict(list)
    override_by_id = {}
    for override in override_rows:
        if not isinstance(override.get("model_id"), str) or not override["model_id"]:
            raise ValueError("model override requires a nonempty model_id")
        requested = override["model_id"]
        model_id = by_name.get(normalized_name(requested))
        if model_id is None:
            model_id = requested
            model_ids.append(model_id)
            by_name[normalized_name(model_id)] = model_id
        if model_id in override_by_id:
            raise ValueError(f"duplicate model override for {model_id}")
        override_by_id[model_id] = override
        for alias in _aliases(override.get("aliases", [])):
            key = normalized_name(alias)
            owner = by_name.get(key)
            if owner is not None and owner != model_id:
                raise ValueError(f"model alias {alias!r} conflicts with {owner}")
            by_name[key] = model_id
            if normalized_name(alias) != normalized_name(model_id):
                aliases[model_id].append(alias)

    fitted = {}
    for row in capability_rows:
        if str(row.get("irt_model", "")) == "fixed_3pl" and str(row.get("fit", "")) == "gpqa":
            capability_name = str(_first(row, "model", "model_id", "name"))
            key = normalized_name(capability_name)
            model_id = by_name.get(key)
            if model_id is None:
                model_id = capability_name
                model_ids.append(model_id)
                by_name[key] = model_id
            if model_id in fitted:
                raise ValueError(f"multiple fitted capabilities resolve to {model_id}")
            fitted[model_id] = row
            if key != normalized_name(model_id):
                aliases[model_id].append(capability_name)

    rows = []
    for model_id in model_ids:
        override = override_by_id.get(model_id, {})
        metadata_names = [model_id, *aliases[model_id]]
        pool_name = override.get("pool_name")
        if isinstance(pool_name, str):
            metadata_names.insert(0, pool_name)
        metadata = next(
            (pool_by_name[normalized_name(name)] for name in metadata_names if normalized_name(name) in pool_by_name),
            {},
        )
        capability = fitted.get(model_id, {})
        row = {column: pd.NA for column in CATALOGUE_COLUMNS}
        row.update({
            "model_id": model_id,
            "display_name": _first(metadata, "display_name", "name") if metadata else model_id,
            "aliases": json.dumps(sorted(set(aliases[model_id])), separators=(",", ":")),
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
        for column in CATALOGUE_COLUMNS:
            if column not in {"model_id", "aliases"} and column in override:
                row[column] = (
                    _reasoning_class(override[column]) if column == "reasoning_class" else override[column]
                )
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


def pool_enabled(value) -> bool:
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
    catalogue_aliases(frame)
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
        correct, total = row.get("gpqa_correct", pd.NA), row.get("gpqa_total", pd.NA)
        if (_present(correct) or _present(total)) and not _valid_counts(correct, total):
            raise ValueError(f"GPQA correct/total counts must be finite integer values at row {index}")
        for column in ("price_in", "price_out"):
            raw_value = row.get(column, pd.NA)
            value = _number(raw_value)
            if _present(raw_value) and (not np.isfinite(value) or value < 0):
                raise ValueError(f"negative {column} at row {index}")
        enabled = any(pool_enabled(row[column]) for column in ("judge_pool", "candidate_pool") if column in row)
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
        posterior_name = next(
            (
                normalized_name(name)
                for name in (str(row["model_id"]), *_aliases(row.get("aliases", pd.NA)))
                if normalized_name(name) in posterior_by_name
            ),
            None,
        )
        if posterior_name is not None:
            values = theta[:, posterior_by_name[posterior_name]]
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
