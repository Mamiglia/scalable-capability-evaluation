"""Hierarchical per-role token and candidate transcript-length models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from scipy.special import expit, logit

from mc_cost.model import PosteriorArtifact


TokenArtifact = PosteriorArtifact
_CLASSES = {"reasoning", "non_reasoning"}


class _TokenData(NamedTuple):
    model: Any
    model_class: Any
    input_x: Any
    output_x: Any
    log_input: Any
    log_output: Any
    reasoning_rows: Any
    reasoning_target: Any
    length_x: Any
    log_length: Any
    num_models: int
    num_classes: int
    has_length: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _source_hashes(source_hashes: Mapping[str, str] | None) -> dict[str, str]:
    if not source_hashes or any(
        not isinstance(key, str) or not key.strip()
        or not isinstance(value, str) or not value.strip()
        for key, value in source_hashes.items()
    ):
        raise ValueError("source hashes must be a nonempty mapping")
    return dict(source_hashes)


def _reasoning_map(source: Mapping[str, str] | pd.DataFrame | None) -> dict[str, str]:
    if source is None:
        return {}
    if isinstance(source, pd.DataFrame):
        required = {"model_id", "reasoning_class"}
        if missing := required - set(source):
            raise ValueError(f"reasoning catalogue lacks columns: {', '.join(sorted(missing))}")
        if source["model_id"].astype(str).duplicated().any():
            raise ValueError("duplicate model_id in reasoning catalogue")
        result = dict(zip(source["model_id"].astype(str), source["reasoning_class"]))
    else:
        result = {str(model): value for model, value in source.items()}
    invalid = sorted(model for model, value in result.items() if value not in _CLASSES)
    if invalid:
        raise ValueError(f"invalid reasoning class for models: {', '.join(invalid)}")
    return result


def _numeric(frame: pd.DataFrame, column: str, *, counts: bool = False) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"token rows lack required column: {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric values in {column}") from None
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {column}")
    if counts and ((values < 0).any() or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{column} must contain non-negative integer counts")
    return values


def _binary(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric(frame, column)
    if not set(values).issubset({0.0, 1.0}):
        raise ValueError(f"{column} must contain only 0/1 values")
    return values


def _standardized(
    frame: pd.DataFrame, feature: str, source: str, *, transform: str = "identity",
) -> tuple[np.ndarray, dict[str, Any]]:
    values = _numeric(frame, source)
    if transform == "log1p":
        if (values < 0).any():
            raise ValueError(f"{source} must be non-negative")
        values = np.log1p(values)
    mean, scale = float(values.mean()), float(values.std()) or 1.0
    return (values - mean) / scale, {
        "source": source, "transform": transform, "mean": mean, "scale": scale,
    }


def _prepare_frame(
    frame: pd.DataFrame, role: str,
    reasoning_classes: Mapping[str, str] | pd.DataFrame | None,
) -> pd.DataFrame:
    if role not in {"candidate", "judge"}:
        raise ValueError("role must be candidate or judge")
    if frame.empty:
        raise ValueError("token rows must contain at least one row")
    if "model" not in frame:
        raise ValueError("token rows lack required column: model")
    result = frame.copy()
    result["model"] = result["model"].astype(str)
    if (result["model"].str.strip() == "").any():
        raise ValueError("model IDs must be nonempty")
    if "role" in result and not result["role"].astype(str).eq(role).all():
        raise ValueError(f"token rows contain a role other than {role}")

    catalogue = _reasoning_map(reasoning_classes)
    models = sorted(result["model"].unique())
    if role == "judge":
        if missing := sorted(set(models) - set(catalogue)):
            raise ValueError(f"judge models missing reasoning class: {', '.join(missing)}")
        mapped = result["model"].map(catalogue)
        if "reasoning_class" in result:
            supplied = result["reasoning_class"]
            conflict = supplied.notna() & supplied.ne(mapped)
            if conflict.any():
                names = sorted(result.loc[conflict, "model"].unique())
                raise ValueError(f"judge reasoning class conflicts with catalogue: {', '.join(names)}")
        result["reasoning_class"] = mapped
    else:
        if "reasoning_class" not in result:
            result["reasoning_class"] = result["model"].map(catalogue)
        elif catalogue:
            result["reasoning_class"] = result["reasoning_class"].fillna(result["model"].map(catalogue))
        missing = sorted(result.loc[result["reasoning_class"].isna(), "model"].unique())
        if missing:
            raise ValueError(f"candidate models missing reasoning class: {', '.join(missing)}")

    invalid = sorted(result.loc[~result["reasoning_class"].isin(_CLASSES), "model"].unique())
    if invalid:
        raise ValueError(f"invalid reasoning class for models: {', '.join(invalid)}")
    counts = result.groupby("model")["reasoning_class"].nunique()
    if (counts > 1).any():
        raise ValueError(f"models have inconsistent reasoning classes: {', '.join(counts[counts > 1].index)}")
    return result


def _feature_data(
    frame: pd.DataFrame, role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str], dict[str, Any]]:
    scales: dict[str, Any] = {}
    base_features: list[str] = []
    base_columns: list[np.ndarray] = []
    if role == "candidate" or "difficulty" in frame:
        values, scales["difficulty_sd"] = _standardized(frame, "difficulty_sd", "difficulty")
        base_features.append("difficulty_sd")
        base_columns.append(values)
    if role == "candidate" or "correct" in frame:
        base_features.append("correct")
        base_columns.append(_binary(frame, "correct"))

    output_features = list(base_features)
    output_x = np.column_stack(base_columns) if base_columns else np.empty((len(frame), 0))
    input_features = list(base_features)
    input_columns = list(base_columns)
    if role == "judge":
        values, scales["transcript_chars_log_sd"] = _standardized(
            frame, "transcript_chars_log_sd", "transcript_chars", transform="log1p",
        )
        input_features.append("transcript_chars_log_sd")
        input_columns.append(values)
    input_x = np.column_stack(input_columns) if input_columns else np.empty((len(frame), 0))

    length_features: list[str] = []
    length_columns: list[np.ndarray] = []
    if role == "candidate":
        for feature, source in (
            ("theta_sd", "theta"), ("release_day_sd", "release_day"),
            ("difficulty_sd", "difficulty"),
        ):
            values, scale = _standardized(frame, feature, source)
            scales[feature] = scale
            length_features.append(feature)
            length_columns.append(values)
        length_features.extend(["correct", "visible_reasoning"])
        length_columns.extend([_binary(frame, "correct"), _binary(frame, "visible_reasoning")])
    length_x = np.column_stack(length_columns) if length_columns else np.empty((len(frame), 0))
    return input_x, output_x, length_x, input_features, output_features, length_features, scales


def _token_model(data: _TokenData) -> None:
    class_input_mean = numpyro.sample(
        "class_input_mean", dist.Normal(5, 2).expand([data.num_classes]).to_event(1),
    )
    class_output_mean = numpyro.sample(
        "class_output_mean", dist.Normal(4, 2).expand([data.num_classes]).to_event(1),
    )
    input_model_scale = numpyro.sample("input_model_scale", dist.HalfNormal(1))
    output_model_scale = numpyro.sample("output_model_scale", dist.HalfNormal(1))
    input_raw = numpyro.sample(
        "input_model_raw", dist.Normal(0, 1).expand([data.num_models]).to_event(1),
    )
    output_raw = numpyro.sample(
        "output_model_raw", dist.Normal(0, 1).expand([data.num_models]).to_event(1),
    )
    input_mean = class_input_mean[data.model_class] + input_model_scale * input_raw
    output_mean = class_output_mean[data.model_class] + output_model_scale * output_raw
    numpyro.deterministic("input_mean", input_mean)
    numpyro.deterministic("output_mean", output_mean)
    if data.input_x.shape[1]:
        input_beta = numpyro.sample(
            "input_beta", dist.Normal(0, 0.5).expand([data.input_x.shape[1]]).to_event(1),
        )
        input_mu = input_mean[data.model] + jnp.sum(input_beta * data.input_x, axis=1)
    else:
        input_mu = input_mean[data.model]
    if data.output_x.shape[1]:
        output_beta = numpyro.sample(
            "output_beta", dist.Normal(0, 0.5).expand([data.output_x.shape[1]]).to_event(1),
        )
        output_mu = output_mean[data.model] + jnp.sum(output_beta * data.output_x, axis=1)
    else:
        output_mu = output_mean[data.model]
    input_sigma = numpyro.sample("input_sigma", dist.HalfNormal(1))
    output_sigma = numpyro.sample("output_sigma", dist.HalfNormal(1))
    numpyro.sample("input_obs", dist.Normal(input_mu, input_sigma), obs=data.log_input)
    numpyro.sample("output_obs", dist.Normal(output_mu, output_sigma), obs=data.log_output)

    if data.reasoning_rows.shape[0]:
        class_reasoning_logit = numpyro.sample(
            "class_reasoning_logit", dist.Normal(0, 2).expand([data.num_classes]).to_event(1),
        )
        reasoning_model_scale = numpyro.sample("reasoning_model_scale", dist.HalfNormal(1))
        reasoning_raw = numpyro.sample(
            "reasoning_model_raw", dist.Normal(0, 1).expand([data.num_models]).to_event(1),
        )
        reasoning_logit = class_reasoning_logit[data.model_class] + reasoning_model_scale * reasoning_raw
        numpyro.deterministic("reasoning_logit", reasoning_logit)
        reasoning_sigma = numpyro.sample("reasoning_sigma", dist.HalfNormal(1))
        numpyro.sample(
            "reasoning_obs",
            dist.Normal(reasoning_logit[data.model[data.reasoning_rows]], reasoning_sigma),
            obs=data.reasoning_target,
        )

    if data.has_length:
        length_class_mean = numpyro.sample(
            "length_class_mean", dist.Normal(5, 2).expand([data.num_classes]).to_event(1),
        )
        length_model_scale = numpyro.sample("length_model_scale", dist.HalfNormal(1))
        length_raw = numpyro.sample(
            "length_model_raw", dist.Normal(0, 1).expand([data.num_models]).to_event(1),
        )
        length_mean = length_class_mean[data.model_class] + length_model_scale * length_raw
        numpyro.deterministic("length_mean", length_mean)
        length_beta = numpyro.sample(
            "length_beta", dist.Normal(0, 0.5).expand([data.length_x.shape[1]]).to_event(1),
        )
        length_sigma = numpyro.sample("length_sigma", dist.HalfNormal(1))
        length_mu = length_mean[data.model] + jnp.sum(length_beta * data.length_x, axis=1)
        numpyro.sample("length_obs", dist.Normal(length_mu, length_sigma), obs=data.log_length)


def _coverage(actual: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> float:
    edge = 1.6448536269514722 * sigma[:, None]
    lower = np.quantile(mean - edge, 0.05, axis=0)
    upper = np.quantile(mean + edge, 0.95, axis=0)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def _coverage_90(
    posterior: dict[str, np.ndarray], data: _TokenData,
) -> dict[str, float]:
    model = np.asarray(data.model)
    input_mean = posterior["input_mean"][:, model]
    output_mean = posterior["output_mean"][:, model]
    if data.input_x.shape[1]:
        input_mean += posterior["input_beta"] @ np.asarray(data.input_x).T
    if data.output_x.shape[1]:
        output_mean += posterior["output_beta"] @ np.asarray(data.output_x).T
    result = {
        "input": _coverage(np.asarray(data.log_input), input_mean, posterior["input_sigma"]),
        "output": _coverage(np.asarray(data.log_output), output_mean, posterior["output_sigma"]),
    }
    reasoning_rows = np.asarray(data.reasoning_rows)
    if reasoning_rows.size:
        reasoning_mean = posterior["reasoning_logit"][:, model[reasoning_rows]]
        result["reasoning"] = _coverage(
            np.asarray(data.reasoning_target), reasoning_mean, posterior["reasoning_sigma"],
        )
    if data.has_length:
        length_mean = (
            posterior["length_mean"][:, model]
            + posterior["length_beta"] @ np.asarray(data.length_x).T
        )
        result["length"] = _coverage(
            np.asarray(data.log_length), length_mean, posterior["length_sigma"],
        )
    return result


def fit_token_model(
    frame: pd.DataFrame, role: str, *,
    reasoning_classes: Mapping[str, str] | pd.DataFrame | None = None,
    source_hashes: Mapping[str, str] | None = None,
    warmup: int = 1000, samples: int = 1000, chains: int = 1, seed: int = 0,
) -> TokenArtifact:
    """Fit the approved role-specific hierarchy and return a portable artifact."""
    hashes = _source_hashes(source_hashes)
    frame = _prepare_frame(frame, role, reasoning_classes)
    input_tokens = _numeric(frame, "input_tokens", counts=True)
    output_tokens = _numeric(frame, "output_tokens", counts=True)
    reasoning_tokens = _numeric(frame, "reasoning_tokens", counts=True)
    if (reasoning_tokens > output_tokens).any():
        raise ValueError("reasoning_tokens must not exceed output_tokens")
    non_reasoning = frame["reasoning_class"].eq("non_reasoning").to_numpy()
    if (reasoning_tokens[non_reasoning] != 0).any():
        raise ValueError("non_reasoning rows must have zero reasoning_tokens")
    if "ordinary_output_tokens" in frame:
        ordinary = _numeric(frame, "ordinary_output_tokens", counts=True)
        if not np.array_equal(ordinary, output_tokens - reasoning_tokens):
            raise ValueError("ordinary_output_tokens plus reasoning_tokens must equal output_tokens")

    models = sorted(frame["model"].unique())
    classes = sorted(frame["reasoning_class"].unique())
    model_map = {model: index for index, model in enumerate(models)}
    class_map = {name: index for index, name in enumerate(classes)}
    model_class_names = (
        frame.drop_duplicates("model").set_index("model")["reasoning_class"].reindex(models).tolist()
    )
    model = frame["model"].map(model_map).to_numpy(dtype=int)
    model_class = np.array([class_map[name] for name in model_class_names], dtype=int)
    input_x, output_x, length_x, input_features, output_features, length_features, scales = (
        _feature_data(frame, role)
    )
    reasoning_rows = np.flatnonzero(frame["reasoning_class"].eq("reasoning").to_numpy())
    reasoning_target = logit(
        (reasoning_tokens[reasoning_rows] + 0.5) / (output_tokens[reasoning_rows] + 1.0),
    )
    if role == "candidate":
        response_chars = _numeric(frame, "response_chars", counts=True)
        log_length = np.log1p(response_chars)
    else:
        log_length = np.empty(0)

    data = _TokenData(
        model=jnp.asarray(model), model_class=jnp.asarray(model_class),
        input_x=jnp.asarray(input_x), output_x=jnp.asarray(output_x),
        log_input=jnp.asarray(np.log1p(input_tokens)), log_output=jnp.asarray(np.log1p(output_tokens)),
        reasoning_rows=jnp.asarray(reasoning_rows), reasoning_target=jnp.asarray(reasoning_target),
        length_x=jnp.asarray(length_x), log_length=jnp.asarray(log_length),
        num_models=len(models), num_classes=len(classes), has_length=role == "candidate",
    )
    mcmc = MCMC(
        NUTS(_token_model), num_warmup=warmup, num_samples=samples,
        num_chains=chains, progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    posterior = {name: np.asarray(values) for name, values in mcmc.get_samples().items()}
    diverging = np.asarray(mcmc.get_extra_fields()["diverging"], dtype=bool)
    diagnostics = {
        "divergences": int(diverging.sum()), "draws": int(diverging.size),
        "warmup": warmup, "samples": samples, "chains": chains, "seed": seed,
        "summary": _jsonable(summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)),
    }
    metadata = {
        "role": role, "models": models, "reasoning_classes": classes,
        "model_reasoning_classes": model_class_names,
        "feature_order": {
            "input": input_features, "output": output_features, "length": length_features,
        },
        "input_features": input_features, "output_features": output_features,
        "length_features": length_features, "training_scales": scales,
        "training_ranges": {
            spec["source"]: [
                float(_numeric(frame, spec["source"]).min()),
                float(_numeric(frame, spec["source"]).max()),
            ]
            for spec in scales.values()
        },
        "source_hashes": hashes, "coverage_90": _coverage_90(posterior, data),
    }
    return TokenArtifact(posterior, metadata, diagnostics)


def _artifact_parts(artifact: TokenArtifact | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(artifact, PosteriorArtifact):
        return artifact.posterior, artifact.metadata
    if not isinstance(artifact, Mapping) or "posterior" not in artifact:
        raise TypeError("artifact must be a TokenArtifact or mapping with posterior arrays")
    return dict(artifact["posterior"]), {key: value for key, value in artifact.items() if key != "posterior"}


def _posterior_value(
    posterior: Mapping[str, Any], name: str, posterior_index: int, default: Any = None,
) -> np.ndarray:
    if name not in posterior:
        if default is None:
            raise ValueError(f"token artifact lacks posterior array: {name}")
        return np.asarray(default)
    values = np.asarray(posterior[name])
    if values.ndim == 0 or posterior_index < 0 or posterior_index >= values.shape[0]:
        raise IndexError(f"posterior_index {posterior_index} is out of range for {name}")
    return np.asarray(values[posterior_index])


def _prediction_rows(rows: pd.DataFrame, metadata: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    missing = {"model", "reasoning_class"} - set(rows)
    if missing:
        raise ValueError(f"prediction rows lack required columns: {', '.join(sorted(missing))}")
    models = rows["model"].astype(str).tolist()
    classes = rows["reasoning_class"].astype(str).tolist()
    allowed = list(metadata.get("reasoning_classes", []))
    invalid = sorted(set(classes) - set(allowed))
    if invalid:
        raise ValueError(f"reasoning classes absent from artifact: {', '.join(invalid)}")
    pairs = pd.DataFrame({"model": models, "reasoning_class": classes}).drop_duplicates()
    if pairs["model"].duplicated().any():
        bad = sorted(pairs.loc[pairs["model"].duplicated(False), "model"].unique())
        raise ValueError(f"models have inconsistent reasoning classes: {', '.join(bad)}")
    known = list(metadata.get("models", []))
    known_classes = metadata.get("model_reasoning_classes")
    if known_classes is not None:
        expected = dict(zip(known, known_classes))
        conflicts = sorted({
            model for model, cls in zip(models, classes)
            if model in expected and expected[model] != cls
        })
        if conflicts:
            raise ValueError(f"reasoning class conflicts with fitted model: {', '.join(conflicts)}")
    return models, classes


def _prediction_features(rows: pd.DataFrame, features: list[str], metadata: Mapping[str, Any]) -> np.ndarray:
    columns = []
    scales = metadata.get("training_scales", metadata.get("feature_scales", {}))
    for feature in features:
        if feature in rows:
            columns.append(_numeric(rows, feature))
            continue
        if feature not in scales:
            raise ValueError(f"prediction rows lack required feature: {feature}")
        spec = scales[feature]
        values = _numeric(rows, spec["source"])
        if spec.get("transform") == "log1p":
            if (values < 0).any():
                raise ValueError(f"{spec['source']} must be non-negative")
            values = np.log1p(values)
        mean, scale = float(spec["mean"]), float(spec["scale"])
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid training scale for {feature}")
        columns.append((values - mean) / scale)
    return np.column_stack(columns) if columns else np.empty((len(rows), 0))


def _latent(
    cache: dict, key: tuple[Any, ...], rng: np.random.Generator, scale: float,
) -> float:
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("posterior model scales must be finite and non-negative")
    if key not in cache:
        cache[key] = float(rng.normal(0, scale))
    return float(cache[key])


def _count_from_log(values: np.ndarray, label: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite simulated {label}")
    with np.errstate(over="raise", invalid="raise"):
        try:
            counts = np.maximum(0, np.rint(np.expm1(values)))
        except FloatingPointError:
            raise ValueError(f"simulated {label} exceed integer range") from None
    if (counts > np.iinfo(np.int64).max).any():
        raise ValueError(f"simulated {label} exceed integer range")
    return counts.astype(np.int64)


def _means(
    posterior: Mapping[str, Any], metadata: Mapping[str, Any], models: list[str],
    classes: list[str], outcome: str, features: np.ndarray, rng: np.random.Generator,
    posterior_index: int, cache: dict,
) -> np.ndarray:
    known = list(metadata.get("models", []))
    class_names = list(metadata.get("reasoning_classes", []))
    known_mean = _posterior_value(posterior, f"{outcome}_mean", posterior_index, np.empty(0))
    class_mean = _posterior_value(posterior, f"class_{outcome}_mean", posterior_index, np.empty(0))
    scale = float(_posterior_value(posterior, f"{outcome}_model_scale", posterior_index, 0.0))
    means = np.empty(len(models))
    for row, (model, cls) in enumerate(zip(models, classes)):
        if model in known and known_mean.size:
            means[row] = known_mean[known.index(model)]
        else:
            if not class_mean.size:
                raise ValueError(f"token artifact lacks {outcome} population means for unseen model {model}")
            means[row] = class_mean[class_names.index(cls)] + _latent(
                cache, ("token", metadata.get("role"), posterior_index, outcome, model), rng, scale,
            )
    beta = _posterior_value(posterior, f"{outcome}_beta", posterior_index, np.zeros(features.shape[1]))
    if beta.shape != (features.shape[1],):
        raise ValueError(f"{outcome}_beta shape does not match feature order")
    return means + features @ beta


def draw_token_usage(
    artifact: TokenArtifact | Mapping[str, Any], rows: pd.DataFrame,
    rng: np.random.Generator, posterior_index: int, latent_cache: dict | None = None,
) -> pd.DataFrame:
    """Draw billed input/output and the reasoning subset for each request."""
    posterior, metadata = _artifact_parts(artifact)
    models, classes = _prediction_rows(rows, metadata)
    cache = {} if latent_cache is None else latent_cache
    input_features = _prediction_features(rows, list(metadata.get("input_features", [])), metadata)
    output_features = _prediction_features(rows, list(metadata.get("output_features", [])), metadata)
    log_input = _means(
        posterior, metadata, models, classes, "input", input_features,
        rng, posterior_index, cache,
    )
    log_output = _means(
        posterior, metadata, models, classes, "output", output_features,
        rng, posterior_index, cache,
    )
    input_sigma = float(_posterior_value(posterior, "input_sigma", posterior_index))
    output_sigma = float(_posterior_value(posterior, "output_sigma", posterior_index))
    if min(input_sigma, output_sigma) < 0 or not np.isfinite([input_sigma, output_sigma]).all():
        raise ValueError("posterior token sigmas must be finite and non-negative")
    input_tokens = _count_from_log(log_input + rng.normal(0, input_sigma, len(rows)), "input tokens")
    output_tokens = _count_from_log(log_output + rng.normal(0, output_sigma, len(rows)), "output tokens")

    reasoning_tokens = np.zeros(len(rows), dtype=np.int64)
    reasoning_rows = np.flatnonzero(np.asarray(classes) == "reasoning")
    if reasoning_rows.size:
        known = list(metadata.get("models", []))
        class_names = list(metadata.get("reasoning_classes", []))
        known_logit = _posterior_value(posterior, "reasoning_logit", posterior_index, np.empty(0))
        class_logit = _posterior_value(posterior, "class_reasoning_logit", posterior_index, np.empty(0))
        scale = float(_posterior_value(posterior, "reasoning_model_scale", posterior_index, 0.0))
        logits = np.empty(reasoning_rows.size)
        for target, row in enumerate(reasoning_rows):
            model, cls = models[row], classes[row]
            if model in known and known_logit.size:
                logits[target] = known_logit[known.index(model)]
            else:
                if not class_logit.size:
                    raise ValueError(f"token artifact lacks reasoning population means for unseen model {model}")
                logits[target] = class_logit[class_names.index(cls)] + _latent(
                    cache, ("token", metadata.get("role"), posterior_index, "reasoning", model), rng, scale,
                )
        sigma = float(_posterior_value(posterior, "reasoning_sigma", posterior_index, 0.0))
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("posterior reasoning sigma must be finite and non-negative")
        logits += rng.normal(0, sigma, reasoning_rows.size)
        reasoning_tokens[reasoning_rows] = np.minimum(
            output_tokens[reasoning_rows],
            np.rint(output_tokens[reasoning_rows] * expit(logits)).astype(np.int64),
        )
    return pd.DataFrame({
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "ordinary_output_tokens": output_tokens - reasoning_tokens,
    }, index=rows.index)


def draw_transcript_lengths(
    candidate_artifact: TokenArtifact | Mapping[str, Any], rows: pd.DataFrame,
    rng: np.random.Generator, posterior_index: int, latent_cache: dict | None = None,
) -> np.ndarray:
    """Draw candidate response lengths, reusing each unseen model effect per trial."""
    posterior, metadata = _artifact_parts(candidate_artifact)
    if metadata.get("role") != "candidate":
        raise ValueError("transcript lengths require a candidate token artifact")
    models, classes = _prediction_rows(rows, metadata)
    cache = {} if latent_cache is None else latent_cache
    features = _prediction_features(rows, list(metadata.get("length_features", [])), metadata)
    known = list(metadata.get("models", []))
    class_names = list(metadata.get("reasoning_classes", []))
    known_mean = _posterior_value(posterior, "length_mean", posterior_index, np.empty(0))
    class_mean = _posterior_value(posterior, "length_class_mean", posterior_index, np.empty(0))
    scale = float(_posterior_value(posterior, "length_model_scale", posterior_index, 0.0))
    means = np.empty(len(rows))
    for row, (model, cls) in enumerate(zip(models, classes)):
        if model in known and known_mean.size:
            means[row] = known_mean[known.index(model)]
        else:
            if not class_mean.size:
                raise ValueError(f"token artifact lacks length population means for unseen model {model}")
            means[row] = class_mean[class_names.index(cls)] + _latent(
                cache, ("length", posterior_index, model), rng, scale,
            )
    beta = _posterior_value(posterior, "length_beta", posterior_index, np.zeros(features.shape[1]))
    if beta.shape != (features.shape[1],):
        raise ValueError("length_beta shape does not match feature order")
    sigma = float(_posterior_value(posterior, "length_sigma", posterior_index))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("posterior length sigma must be finite and non-negative")
    return _count_from_log(means + features @ beta + rng.normal(0, sigma, len(rows)), "transcript lengths")
