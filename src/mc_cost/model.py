"""Hierarchical pairwise-judgment model and portable posterior artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from mc_cost.data import RELEASE_EPOCH


GLOBAL_FEATURES = (
    "oversight_sd", "release_gap_days_sd", "relative_release_days_sd",
    "difficulty_sd", "theta_gap_x_oversight_sd", "theta_gap_x_difficulty_sd",
    "same_family_x_theta_gap_sd", "self_comparison", "reasoning_difference_sd",
)
BASELINE_FEATURES = ()
CORE_FEATURES = tuple(
    feature for feature in GLOBAL_FEATURES
    if feature not in {"release_gap_days_sd", "relative_release_days_sd"}
)
CORE_RELEASE_FEATURES = GLOBAL_FEATURES
FEATURE_COLUMNS = {
    feature: feature.removesuffix("_sd") for feature in GLOBAL_FEATURES
}
_SANCTIONED_SPECS = {
    (BASELINE_FEATURES, False), (CORE_FEATURES, True), (CORE_RELEASE_FEATURES, True),
}


@dataclass(frozen=True)
class JudgmentSpec:
    global_features: tuple[str, ...]
    include_length: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_features", tuple(self.global_features))
        if (self.global_features, self.include_length) not in _SANCTIONED_SPECS:
            raise ValueError("judgment specification is not one of the three sanctioned models")


JUDGMENT_SPECS = {
    "baseline": JudgmentSpec(BASELINE_FEATURES, False),
    "core": JudgmentSpec(CORE_FEATURES, True),
    "core_release": JudgmentSpec(CORE_RELEASE_FEATURES, True),
}


class JudgmentData(NamedTuple):
    y: Any
    judge: Any
    item: Any
    left_model: Any
    right_model: Any
    theta_gap: Any
    global_x: Any
    length_difference: Any
    judge_covariates: Any
    include_length: bool
    num_judges: int
    num_items: int
    num_models: int
    global_features: tuple[str, ...] = ()
    category_maps: dict[str, dict[str, int]] | None = None
    training_scales: dict[str, Any] | None = None
    training_ranges: dict[str, list[float]] | None = None
    source_hashes: dict[str, str] | None = None


@dataclass(frozen=True)
class PosteriorArtifact:
    posterior: dict[str, np.ndarray]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


def _column(frame: pd.DataFrame, feature: str) -> str:
    column = FEATURE_COLUMNS[feature]
    if column not in frame:
        raise ValueError(f"judgments lack required standardized feature: {feature}")
    return column


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="index")
    return dict(value)


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {column}")
    return values


def _standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mean, scale = float(values.mean()), float(values.std()) or 1.0
    return (values - mean) / scale, {"mean": mean, "scale": scale}


def _required_scales(global_features: tuple[str, ...], include_length: bool) -> set[str]:
    required = {"theta_gap"}
    if include_length:
        required.add("length_difference")
    required.update(
        FEATURE_COLUMNS[feature] for feature in global_features if feature.endswith("_sd")
    )
    return required


def _validate_provenance(
    training_scales: Any, source_hashes: Any,
    global_features: tuple[str, ...], include_length: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    scales = _mapping(training_scales)
    hashes = _mapping(source_hashes)
    if not scales:
        raise ValueError("training scales must be a nonempty mapping")
    if not hashes or any(
        not isinstance(key, str) or not key.strip()
        or not isinstance(value, str) or not value.strip()
        for key, value in hashes.items()
    ):
        raise ValueError("source hashes must be a nonempty mapping")
    for column in sorted(_required_scales(global_features, include_length)):
        if column not in scales:
            raise ValueError(f"{column} scale metadata is required")
        try:
            mean, scale = float(scales[column]["mean"]), float(scales[column]["scale"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{column} scale metadata requires numeric mean/scale") from None
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{column} scale metadata requires finite mean and positive scale")
    return scales, {str(key): str(value) for key, value in hashes.items()}


def encode_judgments(
    frame: pd.DataFrame, catalogue: pd.DataFrame, spec: JudgmentSpec, *,
    training_scales: pd.DataFrame | dict[str, Any], source_hashes: dict[str, str],
) -> JudgmentData:
    """Encode prepared rows, preserving the specification's feature order."""
    required = {"chosen_left", "judge", "item", "left_model", "right_model", "theta_gap"}
    if spec.include_length:
        required.add("length_difference")
    missing = required - set(frame)
    if missing:
        raise ValueError(f"judgments lack required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("judgments must contain at least one row")
    training_scales, source_hashes = _validate_provenance(
        training_scales, source_hashes, spec.global_features, spec.include_length,
    )

    labels = {
        "judges": sorted(frame["judge"].astype(str).unique()),
        "items": sorted(frame["item"].astype(str).unique()),
        "models": sorted(set(frame["left_model"].astype(str)) | set(frame["right_model"].astype(str))),
    }
    category_maps = {name: {label: index for index, label in enumerate(values)} for name, values in labels.items()}
    y = _finite(frame, "chosen_left")
    if not set(y).issubset({0.0, 1.0}):
        raise ValueError("chosen_left must contain only 0/1 values")
    theta_gap = _finite(frame, "theta_gap")
    feature_columns = [_column(frame, feature) for feature in spec.global_features]
    global_x = (
        np.column_stack([_finite(frame, column) for column in feature_columns])
        if feature_columns else np.empty((len(frame), 0), dtype=float)
    )
    length_difference = (
        _finite(frame, "length_difference") if spec.include_length else np.zeros(len(frame), dtype=float)
    )

    catalogue_required = {"model_id", "theta", "release"}
    if missing_catalogue := catalogue_required - set(catalogue):
        raise ValueError(f"catalogue lacks required columns: {', '.join(sorted(missing_catalogue))}")
    metadata = catalogue.copy()
    metadata["model_id"] = metadata["model_id"].astype(str)
    if metadata["model_id"].duplicated().any():
        raise ValueError("duplicate model_id in catalogue")
    metadata = metadata.set_index("model_id")
    judge_rows = metadata.reindex(labels["judges"])
    if judge_rows[["theta", "release"]].isna().any(axis=None):
        absent = judge_rows.index[judge_rows[["theta", "release"]].isna().any(axis=1)].tolist()
        raise ValueError(f"judges lack theta/release metadata: {', '.join(absent)}")
    judge_theta = pd.to_numeric(judge_rows["theta"], errors="coerce").to_numpy(dtype=float)
    judge_release = (
        pd.to_datetime(judge_rows["release"], errors="coerce") - RELEASE_EPOCH
    ).dt.days.to_numpy(dtype=float)
    if not np.isfinite(judge_theta).all() or not np.isfinite(judge_release).all():
        raise ValueError("judges have invalid theta/release metadata")
    judge_theta_sd, judge_theta_scale = _standardize(judge_theta)
    judge_release_sd, judge_release_scale = _standardize(judge_release)
    judge_covariates = np.column_stack([judge_theta_sd, judge_release_sd])

    ranges = {"theta_gap": [float(theta_gap.min()), float(theta_gap.max())]}
    for feature, values in zip(spec.global_features, global_x.T):
        ranges[feature] = [float(values.min()), float(values.max())]
    if spec.include_length:
        ranges["length_difference"] = [float(length_difference.min()), float(length_difference.max())]
    ranges.update({
        "judge_theta": [float(judge_theta.min()), float(judge_theta.max())],
        "judge_release_days": [float(judge_release.min()), float(judge_release.max())],
    })

    training_scales.update({
        "judge_theta": judge_theta_scale, "judge_release_days": judge_release_scale,
    })
    return JudgmentData(
        y=jnp.asarray(y, dtype=jnp.int32),
        judge=jnp.asarray(frame["judge"].astype(str).map(category_maps["judges"]).to_numpy()),
        item=jnp.asarray(frame["item"].astype(str).map(category_maps["items"]).to_numpy()),
        left_model=jnp.asarray(frame["left_model"].astype(str).map(category_maps["models"]).to_numpy()),
        right_model=jnp.asarray(frame["right_model"].astype(str).map(category_maps["models"]).to_numpy()),
        theta_gap=jnp.asarray(theta_gap), global_x=jnp.asarray(global_x),
        length_difference=jnp.asarray(length_difference), judge_covariates=jnp.asarray(judge_covariates),
        include_length=spec.include_length, num_judges=len(labels["judges"]),
        num_items=len(labels["items"]), num_models=len(labels["models"]),
        global_features=spec.global_features, category_maps=category_maps,
        training_scales=training_scales, training_ranges=ranges,
        source_hashes=source_hashes,
    )


def judgment_model(data: JudgmentData, conditional_judges: bool = True) -> None:
    """Fixed partially pooled hierarchy for displayed-left choices."""
    covariates = data.judge_covariates if conditional_judges else data.judge_covariates[:, :0]
    judge_design = jnp.concatenate([jnp.ones((data.num_judges, 1)), covariates], axis=1)
    population = numpyro.sample(
        "judge_population", dist.Normal(0, 0.5).expand([2, judge_design.shape[1]]).to_event(2),
    )
    scales = numpyro.sample("judge_scales", dist.HalfNormal(0.5).expand([2]).to_event(1))
    corr = numpyro.sample("judge_corr", dist.LKJCholesky(2, 2.0))
    raw = numpyro.sample(
        "judge_raw", dist.Normal(0, 1).expand([data.num_judges, 2]).to_event(2),
    )
    judge_effects = judge_design @ population.T + (raw @ corr.T) * scales
    numpyro.deterministic("judge_effects", judge_effects)

    if data.global_x.shape[1]:
        global_beta = numpyro.sample(
            "global_beta", dist.Normal(0, 0.5).expand([data.global_x.shape[1]]).to_event(1),
        )
        global_term = jnp.sum(global_beta * data.global_x, axis=1)
    else:
        global_term = 0.0
    if data.include_length:
        length_mean = numpyro.sample("length_mean", dist.Normal(0, 0.5))
        length_scale = numpyro.sample("length_scale", dist.HalfNormal(0.5))
        length_raw = numpyro.sample(
            "length_raw", dist.Normal(0, 1).expand([data.num_judges]).to_event(1),
        )
        length_deviation = length_scale * length_raw
        numpyro.deterministic("length_deviation", length_deviation)
        length_term = (length_mean + length_deviation[data.judge]) * data.length_difference
    else:
        length_term = 0.0
    candidate_scale = numpyro.sample("candidate_scale", dist.HalfNormal(0.5))
    candidate_raw = numpyro.sample(
        "candidate_raw", dist.Normal(0, 1).expand([data.num_models]).to_event(1),
    )
    candidate = candidate_scale * (candidate_raw - candidate_raw.mean())
    numpyro.deterministic("candidate_effects", candidate)
    item_scale = numpyro.sample("item_scale", dist.HalfNormal(0.5))
    item = item_scale * numpyro.sample(
        "item_raw", dist.Normal(0, 1).expand([data.num_items]).to_event(1),
    )
    numpyro.deterministic("item_effects", item)

    logits = (
        judge_effects[data.judge, 0]
        + judge_effects[data.judge, 1] * data.theta_gap
        + global_term
        + length_term
        + candidate[data.left_model] - candidate[data.right_model]
        + item[data.item]
    )
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=data.y)


def _draw_array(posterior: dict[str, np.ndarray], name: str, event_dims: int) -> np.ndarray:
    values = np.asarray(posterior[name])
    if values.ndim == event_dims:
        values = values[None, ...]
    if values.ndim != event_dims + 1:
        raise ValueError(f"{name} has incompatible posterior shape {values.shape}")
    return values


def left_logits(posterior: dict[str, np.ndarray], data: JudgmentData) -> np.ndarray:
    """Return pure-NumPy displayed-left logits with shape ``(draw, row)``."""
    judge_effects = _draw_array(posterior, "judge_effects", 2)
    candidate_effects = _draw_array(posterior, "candidate_effects", 1)
    item_effects = _draw_array(posterior, "item_effects", 1)
    judge = np.asarray(data.judge)
    logits = (
        judge_effects[:, judge, 0]
        + judge_effects[:, judge, 1] * np.asarray(data.theta_gap)
        + candidate_effects[:, np.asarray(data.left_model)]
        - candidate_effects[:, np.asarray(data.right_model)]
        + item_effects[:, np.asarray(data.item)]
    )
    if data.global_x.shape[1]:
        logits = logits + _draw_array(posterior, "global_beta", 1) @ np.asarray(data.global_x).T
    if data.include_length:
        length_slope = (
            _draw_array(posterior, "length_mean", 0)[:, None]
            + _draw_array(posterior, "length_deviation", 1)[:, judge]
        )
        logits = logits + length_slope * np.asarray(data.length_difference)
    return np.asarray(logits)


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


def fit_judgment_model(
    data: JudgmentData, *, warmup: int = 1000, samples: int = 1000,
    chains: int = 1, seed: int = 0, conditional_judges: bool = True,
) -> PosteriorArtifact:
    """Fit with configured NUTS settings and retain reproducibility metadata."""
    _validate_provenance(
        data.training_scales, data.source_hashes, data.global_features, data.include_length,
    )
    mcmc = MCMC(
        NUTS(judgment_model), num_warmup=warmup, num_samples=samples,
        num_chains=chains, progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data, conditional_judges)
    posterior = {name: np.asarray(values) for name, values in mcmc.get_samples().items()}
    grouped = mcmc.get_samples(group_by_chain=True)
    diverging = np.asarray(mcmc.get_extra_fields()["diverging"], dtype=bool)
    diagnostics = {
        "divergences": int(diverging.sum()), "draws": int(diverging.size),
        "warmup": warmup, "samples": samples, "chains": chains, "seed": seed,
        "summary": _jsonable(summary(grouped, group_by_chain=True)),
    }
    metadata = {
        "feature_order": list(data.global_features),
        "category_maps": data.category_maps or {},
        "training_scales": data.training_scales or {},
        "training_ranges": data.training_ranges or {},
        "source_hashes": data.source_hashes or {},
        "conditional_judges": conditional_judges,
    }
    return PosteriorArtifact(posterior, metadata, diagnostics)


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    posterior_path = Path(path)
    if posterior_path.suffix != ".npz":
        posterior_path = posterior_path.with_suffix(".npz")
    return posterior_path, posterior_path.with_suffix(".json")


def save_artifact(artifact: PosteriorArtifact, path: str | Path) -> None:
    """Atomically save numeric/Unicode arrays and JSON-only metadata."""
    posterior_path, metadata_path = _artifact_paths(path)
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values) for name, values in artifact.posterior.items()}
    unsupported = [name for name, values in arrays.items() if values.dtype.kind not in "biufcUS"]
    if unsupported:
        raise TypeError(f"artifact arrays require pickle: {', '.join(sorted(unsupported))}")
    temporary_posterior = posterior_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_posterior, **arrays)
    temporary_posterior.replace(posterior_path)
    payload = {**artifact.metadata, "diagnostics": artifact.diagnostics}
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    temporary_metadata.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False))
    temporary_metadata.replace(metadata_path)


def load_artifact(path: str | Path) -> PosteriorArtifact:
    """Load a portable artifact without enabling NumPy pickle support."""
    posterior_path, metadata_path = _artifact_paths(path)
    with np.load(posterior_path, allow_pickle=False) as stored:
        posterior = {name: stored[name].copy() for name in stored.files}
    payload = json.loads(metadata_path.read_text())
    diagnostics = payload.pop("diagnostics")
    return PosteriorArtifact(posterior, payload, diagnostics)
