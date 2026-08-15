"""Posterior-predictive judgment, usage, and cost simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from mc_cost.data import RELEASE_EPOCH
from mc_cost.model import FEATURE_COLUMNS, JudgmentData, PosteriorArtifact, left_logits
from mc_cost.tokens import draw_token_usage, draw_transcript_lengths


TRIAL_METRICS = (
    "usable_items", "judgment_calls", "candidate_calls", "choice_accuracy",
    "candidate_input_tokens", "candidate_output_tokens",
    "candidate_ordinary_output_tokens", "candidate_reasoning_tokens",
    "judge_input_tokens", "judge_output_tokens",
    "judge_ordinary_output_tokens", "judge_reasoning_tokens", "cost",
)
_TRIAL_COLUMNS = [
    "trial", "configuration", "mode", "comparison_id", "judge", "m1", "m2",
    "model", "reasoning_class", "role", "metric", "value", "warning_state",
]
_WARNING_COLUMNS = ["code", "trial", "row", "source_row", "model", "message"]


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
    missing_price = result[["price_in", "price_out"]].isna().any(axis=1)
    result["cost"] = (
        result["input_tokens"] * result["price_in"]
        + result["output_tokens"] * result["price_out"]
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
    groups = [column for column in trials if column not in {"trial", "value"}]
    values = trials.copy()
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    return values.groupby(groups, dropna=False, sort=False)["value"].agg(
        p5=lambda series: series.quantile(0.05),
        p50=lambda series: series.quantile(0.5),
        p95=lambda series: series.quantile(0.95),
        n_trials="size",
    ).reset_index()


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


def _posterior_count(artifact: PosteriorArtifact | Mapping[str, Any]) -> int:
    posterior, _ = _artifact_parts(artifact)
    counts = [np.asarray(value).shape[0] for value in posterior.values() if np.asarray(value).ndim]
    if not counts or min(counts) < 1:
        raise ValueError("artifact has no posterior draws")
    return min(counts)


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
    required = {"model_id", "release", "reasoning_class"}
    if missing := required - set(catalogue):
        raise ValueError(f"catalogue lacks required columns: {', '.join(sorted(missing))}")
    if catalogue["model_id"].isna().any() or catalogue["model_id"].astype(str).duplicated().any():
        raise ValueError("catalogue model_id values must be unique and non-null")
    result = catalogue.copy()
    result["model_id"] = result["model_id"].astype(str)
    result["release_day"] = (pd.to_datetime(result["release"], errors="coerce") - RELEASE_EPOCH).dt.days
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
    irt: dict[str, np.ndarray], posterior_index: int, rng: np.random.Generator,
    candidate_artifact: Any, cache: LatentCache, trial: int,
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
    a = _draw_value(irt["a"], posterior_index, "IRT discrimination")
    b = _draw_value(irt["b"], posterior_index, "IRT difficulty")
    floor_values = np.asarray(irt["floor"])
    floor = _draw_value(floor_values, posterior_index, "IRT floor") if floor_values.ndim == 2 else floor_values
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
                visible = int(ref.reasoning_class == "reasoning")
            else:
                correct = int(sampled["correct"])
                response_chars = pd.to_numeric(pd.Series([sampled.get("response_chars", np.nan)]), errors="coerce").iloc[0]
                visible_value = sampled.get("visible_reasoning", ref.reasoning_class == "reasoning")
                visible = int(bool(visible_value)) if not pd.isna(visible_value) else int(ref.reasoning_class == "reasoning")
            rows.append({
                "model": ref.model, "item": item, "reasoning_class": ref.reasoning_class,
                "correct": correct, "response_chars": response_chars, "visible_reasoning": visible,
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
            candidate_artifact, prediction_rows, rng, posterior_index,
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


def _scenario_left_probabilities(
    artifact: PosteriorArtifact | Mapping[str, Any], presented: pd.DataFrame,
    metadata: pd.DataFrame, capability: dict[str, float], posterior_index: int,
    cache: LatentCache, trial: int,
) -> np.ndarray:
    if presented.empty:
        return np.empty(0)
    posterior, artifact_metadata = _artifact_parts(artifact)
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
        "self_comparison": (
            presented["left_model"].eq(presented["right_model"])
        ).astype(float).to_numpy(),
        "reasoning_difference": (
            presented["left_visible_reasoning"].to_numpy(dtype=float)
            - presented["right_visible_reasoning"].to_numpy(dtype=float)
        ),
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
    judge_artifact: Any, rng: np.random.Generator, posterior_index: int,
    cache: LatentCache, trial: int, reuse_responses: bool,
) -> pd.DataFrame:
    parts = []
    if not reuse_responses and not cells.empty:
        requests = cells.drop_duplicates(["model", "item", "reasoning_class"]).reset_index(drop=True)
        prediction_rows = _token_prediction_rows(candidate_artifact, requests)
        usage = draw_token_usage(
            candidate_artifact, prediction_rows, rng, posterior_index,
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
            judge_artifact, prediction_rows, rng, posterior_index,
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
        "model": pd.NA, "reasoning_class": pd.NA, "role": pd.NA, "warning_state": "ok",
    }
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, **context: Any) -> None:
        rows.append({**base, "metric": metric, "value": value, **context})

    usable_items = len(presented) / config.presentations
    candidate_calls = int(usage.loc[usage["role"].eq("candidate"), "calls"].sum()) if not usage.empty else 0
    add("usable_items", usable_items)
    add("judgment_calls", len(presented))
    add("candidate_calls", candidate_calls)
    def outcomes(frame: pd.DataFrame) -> tuple[float, int]:
        if frame.empty:
            return np.nan, 0
        tied = frame["m1_theta"].eq(frame["m2_theta"])
        correct = np.where(
            frame["m1_theta"].gt(frame["m2_theta"]), frame["chose_m1"],
            1 - frame["chose_m1"],
        )
        return (float(np.mean(correct[~tied])) if (~tied).any() else np.nan, int(tied.sum()))

    accuracy, ties = outcomes(presented)
    add("choice_accuracy", accuracy)
    add("ties", ties)
    for source in comparisons.itertuples(index=False):
        group = presented.loc[presented["comparison_id"].eq(source.comparison_id)]
        accuracy, ties = outcomes(group)
        context = {
            "comparison_id": source.comparison_id, "judge": source.judge,
            "m1": source.m1, "m2": source.m2,
        }
        add("usable_items", len(group) / config.presentations, **context)
        add("judgment_calls", len(group), **context)
        add("choice_accuracy", accuracy, **context)
        add("ties", ties, **context)
        if config.mode == "manual":
            value = float(group["p_choose_m1"].mean()) if not group.empty else np.nan
            add(
                "p_choose_m1", value, **context,
                warning_state="ok" if np.isfinite(value) else "unavailable",
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
        add(
            "cost", record["cost"], model=record["model"], role=record["role"],
            reasoning_class=record["reasoning_class"],
            warning_state="missing_price" if record["model"] in missing_prices else "ok",
        )
    total_cost = priced["cost"].sum() if len(priced) and not priced["cost"].isna().any() else (0.0 if priced.empty else np.nan)
    add(
        "cost", total_cost, model="__total__", role="all",
        warning_state="missing_price" if missing_prices else "ok",
    )
    return rows


def simulate(
    comparisons: pd.DataFrame | Callable[[np.random.Generator], Any],
    config: SimulationConfig, *, item_ids: Sequence[str], observed_cells: pd.DataFrame,
    capability_models: Sequence[str], capability_draws: np.ndarray, irt: Mapping[str, Any],
    judgment_artifact: PosteriorArtifact | Mapping[str, Any],
    candidate_token_artifact: PosteriorArtifact | Mapping[str, Any],
    judge_token_artifact: PosteriorArtifact | Mapping[str, Any], catalogue: pd.DataFrame,
    seed: int = 0,
) -> SimulationResult:
    """Run fixed-budget posterior-predictive trials with shared latent effects."""
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
    posterior_draws = min(
        len(draws), _posterior_count(judgment_artifact),
        _posterior_count(candidate_token_artifact), _posterior_count(judge_token_artifact),
        len(np.asarray(irt_parts["a"])),
    )
    rng = np.random.default_rng(seed)
    cache = LatentCache(rng)
    trial_records: list[dict[str, Any]] = []
    warning_frames: list[pd.DataFrame] = []
    comparison_count: int | None = None

    for trial in range(config.trials):
        posterior_index = trial % posterior_draws
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
        predictable = selected.loc[predictable_mask]
        required_models = set(predictable[["judge", "m1", "m2"]].to_numpy().ravel())
        if missing := sorted(required_models - set(models)):
            raise ValueError(f"capability draws lack models: {', '.join(missing)}")
        capability = dict(zip(models, draws[posterior_index]))
        attempted = rng.choice(np.asarray(items), config.attempted_items, replace=False).tolist()
        cells = _candidate_cells(
            predictable, attempted, metadata, observed, capability, irt_parts,
            posterior_index, rng, candidate_token_artifact, cache, trial,
        )
        usable = _usable_rows(predictable, attempted, cells)
        presented = _presentation_metadata(counterbalance(usable, config.presentations, rng), cells, metadata)
        probabilities = _scenario_left_probabilities(
            judgment_artifact, presented, metadata, capability, posterior_index, cache, trial,
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
            posterior_index, cache, trial, config.reuse_responses,
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
        "n_trials": config.trials, "exact": True, "warning_state": "ok",
    })
    summary = pd.concat([summary, pd.DataFrame([exact])], ignore_index=True)
    warnings = pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame(columns=_WARNING_COLUMNS)
    warnings = warnings.reindex(columns=_WARNING_COLUMNS)
    warnings = warnings.drop_duplicates(
        ["code", "row", "source_row", "model", "message"], keep="first", ignore_index=True,
    )
    return SimulationResult(trial_frame, summary, warnings)
