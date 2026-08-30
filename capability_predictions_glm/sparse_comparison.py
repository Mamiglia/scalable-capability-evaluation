"""Invariant sparse capability/judge coefficient comparison tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


VALUE = "effective"
TARGET_MEAN = "Q2-Q4 mean"
TARGETS = {"Q2", "Q3", "Q4"}
VARIANTS = {"raw", "pc"}, {"l1", "ridge"}


def _require(frame: pd.DataFrame, columns: set[str], label: object) -> None:
    if missing := sorted(columns - set(frame)):
        raise ValueError(f"{label} missing columns: {missing}")


def _finite(frame: pd.DataFrame, column: str, label: object) -> None:
    if not np.isfinite(frame[column].to_numpy(float)).all():
        raise ValueError(f"{label} contains non-finite {column}")


def backproject_pc(coefficients: pd.DataFrame, loadings: pd.DataFrame) -> pd.DataFrame:
    """Map one effective PC vector into its fold-local composite coordinates."""
    _require(coefficients, {"feature", VALUE}, "PC coefficients")
    _require(loadings, {"component", "feature", "loading"}, "PC loadings")
    coefficients = coefficients.assign(feature=coefficients.feature.astype(str))
    loadings = loadings.assign(
        component=loadings.component.astype(str), feature=loadings.feature.astype(str),
    )
    if coefficients.feature.duplicated().any():
        raise ValueError("PC coefficients contain duplicate components")
    if loadings.duplicated(["component", "feature"]).any():
        raise ValueError("PC loadings contain duplicate component/feature rows")
    _finite(coefficients, VALUE, "PC coefficients")
    _finite(loadings, "loading", "PC loadings")
    matrix = loadings.pivot(index="component", columns="feature", values="loading")
    if matrix.isna().any().any():
        raise ValueError("PC loadings do not contain a complete component/feature matrix")
    if set(coefficients.feature) != set(matrix.index):
        raise ValueError("PC coefficient and loading component inventory differs")
    beta = coefficients.set_index("feature")[VALUE]
    values = matrix.T @ beta.reindex(matrix.index).to_numpy(float)
    return pd.DataFrame({"feature": matrix.columns.astype(str), VALUE: values}).reset_index(drop=True)


def expand_composites(
    coefficients: pd.DataFrame,
    provenance: Mapping[str, object],
    benchmark: str,
) -> pd.DataFrame:
    """Expand one composite vector through benchmark-active original leaves."""
    _require(coefficients, {"feature", VALUE}, "composite coefficients")
    if coefficients.feature.duplicated().any():
        raise ValueError("composite coefficients contain duplicate features")
    _finite(coefficients, VALUE, "composite coefficients")
    composites = provenance.get("final_composites")
    source_blocks = provenance.get("source_blocks")
    if not isinstance(composites, list) or not isinstance(source_blocks, Mapping):
        raise ValueError("composite provenance needs final_composites and source_blocks")
    by_name = {}
    for composite in composites:
        if not isinstance(composite, Mapping):
            raise ValueError("final_composites entries must be mappings")
        name = str(composite.get("output_id", composite.get("output_name", "")))
        if not name or name in by_name:
            raise ValueError("composite provenance contains missing or duplicate output IDs")
        by_name[name] = composite
    values = coefficients.assign(feature=coefficients.feature.astype(str)).set_index("feature")[VALUE]
    if set(values.index) != set(by_name):
        raise ValueError("coefficient and final-composite inventory differs")

    expanded: dict[str, float] = {}
    for composite, coefficient in values.items():
        by_benchmark = by_name[composite].get("leaf_weights_by_benchmark")
        if not isinstance(by_benchmark, Mapping) or benchmark not in by_benchmark:
            raise ValueError(f"composite {composite!r} has no weights for benchmark {benchmark!r}")
        weights = by_benchmark[benchmark]
        if not isinstance(weights, Mapping):
            raise ValueError("benchmark leaf weights must be mappings")
        for leaf, weight in weights.items():
            value = float(weight)
            if not np.isfinite(value):
                raise ValueError("composite provenance contains non-finite leaf weights")
            expanded[str(leaf)] = expanded.get(str(leaf), 0.0) + value * float(coefficient)
    if missing := sorted(set(expanded) - set(map(str, source_blocks))):
        raise ValueError(f"composite provenance lacks source blocks for leaves: {missing}")
    return pd.DataFrame([
        {"feature": leaf, "source_block": str(source_blocks[leaf]), VALUE: value}
        for leaf, value in sorted(expanded.items())
    ], columns=["feature", "source_block", VALUE])


def _variant(directory: Path) -> tuple[str, str]:
    try:
        feature_space, penalty = directory.name.rsplit("-", 1)
    except ValueError as error:
        raise ValueError(f"invalid variant directory: {directory}") from error
    if feature_space not in VARIANTS[0] or penalty not in VARIANTS[1]:
        raise ValueError(f"invalid stable variant: {directory.name}")
    return feature_space, penalty


def _aggregate_files(root: Path, *, capability: bool) -> list[Path]:
    candidates = ([root / "coefficients.csv"] if root.name == "aggregate" else []) + list(
        root.glob("**/aggregate/coefficients.csv")
    )
    output = []
    for path in dict.fromkeys(candidates):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if capability == ({"model_fold", "target_quartile"} <= columns):
            output.append(path)
    if not output:
        raise FileNotFoundError(f"no {'capability' if capability else 'judge'} aggregates under {root}")
    return sorted(output)


def _scope(aggregate: Path) -> str:
    settings_path = aggregate / "settings.json"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text()).get("settings", {})
        if isinstance(settings, Mapping) and settings.get("scope"):
            return str(settings["scope"])
    return aggregate.parent.parent.name


def _vector(
    coefficients: pd.DataFrame,
    feature_space: str,
    loadings_path: Path,
    provenance: Mapping[str, object],
    benchmark: str,
) -> pd.DataFrame:
    values = coefficients[["feature", VALUE]].copy()
    if feature_space == "pc":
        if not loadings_path.is_file():
            raise FileNotFoundError(f"missing fold-local PC loadings: {loadings_path}")
        values = backproject_pc(values, pd.read_csv(loadings_path))
    return expand_composites(values, provenance, benchmark)


def _capability_coefficients(
    aggregate: Path, provenance: Mapping[str, object], feature_space: str,
) -> pd.DataFrame:
    coefficients = pd.read_csv(aggregate / "coefficients.csv", keep_default_na=False)
    _require(
        coefficients,
        {"model_fold", "target_quartile", "benchmark", "feature", VALUE},
        aggregate / "coefficients.csv",
    )
    if "specification" in coefficients:
        coefficients = coefficients[coefficients.specification == "full"]
    keys = ["model_fold", "target_quartile", "benchmark"]
    if coefficients.empty or coefficients.duplicated([*keys, "feature"]).any():
        raise ValueError("capability full coefficients are empty or duplicated")
    targets = coefficients.groupby(["model_fold", "benchmark"]).target_quartile.agg(set)
    if not targets.map(lambda values: values == TARGETS).all():
        raise ValueError("capability coefficients require exactly Q2, Q3, and Q4 per fold/benchmark")
    rows = []
    variant_dir = aggregate.parent
    for values, group in coefficients.groupby(keys, sort=True):
        model_fold, target, benchmark = map(str, values)
        loadings = variant_dir / "cells" / f"{model_fold}__{target}" / "loadings.csv"
        rows.append(_vector(group, feature_space, loadings, provenance, benchmark).assign(
            model_fold=model_fold, target_quartile=target, benchmark=benchmark,
        ))
    output = pd.concat(rows, ignore_index=True)
    averaged = output.groupby(
        ["model_fold", "benchmark", "feature", "source_block"], as_index=False, sort=True,
    )[VALUE].mean().assign(target_quartile=TARGET_MEAN)
    return pd.concat([output, averaged], ignore_index=True)


def _judge_coefficients(
    aggregate: Path, provenance: Mapping[str, object], feature_space: str,
) -> pd.DataFrame:
    coefficients = pd.read_csv(aggregate / "coefficients.csv", keep_default_na=False)
    _require(coefficients, {"fold", "judge", "benchmark", "feature", VALUE}, aggregate)
    keys = ["fold", "judge", "benchmark"]
    if coefficients.empty or coefficients.duplicated([*keys, "feature"]).any():
        raise ValueError("judge coefficients are empty or duplicated")
    rows = []
    variant_dir = aggregate.parent
    for values, group in coefficients.groupby(keys, sort=True):
        fold, judge, benchmark = map(str, values)
        try:
            model_fold, item_fold = fold.rsplit("__item-", 1)
            item_fold = int(item_fold)
        except ValueError as error:
            raise ValueError(f"invalid judge fold name: {fold!r}") from error
        rows.append(_vector(
            group, feature_space, variant_dir / fold / "loadings.csv", provenance, benchmark,
        ).assign(
            model_fold=model_fold, item_fold=item_fold, judge=judge, benchmark=benchmark,
        ))
    return pd.concat(rows, ignore_index=True)


POINT_KEYS = [
    "scope", "variant", "feature_space", "penalty", "model_fold", "item_fold",
    "target_quartile", "judge", "benchmark",
]


def _coefficient_tables(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    points = points.copy()
    capability = points.capability_coefficient.to_numpy(float)
    judge = points.judge_coefficient.to_numpy(float)
    points["sign_quadrant"] = np.select(
        [
            (capability > 0) & (judge > 0),
            (capability > 0) & (judge < 0),
            (capability < 0) & (judge > 0),
            (capability < 0) & (judge < 0),
        ],
        ["positive-positive", "positive-negative", "negative-positive", "negative-negative"],
        default="zero",
    )
    statistics, cosines, contributions = [], [], []
    for values, group in points.groupby(POINT_KEYS, sort=True):
        metadata = dict(zip(POINT_KEYS, values))
        cap = group.capability_coefficient.to_numpy(float)
        judged = group.judge_coefficient.to_numpy(float)
        norm = float(np.linalg.norm(cap) * np.linalg.norm(judged))
        cosine = float(np.dot(cap, judged) / norm) if norm else np.nan
        counts = group.sign_quadrant.value_counts()
        correlations = {
            method: (
                group.capability_coefficient.corr(group.judge_coefficient, method=method)
                if len(group) > 1
                and group.capability_coefficient.nunique() > 1
                and group.judge_coefficient.nunique() > 1
                else np.nan
            )
            for method in ("pearson", "spearman")
        }
        statistics.append(metadata | {
            **correlations,
            **{f"{name.replace('-', '_')}_count": int(counts.get(name, 0)) for name in (
                "positive-positive", "positive-negative", "negative-positive",
                "negative-negative", "zero",
            )},
            "feature_count": len(group),
        })
        cosines.append(metadata | {"cosine": cosine, "feature_count": len(group)})
        contribution = group[["feature", "source_block"]].copy()
        contribution["cosine_contribution"] = cap * judged / norm if norm else np.nan
        contributions.append(contribution.assign(**metadata))
    return {
        "coefficient_points": points,
        "coefficient_statistics": pd.DataFrame(statistics),
        "fold_cosine": pd.DataFrame(cosines),
        "leaf_cosine_contributions": pd.concat(contributions, ignore_index=True),
    }


def _prediction_files(aggregate: Path) -> list[Path]:
    index_path = aggregate / "artifact-index.csv"
    if index_path.is_file():
        index = pd.read_csv(index_path, keep_default_na=False)
        if {"specification", "predictions"} <= set(index):
            paths = index.loc[index.specification == "full", "predictions"].drop_duplicates()
            return [Path(path) if Path(path).is_absolute() else aggregate / path for path in paths]
    path = aggregate / "predictions.parquet"
    return [path] if path.is_file() else []


def _capability_predictions(aggregate: Path) -> pd.DataFrame:
    tables = [pd.read_parquet(path) for path in _prediction_files(aggregate)]
    if not tables:
        raise FileNotFoundError(f"no capability predictions under {aggregate}")
    output = pd.concat(tables, ignore_index=True)
    if "specification" in output:
        output = output[output.specification == "full"]
    _require(output, {
        "model_fold", "target_quartile", "bench", "item", "model_left", "model_right",
        "observed", "probability",
    }, "capability predictions")
    return output


def _pair_identity_columns(left: pd.DataFrame, right_columns: set[str]) -> list[str]:
    return ["model", *[
        name for name in ("epoch", "sample_epoch", "log", "content_hash")
        if all(f"{name}_{side}" in left and f"{name}_{side}" in right_columns
               for side in ("left", "right"))
    ]]


def _pair_keys(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    left = frame[[f"{name}_left" for name in names]].astype(str).agg("\x1f".join, axis=1)
    right = frame[[f"{name}_right" for name in names]].astype(str).agg("\x1f".join, axis=1)
    return frame.assign(
        _left=left, _pair_a=np.minimum(left, right), _pair_b=np.maximum(left, right),
    )


def _raw_pairs(matrix_root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    path = matrix_root / "pair.parquet"
    columns = set(pq.read_schema(path).names)
    names = _pair_identity_columns(predictions, columns)
    required = {"bench", "item", "judge", "pick_left", *{
        f"{name}_{side}" for name in names for side in ("left", "right")
    }}
    if missing := sorted(required - columns):
        raise ValueError(f"raw pair matrix missing columns: {missing}")
    optional = [name for name in ("delta_eci", "eci_difference") if name in columns]
    return pd.read_parquet(path, columns=sorted(required) + optional)


def _accuracy_table(
    aggregate: Path, matrix_root: Path, metadata: dict[str, str],
) -> pd.DataFrame:
    predictions = _capability_predictions(aggregate)
    raw = _raw_pairs(matrix_root, predictions)
    names = _pair_identity_columns(predictions, set(raw))
    predictions = _pair_keys(predictions, names)
    raw = _pair_keys(raw, names)
    keys = ["bench", "item", "_pair_a", "_pair_b"]
    matched = predictions.merge(
        raw,
        on=keys, suffixes=("_capability", "_judge"), validate="many_to_many",
    )
    if matched.empty:
        raise ValueError("capability predictions do not match raw judge pairs")
    observed = matched.observed.to_numpy(int)
    matched["capability_correct"] = (
        (matched.probability.to_numpy(float) >= 0.5) == observed
    ).astype(float)
    aligned_pick = np.where(
        matched._left_capability == matched._left_judge,
        matched.pick_left.to_numpy(float), 1 - matched.pick_left.to_numpy(float),
    )
    matched["judge_correct"] = ((aligned_pick >= 0.5) == observed).astype(float)
    eci = next((name for name in ("delta_eci", "eci_difference") if name in matched), None)
    matched["abs_delta_eci"] = matched[eci].abs() if eci else np.nan
    matched = pd.concat([
        matched.assign(benchmark=matched.bench.astype(str)),
        matched.assign(benchmark="all"),
    ], ignore_index=True)
    keys = ["model_fold", "target_quartile", "judge", "benchmark"]
    specific = matched.groupby(keys, as_index=False, sort=True).agg(
        capability_accuracy=("capability_correct", "mean"),
        judge_accuracy=("judge_correct", "mean"),
        mean_abs_delta_eci=("abs_delta_eci", "mean"), pair_count=("judge_correct", "size"),
    )
    averaged = matched.groupby(
        ["model_fold", "judge", "benchmark"], as_index=False, sort=True,
    ).agg(
        capability_accuracy=("capability_correct", "mean"),
        judge_accuracy=("judge_correct", "mean"),
        mean_abs_delta_eci=("abs_delta_eci", "mean"), pair_count=("judge_correct", "size"),
    ).assign(target_quartile=TARGET_MEAN)
    output = pd.concat([specific, averaged], ignore_index=True).assign(**metadata)
    output["capability_advantage"] = output.capability_accuracy - output.judge_accuracy
    return output[[
        "scope", "variant", "feature_space", "penalty", "model_fold",
        "target_quartile", "judge", "benchmark", "capability_accuracy",
        "judge_accuracy", "capability_advantage", "mean_abs_delta_eci", "pair_count",
    ]]


def build_variant_comparison(
    capability_root: str | Path,
    judge_root: str | Path,
    matrix_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Build invariant original-leaf comparison tables for every matched variant."""
    capability_root, judge_root, matrix_root = map(Path, (
        capability_root, judge_root, matrix_root,
    ))
    provenance = json.loads((matrix_root / "composite-provenance.json").read_text())
    judge_files = {
        _variant(path.parent.parent): path.parent
        for path in _aggregate_files(judge_root, capability=False)
    }
    point_tables, accuracy_tables = [], []
    for path in _aggregate_files(capability_root, capability=True):
        aggregate = path.parent
        variant_dir = aggregate.parent
        feature_space, penalty = _variant(variant_dir)
        judge_aggregate = judge_files.get((feature_space, penalty))
        if judge_aggregate is None:
            raise FileNotFoundError(f"no judge aggregate for {variant_dir.name}")
        scope = _scope(aggregate)
        variant = variant_dir.name
        metadata = {
            "scope": scope, "variant": variant,
            "feature_space": feature_space, "penalty": penalty,
        }
        capability = _capability_coefficients(aggregate, provenance, feature_space)
        judge = _judge_coefficients(judge_aggregate, provenance, feature_space)
        capability_support = {
            values: set(group.feature)
            for values, group in capability.groupby(["model_fold", "benchmark"], sort=True)
        }
        judge_support = {
            values: set(group.feature)
            for values, group in judge.groupby(["model_fold", "benchmark"], sort=True)
        }
        if capability_support != judge_support:
            raise ValueError(f"judge/capability fold, benchmark, or leaf support differs for {variant}")
        points = capability.merge(
            judge,
            on=["model_fold", "benchmark", "feature", "source_block"],
            suffixes=("_capability", "_judge"), validate="many_to_many",
        ).rename(columns={
            f"{VALUE}_capability": "capability_coefficient",
            f"{VALUE}_judge": "judge_coefficient",
        })
        if points.empty:
            raise ValueError(f"no matched coefficient points for {scope}/{variant}")
        point_tables.append(points.assign(**metadata))
        accuracy_tables.append(_accuracy_table(aggregate, matrix_root, metadata))
    tables = _coefficient_tables(pd.concat(point_tables, ignore_index=True))
    accuracy = pd.concat(accuracy_tables, ignore_index=True)
    tables["accuracy"] = accuracy
    tables["cosine_accuracy_eci"] = tables["fold_cosine"].merge(
        accuracy,
        on=[
            "scope", "variant", "feature_space", "penalty", "model_fold",
            "target_quartile", "judge", "benchmark",
        ], validate="many_to_one",
    )
    return tables
