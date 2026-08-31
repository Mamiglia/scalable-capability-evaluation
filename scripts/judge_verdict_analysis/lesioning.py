"""Shared sparse feature, block, and family lesions."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from .config import AnalysisConfig, CONFIG
from .fit import ARTIFACTS as FIT_ARTIFACTS
from .fit import _guard_execution, _load_rows, _outer_folds, _positive_int, _resume_existing
from .fit_common import prepare_features
from .fit_frequentist import JUDGE_TASK, SparseTask, fit_sparse, tune_sparse
from .pooling import POOLING_PARAMETERS


@dataclass(frozen=True)
class Specification:
    name: str
    kind: str
    variable: str
    reference: str
    features: tuple[str, ...]


def _slug(value: str) -> str:
    return value.replace("_", "-").replace("/", "-")


def _source_blocks(row: object) -> tuple[str, ...]:
    value = getattr(row, "source_blocks", None)
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(block, str) for block in parsed):
            raise ValueError("source_blocks must be a JSON list of strings")
        return tuple(parsed)
    return (str(getattr(row, "block")),)


def build_specifications(
    dictionary: pd.DataFrame,
    config: AnalysisConfig = CONFIG,
) -> list[Specification]:
    """Build feature specs plus raw block/family alternatives from leaf blocks."""
    if missing := {"feature", "block"} - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    if dictionary.feature.duplicated().any():
        raise ValueError("feature dictionary contains duplicate features")
    full = tuple(dictionary.feature.astype(str))
    output = [
        specification
        for feature in full
        for specification in (
            Specification(
                f"lesion-feature-{_slug(feature)}", "lesion-feature", feature, "full",
                tuple(name for name in full if name != feature),
            ),
            Specification(
                f"singular-feature-{_slug(feature)}", "singular-feature", feature,
                "intercept-only", (feature,),
            ),
        )
    ]
    if config.feature_space == "pc":
        return output

    memberships = {
        str(row.feature): _source_blocks(row)
        for row in dictionary.itertuples(index=False)
    }
    blocks = list(dict.fromkeys(
        block for feature in full for block in memberships[feature]
    ))
    for block in blocks:
        members = tuple(feature for feature in full if block in memberships[feature])
        member_set = set(members)
        output.extend((
            Specification(
                f"lesion-block-{_slug(block)}", "lesion-block", block, "full",
                tuple(feature for feature in full if feature not in member_set),
            ),
            Specification(
                f"singular-block-{_slug(block)}", "singular-block", block,
                "intercept-only", members,
            ),
        ))
    for family, family_blocks in config.lesion_families:
        wanted = set(family_blocks)
        members = tuple(
            feature for feature in full if wanted.intersection(memberships[feature])
        )
        if not members:
            continue
        member_set = set(members)
        output.extend((
            Specification(
                "no-annotations" if family == "annotations" else f"lesion-family-{family}",
                "lesion-family", family, "full",
                tuple(feature for feature in full if feature not in member_set),
            ),
            Specification(
                f"singular-family-{family}", "singular-family", family,
                "intercept-only", members,
            ),
        ))
    return output


def reference_specifications(
    dictionary: pd.DataFrame,
) -> tuple[Specification, Specification]:
    full = tuple(dictionary.feature.astype(str))
    return (
        Specification("full", "reference", "full", "", full),
        Specification("intercept-only", "reference", "intercept-only", "", ()),
    )


def unique_specifications(
    specifications: list[Specification],
) -> tuple[list[Specification], dict[str, str]]:
    by_features: dict[tuple[str, ...], Specification] = {}
    aliases: dict[str, str] = {}
    for specification in specifications:
        canonical = by_features.setdefault(specification.features, specification)
        if canonical is not specification:
            aliases[specification.name] = canonical.name
    return list(by_features.values()), aliases


def prepare_lesions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dictionary: pd.DataFrame,
    config: AnalysisConfig,
    frozen_hyperparameters: dict[str, float | None] | None = None,
    tuning: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Prepare one fold and tune its full sparse model exactly once."""
    prepared_train, prepared_test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, config.feature_space, config,
    )
    features = prepared_dictionary.feature.astype(str).tolist()
    if frozen_hyperparameters is None:
        frozen_hyperparameters, tuning = tune_sparse(
            prepared_train, features, config.penalty, config,
        )
    if tuning is None:
        raise ValueError("prepared lesions require tuning metadata")
    specifications = build_specifications(prepared_dictionary, config)
    unique, aliases = unique_specifications(specifications)
    return {
        "train": prepared_train,
        "test": prepared_test,
        "dictionary": prepared_dictionary,
        "provenance": provenance,
        "frozen_hyperparameters": frozen_hyperparameters,
        "tuning": tuning,
        "references": reference_specifications(prepared_dictionary),
        "specifications": specifications,
        "unique_specifications": unique,
        "aliases": aliases,
    }


def _log_losses(observed: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probability = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return -(observed * np.log(probability) + (1 - observed) * np.log1p(-probability))


def _paired_predictions(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    specification: Specification,
    fold: str,
    backend: str = "sparse",
) -> pd.DataFrame:
    """Compare aligned held-out predictions with the declared reference."""
    required = {"observed", "probability"}
    if missing := required - set(candidate) | required - set(reference):
        raise ValueError(f"prediction tables missing columns: {sorted(missing)}")
    candidate, reference = candidate.reset_index(drop=True), reference.reset_index(drop=True)
    compared = reference.columns.difference(["probability"])
    if len(candidate) != len(reference) or any(
        column not in candidate or not candidate[column].equals(reference[column])
        for column in compared
    ):
        raise ValueError("candidate predictions are not row-aligned with their reference")
    observed = reference.observed.to_numpy(dtype=int)
    candidate_probability = candidate.probability.to_numpy(dtype=float)
    reference_probability = reference.probability.to_numpy(dtype=float)
    candidate_loss = _log_losses(observed, candidate_probability)
    reference_loss = _log_losses(observed, reference_probability)
    candidate_accuracy = (candidate_probability >= 0.5) == observed
    reference_accuracy = (reference_probability >= 0.5) == observed
    output = reference.drop(columns="probability").copy()
    output[["backend", "fold", "specification", "kind", "variable", "reference"]] = (
        backend, fold, specification.name, specification.kind,
        specification.variable, specification.reference,
    )
    output["reference_probability"] = reference_probability
    output["candidate_probability"] = candidate_probability
    if specification.reference == "full":
        output["delta_log_loss"] = candidate_loss - reference_loss
        output["delta_accuracy"] = reference_accuracy.astype(float) - candidate_accuracy.astype(float)
    elif specification.reference == "intercept-only":
        output["delta_log_loss"] = reference_loss - candidate_loss
        output["delta_accuracy"] = candidate_accuracy.astype(float) - reference_accuracy.astype(float)
    else:
        raise ValueError(f"unknown specification reference: {specification.reference}")
    return output


def _item_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    observed = predictions.observed.to_numpy(dtype=int)
    probability = predictions.probability.to_numpy(dtype=float)
    rows = predictions[["bench", "item"]].copy()
    rows["log_loss"] = _log_losses(observed, probability)
    rows["accuracy"] = ((probability >= 0.5) == observed).astype(float)
    return rows.groupby(["bench", "item"], sort=False, as_index=False).agg(
        log_loss=("log_loss", "mean"), accuracy=("accuracy", "mean"),
        row_count=("accuracy", "size"),
    )


def _paired_item_metrics(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    specification: Specification,
    fold: str,
    backend: str = "sparse",
) -> pd.DataFrame:
    merged = candidate.merge(
        reference, on=["bench", "item"], suffixes=("_candidate", "_reference"),
        validate="one_to_one", how="outer", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("candidate and reference item keys differ")
    output = merged[["bench", "item"]].copy()
    output[["backend", "fold", "specification", "kind", "variable", "reference"]] = (
        backend, fold, specification.name, specification.kind,
        specification.variable, specification.reference,
    )
    output["candidate_log_loss"] = merged.log_loss_candidate
    output["reference_log_loss"] = merged.log_loss_reference
    output["candidate_accuracy"] = merged.accuracy_candidate
    output["reference_accuracy"] = merged.accuracy_reference
    output["row_count"] = merged.row_count_candidate.astype(int)
    if specification.reference == "full":
        output["delta_log_loss"] = output.candidate_log_loss - output.reference_log_loss
        output["delta_accuracy"] = output.reference_accuracy - output.candidate_accuracy
    else:
        output["delta_log_loss"] = output.reference_log_loss - output.candidate_log_loss
        output["delta_accuracy"] = output.candidate_accuracy - output.reference_accuracy
    return output


def _bootstrap_summary(rows: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    metadata = ["backend", "specification", "kind", "variable", "reference"]
    if rows.empty:
        return pd.DataFrame(columns=[
            *metadata, "lesion_metric", "estimate", "lower", "upper",
            "delta_log_loss", "delta_accuracy", "fold_count", "row_count",
        ])
    metric = f"delta_{config.lesion_metric}"
    rng = np.random.default_rng(config.random_seed)
    summaries = []
    for keys, group in rows.groupby(metadata, sort=True):
        weights = group.row_count.to_numpy(dtype=int) if "row_count" in group else np.ones(len(group), int)
        estimates = {
            name: float(np.average(group[name].to_numpy(float), weights=weights))
            for name in ("delta_log_loss", "delta_accuracy")
        }
        clustered = group[["bench", "item"]].copy()
        clustered["count"] = weights
        for name in estimates:
            clustered[name] = group[name].to_numpy(float) * weights
        clustered = clustered.groupby(["bench", "item"], sort=False).sum()
        draws = rng.integers(len(clustered), size=(config.bootstrap_iterations, len(clustered)))
        counts = clustered["count"].to_numpy()[draws].sum(axis=1)
        samples = {
            name: clustered[name].to_numpy()[draws].sum(axis=1) / counts
            for name in estimates
        }
        intervals = {name: np.percentile(sample, [2.5, 97.5]) for name, sample in samples.items()}
        summaries.append({
            **dict(zip(metadata, keys)), "lesion_metric": config.lesion_metric,
            "estimate": estimates[metric], "lower": intervals[metric][0], "upper": intervals[metric][1],
            "delta_log_loss": estimates["delta_log_loss"],
            "delta_accuracy": estimates["delta_accuracy"],
            "fold_count": group.fold.nunique(), "row_count": int(weights.sum()),
        })
    return pd.DataFrame(summaries)


def _atomic(path: Path, write) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings_signature(values: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(
        values, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _config_settings(config: AnalysisConfig, fold_index: int | None = None) -> dict[str, object]:
    settings = {
        "feature_space": config.feature_space,
        "penalty": config.penalty,
        "fold_index": fold_index,
        "model_folds": [[name, list(models)] for name, models in config.model_folds],
        "item_folds": config.item_folds,
        "inner_folds": config.inner_folds,
        "elastic_net_cs": list(config.elastic_net_cs),
        "ridge_cs": list(config.ridge_cs),
        "pooling_svi_steps": config.pooling_svi_steps,
        "pooling_svi_learning_rate": config.pooling_svi_learning_rate,
        "pooling_scale_prior": config.pooling_scale_prior,
        "judge_deviation_penalty": config.judge_deviation_penalty,
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "cell_deviation_penalty": config.cell_deviation_penalty,
        "item_intercept_penalty": config.item_intercept_penalty,
        "glm_max_iter": config.glm_max_iter,
        "random_seed": config.random_seed,
        "source_manifest": config.processed_matrix_manifest_sha256,
    }
    if config.feature_space == "pc":
        settings.update({
            "fit_pca_components": config.fit_pca_components,
            "fit_pca_algorithm": config.fit_pca_algorithm,
            "pc_basis_root": str(config.pc_basis_root),
        })
    if config.lesion_source_signature:
        settings["lesion_source_signature"] = config.lesion_source_signature
    return settings


def _specification_table(plan: dict[str, object]) -> pd.DataFrame:
    aliases = plan["aliases"]
    return pd.DataFrame([
        {
            "specification": spec.name, "kind": spec.kind, "variable": spec.variable,
            "reference": spec.reference, "feature_count": len(spec.features),
            "artifact": aliases.get(spec.name, spec.name),
        }
        for spec in plan["specifications"]
    ])


def _runnable_table(plan: dict[str, object]) -> pd.DataFrame:
    runnable = [plan["references"][1], *plan["unique_specifications"]]
    return pd.DataFrame([
        {
            "spec_index": index, "specification": spec.name, "kind": spec.kind,
            "variable": spec.variable, "reference": spec.reference,
            "features": json.dumps(spec.features),
        }
        for index, spec in enumerate(runnable)
    ])


def _validate_prepared(fold_dir: Path, config: AnalysisConfig, fold_index: int) -> dict[str, object]:
    path = fold_dir / "prepare-manifest.json"
    if not path.is_file():
        raise RuntimeError(f"lesion fold is not prepared: {fold_dir}")
    manifest = json.loads(path.read_text())
    hashes = manifest.get("artifact_sha256")
    fold = _fold_names(config)[fold_index]
    fit_dir = config.fit_dir / f"{config.feature_space}-{config.penalty}" / fold
    _resume_existing(fit_dir, FIT_ARTIFACTS, replace(config, resume_completed_folds=True))
    fit_hash = _sha256(fit_dir / "manifest.json")
    if (
        manifest.get("state") != "prepared"
        or manifest.get("backend") != f"{config.feature_space}-{config.penalty}"
        or manifest.get("config_signature") != _settings_signature(
            _config_settings(config, fold_index))
        or manifest.get("fit_manifest_sha256") != fit_hash
        or not isinstance(hashes, dict)
        or any(not (fold_dir / name).is_file() or _sha256(fold_dir / name) != digest
               for name, digest in hashes.items())
    ):
        raise RuntimeError(f"invalid lesion preparation: {fold_dir}")
    return manifest


def prepare_fold(fold_index: int, config: AnalysisConfig = CONFIG) -> Path:
    """Prepare one immutable fold shard from its completed full-model checkpoint."""
    config = config.validate()
    _guard_execution(config)
    fold_names = _fold_names(config)
    if not 0 <= fold_index < len(fold_names):
        raise ValueError(f"fold_index must be in range({len(fold_names)})")
    fold = fold_names[fold_index]
    fold_dir = config.lesion_dir / "folds" / fold
    if (fold_dir / "manifest.json").is_file():
        _validate_fold(fold_dir, config, fold_index)
        return fold_dir
    if fold_dir.exists():
        _validate_prepared(fold_dir, config, fold_index)
        return fold_dir

    fit_dir = config.fit_dir / f"{config.feature_space}-{config.penalty}" / fold
    _resume_existing(fit_dir, FIT_ARTIFACTS, replace(config, resume_completed_folds=True))
    diagnostics = json.loads((fit_dir / "diagnostics.json").read_text())
    frozen = {
        name: diagnostics[name]
        for name in ("C", "l1_ratio", *POOLING_PARAMETERS)
    }
    tuning = pd.read_csv(fit_dir / "tuning.csv")
    rows, dictionary = _load_rows(config)
    selected_fold, train, test = list(_outer_folds(rows, config))[fold_index]
    if selected_fold != fold:
        raise RuntimeError("outer fold order changed")
    plan = prepare_lesions(train, test, dictionary, config, frozen, tuning)
    pd.testing.assert_frame_equal(
        plan["dictionary"].reset_index(drop=True),
        pd.read_parquet(fit_dir / "feature-dictionary.parquet").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        plan["provenance"]["loadings"].reset_index(drop=True),
        pd.read_csv(fit_dir / "loadings.csv").reset_index(drop=True),
    )

    fold_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{fold}-prepare-", dir=fold_dir.parent))
    try:
        plan["train"].to_parquet(stage / "train.parquet", index=False)
        plan["test"].to_parquet(stage / "test.parquet", index=False)
        plan["dictionary"].to_parquet(stage / "feature-dictionary.parquet", index=False)
        _specification_table(plan).to_csv(stage / "specifications.csv", index=False)
        _runnable_table(plan).to_csv(stage / "run-specifications.csv", index=False)
        plan["tuning"].to_csv(stage / "tuning.csv", index=False)
        (stage / "frozen-hyperparameters.json").write_text(json.dumps(
            plan["frozen_hyperparameters"], indent=2, sort_keys=True,
        ) + "\n")
        (stage / "aliases.json").write_text(json.dumps(
            plan["aliases"], indent=2, sort_keys=True,
        ) + "\n")
        provenance = plan["provenance"]
        provenance["loadings"].to_csv(stage / "loadings.csv", index=False)
        provenance["explained_variance"].to_csv(stage / "explained-variance.csv", index=False)
        (stage / "feature-provenance.json").write_text(json.dumps({
            key: value for key, value in provenance.items()
            if key not in {"loadings", "explained_variance"}
        }, indent=2, sort_keys=True) + "\n")
        full_predictions = pd.read_parquet(fit_dir / "predictions.parquet")
        (stage / "predictions").mkdir()
        (stage / "coefficients").mkdir()
        (stage / "item-metrics").mkdir()
        (stage / "diagnostics").mkdir()
        full_predictions.to_parquet(stage / "predictions/full.parquet", index=False)
        pd.read_csv(fit_dir / "coefficients.csv").to_parquet(
            stage / "coefficients/full.parquet", index=False,
        )
        _item_metrics(full_predictions).to_parquet(
            stage / "item-metrics/full.parquet", index=False,
        )
        (stage / "diagnostics/full.json").write_text(
            (fit_dir / "diagnostics.json").read_text()
        )
        manifest_path = stage / "prepare-manifest.json"
        manifest = {
            "state": "prepared", "fold": fold,
            "backend": f"{config.feature_space}-{config.penalty}",
            "frozen_hyperparameters": plan["frozen_hyperparameters"],
            "config_signature": _settings_signature(_config_settings(config, fold_index)),
            "fit_manifest_sha256": _sha256(fit_dir / "manifest.json"),
            "artifact_sha256": {
                str(path.relative_to(stage)): _sha256(path)
                for path in stage.rglob("*") if path.is_file() and path != manifest_path
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, fold_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return fold_dir


def run_specification(
    task_index: int,
    config: AnalysisConfig = CONFIG,
    *,
    task: SparseTask = JUDGE_TASK,
    write_diagnostics: bool = True,
) -> Path | None:
    """Fit one fold×specification task with frozen full-model hyperparameters."""
    config = config.validate()
    _guard_execution(config)
    first = config.lesion_dir / "folds" / _fold_names(config)[0]
    table = pd.read_csv(first / "run-specifications.csv")
    per_fold = len(table)
    fold_index, spec_index = divmod(task_index, per_fold)
    if fold_index >= len(_fold_names(config)):
        return None
    fold_dir = config.lesion_dir / "folds" / _fold_names(config)[fold_index]
    _validate_prepared(fold_dir, config, fold_index)
    row = pd.read_csv(fold_dir / "run-specifications.csv").iloc[spec_index]
    paths = [
        fold_dir / folder / f"{row.specification}.parquet"
        for folder in ("predictions", "coefficients", "item-metrics")
    ]
    diagnostic = fold_dir / "diagnostics" / f"{row.specification}.json"
    if write_diagnostics:
        paths.append(diagnostic)
    if all(path.is_file() for path in paths):
        return paths[0]
    train = pd.read_parquet(fold_dir / "train.parquet")
    test = pd.read_parquet(fold_dir / "test.parquet")
    all_features = pd.read_parquet(fold_dir / "feature-dictionary.parquet").feature.tolist()
    features = json.loads(row.features)
    frozen = json.loads((fold_dir / "frozen-hyperparameters.json").read_text())
    result = fit_sparse(
        train, test, features, config.penalty, frozen, config, task=task,
    )
    if not result.diagnostics["converged"]:
        raise RuntimeError(
            f"lesion fit did not converge: {row.specification} {result.diagnostics}"
        )
    predictions = result.predictions.drop(columns=all_features, errors="ignore")
    _atomic(paths[0], lambda path: predictions.to_parquet(path, index=False))
    _atomic(paths[1], lambda path: result.coefficients.to_parquet(path, index=False))
    _atomic(paths[2], lambda path: _item_metrics(predictions).to_parquet(path, index=False))
    if write_diagnostics:
        _atomic(diagnostic, lambda path: path.write_text(
            json.dumps(result.diagnostics, indent=2, sort_keys=True) + "\n"
        ))
    return paths[0]


def finalize_fold(fold_index: int, config: AnalysisConfig = CONFIG) -> Path:
    """Validate every specification shard and publish the completed fold manifest."""
    config = config.validate()
    fold_dir = config.lesion_dir / "folds" / _fold_names(config)[fold_index]
    if (fold_dir / "manifest.json").is_file():
        _validate_fold(fold_dir, config, fold_index)
        return fold_dir
    _validate_prepared(fold_dir, config, fold_index)
    table = pd.read_csv(fold_dir / "run-specifications.csv")
    missing = [
        str(fold_dir / folder / f"{specification}.parquet")
        for specification in table.specification
        for folder in ("predictions", "coefficients", "item-metrics")
        if not (fold_dir / folder / f"{specification}.parquet").is_file()
    ]
    missing.extend(
        str(fold_dir / "diagnostics" / f"{specification}.json")
        for specification in table.specification
        if not (fold_dir / "diagnostics" / f"{specification}.json").is_file()
    )
    if missing:
        raise RuntimeError(f"lesion fold is missing {len(missing)} artifacts")
    frozen = json.loads((fold_dir / "frozen-hyperparameters.json").read_text())
    for specification in table.specification:
        diagnostics = json.loads(
            (fold_dir / "diagnostics" / f"{specification}.json").read_text()
        )
        if (
            not diagnostics.get("converged")
            or any(diagnostics.get(name) != value for name, value in frozen.items())
        ):
            raise RuntimeError(f"invalid lesion diagnostics: {specification}")
    manifest_path = fold_dir / "manifest.json"
    manifest = {
        "state": "completed", "fold": _fold_names(config)[fold_index],
        "backend": f"{config.feature_space}-{config.penalty}",
        "frozen_hyperparameters": frozen,
        "config_signature": _settings_signature(_config_settings(config, fold_index)),
        "fit_manifest_sha256": _validate_prepared(
            fold_dir, config, fold_index)["fit_manifest_sha256"],
        "artifacts": {
            str(path.relative_to(fold_dir)): _sha256(path)
            for path in fold_dir.rglob("*") if path.is_file() and path != manifest_path
        },
        "labelled_specifications": len(pd.read_csv(fold_dir / "specifications.csv")),
        "unique_specifications": len(table) - int((table.kind == "reference").sum()),
        "aliases": json.loads((fold_dir / "aliases.json").read_text()),
    }
    _atomic(manifest_path, lambda path: path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ))
    _validate_fold(fold_dir, config, fold_index)
    return fold_dir


def _validate_fold(
    fold_dir: Path,
    config: AnalysisConfig,
    fold_index: int,
) -> dict[str, object] | None:
    if not fold_dir.exists():
        return None
    manifest_path = fold_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"partial fold has no completion manifest: {fold_dir}")
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.get("artifacts")
    prepared = _validate_prepared(fold_dir, config, fold_index)
    expected_files = {
        str(path.relative_to(fold_dir))
        for path in fold_dir.rglob("*") if path.is_file() and path != manifest_path
    }
    if (
        manifest.get("state") != "completed"
        or manifest.get("backend") != f"{config.feature_space}-{config.penalty}"
        or manifest.get("config_signature") != _settings_signature(
            _config_settings(config, fold_index))
        or manifest.get("fit_manifest_sha256") != prepared["fit_manifest_sha256"]
        or not isinstance(hashes, dict)
        or set(hashes) != expected_files
        or any(_sha256(fold_dir / relative) != digest for relative, digest in hashes.items())
    ):
        raise RuntimeError(f"invalid completed fold: {fold_dir}")
    return manifest


def _fold_names(config: AnalysisConfig) -> list[str]:
    return [
        f"{model_name}__item-{item_fold}"
        for model_name, _ in config.model_folds for item_fold in range(config.item_folds)
    ]


def aggregate_lesions(config: AnalysisConfig = CONFIG) -> Path:
    """Aggregate completed item-level lesion deltas across all outer folds."""
    config = config.validate()
    fold_names = _fold_names(config)
    fold_dirs = [config.lesion_dir / "folds" / fold for fold in fold_names]
    fold_manifests = [
        _validate_fold(fold_dir, config, index)
        for index, fold_dir in enumerate(fold_dirs)
    ]
    if any(manifest is None for manifest in fold_manifests):
        raise RuntimeError("not all lesion folds are complete")
    summary_settings = {
        **_config_settings(config),
        "fold_manifests": [_sha256(fold_dir / "manifest.json") for fold_dir in fold_dirs],
        "lesion_metric": config.lesion_metric,
        "bootstrap_iterations": config.bootstrap_iterations,
    }
    output = config.lesion_dir / "lesions.csv"
    summary_artifacts = [
        output, config.lesion_dir / "item-deltas.parquet",
        config.lesion_dir / "artifact-index.csv",
    ]
    manifest_path = config.lesion_dir / "summary-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        hashes = manifest.get("artifact_sha256")
        if (
            manifest.get("state") != "completed"
            or manifest.get("backend") != f"{config.feature_space}-{config.penalty}"
            or manifest.get("settings_signature") != _settings_signature(summary_settings)
            or not isinstance(hashes, dict)
            or set(hashes) != {path.name for path in summary_artifacts}
            or any(not path.is_file() or _sha256(path) != hashes[path.name] for path in summary_artifacts)
        ):
            raise RuntimeError(f"invalid lesion summary: {config.lesion_dir}")
        return output
    if any(path.exists() for path in summary_artifacts):
        raise RuntimeError(f"partial lesion summary has no manifest: {config.lesion_dir}")

    tables = [pd.read_csv(fold_dir / "specifications.csv", keep_default_na=False) for fold_dir in fold_dirs]
    if not tables or any(not table.equals(tables[0]) for table in tables[1:]):
        raise RuntimeError("fold specification tables are missing or differ")
    backend = f"{config.feature_space}-{config.penalty}"
    item_deltas, summaries, index_rows = [], [], []
    for fold, fold_dir in zip(fold_names, fold_dirs):
        references = [
            "full", *dict.fromkeys(
                reference for reference in tables[0].reference if reference != "full"
            ),
        ]
        for reference in references:
            index_rows.append({
                "fold": fold, "specification": reference, "kind": "reference",
                "variable": reference, "reference": "", "artifact": reference,
                "predictions": str(fold_dir / "predictions" / f"{reference}.parquet"),
                "coefficients": str(fold_dir / "coefficients" / f"{reference}.parquet"),
                "item_metrics": str(fold_dir / "item-metrics" / f"{reference}.parquet"),
                "diagnostics": str(fold_dir / "diagnostics" / f"{reference}.json"),
            })
    for row in tables[0].itertuples(index=False):
        specification = Specification(
            row.specification, row.kind, row.variable, row.reference, (),
        )
        paired = []
        for fold, fold_dir in zip(fold_names, fold_dirs):
            candidate = pd.read_parquet(fold_dir / "item-metrics" / f"{row.artifact}.parquet")
            reference = pd.read_parquet(fold_dir / "item-metrics" / f"{row.reference}.parquet")
            paired.append(_paired_item_metrics(
                candidate, reference, specification, fold, backend,
            ))
            index_rows.append({
                "fold": fold, "specification": row.specification, "kind": row.kind,
                "variable": row.variable, "reference": row.reference,
                "artifact": row.artifact,
                "predictions": str(fold_dir / "predictions" / f"{row.artifact}.parquet"),
                "coefficients": str(fold_dir / "coefficients" / f"{row.artifact}.parquet"),
                "item_metrics": str(fold_dir / "item-metrics" / f"{row.artifact}.parquet"),
                "diagnostics": str(fold_dir / "diagnostics" / f"{row.artifact}.json"),
            })
        combined = pd.concat(paired, ignore_index=True)
        item_deltas.append(combined)
        summaries.append(_bootstrap_summary(combined, config))
    config.lesion_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".summary-", dir=config.lesion_dir))
    try:
        staged = [stage / path.name for path in summary_artifacts]
        pd.concat(summaries, ignore_index=True).to_csv(staged[0], index=False)
        pd.concat(item_deltas, ignore_index=True).to_parquet(staged[1], index=False)
        pd.DataFrame(index_rows).to_csv(staged[2], index=False)
        summary_manifest = {
            "state": "completed", "backend": backend,
            "settings_signature": _settings_signature(summary_settings),
            "settings": summary_settings,
            "artifact_sha256": {path.name: _sha256(path) for path in staged},
        }
        staged_manifest = stage / manifest_path.name
        staged_manifest.write_text(json.dumps(summary_manifest, indent=2, sort_keys=True) + "\n")
        for source, target in zip(staged, summary_artifacts):
            os.replace(source, target)
        os.replace(staged_manifest, manifest_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--fold-index", type=int, required=True)
    spec = commands.add_parser("spec")
    spec.add_argument("--task-index", type=int, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--fold-index", type=int, required=True)
    summarize = commands.add_parser("summarize")
    for command in (prepare, spec, finalize, summarize):
        command.add_argument("--processed-matrix-root", type=Path, required=True)
        command.add_argument("--results-dir", type=Path, required=True)
        command.add_argument("--feature-space", choices=("raw", "pc"), required=True)
        command.add_argument("--penalty", choices=("l1", "ridge"), required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--pc-basis-root", type=Path)
        command.add_argument("--pc-components", type=_positive_int)
    args = parser.parse_args(argv)
    config = replace(
        CONFIG, results_dir=args.results_dir,
        feature_space=args.feature_space, penalty=args.penalty,
        processed_matrix_manifest_sha256=args.manifest_sha256,
    )
    if args.feature_space == "pc":
        if args.pc_basis_root is None or args.pc_components is None:
            parser.error("--pc-basis-root and --pc-components are required for PC features")
        config = replace(
            config, pc_basis_root=args.pc_basis_root,
            fit_pca_components=args.pc_components,
        )
    elif args.pc_basis_root is not None or args.pc_components is not None:
        parser.error("PC-only arguments require --feature-space pc")
    config = replace(
        config, processed_matrix_root=args.processed_matrix_root,
        execution_mode="execute", execution_target="swc_hpc",
    )
    if args.command == "prepare":
        prepare_fold(args.fold_index, config)
    elif args.command == "spec":
        run_specification(args.task_index, config)
    elif args.command == "finalize":
        finalize_fold(args.fold_index, config)
    else:
        aggregate_lesions(config)


if __name__ == "__main__":
    main()
