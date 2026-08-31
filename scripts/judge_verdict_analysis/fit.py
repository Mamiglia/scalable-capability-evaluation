"""Safe orchestration for the joint judge-verdict fitting backends."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .config import AnalysisConfig, CONFIG
from .data import load_features
from .fit_common import FitResult, prepare_features
from .fit_frequentist import fit_sparse, tune_sparse
from .hierarchy import outer_roles


ARTIFACTS = (
    "predictions.parquet", "metrics.csv", "coefficients.csv", "selection.csv",
    "tuning.csv", "diagnostics.json", "feature-dictionary.parquet",
    "feature-provenance.json", "loadings.csv", "explained-variance.csv",
    "manifest.json",
)


def _manifest_settings(config: AnalysisConfig) -> dict[str, object]:
    settings = {
        "model_folds": [[name, list(models)] for name, models in config.model_folds],
        "item_folds": config.item_folds,
        "inner_folds": config.inner_folds,
        "fit_n_jobs": config.fit_n_jobs,
        "feature_space": config.feature_space,
        "penalty": config.penalty,
        "elastic_net_cs": list(config.elastic_net_cs),
        "ridge_cs": list(config.ridge_cs),
        "pooling_penalty_ratios": list(config.pooling_penalty_ratios),
        "item_pooling_penalty_ratios": list(config.item_pooling_penalty_ratios),
        "judge_deviation_penalty": config.judge_deviation_penalty,
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "cell_deviation_penalty": config.cell_deviation_penalty,
        "item_intercept_penalty": config.item_intercept_penalty,
        "glm_max_iter": config.glm_max_iter,
        "random_seed": config.random_seed,
        "require_pinned_source_for_fit": config.require_pinned_source_for_fit,
        "processed_matrix_manifest_sha256": config.processed_matrix_manifest_sha256,
    }
    if config.feature_space == "pc":
        settings.update({
            "fit_pca_components": config.fit_pca_components,
            "fit_pca_algorithm": config.fit_pca_algorithm,
            "pc_basis_root": str(config.pc_basis_root),
        })
    return settings


def _ceph_available() -> bool:
    return Path("/ceph/mrsic_flogel").exists()


def _active_remote_target(target: str) -> bool:
    if target == "swc_hpc":
        hostname = socket.gethostname().lower()
        return (
            _ceph_available()
            and bool(os.environ.get("SLURM_JOB_ID"))
            and not hostname.startswith(("gateway", "sgw"))
        )
    return target == "runpod" and bool(os.environ.get("RUNPOD_POD_ID"))


def _guard_execution(config: AnalysisConfig) -> None:
    if config.execution_mode != "execute":
        return
    if config.execution_target == "local_test":
        values = yaml.safe_load(config.processed_matrix_config.read_text())
        pair_path = Path(values["output_root"]) / "pair.parquet"
        if pq.ParquetFile(pair_path).metadata.num_rows > config.local_test_max_rows:
            raise RuntimeError("refusing local fit above local_test_max_rows; use SWC HPC or RunPod")
        return
    if not _active_remote_target(config.execution_target):
        raise RuntimeError(
            f"refusing heavy local fit; execute on configured {config.execution_target} target"
        )
    if not config.require_pinned_source_for_fit:
        return
    values = yaml.safe_load(config.processed_matrix_config.read_text())
    manifest_path = Path(values["output_root"]) / "manifest.json"
    if config.source_hf_revision:
        manifest = json.loads(manifest_path.read_text())
        matches = (manifest.get("source_hf_repo"), manifest.get("source_hf_revision")) == (
            config.source_hf_repo, config.source_hf_revision,
        )
    elif config.processed_matrix_manifest_sha256:
        matches = hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
            config.processed_matrix_manifest_sha256
        )
    else:
        raise RuntimeError(
            "refusing non-local fit without source_hf_revision or processed matrix manifest hash"
        )
    if not matches:
        raise RuntimeError("refusing non-local fit: processed matrix source provenance does not match config")


def _prepared_manifest(backend: str, fold: str, config: AnalysisConfig) -> dict[str, object]:
    return {
        "state": "prepared",
        "backend": backend,
        "fold": fold,
        "target": config.execution_target,
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
        "command": [
            "python", "-c",
            "from scripts.judge_verdict_analysis.fit import run_remote_fits\nrun_remote_fits()",
        ],
        "source_paths": {
            "build_design_matrix_config": str(config.design_matrix_config),
            "processed_matrix_config": str(config.processed_matrix_config),
        },
        "settings": _manifest_settings(config),
    }


def _write_plan_descriptor(config: AnalysisConfig) -> None:
    output = config.fit_dir / "prepared" / f"{config.feature_space}-{config.penalty}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".prepared-", dir=output.parent))
    manifest = _prepared_manifest("", "", config)
    manifest.pop("backend")
    manifest.pop("fold")
    manifest["scope"] = "full_plan"
    manifest["jobs"] = [
        {"backend": f"{config.feature_space}-{config.penalty}",
         "fold": f"{model_name}__item-{item_fold}"}
        for model_name, _ in config.model_folds
        for item_fold in range(config.item_folds)
    ]
    try:
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()
        raise


def _load_rows(config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, dictionary = load_features(
        "pair", None,  # The processed matrix already embodies block selection, including mixed composites.
        features=config.regressors or None,
        splits=config.splits,
        config=config.processed_matrix_config,
    )
    rows = rows[
        rows.judge.isin(config.judges) & rows.bench.isin(config.benchmarks)
    ].reset_index(drop=True)
    features = dictionary.feature.tolist()
    rows = rows.rename(columns=dict(zip(dictionary.column, features)))
    required = {"judge", "bench", "item", "model_left", "model_right", "pick_left", *features}
    if missing := sorted(required - set(rows)):
        raise ValueError(f"pair matrix missing fitting columns: {missing}")
    if rows.empty:
        raise ValueError("no pair rows remain after configured judge and benchmark filtering")
    return rows, dictionary.drop(columns="column", errors="ignore")


def _resume_existing(
    output: Path, required: tuple[str, ...], config: AnalysisConfig,
    backend: str | None = None,
) -> bool:
    """Resume only a complete, hash-valid fold with identical settings."""
    if not output.exists():
        return False
    if not config.resume_completed_folds:
        raise FileExistsError(f"refusing to overwrite {output}")
    if missing := [name for name in required if not (output / name).exists()]:
        raise RuntimeError(f"cannot resume partial output {output}; missing {missing}")
    manifest = json.loads((output / "manifest.json").read_text())
    backend = backend or f"{config.feature_space}-{config.penalty}"
    hashes = manifest.get("artifact_sha256")
    saved_settings = dict(manifest.get("settings", {}))
    expected_settings = _manifest_settings(config)
    for operational in ("resume_completed_folds", "glm_max_iter"):
        saved_settings.pop(operational, None)
        expected_settings.pop(operational, None)
    if (
        manifest.get("state") != "completed"
        or manifest.get("backend") != backend
        or saved_settings != expected_settings
        or not isinstance(hashes, dict)
        or set(hashes) != set(required) - {"manifest.json"}
        or any(_file_sha256(output / name) != digest for name, digest in hashes.items())
    ):
        raise RuntimeError(f"cannot resume stale or corrupt output {output}")
    return True


def run_fold(fold_index: int, config: AnalysisConfig = CONFIG) -> Path:
    """Fit one model-by-item outer fold for a configured sparse variant."""
    config = config.validate()
    _guard_execution(config)
    fold_names = [
        f"{model_name}__item-{item_fold}"
        for model_name, _ in config.model_folds for item_fold in range(config.item_folds)
    ]
    if not 0 <= fold_index < len(fold_names):
        raise ValueError(f"fold_index must be in range({len(fold_names)})")
    output = config.fit_dir / f"{config.feature_space}-{config.penalty}" / fold_names[fold_index]
    if _resume_existing(output, ARTIFACTS, replace(config, resume_completed_folds=True)):
        return output
    rows, dictionary = _load_rows(config)
    fold, train, test = list(_outer_folds(rows, config))[fold_index]
    result, prepared_dictionary, provenance = fit_fold(train, test, dictionary, config)
    _write_result(
        result, train, test, config,
        prepared_dictionary=prepared_dictionary, provenance=provenance,
    )
    return config.fit_dir / result.backend / fold


def aggregate_fits(
    config: AnalysisConfig = CONFIG, backend: str | None = None,
) -> Path:
    """Aggregate all 16 full-fit checkpoints for one sparse variant."""
    config = config.validate()
    backend = backend or f"{config.feature_space}-{config.penalty}"
    fold_names = [
        f"{model_name}__item-{item_fold}"
        for model_name, _ in config.model_folds for item_fold in range(config.item_folds)
    ]
    fold_dirs = [config.fit_dir / backend / fold for fold in fold_names]
    for fold_dir in fold_dirs:
        _resume_existing(
            fold_dir, ARTIFACTS, replace(config, resume_completed_folds=True), backend,
        )
    fold_hashes = {
        fold: _file_sha256(fold_dir / "manifest.json")
        for fold, fold_dir in zip(fold_names, fold_dirs)
    }
    aggregate_artifacts = {
        "predictions.parquet", "metrics.csv", "judge-benchmark.csv",
        "fold-metrics.csv", "uncertainty.csv", "coefficients.csv",
        "selection.csv", "tuning.csv", "artifact-index.csv",
    }
    output = config.fit_dir / backend / "aggregate"
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        hashes = manifest.get("artifact_sha256")
        if (
            manifest.get("state") != "completed"
            or manifest.get("backend") != backend
            or manifest.get("settings") != _manifest_settings(config)
            or manifest.get("fold_manifest_sha256") != fold_hashes
            or not isinstance(hashes, dict)
            or set(hashes) != aggregate_artifacts
            or any(not (output / name).is_file() or _file_sha256(output / name) != digest
                   for name, digest in hashes.items())
        ):
            raise RuntimeError(f"invalid fit aggregate: {output}")
        return output

    predictions, coefficients, selections, tuning, index_rows = [], [], [], [], []
    for fold, fold_dir in zip(fold_names, fold_dirs):
        predictions.append(pd.read_parquet(fold_dir / "predictions.parquet").assign(fold=fold))
        coefficients.append(pd.read_csv(fold_dir / "coefficients.csv").assign(fold=fold))
        selections.append(pd.read_csv(fold_dir / "selection.csv").assign(fold=fold))
        tuning.append(pd.read_csv(fold_dir / "tuning.csv").assign(fold=fold))
        index_rows.append({
            "fold": fold,
            **{name.removesuffix(Path(name).suffix).replace("-", "_"): str(fold_dir / name)
               for name in ARTIFACTS},
        })
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics, judge_benchmark, fold_metrics, uncertainty = summarize_predictions(
        all_predictions, config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".aggregate-", dir=output.parent))
    try:
        all_predictions.to_parquet(stage / "predictions.parquet", index=False)
        metrics.to_csv(stage / "metrics.csv", index=False)
        judge_benchmark.to_csv(stage / "judge-benchmark.csv", index=False)
        fold_metrics.to_csv(stage / "fold-metrics.csv", index=False)
        uncertainty.to_csv(stage / "uncertainty.csv", index=False)
        pd.concat(coefficients, ignore_index=True).to_csv(stage / "coefficients.csv", index=False)
        pd.concat(selections, ignore_index=True).to_csv(stage / "selection.csv", index=False)
        pd.concat(tuning, ignore_index=True).to_csv(stage / "tuning.csv", index=False)
        pd.DataFrame(index_rows).to_csv(stage / "artifact-index.csv", index=False)
        manifest = {
            "state": "completed", "backend": backend,
            "settings": _manifest_settings(config),
            "fold_manifest_sha256": fold_hashes,
            "artifact_sha256": {
                path.name: _file_sha256(path) for path in stage.iterdir()
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
    except BaseException:
        for path in stage.iterdir() if stage.exists() else ():
            path.unlink()
        if stage.exists():
            stage.rmdir()
        raise
    return output


def _outer_folds(frame: pd.DataFrame, config: AnalysisConfig):
    for model_name, models in config.model_folds:
        for item_fold in range(config.item_folds):
            fold = f"{model_name}__item-{item_fold}"
            roles = outer_roles(frame, models, item_fold, config.item_folds)
            train = frame.loc[roles == "train"].assign(fold=fold).reset_index(drop=True)
            test = frame.loc[roles == "test"].assign(fold=fold).reset_index(drop=True)
            if train.empty or test.empty:
                raise ValueError(f"outer fold {fold} has no {'train' if train.empty else 'test'} rows")
            yield fold, train, test


def _json_default(value: object) -> object:
    return value.item() if hasattr(value, "item") else str(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def summarize_predictions(
    predictions: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize OOF predictions with item and crossed-fold uncertainty."""
    required = {"fold", "bench", "judge", "item", "observed", "probability"}
    if missing := required - set(predictions):
        raise ValueError(f"predictions missing aggregation columns: {sorted(missing)}")
    rows = predictions.copy()
    probability = np.clip(rows.probability.to_numpy(float), 1e-12, 1 - 1e-12)
    observed = rows.observed.to_numpy(int)
    rows["accuracy"] = ((probability >= 0.5) == observed).astype(float)
    rows["log_loss"] = -(
        observed * np.log(probability) + (1 - observed) * np.log1p(-probability)
    )
    metrics = pd.DataFrame({
        "metric": ["accuracy", "log_loss"],
        "value": [rows.accuracy.mean(), rows.log_loss.mean()],
    })
    judge_benchmark = rows.groupby(["judge", "bench"], as_index=False).agg(
        accuracy=("accuracy", "mean"), log_loss=("log_loss", "mean"),
        row_count=("accuracy", "size"),
    )
    fold_metrics = rows.groupby("fold", as_index=False).agg(
        accuracy=("accuracy", "mean"), log_loss=("log_loss", "mean"),
        row_count=("accuracy", "size"),
    )
    parts = fold_metrics.fold.str.rsplit("__item-", n=1, expand=True)
    fold_metrics["model_fold"] = parts[0]
    fold_metrics["item_fold"] = parts[1]

    clusters = rows.groupby(["bench", "item"], as_index=False).agg(
        accuracy=("accuracy", "sum"), log_loss=("log_loss", "sum"),
        row_count=("accuracy", "size"),
    )
    rng = np.random.default_rng(config.random_seed)
    draws = rng.integers(
        len(clusters), size=(config.bootstrap_iterations, len(clusters)),
    )
    counts = clusters.row_count.to_numpy()[draws].sum(axis=1)
    uncertainty = []
    for metric in ("accuracy", "log_loss"):
        samples = clusters[metric].to_numpy()[draws].sum(axis=1) / counts
        item_interval = np.percentile(samples, [2.5, 97.5])
        dimensions = {}
        for dimension in ("model_fold", "item_fold"):
            values = np.array([
                np.average(group[metric], weights=group.row_count)
                for _, group in fold_metrics.groupby(dimension, sort=False)
            ])
            dimensions[dimension] = [float(values.min()), float(values.max())]
        estimate = float(metrics.loc[metrics.metric == metric, "value"].item())
        uncertainty.extend([
            {"metric": metric, "method": "item_clustered", "estimate": estimate,
             "lower": item_interval[0], "upper": item_interval[1]},
            {"metric": metric, "method": "model_fold", "estimate": estimate,
             "lower": dimensions["model_fold"][0], "upper": dimensions["model_fold"][1]},
            {"metric": metric, "method": "item_fold", "estimate": estimate,
             "lower": dimensions["item_fold"][0], "upper": dimensions["item_fold"][1]},
            {"metric": metric, "method": "conservative_fold", "estimate": estimate,
             "lower": min(dimensions["model_fold"][0], dimensions["item_fold"][0]),
             "upper": max(dimensions["model_fold"][1], dimensions["item_fold"][1])},
        ])
    return metrics, judge_benchmark, fold_metrics, pd.DataFrame(uncertainty)


def fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dictionary: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[FitResult, pd.DataFrame, dict[str, object]]:
    """Prepare, tune once, and fit one complete outer-fold variant."""
    config = config.validate()
    prepared_train, prepared_test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, config.feature_space, config,
    )
    features = prepared_dictionary.feature.astype(str).tolist()
    frozen, tuning = tune_sparse(prepared_train, features, config.penalty, config)
    result = fit_sparse(
        prepared_train, prepared_test, features, config.penalty,
        frozen, config, tuning,
    )
    selected = tuning[tuning.selected]
    if not result.diagnostics["converged"] or selected.empty or not selected.converged.all():
        raise RuntimeError(f"full fit or selected pooling/C tuning did not converge: {result.fold}")
    return result, prepared_dictionary, provenance | {
        "frozen_hyperparameters": frozen,
    }


def _write_result(
    result: FitResult,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AnalysisConfig,
    *,
    prepared_dictionary: pd.DataFrame,
    provenance: dict[str, object],
    manifest_provenance: dict[str, object] | None = None,
) -> None:
    output = config.fit_dir / result.backend / result.fold
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{result.fold}-", dir=output.parent))
    try:
        result.predictions.to_parquet(stage / "predictions.parquet", index=False)
        result.metrics.to_csv(stage / "metrics.csv", index=False)
        result.coefficients.to_csv(stage / "coefficients.csv", index=False)
        pd.DataFrame({"feature": result.selected_features}).to_csv(stage / "selection.csv", index=False)
        result.tuning.to_csv(stage / "tuning.csv", index=False)
        (stage / "diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n"
        )
        prepared_dictionary.to_parquet(stage / "feature-dictionary.parquet", index=False)
        loadings = provenance["loadings"]
        variance = provenance["explained_variance"]
        assert isinstance(loadings, pd.DataFrame) and isinstance(variance, pd.DataFrame)
        loadings.to_csv(stage / "loadings.csv", index=False)
        variance.to_csv(stage / "explained-variance.csv", index=False)
        json_provenance = {
            key: value for key, value in provenance.items()
            if key not in {"loadings", "explained_variance"}
        }
        (stage / "feature-provenance.json").write_text(
            json.dumps(json_provenance, indent=2, sort_keys=True, default=_json_default) + "\n"
        )
        manifest = {
            "state": "completed",
            "backend": result.backend, "fold": result.fold,
            "train_rows": len(train), "test_rows": len(test),
            "source_hf_repo": config.source_hf_repo,
            "source_hf_revision": config.source_hf_revision,
            "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
            "settings": _manifest_settings(config),
            "frozen_hyperparameters": json_provenance.get("frozen_hyperparameters"),
            "artifact_sha256": {
                path.name: _file_sha256(path) for path in stage.iterdir()
            },
        }
        if manifest_provenance:
            if overlap := set(manifest) & set(manifest_provenance):
                raise ValueError(f"manifest provenance overlaps reserved keys: {sorted(overlap)}")
            manifest.update(manifest_provenance)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
    except BaseException:
        for path in stage.iterdir() if stage.exists() else ():
            path.unlink()
        if stage.exists():
            stage.rmdir()
        raise


def run_fits(config: AnalysisConfig = CONFIG) -> None:
    """Run one configured raw/PC and L1/elastic variant across outer folds."""
    config = config.validate()
    if config.execution_mode == "prepare":
        _write_plan_descriptor(config)
        return
    _guard_execution(config)
    rows, dictionary = _load_rows(config)

    for fold, train, test in _outer_folds(rows, config):
        output = config.fit_dir / f"{config.feature_space}-{config.penalty}" / fold
        if _resume_existing(output, ARTIFACTS, config):
            continue
        result, prepared_dictionary, provenance = fit_fold(
            train, test, dictionary, config,
        )
        _write_result(
            result, train, test, config,
            prepared_dictionary=prepared_dictionary, provenance=provenance,
        )


def run_remote_fits(config: AnalysisConfig = CONFIG) -> None:
    """Execute configured fits after the configured remote-target guard."""
    run_fits(replace(config, execution_mode="execute"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fold", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--results-dir", type=Path, required=True)
        command.add_argument("--feature-space", choices=("raw", "pc"), required=True)
        command.add_argument("--penalty", choices=("l1", "ridge"), required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--pc-basis-root", type=Path)
        command.add_argument("--pc-components", type=_positive_int)
        if name == "fold":
            command.add_argument("--fold-index", type=int, required=True)
            command.add_argument("--processed-matrix-root", type=Path, required=True)
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
    if args.command == "fold":
        run_fold(args.fold_index, replace(
            config, processed_matrix_root=args.processed_matrix_root,
            execution_mode="execute", execution_target="swc_hpc",
        ))
    else:
        aggregate_fits(config)


if __name__ == "__main__":
    main()
