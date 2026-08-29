"""Safe orchestration for the joint judge-verdict fitting backends."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .config import AnalysisConfig, CONFIG
from .data import load_features
from .fit_common import FitResult
from .fit_frequentist import fit_frequentist, select_features
from .hierarchy import HierarchyLevels, outer_roles


ARTIFACTS = (
    "predictions.parquet", "metrics.csv", "coefficients.csv", "selection.csv",
    "tuning.csv", "diagnostics.json", "manifest.json",
)
_COMPLETED_ARTIFACTS = tuple(name for name in ARTIFACTS if name != "manifest.json")


def _manifest_settings(config: AnalysisConfig) -> dict[str, object]:
    return {
        "model_folds": [[name, list(models)] for name, models in config.model_folds],
        "item_folds": config.item_folds,
        "inner_folds": config.inner_folds,
        "fit_n_jobs": config.fit_n_jobs,
        "resume_completed_folds": config.resume_completed_folds,
        "elastic_net_cs": list(config.elastic_net_cs),
        "elastic_net_l1_ratios": list(config.elastic_net_l1_ratios),
        "selection_min_fold_share": config.selection_min_fold_share,
        "ridge_cs": list(config.ridge_cs),
        "judge_deviation_penalty": config.judge_deviation_penalty,
        "benchmark_deviation_penalty": config.benchmark_deviation_penalty,
        "cell_deviation_penalty": config.cell_deviation_penalty,
        "item_intercept_penalty": config.item_intercept_penalty,
        "glm_max_iter": config.glm_max_iter,
        "glm_c": config.glm_c,
        "random_seed": config.random_seed,
        "mcmc_warmup": config.mcmc_warmup,
        "mcmc_samples": config.mcmc_samples,
        "mcmc_chains": config.mcmc_chains,
        "mcmc_method": config.mcmc_method,
        "mcmc_target_accept": config.mcmc_target_accept,
        "mcmc_max_tree_depth": config.mcmc_max_tree_depth,
        "mcmc_init_strategy": config.mcmc_init_strategy,
        "mcmc_chain_method": config.mcmc_chain_method,
        "mcmc_device_type": config.mcmc_device_type,
        "mcmc_gpu_memory_fraction": config.mcmc_gpu_memory_fraction,
        "hibayes_population_prior_scale": config.hibayes_population_prior_scale,
        "hibayes_judge_prior_scale": config.hibayes_judge_prior_scale,
        "hibayes_benchmark_prior_scale": config.hibayes_benchmark_prior_scale,
        "hibayes_cell_prior_scale": config.hibayes_cell_prior_scale,
        "hibayes_item_prior_scale": config.hibayes_item_prior_scale,
        "require_pinned_source_for_fit": config.require_pinned_source_for_fit,
        "processed_matrix_manifest_sha256": config.processed_matrix_manifest_sha256,
    }


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


def write_hibayes_job(
    train_path: Path,
    test_path: Path,
    selected_features: list[str],
    fold: str,
    config: AnalysisConfig,
) -> Path:
    """Load HiBayes only for an explicitly requested Bayesian job."""
    from .fit_hibayes import write_hibayes_job as implementation

    return implementation(train_path, test_path, selected_features, fold, config)


def evaluate_hibayes(
    job_dir: Path, test_rows: pd.DataFrame, config: AnalysisConfig
) -> FitResult:
    """Load HiBayes only when evaluating an explicitly requested Bayesian job."""
    from .fit_hibayes import evaluate_hibayes as implementation

    return implementation(job_dir, test_rows, config)


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
    output = config.fit_dir / "prepared"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".prepared-", dir=output.parent))
    manifest = _prepared_manifest("", "", config)
    manifest.pop("backend")
    manifest.pop("fold")
    manifest["scope"] = "full_plan"
    manifest["jobs"] = [
        {"backend": backend, "fold": f"{model_name}__item-{item_fold}"}
        for backend in config.fit_backends
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


def _load_rows(config: AnalysisConfig) -> tuple[pd.DataFrame, list[str]]:
    rows, dictionary = load_features(
        "pair", config.feature_blocks,
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
    return rows, features


def _resume_existing(
    output: Path, required: tuple[str, ...], config: AnalysisConfig
) -> bool:
    """Skip a complete output only when explicit resume mode is enabled."""
    if not output.exists():
        return False
    if not config.resume_completed_folds:
        raise FileExistsError(f"refusing to overwrite {output}")
    if missing := [name for name in required if not (output / name).exists()]:
        raise RuntimeError(f"cannot resume partial output {output}; missing {missing}")
    return True


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


def _write_result(
    result: FitResult,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AnalysisConfig,
    *,
    existing_job: bool = False,
) -> None:
    output = config.fit_dir / result.backend / result.fold
    if output.exists() and (not existing_job or any((output / name).exists() for name in _COMPLETED_ARTIFACTS)):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{result.fold}-", dir=output.parent))
    manifest = {
        "state": "completed",
        "backend": result.backend, "fold": result.fold,
        "train_rows": len(train), "test_rows": len(test),
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
        "settings": _manifest_settings(config),
    }
    try:
        result.predictions.to_parquet(stage / "predictions.parquet", index=False)
        result.metrics.to_csv(stage / "metrics.csv", index=False)
        result.coefficients.to_csv(stage / "coefficients.csv", index=False)
        pd.DataFrame({"feature": result.selected_features}).to_csv(stage / "selection.csv", index=False)
        result.tuning.to_csv(stage / "tuning.csv", index=False)
        (stage / "diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n"
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if existing_job:
            for path in stage.iterdir():
                os.replace(path, output / path.name)
            stage.rmdir()
        else:
            os.replace(stage, output)
    except BaseException:
        for path in stage.iterdir() if stage.exists() else ():
            path.unlink()
        if stage.exists():
            stage.rmdir()
        raise


def _prepare_hibayes(
    fold: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    tuning: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    output = config.fit_dir / "hibayes" / fold
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = Path(mkdtemp(prefix=f".{fold}-"))
    try:
        train_path, test_path = stage / "train.parquet", stage / "test.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)
        job_dir = write_hibayes_job(train_path, test_path, selected_features, fold, config)
        pd.DataFrame({"feature": selected_features}).to_csv(job_dir / "screen-selection.csv", index=False)
        tuning.to_csv(job_dir / "screen-tuning.csv", index=False)
        manifest = _prepared_manifest("hibayes", fold, config)
        manifest["selected_features"] = selected_features
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        for path in stage.iterdir():
            path.unlink()
        stage.rmdir()


def run_fits(config: AnalysisConfig = CONFIG) -> None:
    """Run approved fits, or prepare external HiBayes jobs without sampling."""
    config = config.validate()
    if config.execution_mode == "prepare":
        _write_plan_descriptor(config)
        return
    _guard_execution(config)
    rows, features = _load_rows(config)

    for fold, train, test in _outer_folds(rows, config):
        selected: list[str] | None = None
        tuning: pd.DataFrame | None = None
        if "frequentist" in config.fit_backends:
            output = config.fit_dir / "frequentist" / fold
            if _resume_existing(output, ARTIFACTS, config):
                selected = pd.read_csv(output / "selection.csv").feature.tolist()
                tuning = pd.read_csv(output / "tuning.csv")
            else:
                frequentist = fit_frequentist(train, test, features, config)
                _write_result(frequentist, train, test, config)
                selected = list(frequentist.selected_features)
                tuning = frequentist.tuning
        if "hibayes" in config.fit_backends:
            if config.hibayes_action == "prepare":
                output = config.fit_dir / "hibayes" / fold
                if _resume_existing(output, (
                    "config.yaml", "train.parquet", "test.parquet",
                    "screen-selection.csv", "screen-tuning.csv", "manifest.json",
                ), config):
                    continue
                if selected is None or tuning is None:
                    selected, tuning = select_features(
                        train, features, config, HierarchyLevels.from_training(train)
                    )
                _prepare_hibayes(fold, train, test, selected, tuning, config)
            else:
                output = config.fit_dir / "hibayes" / fold
                if config.resume_completed_folds and all(
                    (output / name).exists() for name in ARTIFACTS
                ):
                    continue
                result = evaluate_hibayes(config.fit_dir / "hibayes" / fold, test, config)
                _write_result(result, train, test, config, existing_job=True)


def run_remote_fits(config: AnalysisConfig = CONFIG) -> None:
    """Execute configured fits after the configured remote-target guard."""
    run_fits(replace(config, execution_mode="execute"))


def main() -> None:
    """Run the configuration-owned prepare-only fitting entry point."""
    run_fits()


if __name__ == "__main__":
    main()
