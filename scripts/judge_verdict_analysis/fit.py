"""Safe orchestration for the joint judge-verdict fitting backends."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from .config import AnalysisConfig, CONFIG
from .data import load_features
from .fit_common import FitResult
from .fit_frequentist import fit_frequentist
from .fit_hibayes import evaluate_hibayes, write_hibayes_job
from .hierarchy import outer_roles


ARTIFACTS = (
    "predictions.parquet", "metrics.csv", "coefficients.csv", "selection.csv",
    "tuning.csv", "diagnostics.json", "manifest.json",
)


def _manifest_settings(config: AnalysisConfig) -> dict[str, object]:
    return {
        "inner_folds": config.inner_folds,
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
    }


def _active_remote_target(target: str) -> bool:
    if target == "swc_hpc":
        return Path("/ceph/mrsic_flogel").exists()
    return target == "runpod" and bool(os.environ.get("RUNPOD_POD_ID"))


def _require_execution_target(config: AnalysisConfig) -> None:
    if config.execution_mode != "execute":
        return
    if config.execution_target == "local_test":
        return
    if not _active_remote_target(config.execution_target):
        raise RuntimeError(
            f"refusing heavy local fit; execute on configured {config.execution_target} target"
        )


def _load_rows(config: AnalysisConfig) -> tuple[pd.DataFrame, list[str]]:
    rows, dictionary = load_features(
        "pair", config.feature_blocks,
        features=config.regressors or None,
        splits=config.splits,
        config=config.design_matrix_config,
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
    if output.exists() and (not existing_job or any((output / name).exists() for name in ARTIFACTS)):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{result.fold}-", dir=output.parent))
    manifest = {
        "backend": result.backend, "fold": result.fold,
        "train_rows": len(train), "test_rows": len(test),
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
    fold: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str], config: AnalysisConfig
) -> None:
    output = config.fit_dir / "hibayes" / fold
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = Path(mkdtemp(prefix=f".{fold}-"))
    try:
        train_path, test_path = stage / "train.parquet", stage / "test.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)
        write_hibayes_job(train_path, test_path, features, fold, config)
    finally:
        for path in stage.iterdir():
            path.unlink()
        stage.rmdir()


def run_fits(config: AnalysisConfig = CONFIG) -> None:
    """Run approved fits, or prepare external HiBayes jobs without sampling."""
    config = config.validate()
    _require_execution_target(config)
    rows, features = _load_rows(config)
    if config.execution_target == "local_test" and len(rows) > config.local_test_max_rows:
        raise RuntimeError("refusing local fit above local_test_max_rows; use SWC HPC or RunPod")

    for fold, train, test in _outer_folds(rows, config):
        for backend in config.fit_backends:
            if backend == "frequentist":
                if config.execution_mode == "execute":
                    _write_result(fit_frequentist(train, test, features, config), train, test, config)
            elif config.hibayes_action == "prepare":
                _prepare_hibayes(fold, train, test, features, config)
            elif config.execution_mode == "execute":
                result = evaluate_hibayes(config.fit_dir / "hibayes" / fold, test, config)
                _write_result(result, train, test, config, existing_job=True)


def main() -> None:
    """Run the configuration-owned prepare-only fitting entry point."""
    run_fits()


if __name__ == "__main__":
    main()
