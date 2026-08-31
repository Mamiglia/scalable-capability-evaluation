"""Capability-target adapter for shared sparse lesion machinery."""

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp

import pandas as pd

from scripts.judge_verdict_analysis.config import AnalysisConfig, CONFIG
from scripts.judge_verdict_analysis.fit import (
    ARTIFACTS as FIT_ARTIFACTS,
    _guard_execution,
    _load_rows,
    _resume_existing,
)
from scripts.judge_verdict_analysis.fit_common import prepare_features
from scripts.judge_verdict_analysis.lesioning import (
    _config_settings,
    _fold_names,
    _item_metrics,
    _settings_signature,
    _sha256,
    _validate_fold,
    _validate_prepared,
    aggregate_lesions as _aggregate_lesions,
    build_specifications,
    finalize_fold as _finalize_fold,
    run_specification as _run_specification,
    unique_specifications,
)

from .pairwise_wrapper import (
    CAPABILITY_TASK,
    _positive_int,
    _sha256_arg,
    chance_predictions,
    prepare_capability_fold,
    _verify_sha256,
)


@dataclass(frozen=True)
class CapabilityLesionRun:
    config: AnalysisConfig
    responses: Path
    responses_sha256: str
    full_result_root: Path
    output_root: Path
    pc_loadings_sha256: str | None = None
    pc_explained_variance_sha256: str | None = None

    @property
    def analysis(self) -> AnalysisConfig:
        return replace(
            self.config,
            results_dir=self.full_result_root,
            lesion_root=self.output_root,
            lesion_metric="accuracy",
            lesion_source_signature=_settings_signature({
                "responses": str(self.responses),
                "responses_sha256": self.responses_sha256,
                "full_result_root": str(self.full_result_root),
                "output_root": str(self.output_root),
                "pc_loadings_sha256": self.pc_loadings_sha256,
                "pc_explained_variance_sha256": self.pc_explained_variance_sha256,
            }),
        )

    @property
    def backend(self) -> str:
        return f"{self.config.feature_space}-{self.config.penalty}"

    @property
    def lesion_dir(self) -> Path:
        return self.analysis.lesion_dir


def capability_specifications(dictionary, config):
    """Reuse judge lesion inventories with the fixed capability null reference."""
    return [
        replace(specification, reference="null")
        if specification.reference == "intercept-only" else specification
        for specification in build_specifications(dictionary, config)
    ]


def _validate_full_fit(fit_dir: Path, run: CapabilityLesionRun) -> dict[str, object]:
    config = replace(run.analysis, resume_completed_folds=True)
    _resume_existing(fit_dir, FIT_ARTIFACTS, config)
    manifest = json.loads((fit_dir / "manifest.json").read_text())
    expected_hashes = {"responses": run.responses_sha256}
    expected_paths = {"responses": str(run.responses)}
    if config.feature_space == "pc":
        expected_hashes.update({
            "pc_loadings": run.pc_loadings_sha256,
            "pc_explained_variance": run.pc_explained_variance_sha256,
        })
        expected_paths.update({
            "pc_basis": str(config.pc_basis_root),
            "pc_loadings": str(config.pc_basis_root / "loadings.parquet"),
            "pc_explained_variance": str(config.pc_basis_root / "explained-variance.csv"),
        })
    if manifest.get("source_sha256") != expected_hashes:
        raise RuntimeError(f"full fit input hashes do not match: {fit_dir}")
    if manifest.get("source_paths") != expected_paths:
        raise RuntimeError(f"full fit input paths do not match: {fit_dir}")
    return manifest


def _specification_table(specifications, aliases) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "specification": spec.name,
            "kind": spec.kind,
            "variable": spec.variable,
            "reference": spec.reference,
            "feature_count": len(spec.features),
            "artifact": aliases.get(spec.name, spec.name),
        }
        for spec in specifications
    ])


def _runnable_table(specifications) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "spec_index": index,
            "specification": spec.name,
            "kind": spec.kind,
            "variable": spec.variable,
            "reference": spec.reference,
            "features": json.dumps(spec.features),
        }
        for index, spec in enumerate(specifications)
    ])


def _same_hyperparameter(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None or pd.isna(actual)
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _validated_frozen_hyperparameters(
    manifest: dict[str, object], diagnostics: dict[str, object], tuning: pd.DataFrame,
) -> dict[str, float | None]:
    names = ("C", "l1_ratio", "benchmark_deviation_penalty")
    try:
        frozen = {name: diagnostics[name] for name in names}
    except KeyError as error:
        raise RuntimeError("full fit lacks frozen capability hyperparameters") from error
    c = float(frozen["C"])
    ratio = float(frozen["benchmark_deviation_penalty"])
    if not math.isfinite(ratio) or ratio <= 0:
        raise RuntimeError("invalid fitted benchmark ratio")
    penalty = diagnostics.get("penalty")
    if (not math.isfinite(c) or c <= 0
            or (penalty == "l1" and not _same_hyperparameter(frozen["l1_ratio"], 1.0))
            or (penalty == "ridge" and frozen["l1_ratio"] is not None)
            or penalty not in {"l1", "ridge"}):
        raise RuntimeError("invalid frozen capability penalty")
    recorded = manifest.get("frozen_hyperparameters")
    if (not isinstance(recorded, dict) or set(recorded) != set(names)
            or any(not _same_hyperparameter(recorded[name], frozen[name]) for name in names)):
        raise RuntimeError("manifest frozen hyperparameters do not match diagnostics")
    required = {"record_type", "selected", "converged", *names}
    if missing := required - set(tuning):
        raise RuntimeError(f"capability tuning missing columns: {sorted(missing)}")
    selected = tuning[tuning.selected.fillna(False).astype(bool)]
    if (len(selected) != 1 or not bool(selected.converged.iloc[0])
            or any(not _same_hyperparameter(selected.iloc[0][name], frozen[name])
                   for name in names)):
        raise RuntimeError("selected capability tuning does not match frozen hyperparameters")
    variance = tuning[tuning.record_type == "variance_component"]
    if (len(variance) != 1 or not bool(variance.converged.iloc[0])
            or not _same_hyperparameter(
                variance.benchmark_deviation_penalty.iloc[0], ratio,
            )):
        raise RuntimeError("capability variance-component diagnostics are invalid")
    return frozen


def prepare_fold(fold_index: int, run: CapabilityLesionRun) -> Path:
    """Prepare one capability fold from its completed, pinned full fit."""
    config = run.analysis.validate()
    _guard_execution(config)
    fold_names = _fold_names(config)
    if not 0 <= fold_index < len(fold_names):
        raise ValueError(f"fold_index must be in range({len(fold_names)})")
    _verify_sha256(run.responses, run.responses_sha256)
    if config.feature_space == "pc":
        _verify_sha256(config.pc_basis_root / "loadings.parquet", run.pc_loadings_sha256)
        _verify_sha256(
            config.pc_basis_root / "explained-variance.csv",
            run.pc_explained_variance_sha256,
        )
    fold = fold_names[fold_index]
    fold_dir = run.lesion_dir / "folds" / fold
    if (fold_dir / "manifest.json").is_file():
        _validate_fold(fold_dir, config, fold_index)
        return fold_dir
    if fold_dir.exists():
        _validate_prepared(fold_dir, config, fold_index)
        return fold_dir
    fit_dir = config.fit_dir / run.backend / fold
    full_manifest = _validate_full_fit(fit_dir, run)

    pairs, dictionary = _load_rows(config)
    model_index, item_fold = divmod(fold_index, config.item_folds)
    held_out_models = set(config.model_folds[model_index][1])
    train, test = prepare_capability_fold(
        pairs,
        pd.read_parquet(run.responses),
        dictionary,
        held_out_models,
        item_fold,
        config,
        fold,
    )
    train, test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, config.feature_space, config,
    )
    pd.testing.assert_frame_equal(
        prepared_dictionary.reset_index(drop=True),
        pd.read_parquet(fit_dir / "feature-dictionary.parquet").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        provenance["loadings"].reset_index(drop=True),
        pd.read_csv(fit_dir / "loadings.csv").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        provenance["explained_variance"].reset_index(drop=True),
        pd.read_csv(fit_dir / "explained-variance.csv").reset_index(drop=True),
    )
    features = prepared_dictionary.feature.astype(str).tolist()
    full_predictions = pd.read_parquet(fit_dir / "predictions.parquet")
    null_predictions = chance_predictions(test, features)
    pd.testing.assert_frame_equal(
        full_predictions.drop(columns="probability").reset_index(drop=True),
        null_predictions.drop(columns="probability").reset_index(drop=True),
    )
    specifications = capability_specifications(prepared_dictionary, config)
    unique, aliases = unique_specifications(specifications)
    diagnostics = json.loads((fit_dir / "diagnostics.json").read_text())
    tuning = pd.read_csv(fit_dir / "tuning.csv")
    frozen = _validated_frozen_hyperparameters(full_manifest, diagnostics, tuning)

    fold_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{fold}-prepare-", dir=fold_dir.parent))
    try:
        train.to_parquet(stage / "train.parquet", index=False)
        test.to_parquet(stage / "test.parquet", index=False)
        prepared_dictionary.to_parquet(stage / "feature-dictionary.parquet", index=False)
        _specification_table(specifications, aliases).to_csv(
            stage / "specifications.csv", index=False,
        )
        _runnable_table(unique).to_csv(stage / "run-specifications.csv", index=False)
        tuning.to_csv(stage / "tuning.csv", index=False)
        (stage / "frozen-hyperparameters.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        )
        (stage / "aliases.json").write_text(
            json.dumps(aliases, indent=2, sort_keys=True) + "\n"
        )
        for folder in ("predictions", "coefficients", "item-metrics", "diagnostics"):
            (stage / folder).mkdir()
        full_predictions.to_parquet(stage / "predictions/full.parquet", index=False)
        null_predictions.to_parquet(stage / "predictions/null.parquet", index=False)
        full_coefficients = pd.read_csv(fit_dir / "coefficients.csv")
        full_coefficients.to_parquet(stage / "coefficients/full.parquet", index=False)
        full_coefficients.iloc[0:0].to_parquet(
            stage / "coefficients/null.parquet", index=False,
        )
        _item_metrics(full_predictions).to_parquet(
            stage / "item-metrics/full.parquet", index=False,
        )
        null_metrics = _item_metrics(null_predictions)
        null_metrics["accuracy"] = 0.5
        null_metrics.to_parquet(stage / "item-metrics/null.parquet", index=False)
        shutil.copy2(fit_dir / "diagnostics.json", stage / "diagnostics/full.json")
        (stage / "diagnostics/null.json").write_text(
            json.dumps({"reference": "fixed_probability", "probability": 0.5},
                       indent=2, sort_keys=True) + "\n"
        )
        manifest_path = stage / "prepare-manifest.json"
        manifest_path.write_text(json.dumps({
            "state": "prepared",
            "fold": fold,
            "backend": run.backend,
            "config_signature": _settings_signature(_config_settings(config, fold_index)),
            "fit_manifest_sha256": _sha256(fit_dir / "manifest.json"),
            "full_fit_artifact_sha256": full_manifest["artifact_sha256"],
            "artifact_sha256": {
                str(path.relative_to(stage)): _sha256(path)
                for path in stage.rglob("*")
                if path.is_file() and path != manifest_path
            },
        }, indent=2, sort_keys=True) + "\n")
        os.replace(stage, fold_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return fold_dir


def run_specification(task_index: int, run: CapabilityLesionRun) -> Path | None:
    """Run one shared lesion shard with the capability target."""
    return _run_specification(
        task_index, run.analysis, task=CAPABILITY_TASK, write_diagnostics=True,
    )


def finalize_fold(fold_index: int, run: CapabilityLesionRun) -> Path:
    return _finalize_fold(fold_index, run.analysis)


def aggregate_lesions(run: CapabilityLesionRun) -> Path:
    return _aggregate_lesions(run.analysis)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "spec", "finalize", "summarize"):
        command = commands.add_parser(name)
        if name in {"prepare", "finalize"}:
            command.add_argument("--fold-index", type=int, required=True)
        elif name == "spec":
            command.add_argument("--task-index", type=int, required=True)
        command.add_argument("--matrix", type=Path, required=True)
        command.add_argument("--matrix-manifest-sha256", type=_sha256_arg, required=True)
        command.add_argument("--responses", type=Path, required=True)
        command.add_argument("--responses-sha256", type=_sha256_arg, required=True)
        command.add_argument("--full-result-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--feature-space", choices=("raw", "pc"), required=True)
        command.add_argument("--penalty", choices=("l1", "ridge"), required=True)
        command.add_argument("--max-iter", type=_positive_int, default=20_000)
        command.add_argument("--pc-basis-root", type=Path)
        command.add_argument("--pc-components", type=_positive_int)
        command.add_argument("--pc-loadings-sha256", type=_sha256_arg)
        command.add_argument("--pc-explained-variance-sha256", type=_sha256_arg)
    args = parser.parse_args(argv)
    pc_values = (
        args.pc_basis_root,
        args.pc_components,
        args.pc_loadings_sha256,
        args.pc_explained_variance_sha256,
    )
    if args.feature_space == "pc" and any(value is None for value in pc_values):
        parser.error("all PC basis arguments are required for PC features")
    if args.feature_space == "raw" and any(value is not None for value in pc_values):
        parser.error("PC-only arguments require --feature-space pc")
    config = replace(
        CONFIG,
        processed_matrix_root=args.matrix,
        processed_matrix_manifest_sha256=args.matrix_manifest_sha256,
        feature_space=args.feature_space,
        penalty=args.penalty,
        glm_max_iter=args.max_iter,
        execution_mode="execute",
        execution_target="swc_hpc",
        pc_basis_root=args.pc_basis_root,
        fit_pca_components=args.pc_components or CONFIG.fit_pca_components,
    )
    run = CapabilityLesionRun(
        config=config,
        responses=args.responses,
        responses_sha256=args.responses_sha256,
        full_result_root=args.full_result_root,
        output_root=args.output_root,
        pc_loadings_sha256=args.pc_loadings_sha256,
        pc_explained_variance_sha256=args.pc_explained_variance_sha256,
    )
    _guard_execution(run.analysis)
    if args.command == "prepare":
        prepare_fold(args.fold_index, run)
    elif args.command == "spec":
        run_specification(args.task_index, run)
    elif args.command == "finalize":
        finalize_fold(args.fold_index, run)
    else:
        aggregate_lesions(run)


if __name__ == "__main__":
    main()
