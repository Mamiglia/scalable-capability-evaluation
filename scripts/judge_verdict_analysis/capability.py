"""Leakage-safe capability difference for companion verdict fits."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_dm import _deduplicate_judges
from .config import AnalysisConfig, CONFIG
from .fit import (
    ARTIFACTS, _guard_execution, _load_rows, _outer_folds, _positive_int,
    _resume_existing, _write_result, aggregate_fits,
)
from .fit_common import FitResult
from .fit_common import prepare_features
from .fit_frequentist import fit_sparse


CAPABILITY_FEATURE = "capability_difference"


def prepare_capability_difference(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add capability to one outer fold; pass the result to a frozen full-model refit."""
    required = {"bench", "item", "model_left", "model_right", "accuracy_left", "accuracy_right"}
    for name, frame in (("train", train), ("test", test)):
        if missing := sorted(required - set(frame)):
            raise ValueError(f"{name} rows missing capability columns: {missing}")

    sides = []
    for side in ("left", "right"):
        sides.append(train[["bench", "item", f"model_{side}", f"accuracy_{side}"]].rename(
            columns={f"model_{side}": "model", f"accuracy_{side}": "score"},
        ))
    scores = pd.concat(sides, ignore_index=True).drop_duplicates()
    if scores.duplicated(["bench", "model", "item"]).any():
        raise ValueError("outer-training capability scores disagree for a model/item")
    scores["score"] = pd.to_numeric(scores.score, errors="raise")
    if not np.isfinite(scores.score).all():
        raise ValueError("outer-training capability scores must be finite")
    scaling = scores.groupby("bench", sort=False, as_index=False).agg(
        score_mean=("score", "mean"), score_sd=("score", "std"), score_count=("score", "size"),
    )
    if (~np.isfinite(scaling.score_sd) | scaling.score_sd.eq(0)).any():
        raise ValueError("each benchmark needs varying outer-training capability scores")

    parameters = scaling.set_index("bench")

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        matched = frame[["bench"]].join(parameters[["score_mean", "score_sd"]], on="bench")
        if matched[["score_mean", "score_sd"]].isna().any(axis=None):
            raise ValueError("test benchmark has no outer-training capability scaling")
        left = (frame.accuracy_left.to_numpy(float) - matched.score_mean) / matched.score_sd
        right = (frame.accuracy_right.to_numpy(float) - matched.score_mean) / matched.score_sd
        output = frame.copy()
        output[CAPABILITY_FEATURE] = left - right
        return output

    prepared_train, prepared_test = apply(train), apply(test)
    delta_sd = _deduplicate_judges(prepared_train)[CAPABILITY_FEATURE].std(ddof=1)
    if not np.isfinite(delta_sd) or delta_sd == 0:
        raise ValueError("outer-training capability differences need non-zero pair variation")
    prepared_train[CAPABILITY_FEATURE] /= delta_sd
    prepared_test[CAPABILITY_FEATURE] /= delta_sd
    return prepared_train, prepared_test, scaling.assign(delta_sd=float(delta_sd))


def fit_capability_companion(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    frozen_hyperparameters: dict[str, float | None],
    config: AnalysisConfig,
) -> tuple[FitResult, pd.DataFrame]:
    """Add capability and refit only the full model with frozen hyperparameters."""
    if CAPABILITY_FEATURE in features:
        raise ValueError(f"{CAPABILITY_FEATURE} is reserved for the companion fit")
    prepared_train, prepared_test, scaling = prepare_capability_difference(train, test)
    result = fit_sparse(
        prepared_train, prepared_test, [*features, CAPABILITY_FEATURE],
        config.penalty, frozen_hyperparameters, config,
    )
    if not result.diagnostics["converged"]:
        raise RuntimeError("capability companion fit did not converge")
    result.backend = f"{result.backend}-capability"
    return result, scaling


def run_fold(fold_index: int, config: AnalysisConfig = CONFIG) -> Path:
    """Refit one completed full-model fold with capability added and no retuning."""
    config = config.validate()
    _guard_execution(config)
    rows, dictionary = _load_rows(config)
    folds = list(_outer_folds(rows, config))
    if not 0 <= fold_index < len(folds):
        raise ValueError(f"fold_index must be in range({len(folds)})")
    fold, train, test = folds[fold_index]
    primary_backend = f"{config.feature_space}-{config.penalty}"
    backend = f"{primary_backend}-capability"
    output = config.fit_dir / backend / fold
    if _resume_existing(
        output, ARTIFACTS, replace(config, resume_completed_folds=True), backend,
    ):
        return output
    primary = config.fit_dir / primary_backend / fold
    _resume_existing(primary, ARTIFACTS, replace(config, resume_completed_folds=True))
    primary_manifest = json.loads((primary / "manifest.json").read_text())
    frozen = primary_manifest.get("frozen_hyperparameters")
    if not isinstance(frozen, dict):
        raise RuntimeError(f"primary fold has no frozen hyperparameters: {primary}")

    prepared_train, prepared_test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, config.feature_space, config,
    )
    features = prepared_dictionary.feature.astype(str).tolist()
    result, scaling = fit_capability_companion(
        prepared_train, prepared_test, features, frozen, config,
    )
    result.tuning = pd.read_csv(primary / "tuning.csv")
    companion_dictionary = pd.concat([
        prepared_dictionary,
        pd.DataFrame({"feature": [CAPABILITY_FEATURE], "block": ["capability"]}),
    ], ignore_index=True)
    _write_result(
        result, train, test, config,
        prepared_dictionary=companion_dictionary,
        provenance=provenance | {
            "frozen_hyperparameters": frozen,
            "capability_scaling": scaling.to_dict(orient="records"),
            "primary_fit_manifest_sha256": hashlib.sha256(
                (primary / "manifest.json").read_bytes()).hexdigest(),
        },
    )
    return output


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
        CONFIG, results_dir=args.results_dir, feature_space=args.feature_space,
        penalty=args.penalty, processed_matrix_manifest_sha256=args.manifest_sha256,
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
    backend = f"{config.feature_space}-{config.penalty}-capability"
    if args.command == "fold":
        run_fold(args.fold_index, replace(
            config, processed_matrix_root=args.processed_matrix_root,
            execution_mode="execute", execution_target="swc_hpc",
        ))
    else:
        aggregate_fits(config, backend)


if __name__ == "__main__":
    main()
