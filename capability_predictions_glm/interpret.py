"""Fixed-hyperparameter lesions and singular capability predictors."""

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANNOTATION_BLOCKS, CONFIG, TARGET_QUARTILES, AnalysisConfig
from .data import InputTables, load_inputs
from .fit import fit_direct_fixed, fit_pairwise_fixed
from .preprocessing import (
    attach_direct_target,
    attach_pair_target,
    deduplicate_feature_pairs,
    item_quartiles,
    quartile_accuracy,
    split_pairs,
    split_transcripts,
)


FAMILIES = {
    "annotations": {"rubric", "intermediate-errors"},
    "hand-engineered-text": {"surface", "structure", "exposition", "derivation"},
    "clause": {"clause-descriptors", "clause-motifs", "clause-plan"},
    "clause-structure": {"clause-descriptors", "clause-motifs"},
    "planning": {"clause-plan"},
}


@dataclass(frozen=True)
class Specification:
    name: str
    kind: str
    features: tuple[str, ...]


@dataclass
class InterpretationResult:
    direct_predictions: pd.DataFrame
    pair_item_predictions: pd.DataFrame
    coefficients: pd.DataFrame


def _slug(value: str) -> str:
    return value.replace("_", "-").replace("/", "-")


def build_specifications(dictionary: pd.DataFrame) -> list[Specification]:
    """Predeclare full/no-annotation, feature, block, and family models."""
    full = tuple(dictionary.feature.astype(str))
    by_block = {str(block): tuple(group.feature.astype(str))
                for block, group in dictionary.groupby("block", sort=False)}
    specs = [Specification("full", "full", full)]
    no_annotations = tuple(row.feature for row in dictionary.itertuples()
                           if row.block not in ANNOTATION_BLOCKS)
    specs.append(Specification("no-annotations", "lesion-family", no_annotations))
    for feature in full:
        specs.append(Specification(f"lesion-feature-{_slug(feature)}", "lesion-feature",
                                   tuple(name for name in full if name != feature)))
        specs.append(Specification(f"singular-feature-{_slug(feature)}", "singular-feature", (feature,)))
    for block, features in by_block.items():
        specs.append(Specification(f"lesion-block-{_slug(block)}", "lesion-block",
                                   tuple(name for name in full if name not in set(features))))
        specs.append(Specification(f"singular-block-{_slug(block)}", "singular-block", features))
    for family, blocks in FAMILIES.items():
        features = tuple(row.feature for row in dictionary.itertuples() if row.block in blocks)
        if not features:
            continue
        if family != "annotations":  # no-annotations is the same lesion, already named plainly
            specs.append(Specification(f"lesion-family-{family}", "lesion-family",
                                       tuple(name for name in full if name not in set(features))))
        specs.append(Specification(f"singular-family-{family}", "singular-family", features))
    return specs


def read_frozen_penalties(root: str | Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Read the full-set selected ridge settings for each fold×target cell."""
    root = Path(root)
    output = {}
    for path in root.glob("model-fold-*-Q*/tuning.csv"):
        tuning = pd.read_csv(path)
        direct = tuning[(tuning.model == "direct_glm") & (tuning.stage == "ridge")
                        & (tuning.record_type == "tuning") & tuning.selected.fillna(False)]
        pair = tuning[(tuning.model == "pairwise_glm") & (tuning.stage == "ridge")
                      & (tuning.record_type == "tuning") & tuning.selected.fillna(False)]
        if len(direct) != 1 or len(pair) != 1:
            raise ValueError(f"cell does not have one selected ridge setting: {path.parent}")
        fold, target = path.parent.name.rsplit("-", 1)
        output[(fold, target)] = (float(direct.alpha.iloc[0]), float(pair.C.iloc[0]))
    expected = {(fold, target) for fold, _ in CONFIG.model_folds for target in TARGET_QUARTILES}
    if set(output) != expected:
        raise ValueError(f"frozen penalties cover {len(output)} cells, expected {len(expected)}")
    return output


def run_fixed_spec(
    inputs: InputTables,
    config: AnalysisConfig,
    specification: Specification,
    penalties: dict[tuple[str, str], tuple[float, float]],
) -> InterpretationResult:
    """Fit one fixed feature specification over every fold and target."""
    quartiles = item_quartiles(inputs.responses)
    accuracies = quartile_accuracy(inputs.responses, quartiles)
    direct_rows, pair_rows, coefficient_rows = [], [], []
    features = list(specification.features)
    for fold, members in config.model_folds:
        held_out = set(members)
        train_t, test_t = split_transcripts(inputs.transcripts, quartiles, held_out)
        train_p, test_p = split_pairs(inputs.pairs, quartiles, held_out)
        train_p, test_p = deduplicate_feature_pairs(train_p), deduplicate_feature_pairs(test_p)
        for target in TARGET_QUARTILES:
            ridge_alpha, ridge_c = penalties[(fold, target)]
            direct = fit_direct_fixed(
                attach_direct_target(train_t, accuracies, target),
                attach_direct_target(test_t, accuracies, target),
                features, target, config, ridge_alpha=ridge_alpha)
            pair = fit_pairwise_fixed(
                attach_pair_target(train_p, accuracies, target),
                attach_pair_target(test_p, accuracies, target),
                features, target, config, ridge_c=ridge_c)
            direct_rows.append(direct.model_predictions.assign(
                specification=specification.name, kind=specification.kind,
                fold=fold, target_quartile=target))
            row = pair.predictions.copy()
            probability = np.clip(row.probability.to_numpy(float), 1e-12, 1 - 1e-12)
            observed = row.observed.to_numpy(int)
            row["log_loss"] = -(observed * np.log(probability) + (1 - observed) * np.log1p(-probability))
            row["correct"] = (probability >= 0.5) == observed
            pair_rows.append(row.groupby(["bench", "item"], as_index=False).agg(
                log_loss=("log_loss", "mean"), accuracy=("correct", "mean"), pairs=("correct", "size"))
                .assign(specification=specification.name, kind=specification.kind,
                        fold=fold, target_quartile=target))
            coefficient_rows.extend([
                direct.coefficients.assign(specification=specification.name, kind=specification.kind,
                                           task="direct", fold=fold, target_quartile=target),
                pair.coefficients.assign(specification=specification.name, kind=specification.kind,
                                         task="pairwise", fold=fold, target_quartile=target),
            ])
    return InterpretationResult(pd.concat(direct_rows, ignore_index=True),
                                pd.concat(pair_rows, ignore_index=True),
                                pd.concat(coefficient_rows, ignore_index=True))


def write_result(result: InterpretationResult, specification: Specification, output: str | Path) -> None:
    """Atomically write one independent interpretation specification."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{specification.name}-", dir=output.parent))
    try:
        result.direct_predictions.to_parquet(stage / "direct-predictions.parquet", index=False)
        result.pair_item_predictions.to_parquet(stage / "pair-item-predictions.parquet", index=False)
        result.coefficients.to_parquet(stage / "coefficients.parquet", index=False)
        (stage / "manifest.json").write_text(json.dumps({
            "name": specification.name, "kind": specification.kind,
            "features": list(specification.features), "feature_count": len(specification.features),
            "hyperparameters": "frozen from unlesioned full model",
        }, indent=2, sort_keys=True) + "\n")
        stage.replace(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--specification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = replace(CONFIG, input_root=args.input_root)
    inputs = load_inputs(config)
    specs = {spec.name: spec for spec in build_specifications(inputs.dictionary)}
    if args.list:
        print("\n".join(specs))
        return 0
    if not args.specification or not args.output:
        parser.error("--specification and --output are required unless --list is used")
    specification = specs[args.specification]
    result = run_fixed_spec(inputs, config, specification, read_frozen_penalties(args.full_results))
    write_result(result, specification, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
