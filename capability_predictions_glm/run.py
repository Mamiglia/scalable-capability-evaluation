"""End-to-end fitting on a prepared updated-HF input bundle."""

import json
import shutil
import tempfile
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import AnalysisConfig, TARGET_QUARTILES
from .data import InputTables, load_inputs
from .evaluate import ability_table, calibrate_scores, matched_prediction_metrics
from .fit import fit_direct, fit_pairwise, predict_pairwise
from .preprocessing import (
    attach_direct_target,
    attach_pair_target,
    deduplicate_feature_pairs,
    item_quartiles,
    quartile_accuracy,
    split_pairs,
    split_transcripts,
)


METHODS = ["direct_glm", "pairwise_glm", "llm_judge", "q1_accuracy", "training_mean"]


@dataclass
class AnalysisResult:
    predictions: pd.DataFrame
    comparisons: pd.DataFrame
    pair_metrics: pd.DataFrame
    coefficients: pd.DataFrame
    tuning: pd.DataFrame
    calibrations: pd.DataFrame
    item_quartiles: pd.DataFrame


def load_report(inputs: InputTables, config: AnalysisConfig) -> dict[str, object]:
    """Describe data and approximate sparse-design load without fitting or writing."""
    quartiles = item_quartiles(inputs.responses)
    source_items = quartiles[quartiles.quartile == "Q1"]
    source_transcripts, _ = split_transcripts(inputs.transcripts, quartiles, set())
    source_pairs, _ = split_pairs(inputs.pairs, quartiles, set())
    benches = quartiles.bench.nunique()
    designs = {}
    peak_mib = 0.0
    for variant, features in inputs.variants.items():
        direct_columns = benches * (len(features) + 1) + len(source_items)
        pair_columns = len(features) * (benches + 1)
        # CSR data + indices are approximately 12 bytes per nonzero, plus a small row pointer.
        direct_mib = len(source_transcripts) * (len(features) + 2) * 12 / 2**20
        pair_mib = len(source_pairs) * (2 * len(features)) * 12 / 2**20
        peak_mib = max(peak_mib, direct_mib, pair_mib)
        designs[variant] = {
            "direct_columns": direct_columns, "pair_columns": pair_columns,
            "direct_sparse_mib": round(direct_mib, 1), "pair_sparse_mib": round(pair_mib, 1),
        }
    return {
        "rows": {"responses": len(inputs.responses), "q1_transcripts": len(source_transcripts),
                 "q1_judge_rows": len(source_pairs)},
        "feature_counts": {name: len(features) for name, features in inputs.variants.items()},
        "item_counts": quartiles.groupby(["bench", "quartile"]).size().unstack(fill_value=0).to_dict("index"),
        "target_quartiles": list(TARGET_QUARTILES), "outer_folds": len(config.model_folds),
        "designs": designs, "estimated_peak_sparse_matrix_mib": round(peak_mib, 1),
        "model_fits": len(config.model_folds) * len(inputs.variants) * len(TARGET_QUARTILES) * (
            config.inner_folds * (
                len(config.direct_elastic_alphas) * len(config.elastic_l1_ratios)
                + len(config.pair_elastic_cs) * len(config.elastic_l1_ratios)
                + len(config.ridge_alphas) + len(config.logistic_cs) + 2
            ) + 2),
    }


def _jsonable_config(config: AnalysisConfig) -> dict[str, object]:
    values = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def write_results(
    result: AnalysisResult,
    config: AnalysisConfig,
    source_manifest: dict[str, object],
) -> None:
    """Atomically publish the complete set of result tables."""
    output = Path(config.output_root)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing result directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tables = {
            "predictions.csv": result.predictions,
            "comparisons.csv": result.comparisons,
            "pair-metrics.csv": result.pair_metrics,
            "coefficients.csv": result.coefficients,
            "tuning.csv": result.tuning,
            "calibrations.csv": result.calibrations,
            "item-quartiles.csv": result.item_quartiles,
        }
        for name, table in tables.items():
            table.to_csv(stage / name, index=False)
        (stage / "manifest.json").write_text(json.dumps({
            "source_manifest": source_manifest, "config": _jsonable_config(config),
        }, indent=2, sort_keys=True, default=list) + "\n")
        stage.replace(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _direct_predictions(result, fold: str) -> pd.DataFrame:
    return (result.model_predictions
            .assign(fold=fold, target_quartile=result.target_quartile,
                    method="direct_glm", score=lambda x: x.predicted_logit)
            [["fold", "bench", "target_quartile", "method", "model", "score",
              "predicted_accuracy", "observed_accuracy"]])


def _mean_predictions(
    accuracies: pd.DataFrame,
    held_out: set[str],
    target_quartile: str,
    fold: str,
) -> pd.DataFrame:
    target = accuracies[accuracies.quartile == target_quartile]
    rows = []
    for bench, frame in target.groupby("bench", sort=False):
        train = frame[~frame.model.isin(held_out)]
        test = frame[frame.model.isin(held_out)]
        if train.empty or test.empty:
            continue
        output = test[["model", "accuracy"]].rename(columns={"accuracy": "observed_accuracy"})
        output["predicted_accuracy"] = train.accuracy.mean()
        output["score"] = train.accuracy.mean()
        rows.append(output.assign(fold=fold, bench=bench, target_quartile=target_quartile,
                                  method="training_mean"))
    columns = ["fold", "bench", "target_quartile", "method", "model", "score",
               "predicted_accuracy", "observed_accuracy"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def _comparators(source_pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "judge" not in source_pairs:
        raise ValueError("pair table needs judge rows for the LLM comparison")
    output = {str(judge): frame.copy() for judge, frame in source_pairs.groupby("judge", sort=False)}
    output["pooled_judges"] = source_pairs.copy()
    return output


def _pair_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    target: pd.DataFrame,
    metadata: dict[str, str],
) -> list[dict[str, object]]:
    keyed = frame.reset_index(drop=True).copy()
    keyed["_row"] = np.arange(len(keyed))
    labelled = attach_pair_target(keyed, target, metadata["target_quartile"])
    if labelled.empty:
        return []
    selected_probability = probability[labelled._row.to_numpy(int)]
    observed = labelled.left_stronger.to_numpy(int)
    judge_pick = labelled.pick_left.to_numpy(float)
    rows = []
    for method, prediction in (("pairwise_glm", selected_probability), ("llm_judge", judge_pick)):
        for bench, indices in labelled.groupby("bench", sort=False).groups.items():
            idx = labelled.index.get_indexer(indices)
            y, p = observed[idx], np.asarray(prediction)[idx]
            auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
            values = {
                "accuracy": accuracy_score(y, p >= 0.5),
                "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
                "auc": auc,
            }
            rows.extend(metadata | {"bench": bench, "method": method, "metric": metric,
                                    "value": float(value), "pair_count": len(y)}
                        for metric, value in values.items())
    return rows


def run_analysis(
    inputs: InputTables,
    config: AnalysisConfig,
    variants: tuple[str, ...] | None = None,
    folds: tuple[str, ...] | None = None,
    targets: tuple[str, ...] | None = None,
) -> AnalysisResult:
    """Run every fold, target quartile, feature variant, and judge comparator."""
    quartiles = item_quartiles(inputs.responses)
    accuracies = quartile_accuracy(inputs.responses, quartiles)
    source_pairs, _ = split_pairs(inputs.pairs, quartiles, set())
    comparators = _comparators(source_pairs)
    prediction_tables, comparison_tables = [], []
    coefficient_tables, tuning_tables, calibration_tables, pair_metric_rows = [], [], [], []

    for fold, members in config.model_folds:
        if folds and fold not in folds:
            continue
        held_out = set(members)
        train_transcripts, test_transcripts = split_transcripts(inputs.transcripts, quartiles, held_out)
        train_pairs, test_pairs = split_pairs(inputs.pairs, quartiles, held_out)
        train_unique = deduplicate_feature_pairs(train_pairs)
        test_unique = deduplicate_feature_pairs(test_pairs)

        selected_variants = variants or tuple(inputs.variants)
        for variant in selected_variants:
            feature_tuple = inputs.variants[variant]
            features = list(feature_tuple)
            for target_quartile in (targets or TARGET_QUARTILES):
                direct = fit_direct(
                    attach_direct_target(train_transcripts, accuracies, target_quartile),
                    attach_direct_target(test_transcripts, accuracies, target_quartile),
                    features, target_quartile, config)
                pair = fit_pairwise(
                    attach_pair_target(train_unique, accuracies, target_quartile),
                    attach_pair_target(test_unique, accuracies, target_quartile),
                    features, target_quartile, config)
                direct_predictions = _direct_predictions(direct, fold)
                q1_scores = (accuracies[accuracies.quartile == "Q1"]
                             .rename(columns={"accuracy": "score"})
                             .assign(method="q1_accuracy"))
                mean_predictions = _mean_predictions(accuracies, held_out, target_quartile, fold)

                coefficient_tables.extend([
                    direct.coefficients.assign(fold=fold, variant=variant,
                                               target_quartile=target_quartile, model="direct_glm"),
                    pair.coefficients.assign(fold=fold, variant=variant,
                                             target_quartile=target_quartile, model="pairwise_glm"),
                ])
                tuning_tables.extend([
                    direct.tuning.assign(fold=fold, variant=variant,
                                         target_quartile=target_quartile, model="direct_glm"),
                    pair.tuning.assign(fold=fold, variant=variant,
                                       target_quartile=target_quartile, model="pairwise_glm"),
                ])

                for comparator, judge_rows in comparators.items():
                    pair_probability = predict_pairwise(pair, judge_rows)
                    pair_rows = judge_rows.copy()
                    pair_rows["probability"] = pair_probability
                    pair_scores = ability_table(pair_rows, "probability", "pairwise_glm")
                    judge_scores = ability_table(judge_rows, "pick_left", "llm_judge")
                    calibrated = []
                    for scores in (pair_scores, judge_scores, q1_scores):
                        predictions, calibration = calibrate_scores(
                            scores, accuracies, held_out, target_quartile, fold)
                        calibrated.append(predictions)
                        calibration_tables.append(calibration.assign(
                            variant=variant, comparator=comparator))
                    compared = pd.concat(
                        [direct_predictions, *calibrated, mean_predictions], ignore_index=True)
                    compared = compared.assign(variant=variant, comparator=comparator)
                    prediction_tables.append(compared)
                    metrics = matched_prediction_metrics(compared, METHODS, comparison=comparator)
                    comparison_tables.append(metrics.assign(
                        fold=fold, variant=variant, target_quartile=target_quartile,
                        comparator=comparator))
                    pair_metric_rows.extend(_pair_metrics(
                        judge_rows, pair_probability, accuracies,
                        {"fold": fold, "variant": variant, "target_quartile": target_quartile,
                         "comparator": comparator}))

    def concat(tables):
        return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()

    return AnalysisResult(
        predictions=concat(prediction_tables), comparisons=concat(comparison_tables),
        pair_metrics=pd.DataFrame(pair_metric_rows), coefficients=concat(coefficient_tables),
        tuning=concat(tuning_tables), calibrations=concat(calibration_tables),
        item_quartiles=quartiles,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate and describe inputs by default; fit only with explicit --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true", help="run all fits after the load review")
    parser.add_argument("--full-only", action="store_true",
                        help="phase 1: tune and fit only the unlesioned annotated feature set")
    parser.add_argument("--fold", action="append", choices=[name for name, _ in AnalysisConfig().model_folds])
    parser.add_argument("--target", action="append", choices=list(TARGET_QUARTILES))
    args = parser.parse_args(argv)
    values = {}
    if args.input_root:
        values["input_root"] = args.input_root
    if args.output_root:
        values["output_root"] = args.output_root
    config = AnalysisConfig(**({**asdict(AnalysisConfig()), **values}))
    inputs = load_inputs(config)
    print(json.dumps(load_report(inputs, config), indent=2, sort_keys=True))
    if args.execute:
        variants = ("with_llm_annotations",) if args.full_only else None
        write_results(run_analysis(inputs, config, variants,
                                   tuple(args.fold) if args.fold else None,
                                   tuple(args.target) if args.target else None),
                      config, inputs.manifest)
        print(f"wrote {config.output_root}")
    return 0
