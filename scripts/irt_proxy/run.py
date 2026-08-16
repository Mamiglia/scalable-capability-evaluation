"""Command-line orchestration for the IRT easy-to-hard analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from figures import render_figures
from inputs import (
    load_config,
    load_metadata,
    load_responses,
    load_votes,
    validate_inputs,
)
from item_banks import FIT_BENCHMARKS, assign_splits, fit_parameters, irt_cells
from metrics import (
    cross_benchmark_analysis,
    lomo_predictions,
    newer_stronger_analysis,
    partition_analysis,
    rank_metrics,
    value_metrics,
)
from nulls import analytic_null_metrics
from scores import (
    PREDICTOR_ACCURACY_THRESHOLD,
    analysis_frame,
    filter_judges_above_chance,
    fit_saturated_elos,
    item_votes,
    model_accuracy,
)
from simulation import monte_carlo_budgets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "config.yaml"

EXPECTED_CSV = {
    "fit_difficulties.csv",
    "fit_capabilities.csv",
    "item_splits.csv",
    "judge_diagnostics.csv",
    "coverage.csv",
    "elo_scores.csv",
    "rank_metrics.csv",
    "lomo_predictions.csv",
    "value_metrics.csv",
    "quartile_metrics.csv",
    "partition_metrics.csv",
    "partition_predictions.csv",
    "null_metrics.csv",
    "cross_benchmark_metrics.csv",
    "newer_stronger_metrics.csv",
    "monte_carlo_raw.csv",
    "monte_carlo_summary.csv",
}

PARTITION_SPECS = (
    ("tertile", "T", 3, "T2–T3"),
    ("quartile", "Q", 4, "Q2–Q4"),
    ("quintile", "P", 5, "P2–P5"),
)


def load_prefit_tables(
    table_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the saved IRT results needed for a downstream-only refresh."""
    return tuple(
        pd.read_csv(table_directory / filename)
        for filename in (
            "fit_difficulties.csv",
            "fit_capabilities.csv",
            "item_splits.csv",
        )
    )


def partition_item_banks(splits: pd.DataFrame) -> pd.DataFrame:
    """Return individual slices plus each pooled harder-than-first slice."""
    individual = [
        splits.assign(subset=splits[column])
        for column, _, _, _ in PARTITION_SPECS
    ]
    pooled = [
        splits[splits[column] != f"{prefix}1"].assign(subset=pooled_bin)
        for column, prefix, _, pooled_bin in PARTITION_SPECS
    ]
    return pd.concat([*individual, *pooled], ignore_index=True)


def output_directories(
    output_directory: Path,
    explicit_cutoff: float | None,
) -> tuple[Path, Path]:
    """Keep exploratory cutoffs beside, not on top of, the primary results."""
    if explicit_cutoff is None:
        return output_directory / "tables", output_directory / "figures"
    tag = f"cut{round(explicit_cutoff * 100)}"
    return output_directory / "tables" / tag, output_directory / "figures" / tag


def write_outputs(tables: dict[str, pd.DataFrame], output_directory: Path) -> None:
    """Atomically replace the declared analysis tables.

    A full Bayesian run is expensive. Writing through a temporary sibling keeps
    an interrupted run from leaving a half-written CSV under a trusted name.
    """
    if set(tables) != EXPECTED_CSV:
        missing = sorted(EXPECTED_CSV - set(tables))
        extra = sorted(set(tables) - EXPECTED_CSV)
        raise ValueError(f"output table mismatch; missing={missing}, extra={extra}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name not in {
            "fit_difficulties.csv",
            "fit_capabilities.csv",
            "item_splits.csv",
            "judge_diagnostics.csv",
        }:
            if "component" not in table:
                raise ValueError(f"{name} must carry component")
        temporary = output_directory / f".{name}.tmp"
        table.to_csv(temporary, index=False)
        temporary.replace(output_directory / name)


def build_model_info(
    metadata: pd.DataFrame,
    full_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    """Combine recorded release dates with measured full-bank accuracy.

    Release dates are properties of model endpoints, not viewer collections. A
    disagreement between indexes is therefore surfaced instead of resolved by
    arbitrary source order.
    """
    dated = metadata.dropna(subset=["release"])[["model", "release"]]
    conflicts = dated.groupby("model")["release"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"conflicting release dates for {conflicts.index.tolist()}")

    releases = dated.drop_duplicates("model")
    accuracy = full_accuracy[
        (full_accuracy["irt_model"] == "fixed_3pl")
        & (full_accuracy["subset"] == "all")
    ][["fit", "model", "accuracy"]].rename(
        columns={"accuracy": "full_accuracy"},
    )
    return accuracy.merge(releases, on="model", how="left")[[
        "fit",
        "model",
        "release",
        "full_accuracy",
    ]]


def write_irt_posterior(path: Path, fits: dict[str, dict]) -> None:
    arrays = {}
    for prefix, values in fits.items():
        arrays[f"{prefix}__theta"] = values["theta"]
        arrays[f"{prefix}__a"] = values["a"]
        arrays[f"{prefix}__b"] = values["b"]
        arrays[f"{prefix}__floor"] = values["floor"]
        arrays[f"{prefix}__models"] = np.asarray(values["models"], dtype=str)
        arrays[f"{prefix}__items"] = np.asarray(
            [f"{benchmark}:{item}" for benchmark, item in values["items"]], dtype=str,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _complete_coverage(
    coverage: pd.DataFrame,
    frame: pd.DataFrame,
    response_counts: dict[str, int],
) -> pd.DataFrame:
    """Add response-pool and final matched-population counts to graph coverage."""
    key = ["irt_model", "fit", "bank", "judge", "component"]
    matched = (
        frame.groupby(key, as_index=False)["model"]
        .nunique()
        .rename(columns={"model": "n_matched_models"})
    )
    result = coverage.merge(matched, on=key, how="left")
    result["n_matched_models"] = result["n_matched_models"].fillna(0).astype(int)
    result["n_response_models"] = result["fit"].map(response_counts)

    # Eligibility is based on the final matched models, not merely all nodes in
    # the easy-item comparison component.
    result["eligible_rank"] = result["n_matched_models"] >= 3
    result["eligible_lomo"] = result["n_matched_models"] >= 4
    result["reason"] = ""
    result.loc[~result["eligible_lomo"], "reason"] = "fewer than 4 matched models"
    result.loc[~result["eligible_rank"], "reason"] = "fewer than 3 matched models"
    return result


def build_tables(
    responses: pd.DataFrame,
    votes: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    floors: dict[str, float],
    warmup: int,
    samples: int,
    resamples: int,
    seed: int,
    predictor_cutoff: float = PREDICTOR_ACCURACY_THRESHOLD,
    prefit: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Run the analysis in dependency order and return tables plus posteriors."""
    if prefit is None:
        difficulties, capabilities, posteriors = fit_parameters(
            responses,
            floors=floors,
            warmup=warmup,
            samples=samples,
            seed=seed,
        )
        splits = assign_splits(difficulties)
    else:
        difficulties, capabilities, splits = prefit
        posteriors = {}
    rounds, judge_diagnostics = filter_judges_above_chance(
        item_votes(votes), metadata,
    )

    # One canonical response feeds IRT and every measured-accuracy endpoint.
    # Repeated eval epochs would turn the easy baseline into a much more precise
    # benchmark than a model sees in an ordinary single run.
    response_cells = irt_cells(responses)
    primary_items = splits.assign(subset=splits["half"])
    full_items = splits.assign(subset="all")
    primary_accuracy = model_accuracy(response_cells, primary_items)
    partition_accuracy = model_accuracy(
        response_cells,
        partition_item_banks(splits),
    )
    full_accuracy = model_accuracy(response_cells, full_items)

    response_counts = {
        fit_name: response_cells.loc[
            response_cells["benchmark"].isin(benchmarks), "model"
        ].nunique()
        for fit_name, benchmarks in FIT_BENCHMARKS.items()
    }

    elos, coverage = fit_saturated_elos(
        rounds,
        splits,
        primary_accuracy,
        bank="easy",
        threshold=predictor_cutoff,
    )
    frame = analysis_frame(elos, primary_accuracy, coverage)
    coverage = _complete_coverage(coverage, frame, response_counts)
    # Rebuild once so every endpoint row inherits finalized coverage fields.
    frame = analysis_frame(elos, primary_accuracy, coverage)
    ranks = rank_metrics(frame)
    primary_predictions = lomo_predictions(frame)
    values = value_metrics(primary_predictions)

    partition_elos = []
    partition_coverages = []
    half_metrics, half_partition_predictions = partition_analysis(
        elos,
        primary_accuracy,
        coverage,
        predictor_bin="easy",
        target_bins=("hard",),
    )
    partition_metrics = [half_metrics]
    partition_predictions = [half_partition_predictions]
    for _, prefix, n_bins, pooled_bin in PARTITION_SPECS:
        predictor_bin = f"{prefix}1"
        target_bins = tuple(
            f"{prefix}{number}" for number in range(2, n_bins + 1)
        ) + (pooled_bin,)
        bin_elos, bin_coverage = fit_saturated_elos(
            rounds,
            splits,
            partition_accuracy,
            bank=predictor_bin,
            threshold=predictor_cutoff,
        )
        final_frame = analysis_frame(
            bin_elos,
            partition_accuracy,
            bin_coverage,
            predictor_subset=predictor_bin,
            target_subset=target_bins[-1],
        )
        bin_coverage = _complete_coverage(bin_coverage, final_frame, response_counts)
        metrics, bin_predictions = partition_analysis(
            bin_elos,
            partition_accuracy,
            bin_coverage,
            predictor_bin=predictor_bin,
            target_bins=target_bins,
        )
        partition_elos.append(bin_elos)
        partition_coverages.append(bin_coverage)
        partition_metrics.append(metrics)
        partition_predictions.append(bin_predictions)

    partition_metrics = pd.concat(partition_metrics, ignore_index=True)
    partition_predictions = pd.concat(partition_predictions, ignore_index=True)
    all_elos = pd.concat([elos, *partition_elos], ignore_index=True)
    null_scope = (
        partition_predictions["irt_model"].eq("fixed_3pl")
        & partition_predictions["fit"].eq("gpqa")
    )
    null_metrics = analytic_null_metrics(partition_predictions[null_scope])
    quartiles = partition_metrics[
        partition_metrics["partition"] == "quartile"
    ].rename(columns={"target_bin": "target_quartile"})
    cross_benchmark = cross_benchmark_analysis(elos, primary_accuracy)

    # Consensus has no release date or benchmark accuracy of its own, so the
    # temporal stress test is intentionally a per-judge analysis.
    model_info = build_model_info(metadata, full_accuracy)
    newer_stronger = newer_stronger_analysis(
        frame[frame["judge"] != "consensus"],
        model_info,
    )

    monte_carlo_raw, monte_carlo_summary = monte_carlo_budgets(
        rounds,
        splits,
        primary_accuracy,
        resamples=resamples,
        seed=seed,
    )

    return {
        "fit_difficulties.csv": difficulties,
        "fit_capabilities.csv": capabilities,
        "item_splits.csv": splits,
        "judge_diagnostics.csv": judge_diagnostics,
        "coverage.csv": pd.concat([coverage, *partition_coverages], ignore_index=True),
        "elo_scores.csv": all_elos,
        "rank_metrics.csv": ranks,
        "lomo_predictions.csv": primary_predictions,
        "value_metrics.csv": values,
        "quartile_metrics.csv": quartiles,
        "partition_metrics.csv": partition_metrics,
        "partition_predictions.csv": partition_predictions,
        "null_metrics.csv": null_metrics,
        "cross_benchmark_metrics.csv": cross_benchmark,
        "newer_stronger_metrics.csv": newer_stronger,
        "monte_carlo_raw.csv": monte_carlo_raw,
        "monte_carlo_summary.csv": monte_carlo_summary,
    }, posteriors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse run-size overrides while leaving scientific choices in YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--predictor-cutoff",
        type=float,
        help="strict easiest-slice accuracy cutoff; outputs go under cutXX",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete-graph",
        action="store_true",
        help="render provisional results before four-opponent collection finishes",
    )
    parser.add_argument(
        "--reuse-irt",
        action="store_true",
        help="reuse saved IRT tables and refresh only downstream analyses",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, run the analysis, and materialize tables and figures."""
    args = parse_args(argv)
    config = load_config(args.config)
    viewer_root = ROOT / config["paths"]["viewer_data"]
    datasets = config["viewer_datasets"]

    responses = load_responses(ROOT, config)
    votes = load_votes(viewer_root, datasets)
    metadata = load_metadata(viewer_root, datasets)
    counts = validate_inputs(
        responses,
        votes,
        metadata,
        require_four_opponents=not args.allow_incomplete_graph,
    )
    print(counts.to_string(index=False))
    if args.validate_only:
        return 0

    output_directory = ROOT / config["paths"]["output"]
    prefit = (
        load_prefit_tables(output_directory / "tables")
        if args.reuse_irt
        else None
    )
    irt_config = config["irt"]
    tables, posteriors = build_tables(
        responses,
        votes,
        metadata,
        floors=irt_config["floors"],
        warmup=args.warmup if args.warmup is not None else irt_config["warmup"],
        samples=args.samples if args.samples is not None else irt_config["samples"],
        resamples=(
            args.resamples
            if args.resamples is not None
            else config["monte_carlo"]["resamples"]
        ),
        seed=args.seed if args.seed is not None else irt_config["seed"],
        predictor_cutoff=(
            args.predictor_cutoff
            if args.predictor_cutoff is not None
            else PREDICTOR_ACCURACY_THRESHOLD
        ),
        prefit=prefit,
    )
    table_directory, figure_directory = output_directories(
        output_directory,
        args.predictor_cutoff,
    )
    write_outputs(tables, table_directory)
    render_figures(tables, figure_directory)
    if posteriors:
        write_irt_posterior(
            output_directory / "posterior" / "irt_posterior.npz",
            posteriors,
        )

    print("\nJudge inclusion (>50% pair-macro agreement):")
    print(tables["judge_diagnostics.csv"].to_string(index=False))

    consensus = tables["coverage.csv"].query("judge == 'consensus'")
    print("\nPost-filter consensus components:")
    print(consensus[[
        "irt_model",
        "fit",
        "bank",
        "component",
        "n_response_models",
        "n_graph_models",
        "n_component_models",
        "n_matched_models",
    ]].to_string(index=False))
    mode = (
        "downstream tables and split figures"
        if prefit
        else "tables, posterior, and split figures"
    )
    print(f"\nWrote {mode} to {table_directory} and {figure_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
