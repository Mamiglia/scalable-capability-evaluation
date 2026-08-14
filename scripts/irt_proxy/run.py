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
    quartile_analysis,
    rank_metrics,
    value_metrics,
)
from scores import analysis_frame, fit_easy_elos, item_votes, model_accuracy
from simulation import monte_carlo_budgets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "config.yaml"

EXPECTED_CSV = {
    "fit_difficulties.csv",
    "fit_capabilities.csv",
    "item_splits.csv",
    "coverage.csv",
    "elo_scores.csv",
    "rank_metrics.csv",
    "lomo_predictions.csv",
    "value_metrics.csv",
    "quartile_metrics.csv",
    "cross_benchmark_metrics.csv",
    "newer_stronger_metrics.csv",
    "monte_carlo_raw.csv",
    "monte_carlo_summary.csv",
}


def write_outputs(tables: dict[str, pd.DataFrame], output_directory: Path) -> None:
    """Atomically replace the thirteen declared analysis tables.

    A full Bayesian run is expensive. Writing through a temporary sibling keeps
    an interrupted run from leaving a half-written CSV under a trusted name.
    """
    if set(tables) != EXPECTED_CSV:
        missing = sorted(EXPECTED_CSV - set(tables))
        extra = sorted(set(tables) - EXPECTED_CSV)
        raise ValueError(f"output table mismatch; missing={missing}, extra={extra}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name not in {"fit_difficulties.csv", "fit_capabilities.csv", "item_splits.csv"}:
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
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Run the analysis in dependency order and return tables plus posteriors."""
    difficulties, capabilities, posteriors = fit_parameters(
        responses,
        floors=floors,
        warmup=warmup,
        samples=samples,
        seed=seed,
    )
    splits = assign_splits(difficulties)
    rounds = item_votes(votes)

    # The same response attempts feed all outcomes. Only the selected item bank
    # and the benchmark/year macro-averaging change between these tables.
    primary_items = splits.assign(subset=splits["half"])
    quartile_items = splits.assign(subset=splits["quartile"])
    full_items = splits.assign(subset="all")
    primary_accuracy = model_accuracy(responses, primary_items)
    quartile_accuracy = model_accuracy(responses, quartile_items)
    full_accuracy = model_accuracy(responses, full_items)

    response_cells = irt_cells(responses)
    response_counts = {
        fit_name: response_cells.loc[
            response_cells["benchmark"].isin(benchmarks), "model"
        ].nunique()
        for fit_name, benchmarks in FIT_BENCHMARKS.items()
    }

    elos, coverage = fit_easy_elos(rounds, splits, bank="easy")
    frame = analysis_frame(elos, primary_accuracy, coverage)
    coverage = _complete_coverage(coverage, frame, response_counts)
    # Rebuild once so every endpoint row inherits finalized coverage fields.
    frame = analysis_frame(elos, primary_accuracy, coverage)
    ranks = rank_metrics(frame)
    predictions = lomo_predictions(frame)
    values = value_metrics(predictions)

    q1_elos, q1_coverage = fit_easy_elos(rounds, splits, bank="Q1")
    q1_frame = analysis_frame(
        q1_elos,
        quartile_accuracy,
        q1_coverage,
        predictor_subset="Q1",
        target_subset="Q4",
    )
    q1_coverage = _complete_coverage(q1_coverage, q1_frame, response_counts)
    quartiles = quartile_analysis(q1_elos, quartile_accuracy, q1_coverage)
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
        "coverage.csv": pd.concat([coverage, q1_coverage], ignore_index=True),
        "elo_scores.csv": pd.concat([elos, q1_elos], ignore_index=True),
        "rank_metrics.csv": ranks,
        "lomo_predictions.csv": predictions,
        "value_metrics.csv": values,
        "quartile_metrics.csv": quartiles,
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
    parser.add_argument("--validate-only", action="store_true")
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
    counts = validate_inputs(responses, votes, metadata)
    print(counts.to_string(index=False))
    if args.validate_only:
        return 0

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
    )
    output_directory = ROOT / config["paths"]["output"]
    write_outputs(tables, output_directory)
    render_figures(tables, output_directory)
    write_irt_posterior(output_directory / "irt_posterior.npz", posteriors)

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
    print(f"\nWrote {len(EXPECTED_CSV)} tables and 6 figures to {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
