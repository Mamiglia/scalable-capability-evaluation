"""Add deterministic-regressor views from the existing saved fits."""

import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from .config import ROOT
from .contributions import (
    plot_benchmark_grid,
    plot_coefficient_heatmap,
    plot_cross_benchmark_profiles,
    plot_each_block,
    plot_feature_overview as plot_capability_feature_overview,
    plot_groups as plot_capability_groups,
    shared_feature_selection,
)
from .judge_comparison import (
    FILL_ONLY,
    _ordinary,
    _save,
    _style,
    _crossed_cosine_summary,
    coefficient_alignment,
    coefficient_comparison,
    coefficient_statistics,
    plot_comparisons,
)

from scripts.judge_verdict_analysis.result_plots import (
    coefficient_selection_stability,
    load_benchmark_feature_effects,
    plot_benchmark_feature_grid,
    plot_feature_blocks,
    plot_feature_overview as plot_judge_feature_overview,
    plot_group_effects,
    plot_judge_benchmark_grids,
    plot_stability,
    summarize_effective_coefficients,
)


ANNOTATION_BLOCKS = frozenset({"rubric", "intermediate-errors", "annotations"})
PRESENTATION_EXCLUSIONS = frozenset({"bare_answer"})


def deterministic_feature_names(dictionary: pd.DataFrame) -> tuple[str, ...]:
    """Return feature names from the seven non-annotation blocks in saved order."""
    if missing := {"feature", "block"} - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    selected = dictionary[
        ~dictionary.block.isin(ANNOTATION_BLOCKS)
        & ~dictionary.feature.isin(PRESENTATION_EXCLUSIONS)
    ]
    return tuple(selected.feature.astype(str))


def deterministic_rows(
    table: pd.DataFrame, names: tuple[str, ...], *, column: str = "feature",
) -> pd.DataFrame:
    if column not in table:
        raise ValueError(f"table missing deterministic feature column: {column}")
    return table[table[column].isin(names)].copy()


def deterministic_group_rows(table: pd.DataFrame) -> pd.DataFrame:
    if missing := {"variable", "block"} - set(table):
        raise ValueError(f"group table missing columns: {sorted(missing)}")
    return table[
        ~table.block.isin(ANNOTATION_BLOCKS) & ~table.variable.isin(ANNOTATION_BLOCKS)
    ].copy()


def plot_deterministic_effect_scatter(features: pd.DataFrame, output: str | Path) -> Path:
    """Plot the dense singular/unique diagnostic without labels or marker strokes."""
    if missing := {"feature", "singular", "unique"} - set(features):
        raise ValueError(f"feature effects missing columns: {sorted(missing)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limit = max(0.1, math.ceil(float(features[["singular", "unique"]].abs().max().max()) * 10) / 10)
    _style()
    fig, ax = plt.subplots(figsize=(2.1, 1.8), layout="constrained")
    ax.scatter(features.singular, features.unique, s=2, color="#2a78d6",
               alpha=0.14, **FILL_ONLY)
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])
    ax.set_yticks([-limit, 0, limit])
    ax.set_xlabel("Singular improvement")
    ax.set_ylabel("Unique removal loss")
    _ordinary(ax)
    return _save(fig, output, "feature-scatter")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "judge_results/capability_predictions_glm/deterministic-regressors-v1",
    )
    args = parser.parse_args(argv)

    capability_root = ROOT / "judge_results/capability_predictions_glm"
    judge_root = ROOT / "judge_results/verdict_interpretability/readable-eda-20260828"
    comparison_source = capability_root / "judge-comparison-v1"
    contribution_source = capability_root / "contributions-v1"
    dictionary_path = capability_root / "no-common-support-inputs/feature-dictionary.parquet"
    dictionary = pd.read_parquet(dictionary_path)
    names = deterministic_feature_names(dictionary)
    if len(names) != 95:
        raise ValueError(f"expected 95 active deterministic regressors, found {len(names)}")

    comparison_output = args.output / "comparison"
    comparison_tables, comparison_figures = (
        comparison_output / "tables", comparison_output / "figures"
    )
    comparison_tables.mkdir(parents=True, exist_ok=True)
    comparison_figures.mkdir(parents=True, exist_ok=True)
    accuracy_source = comparison_source / "tables/accuracy-by-fold.csv"
    accuracy = pd.read_csv(accuracy_source)
    all_points = pd.read_csv(comparison_source / "tables/coefficient-points.csv")
    points = deterministic_rows(all_points, names)
    capability_coefficients = pd.read_parquet(
        capability_root / "interpret-v1/full/coefficients.parquet"
    )
    judge_paths = sorted((comparison_source / "source/judge-folds").glob(
        "*/reference-coefficients/full.parquet"
    ))
    judge_coefficients = pd.concat([
        pd.read_parquet(path).assign(fold=path.parents[1].name) for path in judge_paths
    ], ignore_index=True)
    support = all_points[["judge", "benchmark", "observed"]].drop_duplicates()
    comparison = deterministic_rows(
        coefficient_comparison(capability_coefficients, judge_coefficients), names,
    ).merge(support, on=["judge", "benchmark"], how="left", validate="many_to_one")
    comparison["observed"] = comparison.observed.fillna(False).astype(bool)
    cosines, drivers = coefficient_alignment(comparison)
    eci = pd.read_csv(comparison_source / "tables/judge-eci.csv")
    plot_comparisons(accuracy, points, cosines, drivers, eci, comparison_figures)
    shutil.copyfile(accuracy_source, comparison_tables / "accuracy-by-fold.csv")
    points.to_csv(comparison_tables / "coefficient-points.csv", index=False)
    coefficient_statistics(points).to_csv(
        comparison_tables / "coefficient-statistics.csv", index=False,
    )
    cosines.to_csv(comparison_tables / "cosine-by-fold.csv", index=False)
    _crossed_cosine_summary(cosines).to_csv(
        comparison_tables / "cosine-summary.csv", index=False,
    )
    driver_summary = drivers.groupby(["judge", "feature"], as_index=False).contribution.mean()
    driver_summary.to_csv(
        comparison_tables / "cosine-feature-contributions.csv", index=False,
    )

    capability_output = args.output / "capability-dissection"
    capability_tables = capability_output / "tables"
    overview = capability_output / "overview"
    by_group = capability_output / "by-feature-group"
    by_benchmark = capability_output / "by-benchmark"
    cross_benchmark = capability_output / "cross-benchmark"
    for directory in (capability_tables, overview, by_group, by_benchmark, cross_benchmark):
        directory.mkdir(parents=True, exist_ok=True)
    for stale in cross_benchmark.glob("*-effect-diamonds*.*"):
        stale.unlink()
    feature_contributions = deterministic_rows(
        pd.read_csv(contribution_source / "tables/feature-contributions.csv"), names,
        column="variable",
    )
    feature_contributions = feature_contributions[feature_contributions.task == "pairwise"].copy()
    group_contributions = deterministic_group_rows(
        pd.read_csv(contribution_source / "tables/group-contributions.csv")
    )
    group_contributions = group_contributions[group_contributions.task == "pairwise"].copy()
    pair_coefficients = deterministic_rows(capability_coefficients, names)
    feature_contributions.to_csv(capability_tables / "feature-contributions.csv", index=False)
    group_contributions.to_csv(capability_tables / "group-contributions.csv", index=False)
    plot_capability_feature_overview(
        feature_contributions, overview, "pairwise", top=8, width=4.0,
    )
    plot_capability_groups(group_contributions, overview, "pairwise")
    plot_each_block(feature_contributions, by_group, "pairwise")
    plot_benchmark_grid(
        feature_contributions, pair_coefficients, by_benchmark, "pairwise",
        count=4, height=3.2,
    )
    shared = shared_feature_selection(feature_contributions, "pairwise", count=8)
    for contribution in ("singular", "unique"):
        plot_cross_benchmark_profiles(
            feature_contributions, shared, cross_benchmark,
            task="pairwise", contribution=contribution,
            diamond_height=3.2, diamond_chunk_size=4,
            heatmap_width=3.2, diamond_width=3.8,
        )
    plot_coefficient_heatmap(
        pair_coefficients, shared, cross_benchmark, task="pairwise", width=3.2,
    )

    judge_output = args.output / "judge-dissection"
    judge_tables = judge_output / "tables"
    judge_overview = judge_output / "overview"
    judge_by_group = judge_output / "by-feature-group"
    judge_by_benchmark = judge_output / "by-benchmark"
    judge_by_judge = judge_output / "by-judge"
    judge_secondary = judge_output / "secondary"
    for directory in (judge_tables, judge_overview, judge_by_group, judge_by_benchmark,
                      judge_by_judge, judge_secondary):
        directory.mkdir(parents=True, exist_ok=True)
    feature_effects = deterministic_rows(
        pd.read_csv(judge_root / "data/model-results/feature-effects.csv"), names,
    )
    group_effects = pd.read_csv(judge_root / "data/model-results/group-effects.csv")
    group_effects = group_effects[
        ~group_effects.group.isin(ANNOTATION_BLOCKS)
    ].copy()
    benchmark_effects = deterministic_rows(load_benchmark_feature_effects(
        judge_root / "lesions/item-deltas.parquet", dictionary_path,
    ), names)
    judge_effects = deterministic_rows(load_benchmark_feature_effects(
        judge_root / "lesion-cells/item-cell-deltas.parquet", dictionary_path,
        by_judge=True,
    ), names)
    full_coefficient_paths = sorted((judge_root / "lesions/fold-artifacts").glob(
        "*/reference-coefficients/full.parquet"
    ))
    full_coefficients = pd.concat([
        pd.read_parquet(path).assign(fold=path.parents[1].name)
        for path in full_coefficient_paths
    ], ignore_index=True)
    full_coefficients = deterministic_rows(full_coefficients, names)
    overall_coefficients = summarize_effective_coefficients(full_coefficients)
    benchmark_coefficients = summarize_effective_coefficients(full_coefficients, by=("benchmark",))
    cell_coefficients = summarize_effective_coefficients(
        full_coefficients, by=("judge", "benchmark"),
    )
    feature_effects.to_csv(judge_tables / "feature-effects.csv", index=False)
    group_effects.to_csv(judge_tables / "group-effects.csv", index=False)
    benchmark_effects.to_csv(judge_tables / "benchmark-feature-effects.csv", index=False)
    judge_effects.to_csv(judge_tables / "judge-benchmark-feature-effects.csv", index=False)
    overall_coefficients.to_csv(judge_tables / "coefficient-effects.csv", index=False)
    benchmark_coefficients.to_csv(judge_tables / "benchmark-coefficients.csv", index=False)
    cell_coefficients.to_csv(judge_tables / "judge-benchmark-coefficients.csv", index=False)
    plot_judge_feature_overview(feature_effects, overall_coefficients, judge_overview, top=8)
    plot_group_effects(group_effects, judge_overview)
    plot_feature_blocks(feature_effects, overall_coefficients, judge_by_group)
    plot_benchmark_feature_grid(
        benchmark_effects, benchmark_coefficients, judge_by_benchmark,
        top=4, height=3.2, label_size=7,
    )
    plot_judge_benchmark_grids(
        judge_effects, cell_coefficients, judge_by_judge,
        top=4, height=3.2, label_size=7,
    )
    plot_deterministic_effect_scatter(feature_effects, judge_secondary)
    selection_paths = sorted((judge_root / "fits/frequentist").glob("*/selection.csv"))
    coefficient_paths = sorted((judge_root / "fits/frequentist").glob("*/coefficients.csv"))
    stability, fold_coefficients = coefficient_selection_stability(
        selection_paths, coefficient_paths,
    )
    stability = deterministic_rows(stability, names)
    fold_coefficients = deterministic_rows(fold_coefficients, names)
    plot_stability(stability, fold_coefficients, judge_overview)

    (args.output / "manifest.json").write_text(json.dumps({
        "feature_scope": "95 active deterministic/non-annotation regressors",
        "excluded_blocks": sorted(ANNOTATION_BLOCKS),
        "excluded_active_features": sorted(PRESENTATION_EXCLUSIONS),
        "mode": "subset views of existing all-regressor fits; no refitting",
        "subfolders": ["comparison", "capability-dissection", "judge-dissection"],
        "comparison_accuracy": "copied unchanged because accuracy is model-level, not feature-level",
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote deterministic subset views to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
