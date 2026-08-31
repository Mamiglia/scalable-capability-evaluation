"""Build the final judge-verdict result tables and manuscript figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import sem, t
import yaml


VARIANTS = {
    "raw-l1": "raw-l1",
    "raw-ridge": "raw-ridge-grid",
    "pc-l1": "pc-l1-global30",
    "pc-ridge": "pc-ridge-global30-grid",
}
LABELS = {
    "raw-l1": "Raw L1", "raw-ridge": "Raw L2",
    "pc-l1": "PC L1", "pc-ridge": "PC L2",
}
PRIMARY_VARIANTS = ("raw-l1", "pc-l1")
IMPORTANT_PCS = (
    "pc_001", "pc_002", "pc_003", "pc_004", "pc_005", "pc_006", "pc_007",
    "pc_008", "pc_022", "pc_029",
)
PC_EFFECT_PCS = (
    "pc_002", "pc_003", "pc_005", "pc_004", "pc_029", "pc_022", "pc_001",
    "pc_016", "pc_006", "pc_007", "pc_018", "pc_026",
)
PC_LABELS = {
    "pc_001": "PC1 · trajectory breadth/length",
    "pc_002": "PC2 · direct efficiency",
    "pc_003": "PC3 · symbolic/backtracking",
    "pc_004": "PC4 · rigorous clarity",
    "pc_005": "PC5 · sequential verification",
    "pc_006": "PC6 · equations vs verification",
    "pc_007": "PC7 · planning vs maths",
    "pc_008": "PC8 · assertion vs equations",
    "pc_016": "PC16 · graph/tool irregularity",
    "pc_018": "PC18 · equations vs correction",
    "pc_022": "PC22 · explicit checking",
    "pc_026": "PC26 · repetition/uncertainty",
    "pc_029": "PC29 · planned backtracking",
}
JUDGE_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama 8B",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v4-flash-0731": "DeepSeek V4",
    "gemma-4-31b-it": "Gemma",
    "gpt-oss-120b": "GPT-OSS",
    "o4-mini": "o4-mini",
}
BENCH_LABELS = {
    "aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
    "swe_bench_mini": "SWE-bench Mini",
}
RAW_HETEROGENEITY_LABELS = {
    "composite-010": "Bare answer / prose length",
    "composite-022": "Graph density / modularity",
    "composite-014": "Relation / type entropy",
    "depth": "Reasoning depth",
    "composite-019": "Response / tool-call length",
    "adjacent_edge_share": "Adjacent-edge share",
    "composite-009": "Graph diameter / path length",
    "test_commands": "Test commands",
}
CMAP = LinearSegmentedColormap.from_list(
    "judge_diverging", ("#2a78d6", "#f4f1eb", "#eb6834"),
)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1,
    "svg.fonttype": "none", "svg.hashsalt": "judge-verdict-20260831",
})


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_position(("outward", 5))
    axis.spines["bottom"].set_position(("outward", 5))
    axis.tick_params(length=2.5, width=0.8)


def _save(figure, root: Path, name: str) -> None:
    output = root / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=300)
    figure.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(figure)
    if "<text" not in (output / f"{name}.svg").read_text():
        raise RuntimeError(f"{name}.svg does not contain editable text")


def _validate(directory: Path, manifest_name: str) -> None:
    manifest = json.loads((directory / manifest_name).read_text())
    if manifest.get("state") != "completed":
        raise RuntimeError(f"incomplete result: {directory}")
    for name, expected in manifest["artifact_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {directory / name}")


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _feature_labels(root: Path) -> dict[str, str]:
    dictionary = pd.read_parquet(root / "metadata/feature-dictionary.parquet")
    output = {}
    for row in dictionary.itertuples():
        members = json.loads(row.leaf_members)
        label = " + ".join(_humanize(member) for member in members)
        output[row.feature] = label
    return output


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance, effects, selections = [], [], []
    for variant, relative in VARIANTS.items():
        directory = root / relative
        _validate(directory / "fit", "manifest.json")
        _validate(directory / "lesions", "summary-manifest.json")
        uncertainty = pd.read_csv(directory / "fit/uncertainty.csv")
        for metric in ("accuracy", "log_loss"):
            item = uncertainty.query("metric == @metric and method == 'item_clustered'").iloc[0]
            fold = uncertainty.query("metric == @metric and method == 'conservative_fold'").iloc[0]
            performance.append({
                "variant": variant, "metric": metric, "estimate": item.estimate,
                "item_lower": item.lower, "item_upper": item.upper,
                "fold_lower": fold.lower, "fold_upper": fold.upper,
            })
        effects.append(pd.read_csv(directory / "lesions/lesions.csv").assign(variant=variant))
        if variant.endswith("l1"):
            selected = pd.read_csv(directory / "fit/selection.csv")
            frequency = selected.feature.value_counts().rename_axis("feature").reset_index(name="folds")
            frequency["fold_share"] = frequency.folds / 16
            selections.append(frequency.assign(variant=variant))
    return pd.DataFrame(performance), pd.concat(effects), pd.concat(selections)


def performance_figure(root: Path, performance: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(3.0, 1.25))
    positions = np.arange(len(PRIMARY_VARIANTS))[::-1]
    table = performance.set_index(["metric", "variant"])
    for axis, metric, label in zip(axes, ("accuracy", "log_loss"), ("Accuracy", "Log loss")):
        for position, variant in zip(positions, PRIMARY_VARIANTS):
            row = table.loc[(metric, variant)]
            axis.errorbar(
                row.estimate, position,
                xerr=[[row.estimate - row.item_lower], [row.item_upper - row.estimate]],
                fmt="o", color="#333333", ms=4, capsize=0, lw=1,
            )
        values = performance[
            (performance.metric == metric) & performance.variant.isin(PRIMARY_VARIANTS)
        ]
        low = np.floor(values.item_lower.min() * 50) / 50
        high = np.ceil(values.item_upper.max() * 50) / 50
        axis.set_xlim(low, high); axis.set_xticks((low, (low + high) / 2, high))
        axis.set_yticks(
            positions,
            [LABELS[variant] for variant in PRIMARY_VARIANTS] if metric == "accuracy" else [],
        )
        axis.set_xlabel(label); _style(axis)
    figure.subplots_adjust(left=0.28, right=0.98, bottom=0.35, top=0.96, wspace=0.24)
    _save(figure, root, "performance")


def _effect_figure(
    root: Path,
    effects: pd.DataFrame,
    variant: str,
    name: str,
    labels: dict[str, str],
    *,
    grouped: bool = False,
    variables: list[str] | tuple[str, ...] | None = None,
) -> None:
    effects = effects[effects.variant == variant]
    if grouped:
        kinds = {"lesion-block", "singular-block", "lesion-family", "singular-family"}
        candidates = effects[effects.kind.isin(kinds)]
        if variables is None:
            variables = candidates.groupby("variable").estimate.apply(
                lambda x: x.abs().max()).nlargest(12).index.tolist()
    else:
        candidates = effects[effects.kind.isin({"lesion-feature", "singular-feature"})]
        if variables is None:
            variables = candidates[candidates.kind == "lesion-feature"].groupby("variable").estimate.apply(
                lambda x: x.abs().max()).nlargest(10).index.tolist()
    variables = list(variables)
    figure, axis = plt.subplots(figsize=(2.9, min(3.1, 1.0 + 0.23 * len(variables))))
    positions = np.arange(len(variables))[::-1]
    plotted = []
    table = candidates.set_index(["kind", "variable"])
    for offset, prefix, color, marker in (
        (-0.08, "lesion", "#222222", "o"),
        (0.08, "singular", "#888888", "s"),
    ):
        for position, variable in zip(positions + offset, variables):
            matches = table.loc[
                [index for index in table.index if index[0].startswith(prefix) and index[1] == variable]
            ] if any(index[0].startswith(prefix) and index[1] == variable for index in table.index) else pd.DataFrame()
            if matches.empty:
                continue
            row = matches.iloc[0]
            plotted.extend([row.lower, row.upper])
            axis.errorbar(
                row.estimate, position,
                xerr=[[row.estimate - row.lower], [row.upper - row.estimate]],
                fmt=marker, color=color, ms=3.5, capsize=0, lw=1,
            )
    axis.axvline(0, color="#999999", lw=0.8, ls="--")
    axis.set_xlabel("Δ log loss"); _style(axis)
    bound = max(0.01, np.ceil(max(abs(np.asarray(plotted))) * 100) / 100)
    axis.set_xlim(-bound, bound); axis.set_xticks((-bound, 0, bound))
    axis.set_yticks(positions, [
        textwrap.fill(labels.get(variable, _humanize(variable)), 22)
        for variable in variables
    ])
    handles = [
        plt.Line2D([], [], marker="o", color="#222222", ls="", label="Unique", ms=4),
        plt.Line2D([], [], marker="s", color="#888888", ls="", label="Singular", ms=4),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.59, right=0.95, bottom=0.2, top=0.97)
    _save(figure, root, name)


def loadings_figure(root: Path, labels: dict[str, str]) -> pd.DataFrame:
    loadings = pd.read_parquet(root / "global-pc-basis/loadings.parquet")
    selected_rows = []
    for component in IMPORTANT_PCS:
        figure, axis = plt.subplots(figsize=(2.7, 3.2))
        values = loadings[loadings.component == component].set_index("feature").loading
        selected = pd.concat([values.nsmallest(4), values.nlargest(4)]).drop_duplicates().sort_values()
        selected_rows.append(pd.DataFrame({
            "component": component, "semantic_label": PC_LABELS[component],
            "feature": selected.index, "loading": selected.values,
            "pole": np.where(selected.values < 0, "negative", "positive"),
        }))
        positions = np.arange(len(selected))
        axis.scatter(selected, positions, color="#333333", s=10, zorder=3)
        axis.hlines(positions, 0, selected, color="#777777", lw=1)
        bound = np.ceil(max(abs(selected)) * 20) / 20
        axis.set_xlim(-bound, bound); axis.set_xticks((-bound, 0, bound))
        axis.set_yticks(positions, [
            textwrap.fill(labels.get(feature, _humanize(feature)), 20)
            for feature in selected.index
        ])
        axis.set_xlabel("Loading")
        number, meaning = PC_LABELS[component].split(" · ", 1)
        figure.text(0.61, 0.97, number + "\n" + textwrap.fill(meaning, 18),
                    color="#666666", fontsize=7, va="top")
        _style(axis)
        figure.subplots_adjust(left=0.61, right=0.96, bottom=0.14, top=0.82)
        _save(figure, root, f"pc{int(component.split('_')[1])}-loadings")
    return pd.concat(selected_rows, ignore_index=True)


def stratified_performance(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.concat([
        pd.read_csv(root / VARIANTS[variant] / "fit/judge-benchmark.csv").assign(variant=variant)
        for variant in PRIMARY_VARIANTS
    ], ignore_index=True)
    summaries = {}
    for dimension in ("judge", "bench"):
        rows = []
        for (variant, value), group in cells.groupby(["variant", dimension], sort=False):
            weight = group.row_count.to_numpy()
            rows.append({
                "variant": variant, dimension: value,
                "accuracy": np.average(group.accuracy, weights=weight),
                "log_loss": np.average(group.log_loss, weights=weight),
                "row_count": weight.sum(),
            })
        summaries[dimension] = pd.DataFrame(rows)
    return cells, summaries["judge"], summaries["bench"]


def ceiling_by_judge(root: Path, cells: pd.DataFrame) -> pd.DataFrame:
    ceiling_path = root.parents[1] / "verdict_dissection/exact_rejudge_20260828/summary.csv"
    ceiling = pd.read_csv(ceiling_path).rename(columns={"benchmark": "bench"})
    rows = []
    for judge, group in cells[cells.variant == "pc-l1"].merge(
        ceiling, on=["judge", "bench"], how="inner").groupby("judge", sort=False):
        weight = group.row_count.to_numpy()
        conservative = np.average(group.wilson_95_low, weights=weight)
        raw = np.average(group.raw_agreement, weights=weight)
        record = {
            "judge": judge, "observed_cells": len(group), "held_out_rows": weight.sum(),
            "conservative_repeat_ceiling": conservative, "raw_repeat_agreement": raw,
        }
        for variant in PRIMARY_VARIANTS:
            variant_rows = cells[(cells.variant == variant) & (cells.judge == judge)]
            accuracy = np.average(variant_rows.accuracy, weights=variant_rows.row_count)
            prefix = variant.replace("-", "_")
            record[f"{prefix}_accuracy"] = accuracy
            record[f"{prefix}_recovery_vs_raw"] = (accuracy - 0.5) / (raw - 0.5)
            record[f"{prefix}_recovery_vs_conservative"] = (
                (accuracy - 0.5) / (conservative - 0.5))
        rows.append(record)
    return pd.DataFrame(rows)


def ceiling_figure(root: Path, table: pd.DataFrame) -> None:
    order = [judge for judge in JUDGE_LABELS if judge in set(table.judge)]
    values = table.set_index("judge").loc[order]
    positions = np.arange(len(order))[::-1]
    figure, axis = plt.subplots(figsize=(2.9, 2.9))
    axis.hlines(
        positions, values.conservative_repeat_ceiling, values.raw_repeat_agreement,
        color="#999999", lw=1,
    )
    axis.plot(values.raw_l1_accuracy, positions - 0.08, "o", color="#2a78d6", ms=4,
              label="Raw L1")
    axis.plot(values.pc_l1_accuracy, positions + 0.08, "o", color="#eb6834", ms=4,
              label="PC L1")
    axis.plot(values.conservative_repeat_ceiling, positions, "s", color="#777777", ms=3.5,
              label="Lower bound")
    axis.plot(values.raw_repeat_agreement, positions, "^", color="#aaaaaa", ms=4,
              label="Repeat")
    axis.set_xlim(0.6, 1); axis.set_xticks((0.6, 0.8, 1.0))
    axis.set_yticks(positions, [JUDGE_LABELS[judge] for judge in order])
    axis.set_xlabel("Accuracy / agreement"); _style(axis)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.4, right=0.98, bottom=0.4, top=0.98)
    _save(figure, root, "accuracy-vs-ceiling")


def performance_strata_figure(root: Path, table: pd.DataFrame, dimension: str) -> None:
    order = list(JUDGE_LABELS if dimension == "judge" else BENCH_LABELS)
    order = [value for value in order if value in set(table[dimension])]
    labels = JUDGE_LABELS if dimension == "judge" else BENCH_LABELS
    figure, axes = plt.subplots(1, 2, figsize=(3.4, 1.3 + 0.24 * len(order)), sharey=True)
    positions = np.arange(len(order))[::-1]
    for axis, metric, axis_label in zip(axes, ("accuracy", "log_loss"), ("Accuracy", "Log loss")):
        for offset, variant, color in ((-0.08, "raw-l1", "#2a78d6"), (0.08, "pc-l1", "#eb6834")):
            values = table[table.variant == variant].set_index(dimension).loc[order]
            axis.plot(values[metric], positions + offset, "o", color=color, ms=4, label=LABELS[variant])
        values = table[metric]
        low = np.floor(values.min() * 20) / 20
        high = np.ceil(values.max() * 20) / 20
        axis.set_xlim(low, high); axis.set_xticks((low, (low + high) / 2, high))
        axis.set_yticks(positions)
        axis.set_xlabel(axis_label); _style(axis)
    axes[0].set_yticklabels([labels[value] for value in order])
    axes[1].tick_params(labelleft=False)
    handles = [
        plt.Line2D([], [], marker="o", color="#2a78d6", ls="", label="Raw L1", ms=4),
        plt.Line2D([], [], marker="o", color="#eb6834", ls="", label="PC L1", ms=4),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.34, right=0.98, bottom=0.3, top=0.97, wspace=0.24)
    _save(figure, root, f"performance-by-{dimension}")


def coefficient_strata(
    root: Path, features: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variance = pd.read_csv(root / "global-pc-basis/explained-variance.csv").set_index("component")
    scales = variance.explained_variance.pow(0.5).to_dict()
    outputs = {"judge": [], "bench": []}
    for variant in PRIMARY_VARIANTS:
        coefficients = pd.read_csv(root / VARIANTS[variant] / "fit/coefficients.csv")
        coefficients = coefficients.rename(columns={"benchmark": "bench"})
        coefficients = coefficients[coefficients.feature.isin(features[variant])]
        observed = pd.read_csv(root / VARIANTS[variant] / "fit/judge-benchmark.csv")[
            ["judge", "bench", "row_count"]
        ]
        coefficients = coefficients.merge(observed, on=["judge", "bench"], how="inner")
        coefficients["slope_per_sd"] = coefficients.effective * coefficients.feature.map(
            scales if variant.startswith("pc") else {}).fillna(1)
        coefficients["weighted"] = coefficients.slope_per_sd * coefficients.row_count
        for dimension in ("judge", "bench"):
            fold = coefficients.groupby(
                ["fold", "feature", dimension], as_index=False,
            ).agg(weighted=("weighted", "sum"), row_count=("row_count", "sum"))
            fold["slope_per_sd"] = fold.weighted / fold.row_count
            summary = fold.groupby(["feature", dimension], as_index=False).agg(
                median=("slope_per_sd", "median"),
                q25=("slope_per_sd", lambda values: values.quantile(0.25)),
                q75=("slope_per_sd", lambda values: values.quantile(0.75)),
                fold_count=("fold", "nunique"),
            )
            outputs[dimension].append(summary.assign(variant=variant))
    return pd.concat(outputs["judge"], ignore_index=True), pd.concat(outputs["bench"], ignore_index=True)


def benchmark_unique_effects(root: Path, features: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for variant in PRIMARY_VARIANTS:
        table = pd.read_parquet(root / VARIANTS[variant] / "lesions/item-deltas.parquet")
        table = table[
            table.kind.eq("lesion-feature") & table.variable.isin(features[variant])
        ].copy()
        table["weighted"] = table.delta_log_loss * table.row_count
        summary = table.groupby(["variable", "bench"], as_index=False).agg(
            weighted=("weighted", "sum"), row_count=("row_count", "sum"),
        )
        summary["unique_delta_log_loss"] = summary.weighted / summary.row_count
        rows.append(summary.assign(variant=variant))
    return pd.concat(rows, ignore_index=True)


def benchmark_effect_heterogeneity(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows, summaries = [], []
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    for variant in PRIMARY_VARIANTS:
        table = pd.read_parquet(root / VARIANTS[variant] / "lesions/item-deltas.parquet")
        table = table[table.kind.isin(("lesion-feature", "singular-feature"))].copy()
        table["weighted"] = table.delta_log_loss * table.row_count
        grouped = table.groupby(["kind", "variable", "bench"], as_index=False).agg(
            weighted=("weighted", "sum"), row_count=("row_count", "sum"))
        grouped["effect"] = grouped.weighted / grouped.row_count
        grouped["effect_type"] = grouped.kind.map({
            "lesion-feature": "Unique", "singular-feature": "Singular",
        })
        long_rows.append(grouped.assign(variant=variant))
        for variable, rows in grouped.groupby("variable"):
            record = {"variant": variant, "variable": variable}
            for kind, prefix in (("lesion-feature", "unique"), ("singular-feature", "singular")):
                values = rows[rows.kind == kind].set_index("bench").effect
                for benchmark in benchmarks:
                    record[f"{prefix}_{benchmark}"] = values.get(benchmark, np.nan)
                record[f"{prefix}_range"] = values.max() - values.min()
                record[f"{prefix}_sd"] = values.std(ddof=0)
                record[f"{prefix}_agentic_minus_reasoning"] = (
                    (values.get("intercode", np.nan) + values.get("swe_bench_mini", np.nan)) / 2
                    - (values.get("aime", np.nan) + values.get("gpqa", np.nan)) / 2
                )
            record["heterogeneity_score"] = max(record["unique_range"], record["singular_range"])
            summaries.append(record)
    return pd.concat(long_rows, ignore_index=True), pd.DataFrame(summaries)


def interpretation_tables(
    root: Path, effects: pd.DataFrame, raw_features: list[str], labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dictionary = pd.read_parquet(root / "metadata/feature-dictionary.parquet").set_index("feature")
    raw_coefficients = pd.read_csv(root / VARIANTS["raw-l1"] / "fit/coefficients.csv")
    raw_population = raw_coefficients[["fold", "feature", "population"]].drop_duplicates().groupby(
        "feature").population.median()
    pc_coefficients = pd.read_csv(root / VARIANTS["pc-l1"] / "fit/coefficients.csv")
    pc_population = pc_coefficients[["fold", "feature", "population"]].drop_duplicates().groupby(
        "feature").population.median()
    variance = pd.read_csv(root / "global-pc-basis/explained-variance.csv").set_index("component")

    def effect(variant: str, kind: str, feature: str):
        return effects[
            (effects.variant == variant) & (effects.kind == kind) & (effects.variable == feature)
        ].iloc[0]

    raw_rows = []
    for feature in raw_features:
        unique, singular = effect("raw-l1", "lesion-feature", feature), effect(
            "raw-l1", "singular-feature", feature)
        raw_rows.append({
            "feature": feature, "label": labels[feature], "block": dictionary.loc[feature, "block"],
            "unique": unique.estimate, "unique_lower": unique.lower, "unique_upper": unique.upper,
            "singular": singular.estimate, "singular_lower": singular.lower,
            "singular_upper": singular.upper, "median_population_slope": raw_population[feature],
        })
    pc_rows = []
    for feature in PC_EFFECT_PCS:
        unique, singular = effect("pc-l1", "lesion-feature", feature), effect(
            "pc-l1", "singular-feature", feature)
        scale = np.sqrt(variance.loc[feature, "explained_variance"])
        pc_rows.append({
            "component": feature, "semantic_label": PC_LABELS[feature],
            "explained_variance_ratio": variance.loc[feature, "explained_variance_ratio"],
            "cumulative_explained_variance_ratio": variance.loc[feature, "cumulative_explained_variance_ratio"],
            "unique": unique.estimate, "unique_lower": unique.lower, "unique_upper": unique.upper,
            "singular": singular.estimate, "singular_lower": singular.lower,
            "singular_upper": singular.upper, "median_population_slope": pc_population[feature],
            "median_population_slope_per_sd": pc_population[feature] * scale,
        })
    return pd.DataFrame(raw_rows), pd.DataFrame(pc_rows)


def heatmap_figure(
    root: Path,
    table: pd.DataFrame,
    variant: str,
    features: list[str],
    dimension: str,
    value: str,
    labels: dict[str, str],
    name: str,
    colorbar_label: str,
) -> None:
    columns = list(JUDGE_LABELS if dimension == "judge" else BENCH_LABELS)
    columns = [column for column in columns if column in set(table[dimension])]
    matrix = table[table.variant == variant].pivot(index="feature" if "feature" in table else "variable",
                                                    columns=dimension, values=value)
    matrix = matrix.reindex(index=features, columns=columns)
    values = matrix.to_numpy(float)
    bound = max(0.01, np.ceil(np.nanmax(np.abs(values)) * 20) / 20)
    width = 3.5 if dimension == "judge" else 2.7
    height = min(3.1, 1.65 + 0.23 * len(features))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(values, cmap=CMAP, vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(np.arange(len(columns)), [
        (JUDGE_LABELS if dimension == "judge" else BENCH_LABELS)[column] for column in columns
    ], rotation=90, ha="center")
    axis.set_yticks(np.arange(len(features)), [
        textwrap.fill(labels.get(feature, _humanize(feature)), 26) for feature in features
    ])
    axis.tick_params(length=0)
    left = 0.45 if dimension == "judge" else 0.5
    figure.subplots_adjust(left=left, right=0.96, bottom=0.47, top=0.98)
    colorbar_axis = figure.add_axes([left, 0.15, 0.96 - left, 0.035])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((-bound, 0, bound)); colorbar.set_label(colorbar_label)
    _save(figure, root, name)


def _model_slug(value: str) -> str:
    value = str(value).split("/")[-1].lower()
    value = value.replace("instruct", "").replace("-it", "").replace("-bf16", "")
    return re.sub(r"[^a-z0-9.]", "", value)


def model_metadata(root: Path, models: set[str]) -> pd.DataFrame:
    records = {}
    for path in (root.parents[2] / "configs/models").glob("*/*.yml"):
        specification = yaml.safe_load(path.read_text()) or {}
        if not specification.get("name") or not specification.get("release_date"):
            continue
        text = path.read_text().lower()
        configured = json.dumps(specification.get("generate_args", {}), sort_keys=True).lower()
        explicit = (
            ("reasoning" in configured and any(value in configured for value in ("true", "medium", "high")))
            or "enable_thinking" in configured
            or "--reasoning-parser" in text
            or ("thinking" in text and "enabled explicitly" in text)
        )
        family = bool(re.search(
            r"(deepseek-r1|(^|/)o[134](?:-|$)|gpt-5|gpt-oss)",
            str(specification["name"]).lower(),
        ))
        records[_model_slug(specification["name"])] = {
            "release_date": pd.Timestamp(str(specification["release_date"]).split("/")[0][:10]),
            "reasoning_configured": explicit or family,
            "metadata_source": str(path),
        }
    missing = [model for model in sorted(models) if _model_slug(model) not in records]
    if missing:
        raise RuntimeError(f"missing model metadata: {missing}")
    return pd.DataFrame([
        {"model": model, **records[_model_slug(model)]} for model in sorted(models)
    ])


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().to_numpy(float)
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * sem(values)) if len(values) >= 3 else np.nan
    return mean, mean - half, mean + half


def prediction_error_analysis(root: Path):
    predictions = pd.read_parquet(root / VARIANTS["pc-l1"] / "fit/predictions.parquet").copy()
    metadata = model_metadata(
        root, set(predictions.model_left) | set(predictions.model_right) | set(predictions.judge),
    ).set_index("model")
    for role in ("left", "right", "judge"):
        column = f"model_{role}" if role in {"left", "right"} else "judge"
        predictions[f"{role}_release"] = predictions[column].map(metadata.release_date)
        predictions[f"{role}_reasoning"] = predictions[column].map(metadata.reasoning_configured)
    predictions["prediction_error"] = (
        (predictions.probability >= 0.5) != predictions.observed
    ).astype(float)
    predictions["prediction_confidence"] = 2 * (predictions.probability - 0.5).abs()
    predictions["judge_correct"] = np.nan
    not_tied = predictions.left_stronger.notna()
    predictions.loc[not_tied, "judge_correct"] = (
        predictions.loc[not_tied, "observed"]
        == (predictions.loc[not_tied, "left_stronger"] > 0).astype(int)
    ).astype(float)
    predictions["absolute_capability_gap"] = predictions.accuracy_difference.abs()
    predictions["pair_release_gap_years"] = (
        predictions.left_release - predictions.right_release
    ).abs().dt.days / 365.25
    predictions["judge_release_distance_years"] = pd.concat([
        (predictions.left_release - predictions.judge_release).abs().dt.days,
        (predictions.right_release - predictions.judge_release).abs().dt.days,
    ], axis=1).max(axis=1) / 365.25
    predictions["any_candidate_newer"] = (
        (predictions.left_release > predictions.judge_release)
        | (predictions.right_release > predictions.judge_release)
    )
    predictions["both_candidates_newer"] = (
        (predictions.left_release > predictions.judge_release)
        & (predictions.right_release > predictions.judge_release)
    )
    predictions["mixed_candidate_reasoning"] = (
        predictions.left_reasoning != predictions.right_reasoning
    )
    predictions["reasoning_mismatch"] = (
        (predictions.left_reasoning != predictions.judge_reasoning)
        | (predictions.right_reasoning != predictions.judge_reasoning)
    )

    predictions["chosen_model"] = np.where(
        predictions.observed == 1, predictions.model_left, predictions.model_right)
    predictions["order_key"] = [
        (judge, bench, item, *sorted((left_log, right_log)))
        for judge, bench, item, left_log, right_log in zip(
            predictions.judge, predictions.bench, predictions.item,
            predictions.log_left, predictions.log_right,
        )
    ]
    order = predictions.groupby("order_key").agg(
        rows=("chosen_model", "size"), choices=("chosen_model", "nunique"))
    predictions["order_consistent"] = predictions.order_key.map(
        order.choices.eq(1).where(order.rows >= 2))

    first = np.minimum(predictions.model_left, predictions.model_right)
    predictions["choice_first"] = np.where(
        predictions.model_left == first, predictions.observed, 1 - predictions.observed)
    predictions["content_key"] = [
        (bench, item, *sorted((left_log, right_log)))
        for bench, item, left_log, right_log in zip(
            predictions.bench, predictions.item, predictions.log_left, predictions.log_right)
    ]
    consensus = predictions.groupby("content_key").choice_first.mean()
    predictions["cross_judge_consensus"] = predictions.content_key.map(
        (consensus - 0.5).abs() * 2)

    factors = {
        "Judge chose weaker model": 1 - predictions.judge_correct,
        "Order-inconsistent verdict": 1 - predictions.order_consistent.astype(float),
        "Configured-reasoning mismatch": predictions.reasoning_mismatch.astype(float),
        "Mixed configured reasoning": predictions.mixed_candidate_reasoning.astype(float),
        "Both candidates newer": predictions.both_candidates_newer.astype(float),
    }
    factor_rows = []
    for factor, values in factors.items():
        frame = predictions.assign(level=values).dropna(subset=["level"])
        for judge, group in frame.groupby("judge"):
            rates = group.groupby("level").prediction_error.mean()
            counts = group.groupby("level").size()
            if not {0.0, 1.0} <= set(rates.index):
                continue
            factor_rows.append({
                "factor": factor, "judge": judge,
                "error_absent": rates[0.0], "error_present": rates[1.0],
                "delta_error": rates[1.0] - rates[0.0],
                "n_absent": counts[0.0], "n_present": counts[1.0],
            })
    factor_by_judge = pd.DataFrame(factor_rows)
    summaries = []
    for factor, group in factor_by_judge.groupby("factor", sort=False):
        mean, lower, upper = _mean_ci(group.delta_error)
        summaries.append({
            "factor": factor, "judge_count": len(group),
            "error_absent": group.error_absent.mean(),
            "error_present": group.error_present.mean(),
            "delta_error": mean, "lower": lower, "upper": upper,
        })
    factor_summary = pd.DataFrame(summaries)

    continuous = {
        "Capability gap": "absolute_capability_gap",
        "Cross-judge consensus": "cross_judge_consensus",
        "Judge–candidate release distance": "judge_release_distance_years",
        "Candidate pair release gap": "pair_release_gap_years",
        "Model confidence": "prediction_confidence",
    }
    quartile_rows = []
    for factor, column in continuous.items():
        quartile = predictions.groupby("judge")[column].transform(
            lambda values: pd.qcut(values, 4, labels=False, duplicates="drop") + 1)
        frame = predictions.assign(quartile=quartile)
        for (judge, level), group in frame.groupby(["judge", "quartile"]):
            quartile_rows.append({
                "factor": factor, "judge": judge, "quartile": int(level),
                "error_rate": group.prediction_error.mean(), "n": len(group),
                "mean_value": group[column].mean(),
            })
    quartile_by_judge = pd.DataFrame(quartile_rows)
    quartile_summary = []
    for (factor, quartile), group in quartile_by_judge.groupby(["factor", "quartile"], sort=False):
        mean, lower, upper = _mean_ci(group.error_rate)
        quartile_summary.append({
            "factor": factor, "quartile": quartile, "judge_count": len(group),
            "error_rate": mean, "lower": lower, "upper": upper,
            "mean_value": group.mean_value.mean(),
        })
    outcome_rows = [{
        "quantity": "overall_prediction_error", "level": "all",
        "n": len(predictions), "value": predictions.prediction_error.mean(),
    }]
    for error, group in predictions.dropna(subset=["judge_correct"]).groupby("prediction_error"):
        outcome_rows.append({
            "quantity": "judge_incorrect_given_prediction_error",
            "level": "error" if error else "correct",
            "n": len(group), "value": 1 - group.judge_correct.mean(),
        })
    for consistency, group in predictions.dropna(subset=["order_consistent"]).groupby("order_consistent"):
        outcome_rows.append({
            "quantity": "prediction_error_given_order_consistency",
            "level": "consistent" if consistency else "inconsistent",
            "n": len(group), "value": group.prediction_error.mean(),
        })
    columns = [
        "bench", "judge", "item", "model_left", "model_right", "observed", "probability",
        "prediction_error", "judge_correct", "order_consistent", "cross_judge_consensus",
        "absolute_capability_gap", "pair_release_gap_years", "judge_release_distance_years",
        "left_reasoning", "right_reasoning", "judge_reasoning", "mixed_candidate_reasoning",
        "reasoning_mismatch", "any_candidate_newer", "both_candidates_newer",
    ]
    return (
        metadata.reset_index(), predictions[columns], factor_by_judge, factor_summary,
        quartile_by_judge, pd.DataFrame(quartile_summary).sort_values(["factor", "quartile"]),
        pd.DataFrame(outcome_rows),
    )


def failure_factor_figure(root: Path, table: pd.DataFrame) -> None:
    table = table.sort_values("delta_error")
    positions = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(2.9, 1.75))
    axis.errorbar(
        table.delta_error, positions,
        xerr=[table.delta_error - table.lower, table.upper - table.delta_error],
        fmt="o", color="#333333", ms=4, capsize=0, lw=1,
    )
    bound = max(0.05, np.ceil(max(abs(table.lower).max(), abs(table.upper).max()) * 20) / 20)
    axis.axvline(0, color="#999999", ls="--", lw=0.8)
    axis.set_xlim(-bound, bound); axis.set_xticks((-bound, 0, bound))
    axis.set_yticks(positions, [textwrap.fill(value, 24) for value in table.factor])
    axis.set_xlabel("Δ prediction error"); _style(axis)
    figure.subplots_adjust(left=0.55, right=0.96, bottom=0.25, top=0.97)
    _save(figure, root, "prediction-failure-factors")


def failure_quartile_figure(root: Path, table: pd.DataFrame) -> None:
    factors = [
        "Capability gap", "Cross-judge consensus",
        "Judge–candidate release distance", "Model confidence",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(4.0, 2.9), sharex=True, sharey=True)
    for axis, factor in zip(axes.flat, factors):
        rows = table[table.factor == factor].sort_values("quartile")
        axis.errorbar(
            rows.quartile, rows.error_rate,
            yerr=[rows.error_rate - rows.lower, rows.upper - rows.error_rate],
            fmt="o-", color="#333333", ms=3.5, capsize=0, lw=1,
        )
        axis.set_xlim(1, 4); axis.set_xticks((1, 2, 3, 4))
        axis.set_ylim(0, 0.5); axis.set_yticks((0, 0.25, 0.5))
        axis.text(0, 1.04, factor, transform=axis.transAxes, color="#666666", fontsize=7)
        _style(axis)
    for axis in axes[1]: axis.set_xlabel("Quartile")
    for axis in axes[:, 0]: axis.set_ylabel("Error rate")
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.95, wspace=0.28, hspace=0.42)
    _save(figure, root, "prediction-failure-quartiles")


def build(root: Path) -> None:
    missing = [relative for relative in VARIANTS.values() if not (root / relative / "fit/manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed variants: {missing}")
    tables = root / "tables"; tables.mkdir(exist_ok=True)
    performance, effects, selections = load_results(root)
    performance.to_csv(tables / "performance.csv", index=False)
    effects.to_csv(tables / "effects.csv", index=False)
    selections.to_csv(tables / "l1-selection-frequency.csv", index=False)
    labels = _feature_labels(root)
    raw_unique = effects[(effects.variant == "raw-l1") & (effects.kind == "lesion-feature")]
    raw_features = raw_unique[raw_unique.lower > 0].nlargest(8, "estimate").variable.tolist()
    strata_features = {"raw-l1": raw_features[:6], "pc-l1": list(IMPORTANT_PCS)}
    raw_interpretation, pc_interpretation = interpretation_tables(
        root, effects, raw_features, labels)
    raw_interpretation.to_csv(tables / "raw-feature-summary.csv", index=False)
    pc_interpretation.to_csv(tables / "pc-summary.csv", index=False)

    cells, by_judge, by_benchmark = stratified_performance(root)
    cells.to_csv(tables / "performance-by-judge-benchmark.csv", index=False)
    by_judge.to_csv(tables / "performance-by-judge.csv", index=False)
    by_benchmark.to_csv(tables / "performance-by-benchmark.csv", index=False)
    ceiling = ceiling_by_judge(root, cells)
    ceiling.to_csv(tables / "ceiling-by-judge.csv", index=False)
    coefficients_by_judge, coefficients_by_benchmark = coefficient_strata(root, strata_features)
    coefficients_by_judge.to_csv(tables / "l1-coefficients-by-judge.csv", index=False)
    coefficients_by_benchmark.to_csv(tables / "l1-coefficients-by-benchmark.csv", index=False)
    unique_by_benchmark = benchmark_unique_effects(root, strata_features)
    unique_by_benchmark.to_csv(tables / "l1-unique-effects-by-benchmark.csv", index=False)
    heterogeneity_long, heterogeneity = benchmark_effect_heterogeneity(root)
    heterogeneity_long.to_csv(tables / "l1-effects-by-benchmark.csv", index=False)
    heterogeneity.to_csv(tables / "l1-benchmark-heterogeneity.csv", index=False)
    (
        metadata, error_cases, failure_by_judge, failure_summary,
        failure_quartiles_by_judge, failure_quartiles, failure_outcomes,
    ) = prediction_error_analysis(root)
    metadata.to_csv(tables / "model-metadata.csv", index=False)
    error_cases.to_parquet(tables / "prediction-error-cases.parquet", index=False)
    failure_by_judge.to_csv(tables / "prediction-failure-factors-by-judge.csv", index=False)
    failure_summary.to_csv(tables / "prediction-failure-factors.csv", index=False)
    failure_quartiles_by_judge.to_csv(
        tables / "prediction-failure-quartiles-by-judge.csv", index=False)
    failure_quartiles.to_csv(tables / "prediction-failure-quartiles.csv", index=False)
    failure_outcomes.to_csv(tables / "prediction-failure-outcomes.csv", index=False)

    performance_figure(root, performance)
    ceiling_figure(root, ceiling)
    performance_strata_figure(root, by_judge, "judge")
    performance_strata_figure(root, by_benchmark, "bench")
    _effect_figure(root, effects, "raw-l1", "raw-feature-effects", labels, variables=raw_features)
    _effect_figure(root, effects, "raw-l1", "raw-group-effects", labels, grouped=True)
    _effect_figure(root, effects, "pc-l1", "pc-effects", PC_LABELS, variables=PC_EFFECT_PCS)
    heatmap_figure(
        root, coefficients_by_judge, "raw-l1", strata_features["raw-l1"], "judge", "median",
        labels, "raw-coefficients-by-judge", "Median slope per SD",
    )
    heatmap_figure(
        root, coefficients_by_judge, "pc-l1", strata_features["pc-l1"], "judge", "median",
        PC_LABELS, "pc-coefficients-by-judge", "Median slope per SD",
    )
    heatmap_figure(
        root, unique_by_benchmark, "raw-l1", strata_features["raw-l1"], "bench",
        "unique_delta_log_loss", labels, "raw-unique-by-benchmark", "Unique Δ log loss",
    )
    heatmap_figure(
        root, unique_by_benchmark, "pc-l1", strata_features["pc-l1"], "bench",
        "unique_delta_log_loss", PC_LABELS, "pc-unique-by-benchmark", "Unique Δ log loss",
    )
    for variant, prefix, label_map in (
        ("raw-l1", "raw", RAW_HETEROGENEITY_LABELS), ("pc-l1", "pc", PC_LABELS),
    ):
        top = heterogeneity[heterogeneity.variant == variant].nlargest(
            8, "heterogeneity_score").variable.tolist()
        for effect_type in ("Unique", "Singular"):
            heatmap_figure(
                root, heterogeneity_long[heterogeneity_long.effect_type == effect_type],
                variant, top, "bench", "effect", label_map,
                f"{prefix}-benchmark-heterogeneity-{effect_type.lower()}",
                f"{effect_type} Δ log loss",
            )
    failure_factor_figure(root, failure_summary)
    failure_quartile_figure(root, failure_quartiles)
    loadings_figure(root, labels).to_csv(tables / "important-pc-loadings.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build(parser.parse_args().root)


if __name__ == "__main__":
    main()
