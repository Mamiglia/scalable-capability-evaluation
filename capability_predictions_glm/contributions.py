"""Summarise and plot singular and unique capability contributions."""

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd

from .config import CONFIG, TARGET_QUARTILES
from .data import load_inputs
from .interpret import FAMILIES, build_specifications


COLORS = {
    "surface": "#2a78d6", "structure": "#eb6834", "rubric": "#1baf7a",
    "exposition": "#eda100", "derivation": "#e87ba4", "intermediate-errors": "#4a3aa7",
    "clause-descriptors": "#5f8f8b", "clause-motifs": "#9c6b4e", "clause-plan": "#666666",
}
EFFECT_COLORS = {"singular": "#2a78d6", "unique": "#eb6834"}
EXCLUDED_FEATURES = {"bare_answer"}


def bootstrap_delta(
    rows: pd.DataFrame,
    unit: str,
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Mean paired delta and percentile interval over equal-weight replication units."""
    values = rows.groupby(unit, sort=False).delta.mean().to_numpy(float)
    if not len(values):
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan, "n_units": 0}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(iterations, len(values)))].mean(axis=1)
    return {"estimate": float(values.mean()), "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)), "n_units": len(values)}


def _direct_null(full_results: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in full_results.glob("model-fold-*-Q*/predictions.csv")]
    rows = pd.concat(frames, ignore_index=True)
    return (rows[(rows.comparator == "pooled_judges") & (rows.method == "training_mean")]
            .drop_duplicates(["fold", "bench", "target_quartile", "model"]))


def _summaries(
    rows: pd.DataFrame,
    unit: str,
    metadata: dict[str, str],
    iterations: int,
) -> list[dict[str, object]]:
    output = []
    for benchmark in ["all", *sorted(rows.bench.astype(str).unique())]:
        benchmark_rows = rows if benchmark == "all" else rows[rows.bench == benchmark]
        for target in ["all", *TARGET_QUARTILES]:
            selected = (benchmark_rows if target == "all"
                        else benchmark_rows[benchmark_rows.target_quartile == target])
            output.append(metadata | {
                "benchmark": benchmark, "target_quartile": target,
                **bootstrap_delta(selected, unit, iterations=iterations),
            })
    return output


def _direct_contribution(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    contribution: str,
) -> pd.DataFrame:
    keys = ["fold", "bench", "target_quartile", "model"]
    columns = keys + ["observed_accuracy", "predicted_accuracy"]
    merged = candidate[columns].merge(reference[columns], on=keys, suffixes=("_candidate", "_reference"),
                                      validate="one_to_one")
    candidate_error = (merged.predicted_accuracy_candidate - merged.observed_accuracy_candidate).abs()
    reference_error = (merged.predicted_accuracy_reference - merged.observed_accuracy_reference).abs()
    merged["delta"] = (reference_error - candidate_error if contribution == "singular"
                       else candidate_error - reference_error)
    return merged


def _pair_contribution(candidate: pd.DataFrame, full: pd.DataFrame | None, contribution: str) -> pd.DataFrame:
    keys = ["fold", "bench", "target_quartile", "item"]
    if contribution == "singular":
        output = candidate[keys].copy()
        output["delta"] = candidate.accuracy.to_numpy(float) - 0.5
    else:
        merged = candidate[keys + ["accuracy"]].merge(
            full[keys + ["accuracy"]], on=keys, suffixes=("_candidate", "_full"), validate="one_to_one")
        output = merged[keys].copy()
        output["delta"] = merged.accuracy_full - merged.accuracy_candidate
    output["item_unit"] = output.bench.astype(str) + "/" + output.item.astype(str)
    return output


def _spec_maps(dictionary: pd.DataFrame):
    full = set(dictionary.feature)
    feature_block = dictionary.set_index("feature").block.to_dict()
    features, groups = {}, {}
    for spec in build_specifications(dictionary):
        values = set(spec.features)
        if spec.kind == "lesion-feature":
            feature = (full - values).pop()
            features.setdefault(feature, {})["unique"] = spec
        elif spec.kind == "singular-feature":
            features.setdefault(spec.features[0], {})["singular"] = spec
        elif spec.kind in {"lesion-block", "singular-block"}:
            group = spec.name.split("-", 2)[2]
            groups.setdefault(("block", group), {})["unique" if spec.kind.startswith("lesion") else "singular"] = spec
        elif spec.name == "no-annotations":
            groups.setdefault(("family", "annotations"), {})["unique"] = spec
        elif spec.kind in {"lesion-family", "singular-family"}:
            group = spec.name.split("-", 2)[2]
            groups.setdefault(("family", group), {})["unique" if spec.kind.startswith("lesion") else "singular"] = spec
    return features, groups, feature_block


def compute_contributions(
    interpret_root: str | Path,
    full_results: str | Path,
    dictionary: pd.DataFrame,
    *,
    iterations: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete feature and group singular/unique contribution tables."""
    interpret_root, full_results = Path(interpret_root), Path(full_results)
    full_direct = pd.read_parquet(interpret_root / "full/direct-predictions.parquet")
    full_pair = pd.read_parquet(interpret_root / "full/pair-item-predictions.parquet")
    null_direct = _direct_null(full_results)
    feature_specs, group_specs, feature_block = _spec_maps(dictionary)
    feature_rows, group_rows = [], []

    def evaluate(spec, contribution, metadata):
        direct = pd.read_parquet(interpret_root / spec.name / "direct-predictions.parquet")
        pair = pd.read_parquet(interpret_root / spec.name / "pair-item-predictions.parquet")
        direct_ref = null_direct if contribution == "singular" else full_direct
        d = _direct_contribution(direct, direct_ref, contribution)
        p = _pair_contribution(pair, None if contribution == "singular" else full_pair, contribution)
        return (_summaries(d, "model", metadata | {"task": "direct", "measure": "delta_mae"}, iterations)
                + _summaries(p, "item_unit", metadata | {"task": "pairwise", "measure": "delta_accuracy"}, iterations))

    for feature, specs in feature_specs.items():
        for contribution, spec in specs.items():
            feature_rows.extend(evaluate(spec, contribution, {
                "variable_type": "feature", "variable": feature,
                "block": feature_block[feature], "contribution": contribution,
            }))
    for (group_type, group), specs in group_specs.items():
        for contribution, spec in specs.items():
            group_rows.extend(evaluate(spec, contribution, {
                "variable_type": group_type, "variable": group,
                "block": group, "contribution": contribution,
            }))
    return pd.DataFrame(feature_rows), pd.DataFrame(group_rows)


def _style() -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "svg.fonttype": "none", "axes.linewidth": 0.7,
                         "scatter.edgecolors": "none", "lines.markeredgewidth": 0})


def _ordinary(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _save(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def _scale(table: pd.DataFrame, task: str) -> pd.DataFrame:
    output = table.copy()
    if task in {"direct", "pairwise"}:
        for column in ("estimate", "lower", "upper"):
            output[column] *= 100
    return output


def plot_feature_overview(
    table: pd.DataFrame, output: Path, task: str, *, top: int = 12, width: float = 3.5,
) -> None:
    frame = table[(table.task == task) & (table.target_quartile == "all")]
    if "benchmark" in frame:
        frame = frame[frame.benchmark == "all"]
    frame = _scale(frame, task)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(width, min(3.2, max(2.0, 0.22 * top + 0.8))),
                             layout="constrained")
    unit = "ΔMAE (pp)" if task == "direct" else "Δ accuracy (pp)"
    for ax, contribution in zip(axes, ("singular", "unique")):
        values = frame[frame.contribution == contribution].nlargest(top, "estimate").sort_values("estimate")
        y = np.arange(len(values))
        point = values.estimate.to_numpy(float)
        error = np.vstack([point - values.lower.to_numpy(float), values.upper.to_numpy(float) - point])
        ax.errorbar(point, y, xerr=error, fmt="o", ms=3.5, lw=1, capsize=0,
                    color=EFFECT_COLORS[contribution])
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_yticks(y, [textwrap.fill(v.replace("_", " "), 17) for v in values.variable], fontsize=7)
        ax.set_xlabel(unit)
        ax.set_title(contribution.capitalize(), loc="left", color="#666666", fontsize=8, pad=2)
        _ordinary(ax)
    _save(fig, output, f"feature-contributions-{task}")


def plot_groups(table: pd.DataFrame, output: Path, task: str) -> None:
    frame = table[(table.task == task) & (table.target_quartile == "all")]
    if "benchmark" in frame:
        frame = frame[frame.benchmark == "all"]
    frame = _scale(frame, task)
    variables = frame.groupby("variable").estimate.apply(lambda x: x.abs().max()).sort_values().index
    _style()
    fig, ax = plt.subplots(figsize=(2.8, 3.0), layout="constrained")
    y = np.arange(len(variables))
    for offset, contribution in ((-0.12, "singular"), (0.12, "unique")):
        values = frame[frame.contribution == contribution].set_index("variable").reindex(variables)
        point = values.estimate.to_numpy(float)
        error = np.vstack([point - values.lower.to_numpy(float), values.upper.to_numpy(float) - point])
        ax.errorbar(point, y + offset, xerr=error, fmt="o", ms=3.5, lw=1, capsize=0,
                    color=EFFECT_COLORS[contribution], label=contribution.capitalize())
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(y, [textwrap.fill(v.replace("-", " "), 20) for v in variables], fontsize=7)
    ax.set_xlabel("ΔMAE (pp)" if task == "direct" else "Δ accuracy (pp)")
    ax.set_ylabel("Feature group")
    _ordinary(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save(fig, output, f"group-contributions-{task}")


def plot_each_block(table: pd.DataFrame, output: Path, task: str) -> None:
    frame = table[(table.task == task) & (table.target_quartile == "all")]
    if "benchmark" in frame:
        frame = frame[frame.benchmark == "all"]
    frame = _scale(frame, task)
    for block, group in frame.groupby("block", sort=False):
        order = group.groupby("variable").estimate.apply(lambda x: x.abs().max()).sort_values().index
        height = min(3.2, max(1.4, 0.105 * len(order) + 0.65))
        _style()
        fig, axes = plt.subplots(1, 2, figsize=(3.5, height), sharey=True, layout="constrained")
        y = np.arange(len(order))
        for column, contribution in enumerate(("singular", "unique")):
            ax = axes[column]
            values = group[group.contribution == contribution].set_index("variable").reindex(order)
            for index, value in enumerate(values.itertuples()):
                _plot_error_point(ax, value.estimate, y[index], value.lower, value.upper,
                                  EFFECT_COLORS[contribution])
            ax.axvline(0, color="#999999", lw=0.6, ls="--")
            ax.set_yticks(y)
            if column == 0:
                ax.set_yticklabels([name.replace("_", " ") for name in order], fontsize=7)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_ylim(0, len(order) - 1)
            ax.tick_params(axis="y", length=0, pad=1)
            limits, ticks = _display_limits(values.estimate.to_numpy(float),
                                             nonnegative=contribution == "singular",
                                             symmetric=False)
            ax.set_xlim(*limits)
            ax.set_xticks(ticks, [_format_tick(value) for value in ticks])
            ax.set_xlabel("MAE (pp)" if task == "direct" else "Accuracy (pp)")
            ax.set_title(contribution.capitalize(), loc="left", color="#666666", fontsize=8, pad=2)
            _ordinary(ax)
        _save(fig, output, f"{task}-{block}-contributions")


def _nice_limit(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(value))
    return next(multiplier * base for multiplier in (1, 2, 5, 10) if value <= multiplier * base)


def _profile_limit(value: float) -> float:
    if value <= 2:
        return max(0.25, math.ceil(value * 4) / 4)
    if value <= 10:
        return math.ceil(value)
    if value <= 50:
        return math.ceil(value / 5) * 5
    return _nice_limit(value)


def _display_limits(values: np.ndarray, *, nonnegative: bool = False, symmetric: bool = True):
    values = np.asarray(values, dtype=float)
    if nonnegative and np.nanmin(values) >= 0:
        upper = _profile_limit(float(np.nanmax(values)) * 1.05)
        return (0, upper), [0, upper]
    if symmetric:
        limit = _profile_limit(float(np.nanmax(np.abs(values))) * 1.05)
        return (-limit, limit), [-limit, 0, limit]
    lower = -_profile_limit(abs(float(np.nanmin(values))) * 1.05) if np.nanmin(values) < 0 else 0
    upper = _profile_limit(float(np.nanmax(values)) * 1.05) if np.nanmax(values) > 0 else 0
    return (lower, upper), list(dict.fromkeys((lower, upper)))


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = f"{abs(value):g}"
    if magnitude.startswith("0."):
        magnitude = magnitude[1:]
    return ("−" if value < 0 else "") + magnitude


def _heatmap_limits(values: np.ndarray, contribution: str):
    return _display_limits(values, nonnegative=contribution == "singular", symmetric=False)


def _cross_feature_count(task: str) -> int:
    return 20 if task == "pairwise" else 10


def _cross_heatmap_height(count: int) -> float:
    return min(4.8, max(1.8, 0.19 * count + 0.9))


def _cross_diamond_height(count: int) -> float:
    return min(9.0, 0.78 * math.ceil(count / 2) + 1.0)


def _symmetric_axis(ax, values: np.ndarray) -> None:
    limit = _nice_limit(float(np.nanmax(np.abs(values))) * 1.05)
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])


def shared_feature_selection(table: pd.DataFrame, task: str, count: int = 10) -> list[str]:
    """Choose one support-first feature set shared by every benchmark row."""
    frame = table[(table.task == task) & (table.target_quartile == "all")
                  & (table.contribution == "unique") & (table.benchmark != "all")]
    frame = frame[~frame.variable.isin(EXCLUDED_FEATURES)]
    summary = cross_benchmark_importance(frame)
    summary["mean_positive"] = summary.mean_positive.fillna(-np.inf)
    summary["peak"] = frame.groupby("variable").estimate.max().reindex(summary.variable).to_numpy()
    return (summary.sort_values(["positive_support", "mean_positive", "peak", "variable"],
                                ascending=[False, False, False, True])
            .head(count).variable.tolist())


def coefficient_fold_summary(
    coefficients: pd.DataFrame,
    task: str,
    benchmark: str,
    variables: list[str],
) -> pd.DataFrame:
    """Average targets within fold, then report the mean and range over outer folds."""
    value = "coefficient" if task == "direct" else "effective"
    term = "benchmark_slope" if task == "direct" else "benchmark_deviation"
    frame = coefficients[(coefficients.task == task)
                         & (coefficients.term == term)
                         & (coefficients.benchmark == benchmark)
                         & coefficients.feature.isin(variables)]
    per_fold = frame.groupby(["feature", "fold"], as_index=False)[value].mean()
    return (per_fold.groupby("feature", as_index=False)[value]
            .agg(mean="mean", lower="min", upper="max", n_folds="size"))


def _plot_error_point(ax, estimate: float, y: float, lower: float, upper: float, color: str):
    interval = ax.errorbar(estimate, y, xerr=[[estimate - lower], [upper - estimate]],
                           fmt="none", lw=0.8, capsize=0, color=color)
    marker, = ax.plot(estimate, y, "o", ms=2.2, color=color, clip_on=False, zorder=3)
    return marker, interval


def plot_benchmark_grid(
    contributions: pd.DataFrame,
    coefficients: pd.DataFrame,
    output: Path,
    task: str,
    *,
    count: int = 10,
    height: float | None = None,
) -> None:
    """Plot four benchmarks by singular, unique, and coefficient columns."""
    output.mkdir(parents=True, exist_ok=True)
    frame = contributions[(contributions.task == task)
                          & (contributions.target_quartile == "all")
                          & (contributions.benchmark != "all")]
    benchmarks = [name for name in ("aime", "gpqa", "intercode", "swe_bench_mini")
                  if name in set(frame.benchmark)]
    labels = {"aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
              "swe_bench_mini": "SWE"}
    _style()
    variables = list(reversed(shared_feature_selection(frame, task, count=count)))
    fig, axes = plt.subplots(len(benchmarks), 3, figsize=(4.0, height or 5.2),
                             layout="constrained", squeeze=False)
    column_points = [[], [], []]
    used_blocks = set()
    for row_index, benchmark in enumerate(benchmarks):
        benchmark_rows = _scale(frame[frame.benchmark == benchmark], task)
        blocks = benchmark_rows.drop_duplicates("variable").set_index("variable").block.to_dict()
        used_blocks.update(blocks[name] for name in variables)
        y = np.arange(len(variables))
        for column, contribution in enumerate(("singular", "unique")):
            ax = axes[row_index, column]
            values = (benchmark_rows[benchmark_rows.contribution == contribution]
                      .set_index("variable").reindex(variables))
            for index, value in enumerate(values.itertuples()):
                _plot_error_point(ax, value.estimate, y[index], value.lower, value.upper,
                                  COLORS[blocks[value.Index]])
            column_points[column].append(values.estimate.to_numpy(float))
        coefficient = (coefficient_fold_summary(coefficients, task, benchmark, variables)
                       .set_index("feature").reindex(variables))
        ax = axes[row_index, 2]
        for index, value in enumerate(coefficient.itertuples()):
            _plot_error_point(ax, value.mean, y[index], value.lower, value.upper,
                              COLORS[blocks[value.Index]])
        column_points[2].append(coefficient["mean"].to_numpy(float))

        tick_labels = [name.replace("_", " ") for name in variables]
        for column, ax in enumerate(axes[row_index]):
            ax.axvline(0, color="#999999", lw=0.6, ls="--")
            ax.set_yticks(y, tick_labels if column == 0 else [], fontsize=7)
            ax.set_ylim(0, len(variables) - 1)
            ax.tick_params(axis="y", length=0, pad=1)
            ax.tick_params(axis="x", labelsize=7, length=2, width=0.6, pad=1)
            if row_index < len(benchmarks) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            _ordinary(ax)
        axes[row_index, 0].text(-0.03, 1.02, labels[benchmark],
                                transform=axes[row_index, 0].transAxes,
                                ha="right", va="bottom", fontsize=7, color="#666666")

    for column, title in enumerate(("Singular", "Unique", "Coefficient")):
        axes[0, column].set_title(title, loc="left", color="#666666", fontsize=7, pad=10)
        limits, ticks = _display_limits(np.concatenate(column_points[column]),
                                        nonnegative=column == 0, symmetric=column == 2)
        for ax in axes[:, column]:
            ax.set_xlim(*limits)
            ax.set_xticks(ticks, [_format_tick(value) for value in ticks])
    unit = "MAE (pp)" if task == "direct" else "Accuracy (pp)"
    axes[-1, 0].set_xlabel(unit)
    axes[-1, 1].set_xlabel(unit)
    axes[-1, 2].set_xlabel("β (1 SD)")
    handles = [Line2D([], [], marker="o", ls="none", ms=3, color=COLORS[block],
                      label=block.replace("-", " "))
               for block in COLORS if block in used_blocks]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    _save(fig, output, "feature-summary-by-benchmark")


def cross_benchmark_importance(rows: pd.DataFrame) -> pd.DataFrame:
    """Count benchmarks with reliably helpful or harmful unique contribution."""
    output = []
    for variable, group in rows.groupby("variable", sort=False):
        positive, negative = group[group.lower > 0], group[group.upper < 0]
        output.append({
            "variable": variable, "block": group.block.iloc[0],
            "positive_support": len(positive), "negative_support": len(negative),
            "positive_benchmarks": ", ".join(sorted(positive.benchmark.astype(str))),
            "negative_benchmarks": ", ".join(sorted(negative.benchmark.astype(str))),
            "mean_positive": positive.estimate.mean(), "mean_negative": negative.estimate.mean(),
        })
    return pd.DataFrame(output)


def plot_cross_benchmark_profiles(
    table: pd.DataFrame,
    variables: list[str],
    output: Path,
    *,
    task: str,
    contribution: str,
    diamond_height: float | None = None,
    diamond_chunk_size: int | None = None,
    heatmap_width: float = 2.8,
    diamond_width: float = 3.2,
) -> None:
    """Plot signed heatmap and positive-effect diamond glyphs for one contribution."""
    output.mkdir(parents=True, exist_ok=True)
    frame = table[(table.task == task) & (table.target_quartile == "all")
                  & (table.contribution == contribution) & (table.benchmark != "all")
                  & table.variable.isin(variables)]
    benchmarks = [name for name in ("aime", "gpqa", "intercode", "swe_bench_mini")
                  if name in set(frame.benchmark)]
    benchmark_labels = {"aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
                        "swe_bench_mini": "SWE"}
    blocks = frame.drop_duplicates("variable").set_index("variable").block.to_dict()
    values = (frame.pivot(index="variable", columns="benchmark", values="estimate")
              .reindex(index=variables, columns=benchmarks) * 100)
    lower = (frame.pivot(index="variable", columns="benchmark", values="lower")
             .reindex(index=variables, columns=benchmarks) * 100)
    upper = (frame.pivot(index="variable", columns="benchmark", values="upper")
             .reindex(index=variables, columns=benchmarks) * 100)
    (heat_lower, heat_upper), heat_ticks = _heatmap_limits(values.to_numpy(float), contribution)
    diamond_limit = _profile_limit(float(np.nanmax(np.abs(values.to_numpy(float)))) * 1.05)

    _style()
    if heat_lower == 0:
        cmap = LinearSegmentedColormap.from_list("positive_effect", ["#f4f1eb", "#2a78d6"])
        norm = Normalize(vmin=0, vmax=heat_upper)
    else:
        cmap = LinearSegmentedColormap.from_list("signed_effect", ["#c65f47", "#f4f1eb", "#2a78d6"])
        norm = TwoSlopeNorm(vmin=heat_lower, vcenter=0, vmax=heat_upper)
    heat_height = _cross_heatmap_height(len(variables))
    fig, ax = plt.subplots(figsize=(heatmap_width, heat_height), layout="constrained")
    image = ax.imshow(values.to_numpy(float), cmap=cmap, norm=norm, aspect="auto")
    significant = (lower.to_numpy(float) > 0) | (upper.to_numpy(float) < 0)
    for row in range(len(variables)):
        for column in range(len(benchmarks)):
            value = values.iat[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 * max(abs(heat_lower), heat_upper) else "#222222")
            if significant[row, column]:
                ax.add_patch(Rectangle((column - 0.48, row - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="#222222", lw=0.8))
    ax.set_xticks(range(len(benchmarks)), [benchmark_labels[name] for name in benchmarks],
                  rotation=90, ha="center")
    ax.set_yticks(range(len(variables)), [name.replace("_", " ") for name in variables], fontsize=7)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    colorbar.set_ticks(heat_ticks)
    colorbar.set_ticklabels([_format_tick(value) for value in heat_ticks])
    measure = "ΔMAE (pp)" if task == "direct" else "Δ accuracy (pp)"
    colorbar.set_label(f"{contribution.capitalize()} {measure}")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, edgecolor="#222222", lw=0.8,
                                  label="95% CI excludes 0")],
               loc="outside upper center", frameon=False)
    _save(fig, output, f"{contribution}-effect-heatmap")

    directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=float)
    chunks = ([variables] if diamond_chunk_size is None else
              [variables[index:index + diamond_chunk_size]
               for index in range(0, len(variables), diamond_chunk_size)])
    for chunk_index, chunk in enumerate(chunks, 1):
        rows = math.ceil(len(chunk) / 2)
        fig, axes = plt.subplots(
            rows, 2, figsize=(diamond_width, diamond_height or _cross_diamond_height(len(chunk))),
            layout="constrained", squeeze=False,
        )
        used_blocks = []
        for ax, variable in zip(axes.flat, chunk):
            block, color = blocks[variable], COLORS[blocks[variable]]
            if block not in used_blocks:
                used_blocks.append(block)
            for radius, line_style in ((1.0, "-"), (0.5, ":")):
                ax.add_patch(Polygon(directions * radius, closed=True, fill=False,
                                     edgecolor="#c8c4bd", lw=0.6, ls=line_style))
            for direction in directions:
                ax.plot([0, direction[0]], [0, direction[1]], color="#d8d4ce", lw=0.5)
            feature_values = values.loc[variable].to_numpy(float)
            radii = np.clip(feature_values, 0, diamond_limit) / diamond_limit
            vertices = directions * radii[:, None]
            ax.add_patch(Polygon(vertices, closed=True, facecolor=color, edgecolor=color,
                                 alpha=0.18, lw=1))
            feature_lower = lower.loc[variable].to_numpy(float)
            for vertex, is_significant in zip(vertices, feature_lower > 0):
                ax.scatter(*vertex, s=8, marker="o" if is_significant else "^",
                           color=color, alpha=1 if is_significant else 0.42,
                           edgecolors="none", linewidths=0, zorder=3)
            ax.text(0, 1.34, textwrap.fill(variable.replace("_", " "), 22),
                    ha="center", va="bottom", fontsize=7)
            for direction, label, alignment in zip(
                directions, ("AIME", "GPQA", "InterCode", "SWE"),
                (("center", "bottom"), ("left", "center"),
                 ("center", "top"), ("right", "center")),
            ):
                ax.text(*(direction * 1.1), label, ha=alignment[0], va=alignment[1], fontsize=7)
            ax.set_xlim(-1.28, 1.28)
            ax.set_ylim(-1.28, 1.28)
            ax.set_aspect("equal")
            ax.axis("off")
        for ax in axes.flat[len(chunk):]:
            ax.remove()
        handles = [Line2D([], [], marker="o", ls="none", ms=3, color=COLORS[block],
                          markeredgewidth=0, label=block.replace("-", " "))
                   for block in used_blocks]
        handles += [Line2D([], [], marker="o", ls="none", ms=3, color="#777777",
                           markeredgewidth=0, label="95% CI > 0"),
                    Line2D([], [], marker="^", ls="none", ms=3, color="#777777",
                           alpha=0.42, markeredgewidth=0, label="Uncertain"),
                    Line2D([], [], color="#c8c4bd", lw=0.8,
                           label=f"Outer ring = {diamond_limit:g} pp")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3,
                   frameon=False, fontsize=7)
        suffix = "" if len(chunks) == 1 else f"-{chunk_index}"
        _save(fig, output, f"{contribution}-effect-diamonds{suffix}")


def plot_coefficient_heatmap(
    coefficients: pd.DataFrame,
    variables: list[str],
    output: Path,
    *,
    task: str,
    width: float = 2.8,
) -> None:
    """Plot effective benchmark-specific feature weights and their fold range."""
    output.mkdir(parents=True, exist_ok=True)
    benchmark_order = [name for name in ("aime", "gpqa", "intercode", "swe_bench_mini")
                       if name in set(coefficients.benchmark.dropna())]
    benchmark_labels = {"aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
                        "swe_bench_mini": "SWE"}
    summaries = [coefficient_fold_summary(coefficients, task, benchmark, variables)
                 .assign(benchmark=benchmark) for benchmark in benchmark_order]
    summary = pd.concat(summaries, ignore_index=True)
    values = (summary.pivot(index="feature", columns="benchmark", values="mean")
              .reindex(index=variables, columns=benchmark_order))
    lower = (summary.pivot(index="feature", columns="benchmark", values="lower")
             .reindex(index=variables, columns=benchmark_order))
    upper = (summary.pivot(index="feature", columns="benchmark", values="upper")
             .reindex(index=variables, columns=benchmark_order))
    limit = _profile_limit(float(np.nanmax(np.abs(values.to_numpy(float)))) * 1.05)
    cmap = LinearSegmentedColormap.from_list("coefficient", ["#c65f47", "#f4f1eb", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    _style()
    height = _cross_heatmap_height(len(variables))
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    image = ax.imshow(values.to_numpy(float), cmap=cmap, norm=norm, aspect="auto")
    stable = (lower.to_numpy(float) > 0) | (upper.to_numpy(float) < 0)
    for row in range(len(variables)):
        for column in range(len(benchmark_order)):
            value = values.iat[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 * limit else "#222222")
            if stable[row, column]:
                ax.add_patch(Rectangle((column - 0.48, row - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="#222222", lw=0.8))
    ax.set_xticks(range(len(benchmark_order)), [benchmark_labels[name] for name in benchmark_order],
                  rotation=90, ha="center")
    ax.set_yticks(range(len(variables)), [name.replace("_", " ") for name in variables], fontsize=7)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    ticks = [-limit, 0, limit]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_tick(value) for value in ticks])
    colorbar.set_label("Effective β (1 SD)")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, edgecolor="#222222", lw=0.8,
                                  label="Fold range excludes 0")],
               loc="outside upper center", frameon=False)
    _save(fig, output, "coefficient-heatmap")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpret-root", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    inputs = load_inputs(type(CONFIG)(**{**CONFIG.__dict__, "input_root": args.input_root}))
    features, groups = compute_contributions(args.interpret_root, args.full_results, inputs.dictionary,
                                             iterations=args.bootstrap)
    features = features[~features.variable.isin(EXCLUDED_FEATURES)].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    tables.mkdir(exist_ok=True)
    features.to_csv(tables / "feature-contributions.csv", index=False)
    groups.to_csv(tables / "group-contributions.csv", index=False)
    coefficients = pd.read_parquet(Path(args.interpret_root) / "full/coefficients.parquet")
    for task in ("direct", "pairwise"):
        task_root = args.output / ("capability" if task == "direct" else "pairwise")
        overview, by_group, by_benchmark = (task_root / "overview", task_root / "by-feature-group",
                                             task_root / "by-benchmark")
        overview.mkdir(parents=True, exist_ok=True)
        by_group.mkdir(exist_ok=True)
        plot_feature_overview(features, overview, task)
        plot_groups(groups, overview, task)
        plot_each_block(features, by_group, task)
        plot_benchmark_grid(features, coefficients, by_benchmark, task)
        cross_benchmark = task_root / "cross-benchmark"
        shared_features = shared_feature_selection(features, task, count=_cross_feature_count(task))
        for contribution in ("singular", "unique"):
            plot_cross_benchmark_profiles(
                features, shared_features, cross_benchmark, task=task, contribution=contribution)
        if task == "pairwise":
            plot_coefficient_heatmap(
                coefficients, shared_features, cross_benchmark, task=task)
    (args.output / "manifest.json").write_text(json.dumps({
        "direct_measure": "positive = held-out MAE reduction",
        "pairwise_measure": "positive = held-out stronger-model classification-accuracy gain (pp)",
        "direct_bootstrap_unit": "model (N=26, all benchmark/target rows kept together)",
        "pairwise_bootstrap_unit": "Q1 benchmark/item",
        "bootstrap_iterations": args.bootstrap,
        "benchmark_plot_selection": "10 shared features ranked by positive unique-effect benchmark support",
        "cross_benchmark_feature_counts": {"capability": 10, "pairwise": 20},
        "coefficient_error_bars": "min-max of four outer-fold means after averaging Q2-Q4 within fold",
        "coefficient_heatmap": "effective population-plus-benchmark weights; outline = fold range excludes zero",
        "cross_benchmark_importance": "unique 95% interval entirely above/below zero",
        "diamond_radius": "positive contribution on a common raw percentage-point scale within each figure",
        "excluded_presentational_features": sorted(EXCLUDED_FEATURES),
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
