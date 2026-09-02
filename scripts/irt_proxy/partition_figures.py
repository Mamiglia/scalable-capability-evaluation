"""Figures for the easiest-slice prediction analyses.

The two predictors get separate scatter panels.  Model colours are shared
between those panels so the same point is easy to follow.  Summary figures use
a second y axis for the observed accuracy spread across models, including the
easiest predictor slice.  Per-judge summaries keep the pooled consensus and
judge-replication views separate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

from irt.plots import GRID, INK, INK2, MUTED, SURFACE


PARTITION_BINS = {
    "halves": ["hard"],
    "tertile": ["T2", "T3"],
    "quartile": ["Q2", "Q3", "Q4"],
    "quintile": ["P2", "P3", "P4", "P5"],
}
POOLED_BINS = {
    "tertile": "T2–T3",
    "quartile": "Q2–Q4",
    "quintile": "P2–P5",
}
PARTITION_FOLDERS = {
    "halves": "halves",
    "tertile": "tertiles",
    "quartile": "quartiles",
    "quintile": "quintiles",
}
FIT_ORDER = ["gpqa"]
METHOD_COLOR = {"elo": "#2a78d6", "easy_accuracy": "#c65353"}
SUMMARY_X_OFFSETS = {"elo": -0.09, "easy_accuracy": 0.0, "model_accuracy": 0.09}
MODEL_ACCURACY_COLOR = "#aaa89f"
MODEL_ACCURACY_POINT_COLOR = "#d0cfc9"
AXIS_PAD = 0.04

FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
}


def _style(ax) -> None:
    """Apply the same quiet axes used by the repository's IRT figures."""
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, length=2)


def _save(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    save_args = {"bbox_inches": "tight", "pad_inches": 0.06, "facecolor": SURFACE}
    fig.savefig(directory / f"{name}.png", **save_args)
    fig.savefig(directory / f"{name}.svg", **save_args)
    plt.close(fig)


def _model_colors(models: pd.Series) -> dict[str, str]:
    """Give every model one stable categorical colour within a figure."""
    names = sorted(pd.unique(models.dropna()))
    samples = np.linspace(0.04, 0.96, len(names))
    return dict(zip(names, (to_hex(color) for color in plt.colormaps["turbo"](samples))))


def _rank_within_component(cell: pd.DataFrame) -> np.ndarray:
    """Rank each predictor separately inside each disconnected component."""
    groups = cell.groupby(["method", "component"])["predictor"]
    ranks = groups.rank(method="average")
    counts = groups.transform("count")
    return np.where(counts > 1, (ranks - 1) / (counts - 1), 0.5)


def _filter_label(data: pd.DataFrame) -> str:
    """Describe the split-specific saturation filter in plain language."""
    predictor_bin = data["predictor_bin"].drop_duplicates().item()
    threshold = data["predictor_accuracy_threshold"].drop_duplicates().item()
    return f"{predictor_bin} >{threshold:.0%}"


def _model_legend_figure(predictions: pd.DataFrame, output_directory: Path) -> None:
    """Standalone colour key for the per-model scatter colours.

    _model_colors sorts model names, so any scatter over the same model pool
    shares this mapping; one key next to the quartile gpqa rank scatter covers
    them all."""
    data = predictions[
        (predictions["irt_model"] == "fixed_3pl")
        & (predictions["judge"] == "consensus")
        & (predictions["partition"] == "quartile")
        & (predictions["fit"] == "gpqa")
        & (predictions["method"].isin(METHOD_COLOR))
    ]
    model_colors = _model_colors(data["model"])
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=4,
                   markeredgewidth=0, color=colour, label=model)
        for model, colour in model_colors.items()
    ]
    rows = -(-len(handles) // 2)
    fig, ax = plt.subplots(figsize=(3.7, 0.14 * rows + 0.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=2, frameon=False,
              fontsize=6, handletextpad=0.3, labelspacing=0.4,
              columnspacing=1.2, labelcolor=INK)
    _save(fig, output_directory / PARTITION_FOLDERS["quartile"],
          "gpqa_rank_scatter_legend")


def _label_points(ax, xs, ys, names) -> None:
    """Direct model labels with leader lines, stacked greedily to avoid overlap."""
    # ponytail: one global vertical stack, so labels never collide even across
    # the left/right sides; fine for inspection figures
    gap = 0.045
    order = sorted(range(len(xs)), key=lambda index: ys[index])
    label_y, previous = [], -np.inf
    for index in order:
        previous = max(ys[index], previous + gap)
        label_y.append(previous)
    overflow = max(0.0, label_y[-1] - 1.02) if label_y else 0.0
    for index, y in zip(order, label_y):
        left = xs[index] > 0.8
        ax.annotate(
            names[index], xy=(xs[index], ys[index]),
            xytext=(xs[index] + (-0.05 if left else 0.05), y - overflow),
            ha="right" if left else "left", va="center",
            fontsize=4, color=INK2, annotation_clip=False,
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": MUTED,
                        "shrinkA": 0, "shrinkB": 1},
        )


def _significance_label(p_value: float, *, show_ns: bool = False) -> str:
    """Use the conventional compact labels without hiding a comparison."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns" if show_ns else ""


def _add_significance_annotations(
    ax,
    panel_metrics: pd.DataFrame,
    panel_nulls: pd.DataFrame,
    target_bins: list[str],
    target_x: np.ndarray,
) -> None:
    """Stack null stars and the method comparison above the plotting area."""
    transform = ax.get_xaxis_transform()
    for position, target_bin in zip(target_x, target_bins):
        target_metrics = panel_metrics[panel_metrics["target_bin"] == target_bin]
        target_nulls = panel_nulls[panel_nulls["target_bin"] == target_bin]
        marker_y = {"elo": 1.015, "easy_accuracy": 1.05}
        for method in METHOD_COLOR:
            null_rows = target_nulls[target_nulls["method"] == method]
            if null_rows.empty:
                continue
            label = _significance_label(null_rows["p_value"].iloc[0])
            if label:
                ax.text(
                    position + SUMMARY_X_OFFSETS[method],
                    marker_y[method],
                    label,
                    color=METHOD_COLOR[method],
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    transform=transform,
                    clip_on=False,
                    zorder=6,
                )

        comparison = target_metrics["comparison_p_value"].dropna()
        if comparison.empty:
            continue
        left = position + SUMMARY_X_OFFSETS["elo"]
        right = position + SUMMARY_X_OFFSETS["easy_accuracy"]
        ax.plot(
            [left, right],
            [1.17, 1.17],
            color=INK,
            linewidth=0.7,
            transform=transform,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (left + right) / 2,
            1.18,
            _significance_label(comparison.iloc[0], show_ns=True),
            color=INK,
            fontsize=6.5,
            ha="center",
            va="bottom",
            transform=transform,
            clip_on=False,
            zorder=6,
        )


def _scatter_figure(
    predictions: pd.DataFrame,
    *,
    partition: str,
    fit: str,
    kind: str,
    output_directory: Path,
    target_bins: list[str] | None = None,
    pooled: bool = False,
    label_models: bool = False,
) -> None:
    """Show Elo and accuracy predictors side by side for one IRT fit."""
    target_bins = target_bins or PARTITION_BINS[partition]
    data = predictions[
        (predictions["irt_model"] == "fixed_3pl")
        & (predictions["judge"] == "consensus")
        & (predictions["partition"] == partition)
        & (predictions["fit"] == fit)
        & (predictions["method"].isin(METHOD_COLOR))
    ].copy()

    point_key = ["fit", "target_bin", "component", "model", "method"]
    if data.duplicated(point_key).any():
        raise ValueError("duplicate model point in partition predictions")
    model_colors = _model_colors(data["model"])
    filter_label = _filter_label(data)

    scale = 2 if label_models else 1
    fig, axes = plt.subplots(
        len(target_bins),
        2,
        figsize=(3.7 * scale, max(1.8, 0.90 * len(target_bins) + 0.55) * scale),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, target_bin in enumerate(target_bins):
        for column_index, method in enumerate(METHOD_COLOR):
            ax = axes[row_index, column_index]
            cell = data[
                (data["target_bin"] == target_bin) & (data["method"] == method)
            ].copy()

            if kind == "rank":
                cell["_x"] = _rank_within_component(cell)
                cell["_y"] = cell["observed"]
            else:
                ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=0.8, zorder=1)
                cell["_x"] = cell["observed"]
                cell["_y"] = cell["prediction"]

            ax.scatter(
                cell["_x"],
                cell["_y"],
                s=12,
                color=cell["model"].map(model_colors),
                alpha=0.68,
                edgecolors="none",
                linewidths=0,
                zorder=3,
            )
            if label_models:
                _label_points(ax, cell["_x"].to_numpy(), cell["_y"].to_numpy(),
                              cell["model"].tolist())

            ax.set_xlim(-AXIS_PAD, 1 + AXIS_PAD)
            ax.set_ylim(-AXIS_PAD, 1 + AXIS_PAD)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            _style(ax)

            if row_index == 0:
                predictor = "Elo predictor" if method == "elo" else "accuracy predictor"
                ax.set_title(f"{fit.upper()} · {predictor}\n{filter_label}", color=INK2,
                             fontsize=8, pad=3)
            if column_index == 0:
                ax.set_ylabel(target_bin, color=INK2, rotation=0, labelpad=10)
            if row_index == len(target_bins) - 1:
                ax.set_xlabel("Predictor rank" if kind == "rank" else "Observed",
                              color=INK2)

    outcome = "Observed accuracy" if kind == "rank" else "LOMO-predicted accuracy"
    fig.supylabel(outcome, x=0.01, color=INK2, fontsize=8)

    directory = output_directory / PARTITION_FOLDERS[partition]
    suffix = "rank" if kind == "rank" else "score_prediction"
    name = (f"{fit}_{suffix}_scatter" + ("_easy_vs_rest" if pooled else "")
            + ("_labelled" if label_models else ""))
    _save(fig, directory, name)


def _panel_order(predictions: pd.DataFrame) -> list[tuple[str, object, int]]:
    """Order human-readable fit/model-count panels without exposing IDs."""
    sizes = (
        predictions.groupby(["fit", "component"])["model"]
        .nunique()
        .rename("n_models")
        .reset_index()
    )
    fit_index = {fit: index for index, fit in enumerate(FIT_ORDER)}
    rows = list(sizes.itertuples(index=False, name=None))
    return sorted(rows, key=lambda row: (fit_index.get(row[0], 99), -row[2]))


def _summary_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    nulls: pd.DataFrame,
    *,
    partition: str,
    fit: str,
    metric: str,
    output_directory: Path,
    target_bins: list[str] | None = None,
    pooled: bool = False,
) -> None:
    """Compare predictor quality while accuracy variance uses the other y axis."""
    target_bins = target_bins or PARTITION_BINS[partition]
    data = predictions[
        (predictions["irt_model"] == "fixed_3pl")
        & (predictions["judge"] == "consensus")
        & (predictions["partition"] == partition)
        & (predictions["fit"] == fit)
        & (predictions["method"].isin(METHOD_COLOR))
    ].copy()
    panels = _panel_order(data[data["method"] == "elo"])

    # Keep disconnected components as narrow vertical small multiples. A
    # side-by-side layout makes the saved canvas much wider than one column.
    n_columns = 1
    n_rows = max(1, len(panels))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(2.2 if pooled else 2.7, 1.75 * n_rows + 0.3),
        squeeze=False,
    )
    predictor_bin = data["predictor_bin"].drop_duplicates().item()
    filter_label = _filter_label(data)
    slice_bins = [predictor_bin, *target_bins]
    x = np.arange(len(slice_bins))
    target_x = x[1:]
    # Put the right-axis label on one central panel only. Repeating it on every
    # right-hand axes makes the labels print on top of one another.
    right_label_panel = min(
        len(panels) - 1,
        (n_rows // 2) * n_columns + (n_columns - 1),
    )

    metric_rows = metrics[
        (metrics["irt_model"] == "fixed_3pl")
        & (metrics["judge"] == "consensus")
        & (metrics["partition"] == partition)
        & (metrics["fit"] == fit)
        & (metrics["metric"] == metric)
        & (metrics["method"].isin(METHOD_COLOR))
    ]
    null_rows = nulls[
        (nulls["irt_model"] == "fixed_3pl")
        & (nulls["judge"] == "consensus")
        & (nulls["partition"] == partition)
        & (nulls["fit"] == fit)
        & (nulls["metric"] == metric)
        & (nulls["method"].isin(METHOD_COLOR))
    ]

    is_rank = metric == "tau_a"
    metric_label = "τ-a" if is_rank else "MAE"
    ylabel = "Kendall τ-a" if is_rank else "LOMO MAE\n(lower is better)"

    for panel_index, (ax, (fit, component, n_models)) in enumerate(
        zip(axes.flat, panels)
    ):
        panel_metrics = metric_rows[
            (metric_rows["fit"] == fit) & (metric_rows["component"] == component)
        ]
        panel_nulls = null_rows[
            (null_rows["fit"] == fit) & (null_rows["component"] == component)
        ]
        for method, name in (("elo", "Elo"),
                             ("easy_accuracy", "accuracy")):
            method_rows = panel_metrics[panel_metrics["method"] == method].set_index(
                "target_bin"
            ).reindex(target_bins)
            values = method_rows["value"]
            method_x = target_x + SUMMARY_X_OFFSETS[method]
            if {"ci_low", "ci_high"} <= set(method_rows):
                lower = (values - method_rows["ci_low"]).clip(lower=0)
                upper = (method_rows["ci_high"] - values).clip(lower=0)
                ax.errorbar(
                    method_x,
                    values,
                    yerr=np.vstack([lower, upper]),
                    fmt="-o",
                    color=METHOD_COLOR[method],
                    markersize=3,
                    linewidth=1,
                    elinewidth=0.8,
                    capsize=0,
                    label=f"{name} {metric_label} (95% model-bootstrap CI)",
                    zorder=4,
                )
            else:
                ax.plot(method_x, values, "-o", color=METHOD_COLOR[method], markersize=3,
                        linewidth=1, label=f"{name} {metric_label}", zorder=4)

        # The right axis shows the actual across-model spread in each target
        # slice.  Light points are models; the darker marker and bar are mean
        # plus/minus one sample standard deviation.
        accuracy_axis = ax.twinx()
        panel_predictions = data[
            (data["method"] == "elo")
            & (data["fit"] == fit)
            & (data["component"] == component)
        ]
        easiest = data[
            (data["method"] == "easy_accuracy")
            & (data["fit"] == fit)
            & (data["component"] == component)
        ].drop_duplicates("model")["predictor"]
        accuracy_by_slice = {
            predictor_bin: easiest,
            **{
                target_bin: panel_predictions[
                    panel_predictions["target_bin"] == target_bin
                ]["observed"]
                for target_bin in target_bins
            },
        }
        for position, slice_bin in enumerate(slice_bins):
            values = accuracy_by_slice[slice_bin].dropna()
            if values.empty:
                continue
            model_x = position + SUMMARY_X_OFFSETS["model_accuracy"]
            jitter = np.linspace(-0.025, 0.025, len(values))
            accuracy_axis.scatter(
                model_x + jitter,
                values,
                s=8,
                color=MODEL_ACCURACY_POINT_COLOR,
                alpha=0.65,
                edgecolors="none",
                linewidths=0,
                zorder=1,
            )
            standard_deviation = values.std(ddof=1) if len(values) > 1 else 0.0
            accuracy_axis.errorbar(
                model_x,
                values.mean(),
                yerr=standard_deviation,
                fmt="s",
                color=MODEL_ACCURACY_COLOR,
                markersize=3,
                linewidth=0.8,
                capsize=0,
                zorder=2,
            )

        ax.text(
            0,
            1.30,
            f"{fit.upper()} · {n_models} models · {filter_label}",
            transform=ax.transAxes,
            fontsize=7,
            color=MUTED,
            ha="left",
            va="bottom",
        )
        ax.set_xticks(x, slice_bins)
        ax.set_xlim(-0.35, len(slice_bins) - 0.65)
        if is_rank:
            ax.axhline(0, color=MODEL_ACCURACY_COLOR, linewidth=0.8,
                       linestyle="--", zorder=1)
            _add_significance_annotations(
                ax, panel_metrics, panel_nulls, target_bins, target_x,
            )
            ax.set_ylim(-1.05, 1.05)
        else:
            candidates = pd.concat([
                panel_metrics.get("ci_high", panel_metrics["value"]),
                panel_nulls["ci_high"],
            ]).dropna()
            top = max(0.30, candidates.max() * 1.08 if len(candidates) else 0.30)
            _add_significance_annotations(
                ax, panel_metrics, panel_nulls, target_bins, target_x,
            )
            ax.set_ylim(-0.01, top)
        accuracy_axis.set_ylim(-AXIS_PAD, 1 + AXIS_PAD)
        accuracy_axis.set_yticks([0, 0.5, 1])
        accuracy_axis.spines[["top", "left"]].set_visible(False)
        accuracy_axis.spines["right"].set_color(MODEL_ACCURACY_COLOR)
        accuracy_axis.tick_params(
            colors=MODEL_ACCURACY_COLOR,
            labelcolor=MODEL_ACCURACY_COLOR,
            length=2,
        )
        _style(ax)

        if panel_index % n_columns == 0:
            ax.set_ylabel(ylabel, color=INK2)
        if panel_index == right_label_panel:
            accuracy_axis.set_ylabel(
                "GPQA accuracy",
                color=MODEL_ACCURACY_COLOR,
            )
        if panel_index % n_columns != n_columns - 1 and n_columns != 1:
            accuracy_axis.set_yticklabels([])
        if panel_index >= len(panels) - n_columns:
            ax.set_xlabel("Item slice", color=INK2)

    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    ci_suffix = " (95% CI)" if {
        "ci_low", "ci_high"
    } <= set(metric_rows) else ""
    handles = [
        plt.Line2D([], [], color=METHOD_COLOR["elo"], marker="o", markersize=3,
                   label=f"Elo {metric_label}{ci_suffix}"),
        plt.Line2D([], [], color=METHOD_COLOR["easy_accuracy"], marker="o",
                   markersize=3, label=f"accuracy {metric_label}{ci_suffix}"),
        plt.Line2D([], [], color=MODEL_ACCURACY_COLOR, marker="s", linestyle="",
                   markersize=3, label="model acc. (mean ± SD)"),
        plt.Line2D([], [], color=METHOD_COLOR["elo"], marker="$*$", linestyle="",
                   markersize=5, label="blue *: Elo vs null"),
        plt.Line2D([], [], color=METHOD_COLOR["easy_accuracy"], marker="$*$",
                   linestyle="", markersize=5,
                   label="red *: acc. vs null"),
        plt.Line2D([], [], color=INK, marker="_", linestyle="", markersize=8,
                   label="bar: Elo vs acc."),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=6)
    kind = "rank" if is_rank else "score"
    suffix = "prediction_easy_vs_rest" if pooled else "prediction_by_slice"
    name = f"{fit}_{kind}_{suffix}"
    _save(fig, output_directory / PARTITION_FOLDERS[partition], name)


def render_partition_figures(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    nulls: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Render split-specific scatters and one dual-y summary per split."""
    with plt.rc_context(FIGURE_STYLE):
        _model_legend_figure(predictions, output_directory)
        for partition in PARTITION_BINS:
            for fit in FIT_ORDER:
                for kind in ("rank", "lomo"):
                    _scatter_figure(
                        predictions,
                        partition=partition,
                        fit=fit,
                        kind=kind,
                        output_directory=output_directory,
                    )
                if partition == "quartile":
                    _scatter_figure(
                        predictions,
                        partition=partition,
                        fit=fit,
                        kind="rank",
                        output_directory=output_directory,
                        label_models=True,
                    )
                    _scatter_figure(
                        predictions,
                        partition=partition,
                        fit=fit,
                        kind="rank",
                        output_directory=output_directory,
                        target_bins=[POOLED_BINS[partition]],
                        pooled=True,
                    )
                for metric in ("tau_a", "mae"):
                    _summary_figure(
                        predictions,
                        metrics,
                        nulls,
                        partition=partition,
                        fit=fit,
                        metric=metric,
                        output_directory=output_directory,
                    )
                if partition in POOLED_BINS:
                    for metric in ("tau_a", "mae"):
                        _summary_figure(
                            predictions,
                            metrics,
                            nulls,
                            partition=partition,
                            fit=fit,
                            metric=metric,
                            output_directory=output_directory,
                            target_bins=[POOLED_BINS[partition]],
                            pooled=True,
                        )
