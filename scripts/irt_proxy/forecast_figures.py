"""Temporal heatmaps of held-out GPQA capability-prediction error."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from irt.plots import INK, MUTED, SURFACE
from mc_cost.catalogue import normalized_name
from mc_cost.ranking import lomo_predictions
from partition_figures import FIGURE_STYLE
from scores import analysis_frame, fit_easy_elos


HORIZONS = ((0, 3, "~0 mo"), (4, 8, "~6 mo"),
            (9, 14, "~12 mo"), (15, 20, "~18 mo"))
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "forecast_error", ("#f7f9fc", "#2a78d6", "#0d3b66"),
)


def forecast_lomo_predictions(
    primary_predictions: pd.DataFrame,
    filtered_rounds: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Refit per-judge Elo without the separate eight-model saturation gate."""
    consensus = primary_predictions[
        primary_predictions["irt_model"].eq("fixed_3pl")
        & primary_predictions["fit"].eq("gpqa")
        & primary_predictions["bank"].eq("easy")
        & primary_predictions["judge"].eq("consensus")
        & primary_predictions["method"].eq("easy_accuracy")
    ].dropna(subset=["model", "predictor", "observed"])
    key = ["irt_model", "fit", "model"]
    accuracies = pd.concat([
        consensus[key + ["predictor"]]
        .rename(columns={"predictor": "accuracy"}).assign(subset="easy"),
        consensus[key + ["observed"]]
        .rename(columns={"observed": "accuracy"}).assign(subset="hard"),
    ], ignore_index=True)
    elos, coverage = fit_easy_elos(
        filtered_rounds, splits, bank="easy", eligible_models=consensus[key],
    )
    return lomo_predictions(analysis_frame(elos, accuracies, coverage))


def _viewer_metadata(metadata: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for model, rows in metadata.groupby("model", sort=False):
        ecis = rows["eci"].dropna().unique()
        releases = pd.to_datetime(
            rows["release"].dropna(), format="mixed",
        ).dt.to_period("M").unique()
        if len(ecis) > 1 or len(releases) > 1:
            raise ValueError(f"conflicting metadata for {model}")
        result[model] = {
            "eci": float(ecis[0]) if len(ecis) else None,
            "release": releases[0] if len(releases) else None,
        }
    return result


def prepare_temporal_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    model_pool: list[Mapping],
) -> pd.DataFrame:
    """Attach ECI/date metadata to future-model GPQA Elo predictions."""
    data = predictions[
        predictions["irt_model"].eq("fixed_3pl")
        & predictions["fit"].eq("gpqa")
        & predictions["bank"].eq("easy")
        & predictions["method"].eq("elo")
        & predictions["judge"].ne("consensus")
    ].dropna(subset=["model", "prediction", "observed"]).copy()
    viewer = _viewer_metadata(metadata)
    pool = {normalized_name(row["name"]): row for row in model_pool}

    def model_metadata(model: str) -> tuple[float | None, pd.Period | None]:
        fallback = viewer.get(model, {})
        matched = pool.get(normalized_name(model), {})
        eci = matched.get("eci") if matched.get("eci") is not None else fallback.get("eci")
        release = matched.get("release") or fallback.get("release")
        if release is not None and not isinstance(release, pd.Period):
            release = pd.Period(pd.to_datetime(release, format="mixed"), freq="M")
        return eci, release

    judge_meta = data["judge"].map(model_metadata)
    target_meta = data["model"].map(model_metadata)
    data["judge_eci"] = [value[0] for value in judge_meta]
    data["judge_release"] = [value[1] for value in judge_meta]
    data["target_eci"] = [value[0] for value in target_meta]
    data["target_release"] = [value[1] for value in target_meta]
    data = data.dropna(subset=[
        "judge_eci", "judge_release", "target_eci", "target_release",
    ])
    data["release_gap_months"] = [
        (target.year - judge.year) * 12 + target.month - judge.month
        for target, judge in zip(data["target_release"], data["judge_release"])
    ]
    data = data[data["release_gap_months"].ge(0)].copy()
    data["horizon"] = data["release_gap_months"].map(
        lambda gap: next((label for low, high, label in HORIZONS if low <= gap <= high), None),
    )
    data = data.dropna(subset=["horizon"])
    data["absolute_error"] = (
        data["prediction"] - data["observed"]
    ).abs().round(12)
    return data.sort_values(["release_gap_months", "judge", "model"]).reset_index(drop=True)


def _edges(values: pd.Series, step: float) -> np.ndarray:
    low = np.floor(values.min() / step) * step
    high = (np.floor(values.max() / step) + 1) * step
    return np.arange(low, high + step / 2, step)


def _save(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", facecolor=SURFACE)
    fig.savefig(directory / f"{name}.svg", facecolor=SURFACE)
    plt.close(fig)


def _heatmap(ax, data, x, y, x_edges, y_edges, vmax):
    cells = data.assign(
        _x=pd.cut(data[x], x_edges, labels=False, right=False, include_lowest=True),
        _y=pd.cut(data[y], y_edges, labels=False, right=False, include_lowest=True),
    ).dropna(subset=["_x", "_y"])
    means = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    counts = np.zeros_like(means, dtype=int)
    for (row, column), values in cells.groupby(["_y", "_x"])["absolute_error"]:
        means[int(row), int(column)] = values.mean() * 100
        counts[int(row), int(column)] = len(values)

    cmap = ERROR_CMAP.with_extremes(bad=SURFACE)
    image = ax.pcolormesh(
        x_edges, y_edges, means, cmap=cmap, vmin=0, vmax=vmax,
        edgecolors=SURFACE, linewidth=0.5,
    )
    for row, column in zip(*np.where(counts)):
        value = means[row, column]
        ax.text(
            (x_edges[column] + x_edges[column + 1]) / 2,
            (y_edges[row] + y_edges[row + 1]) / 2,
            f"{value:.0f}\nn={counts[row, column]}",
            ha="center", va="center", fontsize=5.5,
            color="white" if value > vmax * 0.62 else INK,
        )
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.5)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, length=2)
    return image


def _colorbar(fig, image, axes, vmax):
    bar = fig.colorbar(image, ax=axes, pad=0.04, fraction=0.055, ticks=[0, vmax])
    bar.set_label("Absolute prediction error (pp)")
    bar.outline.set_linewidth(0.5)
    bar.ax.tick_params(colors=MUTED, labelcolor=MUTED, length=2)


def render_temporal_heatmaps(data: pd.DataFrame, output_directory: Path) -> None:
    """Write the three ECI/date forecasting heatmaps as PNG and SVG."""
    if data.empty:
        raise ValueError("no future-model predictions with ECI and release metadata")
    output_directory = Path(output_directory)
    eci_edges = _edges(pd.concat([data["judge_eci"], data["target_eci"]]), 5)
    vmax = max(5, np.ceil(data["absolute_error"].max() * 20) * 5)
    eci_ticks = [eci_edges[0], eci_edges[-1]]

    with plt.rc_context(FIGURE_STYLE):
        facets = [
            (label, data[data["horizon"] == label])
            for _, _, label in HORIZONS
            if (data["horizon"] == label).any()
        ]
        fig, axes = plt.subplots(
            1, len(facets), squeeze=False, sharex=True, sharey=True,
            figsize=(min(4.25, 1.2 * len(facets) + 0.65), 1.75),
            layout="constrained",
        )
        for ax, (label, cell) in zip(axes.flat, facets):
            image = _heatmap(
                ax, cell, "judge_eci", "target_eci", eci_edges, eci_edges, vmax,
            )
            ax.text(
                0.02, 1.02, label, transform=ax.transAxes, va="bottom",
                fontsize=7, clip_on=False,
            )
            ax.set_xticks(eci_ticks)
            ax.set_yticks(eci_ticks)
        fig.supxlabel("Judge ECI")
        fig.supylabel("Target ECI")
        _colorbar(fig, image, list(axes.flat), vmax)
        _save(fig, output_directory, "gpqa_forecast_eci_by_horizon")

        dates = data.copy()
        dates["judge_release_year"] = dates["judge_release"].map(
            lambda date: date.year + (date.month - 1) / 12,
        )
        dates["target_release_year"] = dates["target_release"].map(
            lambda date: date.year + (date.month - 1) / 12,
        )
        release_edges = _edges(pd.concat([
            dates["judge_release_year"], dates["target_release_year"],
        ]), 0.5)
        fig, ax = plt.subplots(figsize=(2.45, 1.9), layout="constrained")
        image = _heatmap(
            ax, dates, "judge_release_year", "target_release_year",
            release_edges, release_edges, vmax,
        )
        ax.set(
            xlabel="Judge release", ylabel="Target release",
            xticks=[release_edges[0], release_edges[-1]],
            yticks=[release_edges[0], release_edges[-1]],
        )
        _colorbar(fig, image, ax, vmax)
        _save(fig, output_directory, "gpqa_forecast_release_dates")

        used_horizons = [
            (low, high, label) for low, high, label in HORIZONS
            if (data["horizon"] == label).any()
        ]
        gap_edges = np.array([used_horizons[0][0] - 0.5] + [
            high + 0.5 for _, high, _ in used_horizons
        ])
        fig, ax = plt.subplots(figsize=(2.45, 1.9), layout="constrained")
        image = _heatmap(
            ax, data, "judge_eci", "release_gap_months",
            eci_edges, gap_edges, vmax,
        )
        ax.set(
            xlabel="Judge ECI", ylabel="Release gap (months)",
            xticks=eci_ticks,
            yticks=[(low + high) / 2 for low, high, _ in used_horizons],
            yticklabels=[label.removeprefix("~").removesuffix(" mo")
                         for _, _, label in used_horizons],
        )
        _colorbar(fig, image, ax, vmax)
        _save(fig, output_directory, "gpqa_forecast_eci_by_release_gap")
