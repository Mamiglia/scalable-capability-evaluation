"""Per-judge views of the easiest-slice prediction analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from irt.plots import INK2
from metrics import BOOTSTRAP_RESAMPLES, kendall_tau_a
from partition_figures import (
    FIGURE_STYLE,
    FIT_ORDER,
    METHOD_COLOR,
    PARTITION_BINS,
    PARTITION_FOLDERS,
    _filter_label,
    _model_colors,
    _save,
    _style,
)


def _kendall_tau_within_components(group: pd.DataFrame) -> float:
    """Pool tau-a pair scores without comparing disconnected Elo scales."""
    score = n_pairs = 0
    for _, component in group.groupby("component", sort=False):
        values = component.dropna(subset=["predictor", "observed"])
        component_pairs = len(values) * (len(values) - 1) // 2
        if component_pairs:
            component_tau = kendall_tau_a(values["predictor"], values["observed"])
            score += component_tau * component_pairs
            n_pairs += component_pairs
    return score / n_pairs if n_pairs else np.nan


def _per_judge_metrics(
    predictions: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Return one component-safe value and model-bootstrap CI per judge."""
    rows = []
    keys = ["partition", "target_bin", "judge", "method"]
    data = predictions[
        predictions["judge"].ne("consensus")
        & predictions["method"].isin(METHOD_COLOR)
    ]
    for values, group in data.groupby(keys, sort=True):
        valid = group.dropna(subset=["prediction", "observed"])
        value = (
            _kendall_tau_within_components(group)
            if metric == "tau_a"
            else (valid["prediction"] - valid["observed"]).abs().mean()
        )
        low, high = _model_bootstrap_interval(group, metric)
        rows.append({
            **dict(zip(keys, values)),
            "value": value,
            "ci_low": low,
            "ci_high": high,
            "n_models": valid["model"].nunique(),
        })
    return pd.DataFrame(rows)


def _model_bootstrap_interval(
    group: pd.DataFrame,
    metric: str,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Resample evaluated models while keeping Elo components separate."""
    random = np.random.default_rng(seed)
    numerator = np.zeros(n_resamples)
    denominator = np.zeros(n_resamples)

    for _, component in group.groupby("component", sort=True):
        valid = component.dropna(subset=["predictor", "prediction", "observed"])
        valid = valid.drop_duplicates("model").sort_values("model")
        count = len(valid)
        if not count:
            continue
        weights = random.multinomial(
            count,
            np.repeat(1 / count, count),
            size=n_resamples,
        )
        if metric == "mae":
            errors = (valid["prediction"] - valid["observed"]).abs().to_numpy()
            numerator += weights @ errors
            denominator += count
            continue

        if count < 2:
            continue
        left, right = np.triu_indices(count, k=1)
        pair_scores = np.sign(
            valid["predictor"].to_numpy()[left]
            - valid["predictor"].to_numpy()[right]
        ) * np.sign(
            valid["observed"].to_numpy()[left]
            - valid["observed"].to_numpy()[right]
        )
        pair_weights = weights[:, left] * weights[:, right]
        numerator += pair_weights @ pair_scores
        denominator += pair_weights.sum(axis=1)

    samples = np.divide(
        numerator,
        denominator,
        out=np.full(n_resamples, np.nan),
        where=denominator > 0,
    )
    samples = samples[np.isfinite(samples)]
    return tuple(np.percentile(samples, [2.5, 97.5])) if len(samples) else (np.nan, np.nan)


def _per_judge_figure(
    predictions: pd.DataFrame,
    nulls: pd.DataFrame,
    *,
    partition: str,
    fit: str,
    metric: str,
    output_directory: Path,
) -> None:
    """Show individual judges and the across-judge mean in separate panels."""
    target_bins = PARTITION_BINS[partition]
    data = predictions[
        (predictions["irt_model"] == "fixed_3pl")
        & (predictions["partition"] == partition)
        & (predictions["fit"] == fit)
    ]
    values = _per_judge_metrics(data, metric).dropna(subset=["value"])
    if values.empty:
        return

    elo_values = values[values["method"] == "elo"]
    judges = sorted(elo_values["judge"].unique())
    judge_colors = _model_colors(pd.Series(judges))
    judge_offsets = dict(zip(judges, np.linspace(-0.10, 0.10, len(judges))))
    x = np.arange(len(target_bins))
    filter_label = _filter_label(data)
    is_rank = metric == "tau_a"

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for judge in judges:
        judge_values = (
            elo_values[elo_values["judge"] == judge]
            .set_index("target_bin")["value"]
            .reindex(target_bins)
        )
        judge_rows = elo_values[elo_values["judge"] == judge].set_index(
            "target_bin"
        ).reindex(target_bins)
        ax.errorbar(
            x + judge_offsets[judge],
            judge_values,
            yerr=np.vstack([
                (judge_values - judge_rows["ci_low"]).clip(lower=0),
                (judge_rows["ci_high"] - judge_values).clip(lower=0),
            ]),
            fmt="-o",
            color=judge_colors[judge],
            alpha=0.58,
            linewidth=0.7,
            elinewidth=0.6,
            markersize=2.5,
            markeredgewidth=0,
            capsize=0,
            zorder=2,
        )

    judge_count = f"{len(judges)} judges"
    ax.set_title(
        f"{fit.upper()} · Elo by judge · {filter_label} · {judge_count}",
        color=INK2,
        fontsize=8,
        pad=3,
    )
    ax.set_xticks(x, target_bins)
    ax.set_xlim(-0.28, len(target_bins) - 0.72)
    ax.set_xlabel("Target item slice", color=INK2)
    _style(ax)

    if is_rank:
        ax.axhline(0, color="#aaa89f", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_ylim(-1.05, 1.05)
        ylabel = "Kendall τ-a"
        kind = "rank"
    else:
        top = max(
            0.30,
            values["ci_high"].max() * 1.08,
        )
        ax.set_ylim(-0.01, top)
        ylabel = "LOMO MAE (lower is better)"
        kind = "score"
    ax.set_ylabel(ylabel, color=INK2)

    handles = [
        plt.Line2D([], [], color=judge_colors[judge], marker="o", linewidth=0.7,
                   markersize=2.5, label=judge)
        for judge in judges
    ]
    handles.append(
        plt.Line2D([], [], color="#222222", marker="o", linewidth=0.7,
                   markersize=2.5, label="judge estimate (95% model-bootstrap CI)")
    )
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(4, len(handles)),
        fontsize=6.5,
    )
    directory = output_directory / PARTITION_FOLDERS[partition] / "per_judge"
    _save(fig, directory, f"{fit}_{kind}_prediction_by_slice")


def render_judge_figures(
    predictions: pd.DataFrame,
    nulls: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Render rank and error summaries with judges as the replication unit."""
    with plt.rc_context(FIGURE_STYLE):
        for partition in PARTITION_BINS:
            for fit in FIT_ORDER:
                for metric in ("tau_a", "mae"):
                    _per_judge_figure(
                        predictions,
                        nulls,
                        partition=partition,
                        fit=fit,
                        metric=metric,
                        output_directory=output_directory,
                    )
