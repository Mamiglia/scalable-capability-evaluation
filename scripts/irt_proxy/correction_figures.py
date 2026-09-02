"""Compact figures for cross-fitted GPQA capability predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from metrics import BOOTSTRAP_RESAMPLES, GAP_BIN_EDGES


FIGURE_STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "svg.fonttype": "none",
}
METHOD_STYLE = {
    "raw_elo": ("Raw Elo", "#777777"),
    "uncorrected": ("Uncorrected", "#eb6834"),
    "corrected": ("Corrected", "#2a78d6"),
}
REFERENCE = "#999999"


def _primary_gpqa(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame["benchmark"].eq("gpqa")].copy()
    if "irt_model" in data and data["irt_model"].eq("fixed_3pl").any():
        data = data[data["irt_model"].eq("fixed_3pl")]
    return data


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 5))
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(direction="out", length=2, width=0.8)


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)


def _capability_predictions(predictions: pd.DataFrame, out_dir: Path) -> None:
    data = _primary_gpqa(predictions)
    methods = ("raw_elo", "corrected")
    fig, axes = plt.subplots(1, 2, figsize=(2.9, 1.7), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        label, color = METHOD_STYLE[method]
        values = data[data["method"].eq(method)].dropna(
            subset=["prediction", "observed"],
        )
        low = (values["prediction"] - values["prediction_p5"]).clip(lower=0)
        high = (values["prediction_p95"] - values["prediction"]).clip(lower=0)
        ax.errorbar(
            values["observed"], values["prediction"],
            yerr=np.vstack([low, high]), fmt="o", color=color,
            linewidth=1, elinewidth=1, markersize=3.5, capsize=0, zorder=2,
        )
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE, linewidth=0.8, zorder=0)
        ax.text(0, 1.05, label, color="#666666", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        _style_axes(ax)
    axes[0].set_ylabel("Predicted accuracy")
    fig.supxlabel("Observed hard accuracy", x=0.56, y=0.17, fontsize=8)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=3.5,
                   color=METHOD_STYLE[method][1], label=METHOD_STYLE[method][0])
            for method in methods
        ] + [Line2D([], [], color=REFERENCE, linestyle="--", linewidth=0.8,
                    label="Ideal")],
        loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False,
        handlelength=1.1, columnspacing=0.8, handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.36, top=0.88, wspace=0.24)
    _save(fig, out_dir, "capability_predictions")


def _gap_curve_table(
    gap_pairs: pd.DataFrame,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Keep all fixed bins, with zero-count estimates left as NaN."""
    rows = []
    bins = GAP_BIN_EDGES[:-1]
    for method in METHOD_STYLE:
        pairs = gap_pairs[gap_pairs["method"].eq(method)].copy()
        models = sorted(set(pairs["model_left"]) | set(pairs["model_right"]))
        model_index = {model: index for index, model in enumerate(models)}
        if models:
            weights = np.random.default_rng(seed).multinomial(
                len(models), np.repeat(1 / len(models), len(models)), size=n_resamples,
            )
            pair_weights = (
                weights[:, pairs["model_left"].map(model_index)]
                * weights[:, pairs["model_right"].map(model_index)]
            )
        else:
            pair_weights = np.empty((n_resamples, 0))
        correct = pairs["ordering_correct"].to_numpy(float)
        for left in bins:
            selected = np.isclose(pairs["gap_bin_left"], left)
            n_pairs = int(selected.sum())
            if n_pairs:
                denominator = pair_weights[:, selected].sum(axis=1)
                samples = np.divide(
                    pair_weights[:, selected] @ correct[selected], denominator,
                    out=np.full(n_resamples, np.nan), where=denominator > 0,
                )
                finite = samples[np.isfinite(samples)]
                low, high = (
                    np.percentile(finite, [2.5, 97.5])
                    if len(finite) else (np.nan, np.nan)
                )
                accuracy = correct[selected].mean()
            else:
                accuracy = low = high = np.nan
            rows.append({
                "method": method,
                "gap_bin_left": left,
                "gap_bin_center": left + (GAP_BIN_EDGES[1] - GAP_BIN_EDGES[0]) / 2,
                "n_pairs": n_pairs,
                "pair_accuracy": accuracy,
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap_models": len(models),
            })
    return pd.DataFrame(rows)


def _gap_curve(gap_pairs: pd.DataFrame, out_dir: Path):
    summary = _gap_curve_table(_primary_gpqa(gap_pairs))
    fig, ax = plt.subplots(figsize=(1.8, 1.6))
    for method, (label, color) in METHOD_STYLE.items():
        values = summary[summary["method"].eq(method)]
        ax.fill_between(
            values["gap_bin_center"], values["ci_low"], values["ci_high"],
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(
            values["gap_bin_center"], values["pair_accuracy"],
            color=color, linewidth=1, label=label,
        )
    ax.axhline(0.5, color=REFERENCE, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("True accuracy gap")
    ax.set_ylabel("Ordering accuracy")
    _style_axes(ax)
    fig.legend(
        *ax.get_legend_handles_labels(), loc="lower center",
        bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
        handlelength=0.7, columnspacing=0.25, handletextpad=0.2,
    )
    fig.subplots_adjust(left=0.27, right=0.95, bottom=0.38, top=0.97)
    _save(fig, out_dir, "gap_curve")
    return fig


def render_correction_figures(
    predictions: pd.DataFrame,
    gap_pairs: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Render the two compact GPQA correction figures without refitting."""
    with plt.rc_context(FIGURE_STYLE):
        _capability_predictions(predictions, Path(out_dir))
        _gap_curve(gap_pairs, Path(out_dir))
