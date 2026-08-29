"""Compact Nature-style figures from saved capability-prediction tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("direct_glm", "pairwise_glm", "llm_judge", "q1_accuracy", "training_mean")
LABELS = {
    "direct_glm": "Direct GLM", "pairwise_glm": "Pair GLM", "llm_judge": "LLM judge",
    "q1_accuracy": "Q1 accuracy", "training_mean": "Mean",
}
COLORS = {
    "direct_glm": "#2a78d6", "pairwise_glm": "#1baf7a", "llm_judge": "#eb6834",
    "q1_accuracy": "#4a3aa7", "training_mean": "#777777",
}
QUARTILE_COLORS = {"Q2": "#2a78d6", "Q3": "#eda100", "Q4": "#4a3aa7"}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
        "axes.linewidth": 0.7,
    })


def _ordinary(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _save(fig, output_dir: Path, name: str) -> list[Path]:
    paths = [output_dir / f"{name}.png", output_dir / f"{name}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _select(frame: pd.DataFrame, variant: str, comparator: str | None) -> pd.DataFrame:
    selected = frame[frame.variant == variant] if "variant" in frame else frame
    if "comparator" not in selected:
        return selected
    available = selected.comparator.dropna().drop_duplicates().tolist()
    chosen = comparator or ("pooled_judges" if "pooled_judges" in available else available[0])
    return selected[selected.comparator == chosen]


def _calibration(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    _style()
    quartiles = [q for q in ("Q2", "Q3", "Q4") if q in set(predictions.target_quartile)]
    fig, axes = plt.subplots(1, len(quartiles), figsize=(3.4, 1.75), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    for ax, quartile in zip(axes.flat, quartiles):
        frame = predictions[predictions.target_quartile == quartile]
        for method in METHODS:
            values = frame[frame.method == method]
            if len(values):
                ax.scatter(values.observed_accuracy, values.predicted_accuracy, s=8,
                           color=COLORS[method], label=LABELS[method], alpha=0.8)
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(quartile, loc="left", color="#666666", fontsize=8, pad=2)
        _ordinary(ax)
    axes[0, 0].set_ylabel("Predicted accuracy")
    fig.supxlabel("Observed accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=min(3, len(handles)), frameon=False)
    return _save(fig, output_dir, "hard-accuracy-calibration")


def _mae(comparisons: pd.DataFrame, output_dir: Path) -> list[Path]:
    _style()
    frame = comparisons[comparisons.metric == "mae"]
    methods = [method for method in METHODS if method in set(frame.method)]
    quartiles = [quartile for quartile in ("Q2", "Q3", "Q4") if quartile in set(frame.target_quartile)]
    summary = frame.groupby(["target_quartile", "method"], as_index=False).value.mean()
    fig, ax = plt.subplots(figsize=(2.6, 1.75), layout="constrained")
    x = np.arange(len(methods))
    offsets = np.linspace(-0.16, 0.16, len(quartiles)) if len(quartiles) > 1 else [0]
    for offset, quartile in zip(offsets, quartiles):
        values = summary[summary.target_quartile == quartile].set_index("method").value.reindex(methods)
        ax.plot(x + offset, values, "o", ms=3.5, color=QUARTILE_COLORS[quartile],
                label=quartile, clip_on=False)
    ax.set_xticks(x, [LABELS[method] for method in methods], rotation=90, ha="center")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    upper = max(0.1, float(summary.value.max()) * 1.1)
    ax.set_ylim(0, upper)
    ax.set_yticks([0, upper])
    ax.set_ylabel("MAE")
    ax.set_xlabel("Method")
    _ordinary(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(quartiles), frameon=False)
    return _save(fig, output_dir, "matched-mae")


def plot_results(
    result_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str = "no_llm_annotations",
    comparator: str | None = None,
) -> list[Path]:
    """Read saved CSVs and render calibration and matched-error figures."""
    result_dir, output_dir = Path(result_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _select(pd.read_csv(result_dir / "predictions.csv"), variant, comparator)
    comparisons = _select(pd.read_csv(result_dir / "comparisons.csv"), variant, comparator)
    return _calibration(predictions, output_dir) + _mae(comparisons, output_dir)
