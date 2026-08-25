"""Plot surface and rolling-surprisal capability correlations by benchmark."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BENCHMARKS = ["gpqa", "aime2024", "aime2025", "aime2026", "intercode",
              "swe_bench_mini"]
BENCHMARK_LABELS = ["GPQA", "AIME\n2024", "AIME\n2025", "AIME\n2026",
                    "InterCode", "SWE-mini"]
SCORERS = ("llama", "qwen", "gemma")
SURFACE_LABELS = {
    "log_chars": "Log characters", "n_words": "Words",
    "personal_pronouns_count": "Personal pronouns", "hedging_count": "Hedging",
    "backtracking_count": "Backtracking", "uncertainty_count": "Uncertainty",
    "confidence_count": "Confidence", "questions_count": "Questions",
    "verification_count": "Verification", "connectives_count": "Connectives",
    "contrast_count": "Contrast terms", "numbered_steps_count": "Numbered steps",
    "bullets_count": "Bullets", "code_chars": "Code characters",
    "latex_chars": "LaTeX characters",
}
SURFACE_RATE_LABELS = {
    "log_chars": "Log characters", "n_words": "Words",
    "personal_pronouns_per_1000_words": "Pronouns / 1k words",
    "hedging_per_1000_words": "Hedges / 1k words",
    "backtracking_per_1000_words": "Backtracks / 1k words",
    "uncertainty_per_1000_words": "Uncertainty / 1k words",
    "confidence_per_1000_words": "Confidence markers / 1k words",
    "questions_per_1000_words": "Questions / 1k words",
    "verification_per_1000_words": "Verification / 1k words",
    "connectives_per_1000_words": "Connectives / 1k words",
    "contrast_per_1000_words": "Contrast / 1k words",
    "numbered_steps_per_1000_words": "Steps / 1k words",
    "bullets_per_1000_words": "Bullets / 1k words",
    "code_chars_per_1000_chars": "Code chars / 1k chars",
    "latex_chars_per_1000_chars": "LaTeX chars / 1k chars",
}


def _panel(ax, values, labels, tag, show_x=False):
    image = ax.imshow(values, cmap=CMAP, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xticks(np.arange(len(BENCHMARKS)),
                  labels=BENCHMARK_LABELS if show_x else [""] * len(BENCHMARKS))
    ax.tick_params(length=0, pad=2)
    if show_x:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    ax.text(0, 1.02, tag, transform=ax.transAxes, color="#555555",
            fontsize=7, ha="left", va="bottom")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "–" if np.isnan(value) else f"{value:+.2f}"
            color = "white" if np.isfinite(value) and abs(value) >= 0.34 else "#202020"
            ax.text(column, row, text, ha="center", va="center", fontsize=5.4,
                    color=color)
    # A heatmap needs its frame; this is the documented exception to offset axes.
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    return image


def make_figure(matrix, output_stem):
    rate_adjusted = set(SURFACE_RATE_LABELS) <= set(matrix.feature)
    surface_labels = SURFACE_RATE_LABELS if rate_adjusted else SURFACE_LABELS
    surface_group = "surface-textual-rate" if rate_adjusted else "surface-textual"
    surface = matrix[matrix.group == surface_group].set_index("feature")
    surface = surface.loc[list(surface_labels)]
    surprisal_features = [
        f"surprisal_rolling_{context}_position_adjusted_mean__{scorer}"
        for context in ("short", "long") for scorer in SCORERS]
    surprisal = matrix.set_index("feature").loc[surprisal_features]
    surprisal_labels = [
        f"{context.capitalize()} · {scorer.capitalize()}"
        for context in ("short", "long") for scorer in SCORERS]

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
        "font.family": "Arial",
    })
    figure = plt.figure(figsize=(3.55, 4.6))
    grid = figure.add_gridspec(2, 1, height_ratios=[15, 6], hspace=0.16,
                              left=0.44, right=0.98, top=0.97, bottom=0.24)
    surface_ax = figure.add_subplot(grid[0])
    surprisal_ax = figure.add_subplot(grid[1])
    _panel(surface_ax, surface[BENCHMARKS].to_numpy(),
           list(surface_labels.values()), "Surface features")
    image = _panel(surprisal_ax, surprisal[BENCHMARKS].to_numpy(),
                   surprisal_labels, "Position-adjusted surprisal", show_x=True)
    color_ax = figure.add_axes([0.57, 0.045, 0.37, 0.018])
    colorbar = figure.colorbar(image, cax=color_ax, orientation="horizontal",
                              ticks=[-0.6, 0, 0.6])
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
    colorbar.set_label("Within-item rank correlation", fontsize=7, labelpad=2)
    colorbar.ax.xaxis.set_label_position("top")

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".svg"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(figure)
    if "<text" not in output_stem.with_suffix(".svg").read_text():
        raise ValueError("SVG text was converted to paths")


CMAP = LinearSegmentedColormap.from_list(
    "muted_diverging", ["#2a78d6", "#f7f5f2", "#eb6834"])
CMAP.set_bad("#dedede")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    make_figure(pd.read_parquet(args.matrix), args.output_stem)
