"""Per-judge predicted-vs-true scatter and original-vs-corrected gap curves."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE = "#2a78d6"
GREY = "#7f7f7f"


def scatter_panels(lomo: pd.DataFrame, corrected: pd.DataFrame, out: Path) -> None:
    original = lomo[
        lomo["irt_model"].eq("fixed_3pl") & lomo["fit"].eq("gpqa")
        & lomo["method"].eq("elo") & lomo["reason"].isna()
    ]
    judges = ["consensus"] + sorted(set(original["judge"]) - {"consensus"})
    panels = [(j, original[original["judge"].eq(j)], GREY) for j in judges]
    panels.append(("corrected (pooled)", corrected, BLUE))

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 3), sharex=True, sharey=True)
    for ax, (label, rows, colour) in zip(axes.flat, panels):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, c="grey", zorder=0)
        ax.plot(rows["observed"], rows["prediction"], "o", ms=3,
                c=colour, mec="none", alpha=0.8)
        short = label.replace("-a4b-it", "").replace("-a3b-instruct", "")
        ax.set_title(f"{short}\n(n={rows['model'].nunique()})",
                     loc="left", fontsize=6.5, color="dimgrey")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.supxlabel("Observed hard accuracy", fontsize=8)
    fig.supylabel("Predicted hard accuracy", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"per_judge_predictions.{suffix}", dpi=300)
    plt.close(fig)


def gap_curves(pairs: pd.DataFrame, out: Path, seed: int = 0) -> None:
    # corrected drawn first so the grey band sits on top where they overlap
    series = [("corrected", BLUE, "corrected"), ("raw_elo", GREY, "original judge Elo")]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.axhline(0.5, ls="--", lw=0.8, c="grey")
    for index, (method, colour, label) in enumerate(series):
        rows = pairs[pairs["method"].eq(method)]
        centres, means, low, high = [], [], [], []
        for centre, bin_rows in rows.groupby("gap_bin_center"):
            correct = bin_rows["ordering_correct"].to_numpy(dtype=float)
            draws = rng.choice(correct, size=(1000, len(correct))).mean(axis=1)
            centres.append(centre)
            means.append(correct.mean())
            low.append(np.percentile(draws, 2.5))
            high.append(np.percentile(draws, 97.5))
        ax.fill_between(centres, low, high, color=colour, alpha=0.22,
                        lw=0, zorder=1 + index)
        ax.plot(centres, means, lw=1, c=colour, label=label, zorder=4)
    ax.set_xlabel("True accuracy gap")
    ax.set_ylabel("Ordering accuracy")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.62))
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"gap_curve_original_vs_corrected.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lomo", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gap-pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lomo = pd.read_csv(args.lomo)
    predictions = pd.read_csv(args.predictions)
    corrected = predictions[predictions["method"].eq("corrected")]
    pairs = pd.read_csv(args.gap_pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    scatter_panels(lomo, corrected, args.out)
    gap_curves(pairs, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
