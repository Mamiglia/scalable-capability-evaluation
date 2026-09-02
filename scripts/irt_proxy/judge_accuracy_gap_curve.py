"""Per-judge gap curves: verdicts scored against true and corrected capability."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE = "#2a78d6"


def panel(ax, votes, capability, n_bins, min_verdicts, line_colour, thin_colour):
    v = votes.copy()
    v["left_value"] = v["left_model"].map(capability)
    v["right_value"] = v["right_model"].map(capability)
    v = v.dropna(subset=["left_value", "right_value", "chosen_left"])
    v = v[v["left_value"] != v["right_value"]]
    v["gap"] = (v["left_value"] - v["right_value"]).abs()
    v["correct"] = np.where(
        v["left_value"] > v["right_value"], v["chosen_left"], 1 - v["chosen_left"])
    edges = np.linspace(0, v["gap"].max() + 1e-9, n_bins + 1)
    v["bin"] = pd.cut(v["gap"], edges)
    per_judge = v.groupby(["bin", "judge"], observed=True)["correct"].agg(["mean", "size"])
    per_judge = per_judge[per_judge["size"] >= min_verdicts]

    ax.axhline(0.5, ls="--", lw=0.8, c="grey")
    for _, rows in per_judge.groupby(level="judge"):
        rows = rows.reset_index()
        ax.plot([b.mid for b in rows["bin"]], rows["mean"], lw=0.6,
                c=thin_colour, alpha=0.7, zorder=1)
    centres, means, low, high = [], [], [], []
    for interval, group in per_judge.groupby(level="bin", observed=True):
        values = group["mean"].to_numpy()
        if len(values) < 3:
            continue
        centres.append(interval.mid)
        means.append(values.mean())
        half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        low.append(values.mean() - half)
        high.append(values.mean() + half)
    ax.fill_between(centres, low, high, color=line_colour, alpha=0.2, lw=0, zorder=2)
    ax.plot(centres, means, lw=1.2, c=line_colour, zorder=3)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.02)


def short(name):
    for suffix in ("-0731", "-a3b-instruct", "-a10b", "-a4b-it"):
        name = name.replace(suffix, "")
    return name


def problem_pairs(votes, capability, min_verdicts=10):
    v = votes.copy()
    v["lv"] = v["left_model"].map(capability)
    v["rv"] = v["right_model"].map(capability)
    v = v.dropna(subset=["lv", "rv", "chosen_left"])
    v = v[v["lv"] != v["rv"]]
    v["gap"] = (v["lv"] - v["rv"]).abs()
    v["correct"] = np.where(v["lv"] > v["rv"], v["chosen_left"], 1 - v["chosen_left"])
    v["stronger"] = np.where(v["lv"] > v["rv"], v["left_model"], v["right_model"])
    v["weaker"] = np.where(v["lv"] > v["rv"], v["right_model"], v["left_model"])
    pairs = v.groupby(["stronger", "weaker"]).agg(
        gap=("gap", "first"), accuracy=("correct", "mean"), n=("correct", "size"),
    ).reset_index()
    pairs = pairs[pairs["n"] >= min_verdicts]
    return pairs[(pairs["accuracy"] < 0.45) & (pairs["gap"] >= 0.12)].sort_values("gap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--min-verdicts", type=int, default=20)
    args = parser.parse_args()

    votes = pd.read_parquet(args.input / "votes.parquet")[
        ["judge", "left_model", "right_model", "chosen_left"]
    ]
    responses = pd.read_parquet(args.input / "responses.parquet")
    items = pd.read_csv(args.reference / "items.csv")
    hard_items = set(items[
        items["irt_model"].eq("fixed_3pl") & items["half"].eq("hard")
    ]["item"])
    hard = responses[responses["item"].isin(hard_items)].groupby("model")["correct"].mean()
    predictions = pd.read_csv(args.predictions)
    corrected = predictions[predictions["method"].eq("corrected")]
    corrected = corrected.set_index("model")["prediction"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    panel(axes[0], votes, hard, args.bins, args.min_verdicts, "black", "lightgrey")
    axes[0].set_title("vs true hard accuracy", loc="left", fontsize=7, color="dimgrey")
    axes[0].set_ylabel("Judge accuracy")
    panel(axes[1], votes, corrected, args.bins, args.min_verdicts, BLUE, "#b9d4f2")
    axes[1].set_title("vs corrected capability", loc="left", fontsize=7, color="dimgrey")
    for ax in axes:
        ax.set_xlabel("Capability gap", labelpad=2)

    # name the below-chance pairs under the left axis at their gap position,
    # one row per systematically under-picked (stronger) model
    labels = problem_pairs(votes, hard)
    lines = []
    for stronger, group in labels.groupby("stronger", sort=False):
        entries = ", ".join(
            f"{short(row['weaker'])} ({row['gap']:.2f})"
            for _, row in group.sort_values("gap").iterrows()
        )
        lines.append(f"{short(stronger)} loses to: {entries}")
    caption = "below-chance comparisons (gap in parentheses) — " + "; ".join(lines)
    import textwrap
    wrapped = textwrap.fill(caption, width=130)
    fig.text(0.09, 0.16, wrapped, fontsize=5, color="dimgrey", va="top")
    fig.subplots_adjust(bottom=0.30)
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"gap_curve_judge_accuracy.{suffix}", dpi=300)


if __name__ == "__main__":
    main()
