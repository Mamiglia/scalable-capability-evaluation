"""Pair-level view: raw verdict accuracy per pair, and corrected ordering."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gap-pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    votes = pd.read_parquet(args.input / "votes.parquet")
    responses = pd.read_parquet(args.input / "responses.parquet")
    items = pd.read_csv(args.reference / "items.csv")
    hard_items = set(items[
        items["irt_model"].eq("fixed_3pl") & items["half"].eq("hard")
    ]["item"])
    hard = responses[responses["item"].isin(hard_items)].groupby("model")["correct"].mean()

    v = votes[["left_model", "right_model", "chosen_left"]].copy()
    v["left_hard"] = v["left_model"].map(hard)
    v["right_hard"] = v["right_model"].map(hard)
    v = v.dropna()
    v = v[v["left_hard"] != v["right_hard"]]
    v["gap"] = (v["left_hard"] - v["right_hard"]).abs()
    v["correct"] = np.where(
        v["left_hard"] > v["right_hard"], v["chosen_left"], 1 - v["chosen_left"])
    v["pair"] = [
        "|".join(sorted(p)) for p in zip(v["left_model"], v["right_model"])
    ]
    pairs = v.groupby("pair").agg(
        gap=("gap", "first"), accuracy=("correct", "mean"), n=("correct", "size"),
    )
    pairs = pairs[pairs["n"] >= 10]

    gp = pd.read_csv(args.gap_pairs)
    corrected = gp[gp["method"].eq("corrected")].copy()
    oriented = np.sign(corrected["true_gap"]) * corrected["estimated_gap"]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
    ax = axes[0]
    ax.axhline(0.5, ls="--", lw=0.8, c="grey")
    ax.plot(pairs["gap"], pairs["accuracy"], "o", ms=2, c="dimgrey",
            mec="none", alpha=0.4)
    ax.set_xlabel("True hard-accuracy gap")
    ax.set_ylabel("Judge accuracy on the pair")
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(f"raw verdicts ({len(pairs)} judged pairs)",
                 loc="left", fontsize=7, color="dimgrey")

    ax = axes[1]
    ax.axhline(0, ls="--", lw=0.8, c="grey")
    ax.plot(corrected["absolute_true_gap"], oriented, "o", ms=2, c=BLUE,
            mec="none", alpha=0.4)
    wrong = int((oriented < 0).sum())
    ax.set_xlabel("True hard-accuracy gap")
    ax.set_ylabel("Corrected gap, aligned with truth")
    ax.set_title(f"corrected ({wrong}/{len(corrected)} misordered)",
                 loc="left", fontsize=7, color="dimgrey")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"gap_pairs_raw_vs_corrected.{suffix}", dpi=300)


if __name__ == "__main__":
    main()
