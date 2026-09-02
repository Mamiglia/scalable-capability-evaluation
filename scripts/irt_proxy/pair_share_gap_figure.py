"""Per-pair judge agreement vs GPQA capability gap.

Each point is a model pair from the viewer GPQA collections. x is the gap in
true GPQA accuracy between the two models (the accuracy recorded per evaluated
model in each collection's index.json); y is the share of per-question judge
votes favouring the higher-accuracy model, averaged first within judge then
across judges. Colour marks pairs the judges ordered right (share > 0.5).

DeepSeek V4 Flash judges but is never an evaluated candidate, so it is dropped
from the pairs; the -low/-high/-preview effort variants are kept as candidates.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inputs import canonical_model, is_deepseek, load_votes  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",
})
RIGHT, WRONG = "#2a78d6", "#eb6834"


def gpqa_accuracy(viewer_root: Path, datasets) -> pd.Series:
    """One GPQA-accuracy label per evaluated model from the viewer index.json.

    Read directly (not via load_metadata) so DeepSeek keeps an accuracy label
    when it is included as a candidate.
    """
    rows = []
    for ds in datasets:
        index = json.loads((Path(viewer_root) / ds / "index.json").read_text())
        for raw_model, meta in index.get("models", {}).items():
            if meta.get("accuracy") is not None:
                rows.append((canonical_model(raw_model), meta["accuracy"]))
    return pd.DataFrame(rows, columns=["model", "accuracy"]).groupby("model")["accuracy"].mean()


def per_pair(viewer_root: Path, datasets, accuracy: pd.Series,
             include_deepseek: bool = False) -> pd.DataFrame:
    votes = pd.concat(
        [load_votes(viewer_root, {ds: "gpqa"}, all_candidates=True) for ds in datasets],
        ignore_index=True,
    )
    votes = votes[votes["parse_ok"]].copy()
    if not include_deepseek:
        votes = votes[~votes["major"].map(is_deepseek) & ~votes["minor"].map(is_deepseek)]
    votes["major_acc"] = votes["major"].map(accuracy)
    votes["minor_acc"] = votes["minor"].map(accuracy)
    votes = votes.dropna(subset=["major_acc", "minor_acc", "picked_major"])
    votes = votes[votes["major_acc"] != votes["minor_acc"]]
    # picked_major is 1 when the judge picked the major model, independent of
    # presentation order (chosen_left tracks the physical slot and cancels
    # across the counterbalanced orders).
    votes["fav_strong"] = np.where(
        votes["major_acc"] > votes["minor_acc"],
        votes["picked_major"], 1 - votes["picked_major"])
    # Drop judges that favour the stronger model below chance pooled over every
    # comparison: they carry no capability signal and only add noise to the mean.
    judge_mean = votes.groupby("judge")["fav_strong"].mean()
    below = sorted(judge_mean[judge_mean < 0.5].index)
    if below:
        print("dropped below-chance judges:", below)
    votes = votes[votes["judge"].isin(judge_mean[judge_mean >= 0.5].index)]
    votes["gap"] = (votes["major_acc"] - votes["minor_acc"]).abs()
    votes["pair"] = ["|".join(sorted(p)) for p in zip(votes["major"], votes["minor"])]
    by_judge = votes.groupby(["pair", "judge"]).agg(
        share=("fav_strong", "mean"), gap=("gap", "first"))
    pairs = by_judge.groupby("pair").agg(
        gap=("gap", "first"), share=("share", "mean"), n_judges=("share", "size"))
    return pairs, votes[["gap", "fav_strong", "judge"]].reset_index(drop=True)


def logistic_grid(gap, fav_strong, xmax):
    """P(favour stronger) vs gap from a near-unregularised logistic fit."""
    model = LogisticRegression(C=1e6)
    model.fit(np.asarray(gap).reshape(-1, 1), np.asarray(fav_strong))
    grid = np.linspace(0, xmax, 200)
    return grid, model.predict_proba(grid.reshape(-1, 1))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path,
                        default=HERE.parent.parent / "interactive_dashboards/transcript_viewer/data")
    parser.add_argument("--datasets", nargs="+",
                        default=["gpqa-pilot-rd2", "data-expansion-gpqa"])
    parser.add_argument("--out", type=Path, default=HERE.parent.parent / "docs/pair_share_gap.pdf")
    parser.add_argument("--include-deepseek", action="store_true",
                        help="keep DeepSeek V4 Flash as an evaluated candidate")
    parser.add_argument("--fit", choices=["none", "averaged", "per-judge", "both"],
                        default="none", help="logistic fit lines to overlay")
    args = parser.parse_args()

    accuracy = gpqa_accuracy(args.viewer, args.datasets)
    pairs, votes = per_pair(args.viewer, args.datasets, accuracy,
                            include_deepseek=args.include_deepseek)
    right = pairs[pairs["share"] > 0.5]
    wrong = pairs[pairs["share"] <= 0.5]
    print(f"{len(pairs)} pairs, {len(wrong)} ranked wrong")
    clear = pairs[pairs["gap"] > 0.05]
    clear_wrong = int((clear["share"] <= 0.5).sum())
    print(f"gap > 5%: {len(clear)} pairs, {clear_wrong} ranked wrong "
          f"({1 - clear_wrong / len(clear):.1%} correct)")

    gap_max = pairs["gap"].max()
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    # Trim the visible spines to the data range so the axis lines stop at the
    # end ticks instead of running out past the padded limits.
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, gap_max)
    ax.axhline(0.5, ls="--", lw=0.8, c="grey")
    ax.axvline(0.05, ls="--", lw=0.8, c="grey")
    ax.plot(right["gap"], right["share"], "o", ms=4, c=RIGHT, mec="none",
            clip_on=False, label="pair ranked correctly")
    ax.plot(wrong["gap"], wrong["share"], "x", ms=4, mew=1.1, c=WRONG,
            clip_on=False, label="pair ranked wrong")
    if args.fit != "none":
        curves = {}
        for judge, rows in votes.groupby("judge"):
            grid, curves[judge] = logistic_grid(rows["gap"], rows["fav_strong"], gap_max)
        if args.fit in ("per-judge", "both"):
            for i, prob in enumerate(curves.values()):
                ax.plot(grid, prob, c="0.55", lw=0.6, alpha=0.5,
                        label="per-judge fit" if i == 0 else None)
        if args.fit in ("averaged", "both"):
            # mean across judges, matching how the scatter averages the shares.
            ax.plot(grid, np.mean(list(curves.values()), axis=0), c="black",
                    lw=1.2, label="averaged fit")
    ax.set_xlabel("Δ GPQA accuracy")
    ax.set_ylabel("Share favouring stronger model")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.015, gap_max + 0.03)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
