#!/usr/bin/env python3
"""Features-only baseline: order a pair by superficial transcript features alone.

The target is which member of the pair scores higher, not which one the judge
picked, so the number sits beside a judge's oriented accuracy instead of
measuring mimicry of the judge. Predictions are out-of-fold with folds grouped by
pair, and judges and baseline are scored on the same comparisons.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix(".yaml")
# lbfgs default cap is low enough to stop short of convergence on the wider variant.
SOLVER_ITERATIONS = 1000
CI_PERCENTILES = (2.5, 97.5)  # 95% band
ALL_SLICE = "all_comparisons"
LARGE_GAP = "large_gap"
LOW_GAP = "low_gap_comparisons"


def build_estimator(name: str, seed: int):
    """Estimator by config name."""
    if name == "logistic":
        return LogisticRegression(max_iter=SOLVER_ITERATIONS)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown estimator {name}")


def out_of_fold(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    estimator,
    grid: dict,
    outer_folds: int,
    inner_folds: int,
) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold predictions, folds grouped so no pair spans training and test.

    Hyperparameters come from an inner grouped search inside each training fold.
    """
    prediction = np.full(len(target), -1, dtype=int)
    chosen = []
    for train, test in GroupKFold(n_splits=outer_folds).split(features, target, groups):
        shared = set(groups[train]) & set(groups[test])
        if shared:
            raise ValueError(f"{len(shared)} pairs are in both training and test, e.g. {sorted(shared)[:3]}")
        search = GridSearchCV(
            estimator, grid, cv=GroupKFold(n_splits=inner_folds), scoring="accuracy", n_jobs=-1
        )
        search.fit(features[train], target[train], groups=groups[train])
        prediction[test] = search.predict(features[test])
        chosen.append(dict(search.best_params_))
    unscored = int((prediction < 0).sum())
    if unscored:
        raise ValueError(f"{unscored} comparisons were never in a test fold")
    return prediction, chosen


def clustered_accuracy(
    correct: np.ndarray, groups: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Accuracy with a 95% band from resampling whole pairs with replacement.

    Groups are sorted before resampling, so every quantity scored on the same rows
    shares the same resamples and the comparisons between them stay paired.
    """
    codes, uniques = pd.factorize(pd.Series(groups), sort=True)
    hits = np.bincount(codes, weights=correct, minlength=len(uniques))
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    draws = np.random.default_rng(seed).integers(len(uniques), size=(resamples, len(uniques)))
    band = np.percentile(hits[draws].sum(axis=1) / counts[draws].sum(axis=1), CI_PERCENTILES)
    return float(hits.sum() / counts.sum()), float(band[0]), float(band[1])


def slices(gap: pd.Series, threshold: float) -> dict[str, pd.Series]:
    """Row masks for the whole set and the two capability-gap slices."""
    return {
        ALL_SLICE: pd.Series(True, index=gap.index),
        LARGE_GAP: gap >= threshold,
        LOW_GAP: gap < threshold,
    }


def score(
    frame: pd.DataFrame, correct: pd.Series, group_column: str, threshold: float, resamples: int, seed: int
) -> list[dict]:
    """One row per slice: accuracy, band, and how much the slice holds."""
    rows = []
    for name, mask in slices(frame["capability_gap"], threshold).items():
        if not mask.any():
            continue
        accuracy, low, high = clustered_accuracy(
            correct[mask].to_numpy(float), frame.loc[mask, group_column].to_numpy(), resamples, seed
        )
        rows.append(
            {
                "slice": name,
                "n_rows": int(mask.sum()),
                "n_pairs": int(frame.loc[mask, group_column].nunique()),
                "accuracy": accuracy,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def evaluate(config: dict, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit every variant, then score baseline and judges on the same comparisons."""
    keys = list(config["comparison_keys"])
    group_column = config["group_column"]
    judge_column = config["judge_column"]
    threshold = float(config["low_gap_threshold"])
    resamples = int(config["bootstrap_resamples"])
    seed = int(config["random_seed"])

    difference = frame[config["target_column"]]
    if (difference == 0).any():
        raise ValueError(f"{int((difference == 0).sum())} comparisons have no higher-capability member")
    frame = frame.assign(left_higher=(difference > 0).astype(int), capability_gap=difference.abs())
    # The judge is right when it picked the member with the higher capability score.
    frame["judge_correct"] = (frame[config["judge_choice_column"]] == frame["left_higher"]).astype(int)

    # One row per comparison: the features do not depend on which judge saw it, so
    # fitting on the judge-level frame would just weight comparisons by coverage.
    comparisons = frame.drop_duplicates(keys).reset_index(drop=True)
    groups = comparisons[group_column].to_numpy()
    target = comparisons["left_higher"].to_numpy()

    labels, chosen = [], {}
    for variant, columns in config["variants"].items():
        missing = [column for column in columns if column not in comparisons]
        if missing:
            raise ValueError(f"Variant {variant} wants columns the frame does not have: {missing}")
        features = comparisons[columns].to_numpy(float)
        for estimator_name, grid in config["estimators"].items():
            prediction, folds = out_of_fold(
                features,
                target,
                groups,
                build_estimator(estimator_name, seed),
                grid,
                int(config["outer_folds"]),
                int(config["inner_folds"]),
            )
            label = f"{variant}|{estimator_name}"
            comparisons[label] = (prediction == target).astype(int)
            labels.append(label)
            chosen[label] = folds

    # Comparisons every judge saw, so the baseline and no judge are scored on
    # different work.
    frame_keys = pd.MultiIndex.from_frame(frame[keys])
    comparison_keys = pd.MultiIndex.from_frame(comparisons[keys])
    common = set.intersection(
        *(set(pd.MultiIndex.from_frame(block[keys])) for _, block in frame.groupby(judge_column))
    )

    def rows_for(judge_rows: pd.DataFrame, baseline_rows: pd.DataFrame, comparison_set: str) -> list[dict]:
        out = [
            {"comparison_set": comparison_set, "method": "judge", "name": judge, **row}
            for judge, block in judge_rows.groupby(judge_column)
            for row in score(block, block["judge_correct"], group_column, threshold, resamples, seed)
        ]
        out += [
            {"comparison_set": comparison_set, "method": "baseline", "name": label, **row}
            for label in labels
            for row in score(baseline_rows, baseline_rows[label], group_column, threshold, resamples, seed)
        ]
        return out

    rows = rows_for(
        frame[frame_keys.isin(common)], comparisons[comparison_keys.isin(common)], "all_judges_common"
    )
    # Same numbers on each judge's own comparisons, in case the common set is not
    # representative of what a judge with thinner coverage was scored on.
    for judge, block in frame.groupby(judge_column):
        own = set(pd.MultiIndex.from_frame(block[keys]))
        rows += rows_for(block, comparisons[comparison_keys.isin(own)], f"matched|{judge}")
    return pd.DataFrame(rows), chosen


def figure(results: pd.DataFrame, path: Path, threshold: float) -> None:
    """Judges as dots with clustered bands; each baseline variant as a line and band."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    common = results[results["comparison_set"] == "all_judges_common"]
    judges = common[common["method"] == "judge"]
    baseline = common[common["method"] == "baseline"]
    order = judges[judges["slice"] == ALL_SLICE].sort_values("accuracy")["name"].tolist()
    # Show each variant at its stronger estimator: the baseline gets its best shot.
    variants = {}
    for variant in sorted({name.split("|")[0] for name in baseline["name"]}):
        block = baseline[baseline["name"].str.startswith(f"{variant}|")]
        overall = block[block["slice"] == ALL_SLICE]
        variants[variant] = overall.loc[overall["accuracy"].idxmax(), "name"]
    print("  figure shows:", variants)

    panels = [(ALL_SLICE, "all comparisons"), (LARGE_GAP, f"gap ≥ {threshold:g} ECI"),
              (LOW_GAP, f"gap < {threshold:g} ECI")]
    styles = {"length_only": ("#8c8c8c", "--"), "length_and_style": ("#2a2a2a", "-")}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2, 2.3), sharey=True)
    for ax, (name, tag) in zip(axes, panels, strict=True):
        judge_slice = judges[judges["slice"] == name].set_index("name").reindex(order)
        positions = np.arange(len(order))
        ax.errorbar(
            positions,
            judge_slice["accuracy"],
            yerr=[judge_slice["accuracy"] - judge_slice["ci_low"], judge_slice["ci_high"] - judge_slice["accuracy"]],
            fmt="o", ms=3, lw=1, capsize=0, color="#2a78d6", zorder=3,
        )
        for variant, label in variants.items():
            row = baseline[(baseline["slice"] == name) & (baseline["name"] == label)].iloc[0]
            colour, style = styles[variant]
            ax.axhline(row["accuracy"], lw=1, ls=style, color=colour, zorder=2)
            ax.fill_between([-0.5, len(order) - 0.5], row["ci_low"], row["ci_high"],
                            color=colour, alpha=0.18, lw=0, zorder=1)
        ax.axhline(0.5, lw=0.8, ls=":", color="grey", zorder=0)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_xticks(positions, order, rotation=90)
        ax.set_title(tag, fontsize=7, color="grey", loc="left", pad=3)
    axes[0].set_ylabel("pair ordering accuracy")
    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8])
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=3, color="#2a78d6", label="LLM judge"),
        plt.Line2D([], [], lw=1, ls="--", color="#8c8c8c", label="length only"),
        plt.Line2D([], [], lw=1, ls="-", color="#2a2a2a", label="length + style"),
        plt.Line2D([], [], lw=0.8, ls=":", color="grey", label="chance"),
        plt.Line2D([], [], lw=4, alpha=0.35, color="grey", label="95% CI, clustered by pair"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"  {path.with_suffix(suffix)}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    frame = pd.read_csv(ROOT / config["frame"])
    print(f"Read {len(frame):,} rows from {config['frame']}")
    results, chosen = evaluate(config, frame)

    out = ROOT / config["out_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Wrote {len(results):,} rows to {out}")
    for label, folds in chosen.items():
        print(f"  {label} per fold: {folds}")
    common = results[(results["comparison_set"] == "all_judges_common") & (results["slice"] == ALL_SLICE)]
    for _, row in common.sort_values("accuracy").iterrows():
        print(f"  {row['name']:<34} {row['accuracy']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
    figure(results, ROOT / config["out_figure"], float(config["low_gap_threshold"]))


if __name__ == "__main__":
    main()
