#!/usr/bin/env python3
"""Features-only baseline: order a pair by superficial transcript features alone.

The target is which member of the pair scores higher, not which one the judge
picked, so the number sits beside a judge's oriented accuracy instead of measuring
mimicry of the judge. Predictions are always held out, under two protocols — folds
grouped by pair, and one fold per model — and judges and baseline are scored on the
same comparisons.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix(".yaml")
# lbfgs default cap is low enough to stop short of convergence on the wider variants.
SOLVER_ITERATIONS = 1000
CI_PERCENTILES = (2.5, 97.5)
COMMON_SET = "all_judges_common"
ALL_SLICE = "all_comparisons"
LARGE_GAP = "large_gap"
LOW_GAP = "low_gap_comparisons"


def build_estimator(name: str, seed: int):
    """Estimator by config name."""
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=SOLVER_ITERATIONS))
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown estimator {name}")


def pair_grouped(comparisons: pd.DataFrame, config: dict, settings: dict):
    """Folds holding out whole pairs, so no pair is in both training and test."""
    groups = comparisons[config["group_column"]].to_numpy()
    for train, test in GroupKFold(n_splits=int(settings["outer_folds"])).split(comparisons, groups=groups):
        shared = set(groups[train]) & set(groups[test])
        if shared:
            raise ValueError(f"{len(shared)} pairs are in both training and test, e.g. {sorted(shared)[:3]}")
        yield train, test


def leave_one_model_out(comparisons: pd.DataFrame, config: dict, settings: dict):
    """One fold per model: test on its comparisons, train on those it is absent from.

    Pair grouping holds out pairs, not models, so both members of a held-out pair are
    still in training through their other pairs. This protocol is what the baseline
    scores on a model it has never seen. A comparison has two models, so it is tested
    twice, once against each.
    """
    sides = comparisons[config["model_columns"]].to_numpy()
    for model in sorted(set(sides.ravel())):
        held = (sides == model).any(axis=1)
        train, test = np.flatnonzero(~held), np.flatnonzero(held)
        if (sides[train] == model).any():
            raise ValueError(f"{model} is in its own training fold")
        yield train, test


PROTOCOLS = {"pair_grouped": pair_grouped, "leave_one_model_out": leave_one_model_out}


def cross_validated(
    features: np.ndarray,
    target: np.ndarray,
    splits,
    inner_groups: np.ndarray,
    estimator,
    grid: dict,
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Held-out predictions over the given splits, as (row index, was it right, fold choices).

    Hyperparameters come from an inner pair-grouped search inside each training fold.
    """
    index, correct, chosen = [], [], []
    for train, test in splits:
        search = GridSearchCV(
            estimator, grid, cv=GroupKFold(n_splits=inner_folds), scoring="accuracy", n_jobs=-1
        )
        search.fit(features[train], target[train], groups=inner_groups[train])
        index.append(test)
        correct.append((search.predict(features[test]) == target[test]).astype(int))
        chosen.append(dict(search.best_params_))
    index = np.concatenate(index)
    if not np.array_equal(np.unique(index), np.arange(len(target))):
        raise ValueError("some comparisons were never held out")
    return index, np.concatenate(correct), chosen


def clustered_accuracy(
    correct: np.ndarray, groups: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Accuracy with a band from resampling whole pairs with replacement.

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
    """Fit every variant under every protocol, then score baseline and judges alike."""
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
    inner_groups = comparisons[group_column].to_numpy()
    target = comparisons["left_higher"].to_numpy()

    held_out, chosen = {}, {}
    for protocol, settings in config["protocols"].items():
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol {protocol}, have {sorted(PROTOCOLS)}")
        for variant, columns in config["variants"].items():
            missing = [column for column in columns if column not in comparisons]
            if missing:
                raise ValueError(f"Variant {variant} wants columns the frame does not have: {missing}")
            features = comparisons[columns].to_numpy(float)
            for estimator_name, grid in config["estimators"].items():
                index, correct, folds = cross_validated(
                    features,
                    target,
                    PROTOCOLS[protocol](comparisons, config, settings),
                    inner_groups,
                    build_estimator(estimator_name, seed),
                    grid,
                    int(config["inner_folds"]),
                )
                label = (protocol, f"{variant}|{estimator_name}")
                held_out[label] = comparisons.iloc[index].reset_index(drop=True).assign(correct=correct)
                chosen[label] = folds

    # Comparisons every judge saw, so the baseline and no judge are scored on
    # different work.
    common = set.intersection(
        *(set(pd.MultiIndex.from_frame(block[keys])) for _, block in frame.groupby(judge_column))
    )

    def rows_for(judge_rows: pd.DataFrame, allowed: set, comparison_set: str) -> list[dict]:
        out = [
            {"comparison_set": comparison_set, "method": "judge", "protocol": "", "name": judge, **row}
            for judge, block in judge_rows.groupby(judge_column)
            for row in score(block, block["judge_correct"], group_column, threshold, resamples, seed)
        ]
        for (protocol, name), predictions in held_out.items():
            kept = predictions[pd.MultiIndex.from_frame(predictions[keys]).isin(allowed)]
            out += [
                {"comparison_set": comparison_set, "method": "baseline", "protocol": protocol, "name": name, **row}
                for row in score(kept, kept["correct"], group_column, threshold, resamples, seed)
            ]
        return out

    frame_keys = pd.MultiIndex.from_frame(frame[keys])
    rows = rows_for(frame[frame_keys.isin(common)], common, COMMON_SET)
    # Same numbers on each judge's own comparisons, in case the common set is not
    # representative of what a judge with thinner coverage was scored on.
    for judge, block in frame.groupby(judge_column):
        rows += rows_for(block, set(pd.MultiIndex.from_frame(block[keys])), f"matched|{judge}")
    return pd.DataFrame(rows), chosen


def figure(results: pd.DataFrame, path: Path, threshold: float, baselines: list[dict]) -> None:
    """Judges and baselines as points on one categorical axis, with clustered bands."""
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
    common = results[results["comparison_set"] == COMMON_SET]
    judges = common[common["method"] == "judge"]
    baseline = common[common["method"] == "baseline"]
    overall = common[common["slice"] == ALL_SLICE]

    judge_names = judges[judges["slice"] == ALL_SLICE].sort_values("accuracy")["name"].tolist()
    # Each configured line at whichever estimator scores higher: the baseline's best shot.
    picked = []
    for spec in baselines:
        block = overall[
            (overall["protocol"] == spec["protocol"])
            & overall["name"].str.startswith(f"{spec['variant']}|")
        ]
        best = block.loc[block["accuracy"].idxmax()]
        picked.append((best["accuracy"], spec["label"], spec["protocol"], best["name"]))
    picked.sort()
    print("  figure shows:", [(label, name) for _, label, _, name in picked])

    gap = 1.0  # blank column separating the judges from the baselines
    judge_x = np.arange(len(judge_names))
    baseline_x = len(judge_names) + gap + np.arange(len(picked))
    panels = [
        (ALL_SLICE, "all comparisons"),
        (LARGE_GAP, f"gap ≥ {threshold:g} ECI"),
        (LOW_GAP, f"gap < {threshold:g} ECI"),
    ]
    styles = {"LLM judge": ("#2a78d6", "o"), "features-only baseline": ("#4a4a4a", "s")}

    def draw(ax, block: pd.DataFrame, positions: np.ndarray, kind: str) -> None:
        colour, marker = styles[kind]
        ax.errorbar(
            positions,
            block["accuracy"],
            yerr=[block["accuracy"] - block["ci_low"], block["ci_high"] - block["accuracy"]],
            fmt=marker, ms=3, lw=1, capsize=0, color=colour, zorder=3,
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5, 2.6), sharey=True)
    limits = []
    for ax, (name, tag) in zip(axes, panels, strict=True):
        judge_block = judges[judges["slice"] == name].set_index("name").reindex(judge_names)
        baseline_block = pd.concat(
            [
                baseline[
                    (baseline["slice"] == name)
                    & (baseline["protocol"] == protocol)
                    & (baseline["name"] == fit)
                ]
                for _, _, protocol, fit in picked
            ]
        )
        draw(ax, judge_block, judge_x, "LLM judge")
        draw(ax, baseline_block, baseline_x, "features-only baseline")
        limits += judge_block["ci_low"].tolist() + judge_block["ci_high"].tolist()
        limits += baseline_block["ci_low"].tolist() + baseline_block["ci_high"].tolist()
        ax.axhline(0.5, lw=0.8, ls="--", color="grey", zorder=0)
        ax.set_xlim(-0.5, baseline_x[-1] + 0.5)
        ax.set_xticks(
            np.concatenate([judge_x, baseline_x]),
            judge_names + [label for _, label, _, _ in picked],
            rotation=90,
        )
        ax.set_title(tag, fontsize=7, color="grey", loc="left", pad=3)
    pad = 0.02
    axes[0].set_ylim(min(limits + [0.5]) - pad, max(limits + [0.5]) + pad)
    axes[0].set_ylabel("pair ordering accuracy")
    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8])
    handles = [
        plt.Line2D([], [], ls="", marker=marker, ms=3, color=colour, label=kind)
        for kind, (colour, marker) in styles.items()
    ] + [
        plt.Line2D([], [], lw=0.8, ls="--", color="grey", label="chance"),
        plt.Line2D(
            [], [], lw=1, color="grey",
            label=f"{CI_PERCENTILES[1] - CI_PERCENTILES[0]:g}% CI, clustered by pair",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
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
    for (protocol, name), folds in chosen.items():
        counts = pd.Series([str(sorted(fold.items())) for fold in folds]).value_counts()
        print(f"  {protocol} {name}: {dict(counts)}")
    common = results[(results["comparison_set"] == COMMON_SET) & (results["slice"] == ALL_SLICE)]
    for _, row in common.sort_values("accuracy").iterrows():
        print(f"  {row['protocol']:<21}{row['name']:<36}{row['accuracy']:.4f} "
              f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
    figure(results, ROOT / config["out_figure"], float(config["low_gap_threshold"]), config["figure_baselines"])


if __name__ == "__main__":
    main()
