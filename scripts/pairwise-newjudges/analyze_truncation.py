"""Digest and compare the paired transcript-truncation campaign."""

import argparse
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log
from scipy.stats import t

from judge.dataset import truncate_pair
from judge.util import extract_transcript, first_correct_by_id
from mc_cost.gpqa import model_family


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hibayes"))
from style_features import feature_rates  # noqa: E402


plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
})


FRACTIONS = (0.1, 0.5, 0.9, 1.0)
CONDITION = re.compile(r"trunc_(first|end)_(10|50|90)$")
CONDITION_ORDER = (
    "first_10", "first_50", "first_90",
    "end_10", "end_50", "end_90", "original",
)
STYLE_CATEGORIES = (
    "personal_pronouns", "hedging", "backtracking",
    "uncertainty", "confidence", "questions",
)
SCALED_FEATURES = (
    "capability_gap", "total_length", "length_difference_stronger",
    *[f"{category}_difference_stronger" for category in STYLE_CATEGORIES],
)
PREDICTORS = (
    "truth_side", "truth_side_x_total_length", "total_length_sd",
    "capability_gap_sd", "length_difference_stronger_sd", "same_model_family",
    *[f"{category}_difference_stronger_sd" for category in STYLE_CATEGORIES],
)
GAP_CONDITIONS = ("original", "first_90", "end_90")
CONDITION_LABELS = {
    "original": "Original", "first_90": "First 90%", "end_90": "End 90%",
}


def collect_logs(root: Path, read_log=read_eval_log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return verdict rows and one success row per observed pair directory."""
    rows, runs = [], []
    for condition_dir in sorted(p for p in Path(root).iterdir()
                                if p.is_dir() and CONDITION.fullmatch(p.name)):
        direction, percent = CONDITION.fullmatch(condition_dir.name).groups()
        pair_dirs = sorted({path.parent for path in condition_dir.rglob("*.eval")})
        for pair_dir in pair_dirs:
            successes = []
            for path in sorted(pair_dir.glob("*.eval")):
                try:
                    if read_log(path, header_only=True).status == "success":
                        successes.append(path)
                except Exception:
                    continue
            path = successes[-1] if successes else None
            runs.append({
                "condition": condition_dir.name,
                "judge": pair_dir.parent.name,
                "pair": pair_dir.name,
                "success": path is not None,
                "log_file": str(path or ""),
            })
            if path is None:
                continue
            log = read_log(path)
            metadata = log.eval.metadata or {}
            fraction = float(metadata.get("truncate_frac"))
            keep = metadata.get("truncate_keep", "first")
            if round(100 * fraction) != int(percent) or keep != ("last" if direction == "end" else "first"):
                raise ValueError(f"condition metadata mismatch in {path}")
            for sample in log.samples or []:
                item, target = str(sample.id).rsplit("_", 1)
                score = next(iter((sample.scores or {}).values()), None)
                picked = getattr(score, "answer", None)
                parse_ok = picked in ("1", "2")
                rows.append({
                    "condition": condition_dir.name,
                    "direction": direction,
                    "fraction": fraction,
                    "judge": pair_dir.parent.name,
                    "pair": pair_dir.name,
                    "item": item,
                    "order": int(target),
                    "picked": picked if parse_ok else "",
                    "picked_major": picked == target if parse_ok else np.nan,
                    "parse_ok": parse_ok,
                    "log_file": str(path),
                })
    row_frame = pd.DataFrame(rows).sort_values(
        ["condition", "judge", "pair", "item", "order"], ignore_index=True)
    run_frame = pd.DataFrame(runs).sort_values(
        ["condition", "judge", "pair"], ignore_index=True)
    return row_frame, run_frame


def baseline_rows(verdicts: pd.DataFrame, sample_ids: dict) -> pd.DataFrame:
    """Return the selected original-transcript baseline as both directions' 100%."""
    rows = verdicts[verdicts.condition == "a_baseline"].copy()
    rows["pair"] = rows.major.astype(str) + "-vs-" + rows.minor.astype(str)
    rows["item"] = rows.item_id.astype(str)
    selected = {(pair, str(item)) for pair, items in sample_ids.items() for item in items}
    rows = rows[[
        (pair, item) in selected for pair, item in zip(rows.pair, rows.item)
    ]]
    rows["fraction"] = 1.0
    columns = ["condition", "judge", "pair", "item", "order", "picked",
               "picked_major", "parse_ok", "fraction"]
    return pd.concat([
        rows[columns].assign(direction="first"),
        rows[columns].assign(direction="end"),
    ], ignore_index=True)


def _counterbalanced_cells(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["judge", "direction", "fraction", "pair", "item"]
    parsed = rows[rows.parse_ok & rows.picked_major.notna()]
    cells = (parsed.groupby(keys, as_index=False)
                   .agg(score=("picked_major", "mean"),
                        rows=("order", "size"), orders=("order", "nunique")))
    return cells[(cells.rows == 2) & (cells.orders == 2)].drop(columns=["rows", "orders"])


def experiment_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """One shared counterbalanced cell set across judges and all seven conditions."""
    cells = _counterbalanced_cells(rows).rename(columns={"score": "observed"})
    cells["condition"] = np.where(
        cells.fraction.eq(1), "original",
        cells.direction + "_" + (100 * cells.fraction).round().astype(int).astype(str),
    )
    cells = cells.drop_duplicates(["judge", "condition", "pair", "item"])
    ready = cells.groupby("judge").condition.nunique()
    cells = cells[cells.judge.isin(ready[ready == len(CONDITION_ORDER)].index)]
    groups = list(cells.groupby(["judge", "condition"]))
    common = set.intersection(*[
        set(zip(group.pair, group.item)) for _, group in groups
    ]) if groups else set()
    cells = cells[[key in common for key in zip(cells.pair, cells.item)]]
    order = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    cells["condition_order"] = cells.condition.map(order)
    return cells.sort_values(
        ["judge", "condition_order", "pair", "item"], ignore_index=True
    ).drop(columns="condition_order")


def overall_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-judge accuracy and mean intervals with judges as replication units."""
    by_judge = (cells.groupby(["judge", "condition"], as_index=False)
                     .agg(accuracy=("observed", "mean"), n_cells=("item", "size")))
    rows = []
    for condition in CONDITION_ORDER:
        values = by_judge[by_judge.condition == condition].accuracy
        mean = values.mean()
        if len(values) < 3:
            low = high = np.nan
        else:
            half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            low, high = mean - half, mean + half
        rows.append({
            "condition": condition, "accuracy": mean, "ci_low": low, "ci_high": high,
            "n_judges": values.size,
            "n_cells": int(by_judge[by_judge.condition == condition].n_cells.min())
            if len(values) else 0,
        })
    return by_judge, pd.DataFrame(rows)


def build_glm_frame(cells: pd.DataFrame, transcripts: dict,
                    comparisons: dict[str, tuple[str, str]], capabilities: dict,
                    reference: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    """Recompute frozen-GLM predictors from the exact text shown per pair."""
    rows = []
    for cell in cells.drop_duplicates(["judge", "condition", "pair", "item"]).itertuples():
        major, minor = comparisons[cell.pair]
        major_text = transcripts[(major, cell.item)]
        minor_text = transcripts[(minor, cell.item)]
        if cell.condition == "original":
            shown = major_text, minor_text
        elif cell.condition == "first_90":
            shown = truncate_pair(major_text, minor_text, 0.9, keep="first")
        elif cell.condition == "end_90":
            shown = truncate_pair(major_text, minor_text, 0.9, keep="last")
        else:
            raise ValueError(f"No GLM transcript treatment for {cell.condition}")
        for order in (1, 2):
            left_model, right_model = (major, minor) if order == 1 else (minor, major)
            left, right = shown if order == 1 else shown[::-1]
            truth_side = 1.0 if order == 1 else -1.0
            left_rates, right_rates = feature_rates(left, lexicon), feature_rates(right, lexicon)
            row = {
                "judge": cell.judge, "condition": cell.condition, "pair": cell.pair,
                "item": cell.item, "order": order, "truth_side": truth_side,
                "left_chars": len(left), "right_chars": len(right),
                "capability_gap": capabilities[major] - capabilities[minor],
                "total_length": np.log1p(len(left) + len(right)),
                "length_difference_stronger": (
                    np.log1p(len(left)) - np.log1p(len(right))
                ) * truth_side,
                "same_model_family": int(
                    left_model != right_model
                    and model_family(left_model) == model_family(right_model)
                ),
            }
            for category in STYLE_CATEGORIES:
                row[f"{category}_difference_stronger"] = (
                    left_rates[category] - right_rates[category]
                ) * truth_side
            rows.append(row)
    frame = pd.DataFrame(rows)
    for feature in SCALED_FEATURES:
        frame[f"{feature}_sd"] = (
            frame[feature] - reference[feature].mean()
        ) / reference[feature].std(ddof=1)
    frame["truth_side_x_total_length"] = frame.truth_side * frame.total_length_sd
    if not np.isfinite(frame[list(PREDICTORS)].to_numpy()).all():
        raise ValueError("Counterfactual GLM predictors are not finite")
    return frame


def load_transcripts(config: dict, root: Path = ROOT,
                     read_log=read_eval_log) -> dict[tuple[str, str], str]:
    """Replay the exact first-correct, final-text transcript selection."""
    required: dict[str, set[str]] = {}
    for major, minor in config["comparisons"]:
        items = config["sample_ids"][f"{major}-vs-{minor}"]
        required.setdefault(major, set()).update(map(str, items))
        required.setdefault(minor, set()).update(map(str, items))
    transcripts = {}
    for model, items in sorted(required.items()):
        grouped = first_correct_by_id(read_log(root / config["transcripts"][model]))
        missing = items - set(map(str, grouped))
        if missing:
            raise ValueError(f"Missing {len(missing)} selected transcripts for {model}")
        for item in items:
            sample = grouped[item][0]
            transcripts[(model, item)] = extract_transcript(
                sample, include_reasoning=False
            )[:config["max_length"]]
    return transcripts


def collapse_predictions(cells: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Average the frozen GLM's two presentation-order predictions per cell."""
    keys = ["judge", "condition", "pair", "item"]
    predictions = (frame.groupby(keys, as_index=False)
                         .agg(predicted=("predicted", "mean"),
                              capability_gap=("capability_gap", "first")))
    return cells.merge(predictions, on=keys, how="inner", validate="one_to_one")


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def make_overall_figure(cells: pd.DataFrame):
    """Overall first/end retention curves with judge-level uncertainty."""
    by_judge, summary = overall_tables(cells)
    fig, ax = plt.subplots(figsize=(2.2, 2.4))
    colors = {"first": "#2a78d6", "end": "#eb6834"}
    fractions = [10, 50, 90, 100]
    for direction in ("first", "end"):
        conditions = [f"{direction}_{fraction}" for fraction in fractions[:-1]] + ["original"]
        for judge in sorted(by_judge.judge.unique()):
            values = (by_judge[by_judge.judge == judge].set_index("condition")
                      .reindex(conditions).accuracy)
            ax.plot(fractions, values, color=colors[direction], lw=0.7, marker="o",
                    ms=2.2, alpha=0.28, zorder=2)
        means = summary.set_index("condition").reindex(conditions)
        yerr = np.vstack([means.accuracy - means.ci_low,
                          means.ci_high - means.accuracy])
        yerr = yerr if np.isfinite(yerr).all() else None
        ax.errorbar(
            fractions, means.accuracy, yerr=yerr, color=colors[direction],
            lw=1, marker="o", ms=3.2, capsize=0, label=f"{direction.title()} retained",
            zorder=4,
        )
    ax.axhline(0.5, color="#888888", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.set(xlim=(10, 100), ylim=(0, 1), xticks=fractions, yticks=[0, 0.5, 1],
           xlabel="Transcript retained (%)", ylabel="Judge accuracy")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.97, bottom=0.30)
    return fig


def gap_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ECI-gap tertiles with judges retained as the replication units."""
    cells = cells[cells.condition.isin(GAP_CONDITIONS)].copy()
    pair_gaps = cells.groupby("pair").capability_gap.first().sort_values()
    bins = pd.qcut(pair_gaps.rank(method="first"), 3, labels=False)
    cells["gap_bin"] = cells.pair.map(bins)
    by_judge = (cells.groupby(["judge", "condition", "gap_bin"], as_index=False)
                     .agg(gap_mean=("capability_gap", "mean"),
                          observed=("observed", "mean"),
                          predicted=("predicted", "mean")))
    rows = []
    for (condition, gap_bin), group in by_judge.groupby(["condition", "gap_bin"]):
        row = {"condition": condition, "gap_bin": int(gap_bin),
               "gap_mean": group.gap_mean.mean(), "n_judges": len(group)}
        for value in ("observed", "predicted"):
            mean = group[value].mean()
            if len(group) < 3:
                low = high = np.nan
            else:
                half = (t.ppf(0.975, len(group) - 1) * group[value].std(ddof=1)
                        / np.sqrt(len(group)))
                low, high = mean - half, mean + half
            row |= {value: mean, f"{value}_ci_low": low, f"{value}_ci_high": high}
        rows.append(row)
    return by_judge, pd.DataFrame(rows)


def make_gap_figure(cells: pd.DataFrame):
    """Observed and frozen-GLM gap curves for original and 90% treatments."""
    by_judge, summary = gap_tables(cells)
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.4), sharex=True, sharey=True)
    palette = ("#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7")
    judges = sorted(by_judge.judge.unique())
    judge_colors = dict(zip(judges, palette, strict=False))
    maximum = max(10.0, float(summary.gap_mean.max()))
    x_max = np.ceil(maximum / 10) * 10
    for ax, condition in zip(axes, GAP_CONDITIONS, strict=True):
        for judge in judges:
            rows = by_judge[(by_judge.judge == judge) &
                            (by_judge.condition == condition)].sort_values("gap_bin")
            ax.plot(rows.gap_mean, rows.observed, color=judge_colors[judge], marker="o",
                    ms=2.2, lw=0.7, alpha=0.35, zorder=2)
            ax.plot(rows.gap_mean, rows.predicted, color=judge_colors[judge], marker="o",
                    ms=2.2, lw=0.7, ls="--", alpha=0.35, zorder=2)
        rows = summary[summary.condition == condition].sort_values("gap_bin")
        for value, color, linestyle in (
            ("observed", "#111111", "-"), ("predicted", "#666666", "--")
        ):
            low, high = f"{value}_ci_low", f"{value}_ci_high"
            yerr = np.vstack([rows[value] - rows[low], rows[high] - rows[value]])
            yerr = yerr if np.isfinite(yerr).all() else None
            ax.errorbar(rows.gap_mean, rows[value], yerr=yerr, color=color, marker="o",
                        ms=3, lw=1, capsize=0, linestyle=linestyle, zorder=4)
        ax.axhline(0.5, color="#888888", lw=0.8, ls=":", zorder=1)
        ax.text(0.03, 0.96, CONDITION_LABELS[condition], transform=ax.transAxes,
                color="#666666", va="top", fontsize=8)
        ax.set(xlim=(0, x_max), ylim=(0, 1), xticks=[0, x_max / 2, x_max],
               yticks=[0, 0.5, 1], xlabel="Capability gap (ECI)")
        _style_axes(ax)
    axes[0].set_ylabel("Judge accuracy")
    handles = [
        plt.Line2D([], [], color="#111111", marker="o", ms=3, lw=1,
                   label="Observed mean"),
        plt.Line2D([], [], color="#666666", marker="o", ms=3, lw=1, ls="--",
                   label="Frozen GLM"),
        *[plt.Line2D([], [], color=judge_colors[judge], lw=1, label=judge)
          for judge in judges],
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.7, columnspacing=0.8)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.97, bottom=0.32, wspace=0.24)
    return fig


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    values = pd.Series(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def matched_estimates(rows: pd.DataFrame) -> pd.DataFrame:
    """Pair-clustered curves using cells shared by all four fractions."""
    cells = _counterbalanced_cells(rows)
    output = []
    for (judge, direction), group in cells.groupby(["judge", "direction"]):
        if set(group.fraction) != set(FRACTIONS):
            continue
        scores = {
            fraction: block.set_index(["pair", "item"]).score
            for fraction, block in group.groupby("fraction")
        }
        common = set.intersection(*(set(series.index) for series in scores.values()))
        if not common:
            continue
        common = sorted(common)
        reference = scores[1.0].reindex(common)
        for fraction in FRACTIONS:
            current = scores[fraction].reindex(common)
            pair_scores = current.groupby(level="pair").mean()
            pair_deltas = (current - reference).groupby(level="pair").mean()
            accuracy, accuracy_low, accuracy_high = _mean_ci(pair_scores)
            delta, delta_low, delta_high = _mean_ci(pair_deltas)
            output.append({
                "judge": judge,
                "direction": direction,
                "fraction": fraction,
                "accuracy": accuracy,
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "delta_vs_100": delta,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "n_cells": len(common),
                "n_pairs": len(pair_scores),
            })
    columns = [
        "judge", "direction", "fraction", "accuracy", "accuracy_ci_low",
        "accuracy_ci_high", "delta_vs_100", "delta_ci_low", "delta_ci_high",
        "n_cells", "n_pairs",
    ]
    return pd.DataFrame(output, columns=columns).sort_values(
        ["judge", "direction", "fraction"], ignore_index=True)


def matched_direction_contrasts(rows: pd.DataFrame) -> pd.DataFrame:
    """End-minus-first contrasts using cells shared by all eight conditions."""
    cells = _counterbalanced_cells(rows)
    output = []
    required = {(direction, fraction) for direction in ("first", "end")
                for fraction in FRACTIONS}
    for judge, group in cells.groupby("judge"):
        if set(zip(group.direction, group.fraction)) != required:
            continue
        scores = {
            (direction, fraction): block.set_index(["pair", "item"]).score
            for (direction, fraction), block in group.groupby(["direction", "fraction"])
        }
        common = sorted(set.intersection(*(set(series.index) for series in scores.values())))
        for fraction in FRACTIONS:
            differences = (scores[("end", fraction)].reindex(common)
                           - scores[("first", fraction)].reindex(common))
            pair_differences = differences.groupby(level="pair").mean()
            mean, low, high = _mean_ci(pair_differences)
            output.append({
                "judge": judge,
                "fraction": fraction,
                "end_minus_first": mean,
                "ci_low": low,
                "ci_high": high,
                "n_cells": len(common),
                "n_pairs": len(pair_differences),
            })
    columns = ["judge", "fraction", "end_minus_first", "ci_low", "ci_high",
               "n_cells", "n_pairs"]
    return pd.DataFrame(output, columns=columns).sort_values(
        ["judge", "fraction"], ignore_index=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path,
                        default=root / "logs/pairwise-newjudges/manipulations")
    parser.add_argument("--configs", type=Path,
                        default=root / "scripts/pairwise-newjudges/configs/manipulations")
    parser.add_argument("--baseline", type=Path,
                        default=root / "judge_results/judging_variations/verdicts.csv")
    parser.add_argument("--subset", type=Path,
                        default=root / "scripts/pairwise-newjudges/configs/manipulations/trunc_first_10.yaml")
    parser.add_argument("--out", type=Path,
                        default=root / "judge_results/robustness/truncation")
    args = parser.parse_args()

    rows, runs = collect_logs(args.logs)
    subset = yaml.safe_load(args.subset.read_text())
    baseline = baseline_rows(pd.read_csv(args.baseline), subset["sample_ids"])
    baseline = baseline[baseline.judge.isin(runs.judge.unique())]
    rows = pd.concat([rows, baseline], ignore_index=True)
    estimates = matched_estimates(rows)
    contrasts = matched_direction_contrasts(rows)
    cells = _counterbalanced_cells(rows)
    judges = sorted(runs.judge.unique())
    coverage = []
    for condition in [f"trunc_{direction}_{percent}"
                      for direction in ("first", "end") for percent in (10, 50, 90)]:
        cfg = yaml.safe_load((args.configs / f"{condition}.yaml").read_text())
        expected_pairs = len(cfg["comparisons"])
        expected_rows = 2 * sum(len(items) for items in cfg["sample_ids"].values())
        for judge in judges:
            run = runs[(runs.condition == condition) & (runs.judge == judge)]
            verdict = rows[(rows.condition == condition) & (rows.judge == judge)]
            complete = cells[(cells.judge == judge) &
                             (cells.direction == condition.split("_")[1]) &
                             (cells.fraction == int(condition.rsplit("_", 1)[1]) / 100)]
            coverage.append({
                "condition": condition,
                "judge": judge,
                "successful_pairs": int(run.success.sum()) if len(run) else 0,
                "expected_pairs": expected_pairs,
                "rows": len(verdict),
                "expected_rows": expected_rows,
                "parse_rate": float(verdict.parse_ok.mean()) if len(verdict) else np.nan,
                "complete_cells": len(complete),
            })
    expected_pairs = len(subset["comparisons"])
    expected_rows = 2 * sum(len(items) for items in subset["sample_ids"].values())
    for judge in judges:
        verdict = baseline[(baseline.direction == "first") & (baseline.judge == judge)]
        complete = cells[(cells.judge == judge) & (cells.direction == "first") &
                         (cells.fraction == 1.0)]
        coverage.append({
            "condition": "a_baseline",
            "judge": judge,
            "successful_pairs": verdict.pair.nunique(),
            "expected_pairs": expected_pairs,
            "rows": len(verdict),
            "expected_rows": expected_rows,
            "parse_rate": float(verdict.parse_ok.mean()) if len(verdict) else np.nan,
            "complete_cells": len(complete),
        })
    coverage = pd.DataFrame(coverage)

    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "verdicts.csv", index=False)
    coverage.to_csv(args.out / "coverage.csv", index=False)
    estimates.to_csv(args.out / "estimates.csv", index=False)
    contrasts.to_csv(args.out / "direction_contrasts.csv", index=False)
    print(coverage.to_string(index=False))
    print("\nMatched complete curves:")
    shown = estimates.copy()
    for column in ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "delta_vs_100",
                   "delta_ci_low", "delta_ci_high"):
        shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    print(shown.to_string(index=False))
    print("\nMatched end-minus-first contrasts:")
    shown = contrasts.copy()
    for column in ("end_minus_first", "ci_low", "ci_high"):
        shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    print(shown.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
