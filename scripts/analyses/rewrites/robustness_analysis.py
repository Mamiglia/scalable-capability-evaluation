"""Compare original and style-standardised transcript judging.

The verdict CSV must contain ``bench, condition, judge, major, minor, item_id,
order, picked, picked_major, parse_ok, len_major, len_minor``.  The campaign
YAML must contain ``{"sample_ids": {"major-vs-minor": ["item", ...]}}``.
Each style-rate CSV is keyed by ``model,item`` and contains ``n_chars``,
``reasoning_chars`` and the six categories in ``STYLE_CATEGORIES``.  Frozen
GLM inputs are the baseline correctness frame, posterior summary and coords.

Build the rewrite rates from the actual eval transcripts, preserving the
``n_chars`` cross-check used by the analysis::

  uv run python scripts/hibayes/style_rates.py \
    --condition-config scripts/pairwise-newjudges/configs/manipulations/style_standardised_gpt.yaml \
    --out judge_results/robustness/style_rates_style_standardised_gpt.csv
"""

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import t


CONDITIONS = ("a_baseline", "style_standardised_gpt")
CONDITION_LABELS = ("Original", "Style-standardised")
JUDGE_ORDER = ("qwen3-8b", "qwen3-32b", "gpt-oss-20b")
JUDGE_LABELS = {"qwen3-8b": "Qwen3 8B", "qwen3-32b": "Qwen3 32B",
                "gpt-oss-20b": "GPT-OSS 20B"}
JUDGE_COLORS = {"qwen3-8b": "#1baf7a", "qwen3-32b": "#e87ba4",
                "gpt-oss-20b": "#eb6834"}
STYLE_CATEGORIES = (
    "personal_pronouns", "hedging", "backtracking", "uncertainty",
    "confidence", "questions",
)
STYLE_KEYS = ("model", "item", "n_chars", "reasoning_chars")
TRANSCRIPT_FEATURES = (
    "total_length", "length_difference_stronger",
    *[f"{category}_difference_stronger" for category in STYLE_CATEGORIES],
)
PREDICTORS = (
    "truth_side", "truth_side_x_total_length", "total_length_sd",
    "capability_gap_sd", "length_difference_stronger_sd", "same_model_family",
    "personal_pronouns_difference_stronger_sd", "hedging_difference_stronger_sd",
    "backtracking_difference_stronger_sd", "uncertainty_difference_stronger_sd",
    "confidence_difference_stronger_sd", "questions_difference_stronger_sd",
)
SCALED_FEATURES = (
    "capability_gap", *TRANSCRIPT_FEATURES,
)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hibayes"))
CAMPAIGN = (
    ROOT / "scripts/pairwise-newjudges/configs/manipulations/style_standardised_gpt.yaml"
)
STYLE_RATES = {
    "a_baseline": ROOT / "analysis/hibayes/style_rates.csv",
    "style_standardised_gpt": (
        ROOT / "judge_results/robustness/style_rates_style_standardised_gpt.csv"
    ),
}
REFERENCE = ROOT / "analysis/hibayes/gpqa_correctness.csv"
GLM_DIR = (ROOT / "analysis/hibayes/run_correctness_full/models/"
           "judge_feature_logistic_model_full")
GLM_SUMMARY = GLM_DIR / "diagnostics/inference_data_summary.csv"
GLM_COORDS = ROOT / "analysis/hibayes/run_correctness_full/coords.json"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
})


def validate_campaign(campaign_cells: dict) -> None:
    """Refuse the historical 1,192-cell scope or any other partial campaign."""
    valid = (
        len(campaign_cells) == 50
        and all(len(items) == 30 and len(set(items)) == 30
                for items in campaign_cells.values())
    )
    if not valid:
        counts = [len(items) for items in campaign_cells.values()]
        raise ValueError(
            "The style-standardisation campaign must contain exactly 50 pairs × 30 "
            f"items, unique within pair (got {len(campaign_cells)} pairs, "
            f"{sum(counts)} pair-items)"
        )


def load_campaign(path: Path) -> dict:
    """Load and validate the cluster campaign's shared sample IDs."""
    campaign_cells = yaml.safe_load(path.read_text())["sample_ids"]
    validate_campaign(campaign_cells)
    return campaign_cells


def validate_style_rates(style_rates: pd.DataFrame, path: Path) -> None:
    """Require the exact feature schema used by the frozen correctness GLM."""
    missing_keys = set(STYLE_KEYS) - set(style_rates.columns)
    categories = set(style_rates.columns) - set(STYLE_KEYS)
    if (missing_keys or categories != set(STYLE_CATEGORIES)
            or style_rates.columns.duplicated().any()):
        raise ValueError(
            f"{path} must contain metadata {STYLE_KEYS} and exactly the canonical "
            f"style categories {STYLE_CATEGORIES}; got {tuple(style_rates.columns)}"
        )
    if style_rates.duplicated(["model", "item"]).any():
        raise ValueError(f"{path} contains duplicate model,item rows")


def summarise(verdicts: pd.DataFrame, campaign_cells: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired condition estimates and pre-pairing coverage diagnostics."""
    allowed = {(pair, item) for pair, items in campaign_cells.items() for item in items}
    rows = verdicts.copy()
    rows["pair"] = rows.major + "-vs-" + rows.minor
    rows = rows[[key in allowed for key in zip(rows.pair, rows.item_id)]]
    if rows.parse_ok.dtype == object:
        rows["parse_ok"] = rows.parse_ok.map(lambda x: str(x).lower() == "true")

    expected_rows = 2 * len(allowed)
    coverage = []
    cells_by_condition = {}
    row_counts = {}
    for judge in rows.judge.unique():
        for condition in CONDITIONS:
            group = rows[(rows.judge == judge) & (rows.condition == condition)]
            parsed = group[group.parse_ok & group.picked_major.notna()].copy()
            parsed["picked_major"] = parsed.picked_major.astype(float)
            keys = ["pair", "major", "minor", "item_id"]
            valid = parsed.groupby(keys).filter(
                lambda cell: len(cell) == 2 and cell.order.nunique() == 2
            )
            cells = (valid.groupby(keys, as_index=False)
                          .agg(score=("picked_major", "mean")))
            cells_by_condition[(judge, condition)] = cells
            row_counts[(judge, condition)] = len(group)
            coverage.append({
                "judge": judge,
                "condition": condition,
                "expected_rows": expected_rows,
                "observed_rows": len(group),
                "parsed_rows": len(parsed),
                "parse_rate": len(parsed) / expected_rows,
                "complete_cells": len(cells),
            })

    ready_judges = [
        judge for judge in rows.judge.unique()
        if all(row_counts[(judge, condition)] == expected_rows for condition in CONDITIONS)
    ]
    common = set.intersection(*[
        set(zip(cells_by_condition[(judge, condition)].pair,
                cells_by_condition[(judge, condition)].item_id))
        for judge in ready_judges for condition in CONDITIONS
    ]) if ready_judges else set()

    estimates = []
    for judge in ready_judges:
        paired = {}
        for condition in CONDITIONS:
            cells = cells_by_condition[(judge, condition)]
            paired[condition] = cells[
                [key in common for key in zip(cells.pair, cells.item_id)]
            ].set_index(["pair", "item_id"]).score
        baseline = paired["a_baseline"]

        for condition in CONDITIONS:
            scores = paired[condition]
            delta = scores - baseline
            accuracy = scores.mean()
            change = delta.mean()
            estimates.append({
                "judge": judge,
                "condition": condition,
                "accuracy": accuracy,
                "delta_vs_baseline": change,
                "n_cells": len(scores),
                "n_pairs": scores.index.get_level_values("pair").nunique(),
            })

    return pd.DataFrame(estimates), pd.DataFrame(coverage)


def _t_interval(values: pd.Series) -> tuple[float, float]:
    """95% t interval across judges; no inferential interval below N=3."""
    values = values.dropna().astype(float)
    if len(values) < 3:
        return math.nan, math.nan
    mean = values.mean()
    half_width = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def accuracy_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy and paired change, with judges as the replication unit."""
    by_judge = (cells.groupby(["judge", "condition"], as_index=False)
                     .agg(accuracy=("observed", "mean"), n_cells=("item", "size")))
    baseline = (by_judge[by_judge.condition == "a_baseline"]
                .set_index("judge").accuracy)
    by_judge["delta_vs_baseline"] = by_judge.accuracy - by_judge.judge.map(baseline)

    summary = []
    for condition in CONDITIONS:
        rows = by_judge[by_judge.condition == condition]
        ci_lo, ci_hi = _t_interval(rows.accuracy)
        delta_lo, delta_hi = _t_interval(rows.delta_vs_baseline)
        summary.append({
            "condition": condition,
            "accuracy": rows.accuracy.mean(),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "delta": rows.delta_vs_baseline.mean(),
            "delta_ci_lo": delta_lo,
            "delta_ci_hi": delta_hi,
            "n_judges": rows.judge.nunique(),
            "n_cells": rows.n_cells.min() if not rows.empty else 0,
        })
    return by_judge, pd.DataFrame(summary)


def _style_axes(ax) -> None:
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_position(("outward", 5))


def _mean_errorbar(ax, x, row, value: str, lo: str, hi: str) -> None:
    lower, upper = row[lo], row[hi]
    yerr = None
    if np.isfinite(lower) and np.isfinite(upper):
        yerr = [[row[value] - lower], [upper - row[value]]]
    ax.errorbar(
        [x], [row[value]], yerr=yerr, color="#111111", marker="D", ms=3.2,
        lw=1, capsize=0, linestyle="none", zorder=5,
    )


def make_accuracy_figure(cells: pd.DataFrame):
    """Original versus rewrite accuracy and paired judge-level change."""
    by_judge, summary = accuracy_tables(cells)
    present_set = set(by_judge.judge)
    present = [judge for judge in JUDGE_ORDER if judge in present_set]
    present += sorted(present_set - set(present))
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(
        1, 2, figsize=(3.8, 2.4), gridspec_kw={"width_ratios": [1.3, 1.0]}
    )

    for judge in present:
        rows = (by_judge[by_judge.judge == judge]
                .set_index("condition").reindex(CONDITIONS))
        axes[0].plot(
            x, rows.accuracy, color=JUDGE_COLORS.get(judge, "#777777"),
            marker="o", ms=3, lw=0.9, label=JUDGE_LABELS.get(judge, judge), zorder=3,
        )
    for condition_index, condition in enumerate(CONDITIONS):
        row = summary[summary.condition == condition].iloc[0]
        _mean_errorbar(axes[0], condition_index, row, "accuracy", "ci_lo", "ci_hi")
    axes[0].axhline(0.5, color="#888888", lw=0.8, ls=(0, (2, 2)), zorder=1)
    axes[0].set(
        xlim=(-0.25, len(CONDITIONS) - 0.75), ylim=(0, 1), xticks=x,
        xticklabels=CONDITION_LABELS, yticks=[0, 0.5, 1], ylabel="Judge accuracy",
    )

    rewrite = by_judge[by_judge.condition == "style_standardised_gpt"]
    offsets = np.linspace(-0.06, 0.06, len(rewrite)) if len(rewrite) > 1 else [0]
    for offset, (_, row) in zip(offsets, rewrite.iterrows(), strict=True):
        axes[1].plot(
            offset, row.delta_vs_baseline, color=JUDGE_COLORS.get(row.judge, "#777777"),
            marker="o", ms=3, linestyle="none", zorder=3,
        )
    rewrite_summary = summary[summary.condition == "style_standardised_gpt"].iloc[0]
    _mean_errorbar(
        axes[1], 0, rewrite_summary, "delta", "delta_ci_lo", "delta_ci_hi"
    )
    axes[1].axhline(0, color="#888888", lw=0.8, ls=(0, (2, 2)), zorder=1)
    delta_values = rewrite.delta_vs_baseline.abs().dropna().tolist()
    delta_values += [
        abs(value) for value in (
            rewrite_summary.delta_ci_lo, rewrite_summary.delta_ci_hi,
        ) if np.isfinite(value)
    ]
    limit = max(0.1, math.ceil(max(delta_values, default=0.1) * 10) / 10)
    axes[1].set(
        xlim=(-0.25, 0.25), ylim=(-limit, limit), xticks=[0],
        xticklabels=["Style-standardised"], yticks=[-limit, 0, limit],
        ylabel="Accuracy change",
    )
    for ax, tag in zip(axes, ("a", "b"), strict=True):
        _style_axes(ax)
        ax.text(-0.18, 1.03, tag, transform=ax.transAxes, color="#666666",
                fontsize=8, va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#111111", marker="D", ms=3.2,
                              lw=0, label="Mean ± 95% CI"))
    labels.append("Mean ± 95% CI")
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.94, bottom=0.34, wspace=0.52)
    return fig


def predict_glm(frame: pd.DataFrame, summary_path: Path, coords_path: Path) -> pd.Series:
    """Posterior-mean prediction from the baseline hierarchical correctness GLM."""
    coords = json.loads(coords_path.read_text())
    summary_frame = pd.read_csv(summary_path)
    summary = summary_frame.set_index(summary_frame.columns[0])["mean"]
    judge_index = {name: index for index, name in enumerate(coords["judge"])}
    intercepts = {name: summary[f"judge_intercepts[{index}]"]
                  for name, index in judge_index.items()}
    effects = {
        name: np.array([
            summary[f"judge_predictor_effects[{index}, {predictor_index}]"]
            for predictor_index in range(len(PREDICTORS))
        ])
        for name, index in judge_index.items()
    }
    item_effects = {item: summary[f"item_effects[{item}]"] for item in coords["item"]}
    logits = np.array([intercepts[judge] for judge in frame.judge])
    logits += np.array([item_effects[item] for item in frame.item])
    logits += np.array([
        effects[judge] @ row
        for judge, row in zip(frame.judge, frame[list(PREDICTORS)].to_numpy(), strict=True)
    ])
    return pd.Series(expit(logits), index=frame.index, name="predicted")


def build_designs(verdicts: pd.DataFrame, style_paths: dict[str, Path],
                  reference_path: Path) -> pd.DataFrame:
    """Recompute rewrite predictors on the baseline model's training scales."""
    from prepare_correctness import difference_columns, to_correctness
    from prepare_from_verdicts import build, load_capabilities, CAPABILITY_CONFIG

    reference = pd.read_csv(reference_path)
    capabilities = load_capabilities(CAPABILITY_CONFIG, "eci")
    frames = []
    for condition in CONDITIONS:
        selected = verdicts[verdicts.condition == condition].copy()
        if selected.empty:
            raise ValueError(f"No verdicts for {condition}")
        selected["condition"] = "a_baseline"
        style_rates = pd.read_csv(style_paths[condition])
        validate_style_rates(style_rates, style_paths[condition])
        frame, _ = build(
            selected, capabilities, "gpqa", ["a_baseline"], style_rates=style_rates,
        )
        frame, _ = to_correctness(frame, difference_columns(style_rates))
        for feature in SCALED_FEATURES:
            frame[f"{feature}_sd"] = (
                frame[feature] - reference[feature].mean()
            ) / reference[feature].std(ddof=1)
        frame["truth_side_x_total_length"] = frame.truth_side * frame.total_length_sd
        frame["transcript_condition"] = condition
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result[list(PREDICTORS)].to_numpy()).all():
        raise ValueError("Counterfactual GLM predictors are not finite")
    return result


def paired_cells(frame: pd.DataFrame, campaign_cells: dict) -> pd.DataFrame:
    """Average orders after enforcing one common cell set across the experiment."""
    allowed = {
        ("|".join(sorted(pair.split("-vs-", 1))), item)
        for pair, items in campaign_cells.items() for item in items
    }
    rows = frame[
        [key in allowed for key in zip(frame.model_pair, frame.item)]
    ].rename(columns={"transcript_condition": "condition"})
    keys = ["judge", "condition", "model_pair", "item"]
    valid = rows.groupby(keys).filter(lambda cell: len(cell) == 2 and cell.order.nunique() == 2)
    cells = (valid.groupby(keys, as_index=False)
                  .agg(observed=("is_correct", "mean"),
                       predicted=("predicted", "mean"),
                       capability_gap=("capability_gap", "first")))
    ready = cells.groupby("judge").condition.nunique()
    cells = cells[cells.judge.isin(ready[ready == len(CONDITIONS)].index)]
    common = set.intersection(*[
        set(zip(group.model_pair, group.item))
        for _, group in cells.groupby(["judge", "condition"])
    ]) if not cells.empty else set()
    return cells[[key in common for key in zip(cells.model_pair, cells.item)]].reset_index(drop=True)


def gap_by_judge(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-judge tertile curves on one shared capability-gap binning."""
    pair_gaps = cells.groupby("model_pair").capability_gap.first().sort_values()
    bins = pd.qcut(pair_gaps.rank(method="first"), 3, labels=False)
    rows = cells.assign(gap_bin=cells.model_pair.map(bins))
    return (rows.groupby(["judge", "condition", "gap_bin"], as_index=False)
                .agg(gap_mean=("capability_gap", "mean"),
                     observed=("observed", "mean"),
                     predicted=("predicted", "mean")))


def gap_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Tertile curves, treating judges as the replication unit."""
    per_judge = gap_by_judge(cells)
    result = []
    for (condition, gap_bin), group in per_judge.groupby(["condition", "gap_bin"]):
        n = len(group)
        scale = t.ppf(0.975, n - 1) / math.sqrt(n) if n >= 3 else math.nan
        result.append({
            "condition": condition, "gap_bin": int(gap_bin),
            "gap_mean": group.gap_mean.mean(), "n_judges": n,
            "observed": group.observed.mean(),
            "obs_ci_lo": group.observed.mean() - scale * group.observed.std(ddof=1),
            "obs_ci_hi": group.observed.mean() + scale * group.observed.std(ddof=1),
            "predicted": group.predicted.mean(),
            "pred_ci_lo": group.predicted.mean() - scale * group.predicted.std(ddof=1),
            "pred_ci_hi": group.predicted.mean() + scale * group.predicted.std(ddof=1),
        })
    return pd.DataFrame(result)


def make_gap_figure(gaps: pd.DataFrame, by_judge: pd.DataFrame | None = None):
    """Observed and frozen-GLM gap curves, retaining judge-level traces."""
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.4), sharex=True, sharey=True)
    max_gap = max(10.0, float(gaps.gap_mean.max()))
    x_max = (math.floor(max_gap / 10) + 1) * 10
    for ax, condition, label in zip(axes, CONDITIONS, CONDITION_LABELS, strict=True):
        rows = gaps[gaps.condition == condition].sort_values("gap_bin")
        if by_judge is not None:
            present = [judge for judge in JUDGE_ORDER if judge in set(by_judge.judge)]
            present += sorted(set(by_judge.judge) - set(present))
            for judge in present:
                unit = by_judge[(by_judge.condition == condition) &
                                (by_judge.judge == judge)].sort_values("gap_bin")
                if unit.empty:
                    continue
                color = JUDGE_COLORS.get(judge, "#777777")
                ax.plot(unit.gap_mean, unit.observed, color=color, marker="o",
                        ms=2.2, lw=0.7, alpha=0.42, zorder=2)
                ax.plot(unit.gap_mean, unit.predicted, color=color, marker="o",
                        ms=2.2, lw=0.7, ls="--", alpha=0.42, zorder=2)
        for value, lo, hi, color, linestyle in (
            ("observed", "obs_ci_lo", "obs_ci_hi", "#111111", "-"),
            ("predicted", "pred_ci_lo", "pred_ci_hi", "#666666", "--"),
        ):
            finite_ci = rows[[lo, hi]].notna().all().all()
            yerr = None if not finite_ci else [rows[value] - rows[lo], rows[hi] - rows[value]]
            ax.errorbar(
                rows.gap_mean, rows[value], yerr=yerr,
                color=color, marker="o", ms=3, lw=1, capsize=0,
                linestyle=linestyle, zorder=3,
            )
        ax.axhline(0.5, color="#888888", lw=0.8, ls=":", zorder=1)
        ax.text(0.03, 0.96, label, transform=ax.transAxes, color="#666666",
                fontsize=8, va="top")
        ax.set(xlim=(0, x_max), ylim=(0, 1), xticks=[0, x_max / 2, x_max],
               yticks=[0, 0.5, 1])
        _style_axes(ax)
    axes[0].set_ylabel("Judge accuracy")
    fig.supxlabel("Capability gap (ECI)", x=0.56, y=0.205, fontsize=8)
    handles = [
        plt.Line2D([], [], color="#111111", marker="o", ms=3, lw=1,
                   label="Observed mean"),
        plt.Line2D([], [], color="#666666", marker="o", ms=3, lw=1, ls="--",
                   label="Frozen GLM"),
    ]
    if by_judge is not None:
        present = [judge for judge in JUDGE_ORDER if judge in set(by_judge.judge)]
        present += sorted(set(by_judge.judge) - set(present))
        handles += [
            plt.Line2D([], [], color=JUDGE_COLORS.get(judge, "#777777"), lw=1,
                       label=JUDGE_LABELS.get(judge, judge))
            for judge in present
        ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.14, right=0.975, top=0.96, bottom=0.38, wspace=0.28)
    return fig


def write_outputs(cells: pd.DataFrame, coverage: pd.DataFrame, out: Path) -> None:
    """Write both analyses from the same complete cell intersection."""
    out.mkdir(parents=True, exist_ok=True)
    accuracy_by_judge, accuracy_summary = accuracy_tables(cells)
    by_judge = gap_by_judge(cells)
    gaps = gap_summary(cells)
    overall = (cells.groupby(["judge", "condition"], as_index=False)
                    .agg(observed=("observed", "mean"),
                         predicted=("predicted", "mean"), n_cells=("item", "size")))
    baseline = overall[overall.condition == "a_baseline"].set_index("judge")
    overall["observed_delta"] = overall.observed - overall.judge.map(baseline.observed)
    overall["predicted_delta"] = overall.predicted - overall.judge.map(baseline.predicted)

    accuracy_by_judge.to_csv(out / "accuracy_by_judge.csv", index=False)
    accuracy_summary.to_csv(out / "accuracy_summary.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    cells.to_csv(out / "cell_predictions.csv", index=False)
    by_judge.to_csv(out / "gap_by_judge.csv", index=False)
    gaps.to_csv(out / "gap_estimates.csv", index=False)
    overall.to_csv(out / "overall_observed_vs_glm.csv", index=False)

    figures = {
        "accuracy_original_vs_rewrite": make_accuracy_figure(cells),
        "gap_curves": make_gap_figure(gaps, by_judge),
    }
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png", dpi=300)
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)


def analyse(verdicts: pd.DataFrame, campaign_cells: dict, out: Path,
            style_paths: dict[str, Path] = STYLE_RATES,
            reference_path: Path = REFERENCE,
            summary_path: Path = GLM_SUMMARY,
            coords_path: Path = GLM_COORDS) -> None:
    """Build one shared complete-cell analysis from finished judging outputs."""
    validate_campaign(campaign_cells)
    _, coverage = summarise(verdicts, campaign_cells)
    complete = coverage[coverage.observed_rows == coverage.expected_rows]
    ready = complete.groupby("judge").condition.nunique()
    ready = set(ready[ready == len(CONDITIONS)].index)
    if not ready:
        raise ValueError("No judge has complete original and style-standardised rows")
    verdicts = verdicts[verdicts.judge.isin(ready)]
    designs = build_designs(verdicts, style_paths, reference_path)
    designs["predicted"] = predict_glm(designs, summary_path, coords_path)
    cells = paired_cells(designs, campaign_cells)
    if cells.empty:
        raise ValueError("No common complete cell intersection across judges and conditions")
    write_outputs(cells, coverage, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path,
                        default=Path("judge_results/robustness/style_standardised_gpt/verdicts.csv"))
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--out", type=Path,
                        default=Path("judge_results/robustness/style_standardised_gpt"))
    args = parser.parse_args()
    cells = load_campaign(args.campaign)
    verdicts = pd.read_csv(args.verdicts)
    analyse(verdicts, cells, args.out)


if __name__ == "__main__":
    main()
