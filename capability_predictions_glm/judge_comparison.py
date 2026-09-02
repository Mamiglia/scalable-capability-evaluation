"""Compare pairwise capability and judge-verdict GLMs from saved artifacts."""

import argparse
from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t

from .config import CONFIG, ROOT, TARGET_QUARTILES
from .data import validate_manifest
from .plots import _ordinary, _style
from .preprocessing import attach_pair_target, item_quartiles, quartile_accuracy, split_pairs


JUDGE_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v4-flash-0731": "DeepSeek V4",
    "gemma-4-31b-it": "Gemma 4",
    "gpt-oss-120b": "GPT-OSS",
    "ling-2.6-flash": "Ling",
    "o4-mini": "o4-mini",
}
SHORT_JUDGE_LABELS = {
    "Llama-3.1-8B-Instruct": "L8", "deepseek-r1-0528": "R1",
    "deepseek-v4-flash-0731": "V4", "gemma-4-31b-it": "G4",
    "gpt-oss-120b": "OSS", "o4-mini": "o4",
}
BENCHMARK_LABELS = {
    "aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
    "swe_bench_mini": "SWE",
}
METHOD_COLORS = {"pairwise_glm": "#1baf7a", "llm_judge": "#eb6834"}
METHOD_LABELS = {"pairwise_glm": "Capability GLM", "llm_judge": "LLM judge"}
QUADRANT_COLORS = {
    "positive_positive": "#1baf7a", "positive_negative": "#eb6834",
    "negative_positive": "#eda100", "negative_negative": "#4a3aa7",
}
FILL_ONLY = {"edgecolors": "none", "linewidths": 0}
ECI_MODELS = {
    "deepseek-r1-0528": "DeepSeek-R1 (May 2025)",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash 0731",
    "gemma-4-31b-it": "Gemma 4 31B IT",
    "gpt-oss-120b": "gpt-oss-120b",
    "o4-mini": "o4-mini",
}
ECI_LABEL_POSITIONS = {
    "deepseek-r1-0528": (143.8, 0.11, "left"),
    "gemma-4-31b-it": (140.8, 0.46, "center"),
    "gpt-oss-120b": (136.0, 0.12, "left"),
    "o4-mini": (147.0, 0.18, "left"),
    "deepseek-v4-flash-0731": (154.5, 0.42, "right"),
}
ECI_SHORT_LABELS = {
    "deepseek-r1-0528": "R1", "deepseek-v4-flash-0731": "V4",
    "gemma-4-31b-it": "G4", "gpt-oss-120b": "OSS", "o4-mini": "o4",
}


def strict_pair_accuracy(
    pairs: pd.DataFrame,
    coefficients: pd.DataFrame,
    held_out_models: set[str],
) -> pd.DataFrame:
    """Score the capability GLM and judges on pairs touching a held-out model."""
    required = {
        "bench", "judge", "model_left", "model_right", "left_stronger", "pick_left",
    }
    if missing := required - set(pairs):
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    if missing := {"benchmark", "feature", "effective"} - set(coefficients):
        raise ValueError(f"coefficient table missing columns: {sorted(missing)}")
    eligible = pairs[
        pairs.model_left.isin(held_out_models) | pairs.model_right.isin(held_out_models)
    ]
    rows = []
    for (bench, judge), frame in eligible.groupby(["bench", "judge"], sort=False):
        weights = coefficients[coefficients.benchmark == bench]
        feature_columns = [f"d_{feature}" for feature in weights.feature]
        if missing := set(feature_columns) - set(frame):
            raise ValueError(f"pair table missing feature columns: {sorted(missing)}")
        logits = frame[feature_columns].to_numpy(float) @ weights.effective.to_numpy(float)
        observed = frame.left_stronger.to_numpy(int)
        for method, predicted in (
            ("pairwise_glm", logits >= 0),
            ("llm_judge", frame.pick_left.to_numpy(float) >= 0.5),
        ):
            rows.append({
                "bench": bench,
                "judge": judge,
                "method": method,
                "accuracy": float(np.mean(predicted == observed)),
                "pair_count": len(frame),
            })
    return pd.DataFrame(rows)


def coefficient_comparison(
    capability: pd.DataFrame,
    judges: pd.DataFrame,
) -> pd.DataFrame:
    """Match capability and judge coefficients within the same outer model fold."""
    capability_required = {"fold", "target_quartile", "benchmark", "feature", "effective"}
    judge_required = {"fold", "judge", "benchmark", "feature", "effective"}
    if missing := capability_required - set(capability):
        raise ValueError(f"capability coefficients missing columns: {sorted(missing)}")
    if missing := judge_required - set(judges):
        raise ValueError(f"judge coefficients missing columns: {sorted(missing)}")
    selected = capability[capability.benchmark.notna() & capability.feature.ne("bare_answer")]
    if "task" in selected:
        selected = selected[selected.task == "pairwise"]
    selected = (selected.groupby(["fold", "benchmark", "feature"], as_index=False)
                .effective.mean()
                .rename(columns={"fold": "model_fold", "effective": "capability"}))
    judge_rows = judges[judges.feature.ne("bare_answer")].copy()
    dimensions = judge_rows.fold.astype(str).str.split("__", n=1, expand=True)
    if dimensions.shape[1] != 2 or dimensions.isna().any().any():
        raise ValueError("judge coefficient folds must use MODEL_FOLD__ITEM_FOLD")
    judge_rows["model_fold"] = dimensions[0].to_numpy()
    judge_rows["item_fold"] = dimensions[1].to_numpy()
    judge_rows = judge_rows.rename(columns={"effective": "judge_coefficient"})
    return (judge_rows.merge(selected, on=["model_fold", "benchmark", "feature"],
                             how="inner", validate="many_to_one")
            [["fold", "model_fold", "item_fold", "judge", "benchmark", "feature",
              "capability", "judge_coefficient"]]
            .sort_values(["fold", "judge", "benchmark", "feature"], kind="stable")
            .reset_index(drop=True))


def coefficient_alignment(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold-level cosines and signed feature contributions to pooled cosine."""
    required = {
        "fold", "model_fold", "item_fold", "judge", "benchmark", "feature",
        "capability", "judge_coefficient",
    }
    if missing := required - set(comparison):
        raise ValueError(f"coefficient comparison missing columns: {sorted(missing)}")
    has_support = "observed" in comparison
    cosine_rows, contribution_rows = [], []
    for (fold, model_fold, item_fold, judge), frame in comparison.groupby(
        ["fold", "model_fold", "item_fold", "judge"], sort=False,
    ):
        for benchmark in ["all", *frame.benchmark.drop_duplicates().tolist()]:
            values = frame if benchmark == "all" else frame[frame.benchmark == benchmark]
            left = values.capability.to_numpy(float)
            right = values.judge_coefficient.to_numpy(float)
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            cosine = float(left @ right / denominator) if denominator else np.nan
            cosine_rows.append({
                "fold": fold, "model_fold": model_fold, "item_fold": item_fold,
                "judge": judge, "benchmark": benchmark, "cosine": cosine,
                "observed": bool(values.observed.all()) if has_support else True,
            })
            if benchmark == "all" and denominator:
                contributions = values.assign(
                    contribution=left * right / denominator,
                ).groupby("feature", as_index=False).contribution.sum()
                contribution_rows.extend(
                    contributions.assign(fold=fold, model_fold=model_fold,
                                         item_fold=item_fold, judge=judge).to_dict("records")
                )
    return pd.DataFrame(cosine_rows), pd.DataFrame(contribution_rows)


def coefficient_statistics(points: pd.DataFrame) -> pd.DataFrame:
    """Describe coefficient agreement overall and within each benchmark."""
    required = {"judge", "benchmark", "feature", "capability", "judge_coefficient", "observed"}
    if missing := required - set(points):
        raise ValueError(f"coefficient points missing columns: {sorted(missing)}")
    overall = (points.groupby(["judge", "feature"], as_index=False)
               .agg(capability=("capability", "mean"),
                    judge_coefficient=("judge_coefficient", "mean"),
                    observed=("observed", "all"))
               .assign(benchmark="all"))
    rows = []
    for (judge, benchmark), frame in pd.concat([overall, points], ignore_index=True).groupby(
        ["judge", "benchmark"], sort=False,
    ):
        capability = frame.capability.to_numpy(float)
        verdict = frame.judge_coefficient.to_numpy(float)
        positive_capability, positive_verdict = capability >= 0, verdict >= 0
        rows.append({
            "judge": judge, "benchmark": benchmark,
            "pearson": float(pearsonr(capability, verdict).statistic),
            "spearman": float(spearmanr(capability, verdict).statistic),
            "same_sign_share": float(np.mean(positive_capability == positive_verdict)),
            "positive_positive": int(np.sum(positive_capability & positive_verdict)),
            "positive_negative": int(np.sum(positive_capability & ~positive_verdict)),
            "negative_positive": int(np.sum(~positive_capability & positive_verdict)),
            "negative_negative": int(np.sum(~positive_capability & ~positive_verdict)),
            "n_features": len(frame), "observed": bool(frame.observed.all()),
        })
    return pd.DataFrame(rows)


def build_tables(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    capability_coefficients: pd.DataFrame,
    judge_coefficients: pd.DataFrame,
    dictionary: pd.DataFrame,
    model_folds: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build strict accuracy, coefficient, cosine, and driver tables."""
    quartiles = item_quartiles(responses)
    accuracies = quartile_accuracy(responses, quartiles)
    q1_pairs, _ = split_pairs(pairs, quartiles, set())
    accuracy_rows = []
    for fold, members in model_folds:
        for target in TARGET_QUARTILES:
            labelled = attach_pair_target(q1_pairs, accuracies, target)
            coefficients = capability_coefficients[
                (capability_coefficients.fold == fold)
                & (capability_coefficients.target_quartile == target)
                & capability_coefficients.benchmark.notna()
                & capability_coefficients.effective.notna()
            ]
            if "task" in coefficients:
                coefficients = coefficients[coefficients.task == "pairwise"]
            rows = strict_pair_accuracy(labelled, coefficients, set(members))
            accuracy_rows.append(rows.assign(fold=fold, target_quartile=target))
    accuracy = pd.concat(accuracy_rows, ignore_index=True)
    support = (pairs[["judge", "bench"]].drop_duplicates()
               .rename(columns={"bench": "benchmark"}).assign(observed=True))
    comparison = coefficient_comparison(capability_coefficients, judge_coefficients).merge(
        support, on=["judge", "benchmark"], how="left", validate="many_to_one",
    )
    comparison["observed"] = comparison.observed.fillna(False).astype(bool)
    if missing := {"feature", "block"} - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    points = (comparison.groupby(["judge", "benchmark", "feature", "observed"], as_index=False)
              [["capability", "judge_coefficient"]].mean()
              .merge(dictionary[["feature", "block"]], on="feature", how="left",
                     validate="many_to_one"))
    if points.block.isna().any():
        raise ValueError("feature dictionary is missing compared coefficients")
    cosines, contributions = coefficient_alignment(comparison)
    return accuracy, points, cosines, contributions


def _save(fig, output_dir: Path, name: str) -> Path:
    base = output_dir / name
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)
    return base


def _t_interval(values: pd.Series, lower: float, upper: float) -> tuple[float, float]:
    if len(values) < 3 or values.nunique() == 1:
        mean = float(values.mean())
        return mean, mean
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    mean = float(values.mean())
    return max(lower, mean - margin), min(upper, mean + margin)


def _accuracy_marks(ax, frame: pd.DataFrame, judges: list[str]) -> None:
    positions = np.arange(len(judges))
    means: dict[str, list[float]] = {}
    for method, offset in (("pairwise_glm", -0.11), ("llm_judge", 0.11)):
        values = []
        for position, judge in zip(positions, judges):
            fold_values = frame[(frame.judge == judge) & (frame.method == method)].accuracy
            if fold_values.empty:
                values.append(np.nan)
                continue
            mean = float(fold_values.mean())
            low, high = _t_interval(fold_values, 0, 1)
            ax.scatter(fold_values, np.full(len(fold_values), position + offset), s=2,
                       color=METHOD_COLORS[method], alpha=0.18, **FILL_ONLY)
            ax.errorbar(mean, position + offset, xerr=[[mean - low], [high - mean]],
                        fmt="none", color=METHOD_COLORS[method], lw=1, capsize=0,
                        clip_on=False)
            ax.scatter(mean, position + offset, s=8, color=METHOD_COLORS[method],
                       clip_on=False, zorder=3, **FILL_ONLY)
            values.append(mean)
        means[method] = values
    for position, capability, judge in zip(positions, means["pairwise_glm"], means["llm_judge"]):
        if np.isfinite(capability) and np.isfinite(judge):
            ax.plot([capability, judge], [position - 0.11, position + 0.11],
                    color="#aaaaaa", lw=0.6, zorder=0)
    ax.axvline(0.5, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(0.3, 1)
    ax.set_xticks([0.3, 1])
    ax.set_ylim(len(judges) - 0.5, -0.5)
    ax.set_yticks(positions)
    _ordinary(ax)


def _scatter_panel(ax, frame: pd.DataFrame, limit: float) -> None:
    ax.scatter(frame.capability, frame.judge_coefficient, s=2, color="#2a78d6",
               alpha=0.14, clip_on=False, **FILL_ONLY)
    ax.plot([-limit, limit], [-limit, limit], color="#aaaaaa", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])
    ax.set_yticks([-limit, 0, limit])
    _ordinary(ax)


def _crossed_cosine_summary(cosines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (judge, benchmark), frame in cosines.groupby(["judge", "benchmark"], sort=False):
        intervals = []
        for dimension in ("model_fold", "item_fold"):
            values = frame.groupby(dimension).cosine.mean()
            intervals.append(_t_interval(values, -1, 1))
        rows.append({
            "judge": judge, "benchmark": benchmark, "estimate": float(frame.cosine.mean()),
            "lower": min(value[0] for value in intervals),
            "upper": max(value[1] for value in intervals),
            "observed": bool(frame.observed.all()) if "observed" in frame else True,
        })
    return pd.DataFrame(rows)


def plot_comparisons(
    accuracy: pd.DataFrame,
    coefficient_points: pd.DataFrame,
    cosines: pd.DataFrame,
    contributions: pd.DataFrame,
    eci: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Render the approved story-first figures and compact diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = []
    judges = accuracy.judge.drop_duplicates().tolist()
    judge_labels = [JUDGE_LABELS.get(value, value) for value in judges]
    quartiles = [value for value in ("Q2", "Q3", "Q4")
                 if value in set(accuracy.target_quartile)]
    benches = [value for value in ("aime", "gpqa", "intercode", "swe_bench_mini")
               if value in set(accuracy.bench)]

    overall = (accuracy.groupby(["fold", "target_quartile", "judge", "method"], as_index=False)
               .accuracy.mean())
    fig, axes = plt.subplots(1, len(quartiles), figsize=(3.0, 1.75), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    for column, (ax, quartile) in enumerate(zip(axes.flat, quartiles)):
        _accuracy_marks(ax, overall[overall.target_quartile == quartile], judges)
        ax.set_title(quartile, loc="left", color="#666666", fontsize=7, pad=1)
    axes[0, 0].set_yticks(range(len(judges)), judge_labels, fontsize=7)
    for ax in axes.flat[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel("Macro accuracy")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, markeredgewidth=0,
               label=METHOD_LABELS[method])
        for method, color in METHOD_COLORS.items()
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=7)
    outputs.append(_save(fig, output_dir, "accuracy-by-judge-overall"))

    accuracy_mean = (accuracy.groupby(["bench", "target_quartile", "judge", "method"], as_index=False)
                     .accuracy.mean()
                     .pivot(index=["judge", "bench", "target_quartile"],
                            columns="method", values="accuracy").reset_index())
    accuracy_mean["delta"] = 100 * (accuracy_mean.pairwise_glm - accuracy_mean.llm_judge)
    columns = pd.MultiIndex.from_product([benches, quartiles], names=["bench", "target_quartile"])
    delta = (accuracy_mean.pivot(index="judge", columns=["bench", "target_quartile"], values="delta")
             .reindex(index=judges, columns=columns))
    delta_values = np.ma.masked_invalid(delta.to_numpy(float))
    delta_scale = max(5.0, 5 * math.ceil(float(np.nanmax(np.abs(delta_values))) / 5))
    delta_cmap = LinearSegmentedColormap.from_list("accuracy_delta", ["#2a78d6", "#f7f7f3", "#eb6834"])
    fig, ax = plt.subplots(figsize=(3.0, 1.85), layout="constrained")
    image = ax.imshow(delta_values, cmap=delta_cmap,
                      norm=TwoSlopeNorm(vmin=-delta_scale, vcenter=0, vmax=delta_scale), aspect="auto")
    ax.set_xticks(range(len(columns)), [target for _, target in columns],
                  rotation=90, ha="center", fontsize=7)
    ax.set_yticks(range(len(judges)), judge_labels, fontsize=7)
    top = ax.secondary_xaxis("top")
    top.set_xticks([1, 4, 7, 10], [BENCHMARK_LABELS[value] for value in benches])
    top.tick_params(length=0, labelsize=7, pad=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03,
                            ticks=[-delta_scale, 0, delta_scale])
    colorbar.set_label("Accuracy Δ (pp)", fontsize=7)
    outputs.append(_save(fig, output_dir, "accuracy-advantage-by-benchmark"))

    stats = coefficient_statistics(coefficient_points)
    coefficient_judges = coefficient_points.judge.drop_duplicates().tolist()
    overall_stats = (stats[stats.benchmark == "all"].set_index("judge")
                     .reindex(coefficient_judges).reset_index())
    positions = np.arange(len(coefficient_judges))
    fig = plt.figure(figsize=(3.0, 2.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.35], height_ratios=[1, 1.05])
    correlation_ax, quadrant_ax, benchmark_ax = (
        fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    )
    correlation_ax.scatter(overall_stats.pearson, positions, s=9, color="#222222",
                           alpha=0.78, **FILL_ONLY)
    correlation_low = min(0.0, math.floor((float(overall_stats.pearson.min()) - 0.02) * 10) / 10)
    correlation_high = max(0.1, math.ceil((float(overall_stats.pearson.max()) + 0.02) * 10) / 10)
    correlation_ax.axvline(0, color="#999999", lw=0.8, ls="--")
    correlation_ax.set_xlim(correlation_low, correlation_high)
    correlation_ax.set_xticks(sorted(set([correlation_low, 0, correlation_high])))
    correlation_ax.set_ylim(len(positions) - 0.5, -0.5)
    correlation_ax.set_yticks(positions, [SHORT_JUDGE_LABELS.get(value, value) for value in coefficient_judges],
                              fontsize=7)
    correlation_ax.set_xlabel("Pearson r")
    _ordinary(correlation_ax)

    quadrant_names = ["positive_positive", "positive_negative", "negative_positive", "negative_negative"]
    quadrant_labels = ["++", "+−", "−+", "−−"]
    left = np.zeros(len(overall_stats))
    for name, label in zip(quadrant_names, quadrant_labels):
        count = overall_stats[name].to_numpy(int)
        quadrant_ax.barh(positions, count, left=left, height=0.58,
                         color=QUADRANT_COLORS[name], edgecolor="none", label=label)
        for y, start, width in zip(positions, left, count):
            if width >= 8:
                quadrant_ax.text(start + width / 2, y, str(width), ha="center", va="center",
                                 fontsize=7, color="white")
        left += count
    maximum_features = int(overall_stats.n_features.max())
    quadrant_ax.set_xlim(0, maximum_features)
    quadrant_ax.set_xticks([0, maximum_features])
    quadrant_ax.set_ylim(len(positions) - 0.5, -0.5)
    quadrant_ax.set_yticks(positions, [])
    quadrant_ax.set_xlabel("Features")
    quadrant_ax.set_title("Sign quadrants", loc="left", color="#666666", fontsize=7, pad=1)
    _ordinary(quadrant_ax)

    benchmark_stats = stats[stats.benchmark != "all"]
    benchmark_matrix = (benchmark_stats.pivot(index="judge", columns="benchmark", values="pearson")
                        .reindex(index=coefficient_judges, columns=benches))
    benchmark_scale = max(0.1, math.ceil(float(np.nanmax(np.abs(benchmark_matrix))) * 10) / 10)
    coefficient_cmap = LinearSegmentedColormap.from_list(
        "coefficient_alignment", ["#2a78d6", "#f7f7f3", "#eb6834"]
    )
    benchmark_image = benchmark_ax.imshow(
        benchmark_matrix, cmap=coefficient_cmap,
        norm=TwoSlopeNorm(vmin=-benchmark_scale, vcenter=0, vmax=benchmark_scale), aspect="auto",
    )
    benchmark_ax.set_xticks(range(len(benches)), [BENCHMARK_LABELS[value] for value in benches],
                            rotation=90, ha="center", fontsize=7)
    benchmark_ax.set_yticks(range(len(coefficient_judges)),
                            [SHORT_JUDGE_LABELS.get(value, value) for value in coefficient_judges], fontsize=7)
    benchmark_ax.set_title("By-benchmark r · † pooled", loc="left",
                           color="#666666", fontsize=7, pad=1)
    support = (benchmark_stats.pivot(index="judge", columns="benchmark", values="observed")
               .reindex(index=coefficient_judges, columns=benches))
    for row, judge in enumerate(coefficient_judges):
        for column, benchmark in enumerate(benches):
            if not bool(support.loc[judge, benchmark]):
                benchmark_ax.text(column + 0.37, row - 0.22, "†", ha="center", va="center",
                                  fontsize=7, color="#555555")
    colorbar = fig.colorbar(benchmark_image, ax=benchmark_ax, fraction=0.025, pad=0.025,
                            ticks=[-benchmark_scale, 0, benchmark_scale])
    colorbar.set_label("Pearson r", fontsize=7)
    quadrant_handles = [Patch(facecolor=QUADRANT_COLORS[name], edgecolor="none", label=label)
                        for name, label in zip(quadrant_names, quadrant_labels)]
    fig.legend(handles=quadrant_handles, loc="outside lower center", ncol=4,
               frameon=False, fontsize=7)
    outputs.append(_save(fig, output_dir, "coefficient-alignment-summary"))

    pooled = (coefficient_points.groupby(["judge", "feature"], as_index=False)
              .agg(capability=("capability", "mean"),
                   judge_coefficient=("judge_coefficient", "mean"),
                   observed=("observed", "all")))
    maximum = float(pooled[["capability", "judge_coefficient"]].abs().max().max())
    limit = max(0.1, math.ceil(maximum * 10) / 10)
    rows = math.ceil(len(coefficient_judges) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(2.9, 2.8), sharex=True, sharey=True,
                             squeeze=False)
    fig.subplots_adjust(left=0.19, right=0.94, bottom=0.25, top=0.80,
                        wspace=0.38, hspace=0.78)
    stats_index = overall_stats.set_index("judge")
    for index, (ax, judge) in enumerate(zip(axes.flat, coefficient_judges)):
        frame = pooled[pooled.judge == judge]
        _scatter_panel(ax, frame, limit)
        dagger = "†" if not bool(frame.observed.all()) else ""
        row = stats_index.loc[judge]
        ax.set_title(f"{SHORT_JUDGE_LABELS.get(judge, judge)}{dagger}\n"
                     f"r={row.pearson:.2f}\ns={row.same_sign_share:.2f}",
                     loc="left", color="#666666", fontsize=7, pad=1)
        if index % 3:
            ax.tick_params(axis="y", labelleft=False)
        if index // 3 < rows - 1:
            ax.tick_params(axis="x", labelbottom=False)
    for ax in axes.flat[len(coefficient_judges):]:
        ax.set_visible(False)
    fig.supxlabel("Capability coefficient", y=0.09)
    fig.supylabel("Judge coefficient", x=0.02)
    fig.text(0.5, 0.015, "s = same sign share; † pooled", ha="center", va="bottom",
             fontsize=7, color="#666666")
    outputs.append(_save(fig, output_dir, "coefficient-scatter-diagnostic"))

    cosine_summary = _crossed_cosine_summary(cosines)
    overall_cosine = cosine_summary[cosine_summary.benchmark == "all"].merge(
        eci, on="judge", how="inner"
    )
    x_low = 5 * math.floor(float(overall_cosine.eci_ci_low.min()) / 5)
    x_high = 5 * math.ceil(float(overall_cosine.eci_ci_high.max()) / 5)
    y_low = min(0.0, math.floor((float(overall_cosine.lower.min()) - 0.02) * 10) / 10)
    y_high = max(0.1, math.ceil((float(overall_cosine.upper.max()) + 0.02) * 10) / 10)
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.75), layout="constrained",
                             gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].errorbar(overall_cosine.eci, overall_cosine.estimate,
                     xerr=np.vstack((overall_cosine.eci - overall_cosine.eci_ci_low,
                                     overall_cosine.eci_ci_high - overall_cosine.eci)),
                     yerr=np.vstack((overall_cosine.estimate - overall_cosine.lower,
                                     overall_cosine.upper - overall_cosine.estimate)),
                     fmt="none", color="#555555", lw=1, capsize=0)
    axes[0].scatter(overall_cosine.eci, overall_cosine.estimate, s=10, color="#222222",
                    alpha=0.8, zorder=3, **FILL_ONLY)
    for row in overall_cosine.itertuples():
        x_text, y_text, alignment = ECI_LABEL_POSITIONS[row.judge]
        label = ECI_SHORT_LABELS.get(row.judge, row.judge) + ("†" if not row.observed else "")
        axes[0].annotate(label, (row.eci, row.estimate), xytext=(x_text, y_text),
                         textcoords="data", ha=alignment, fontsize=7,
                         arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.5})
    axes[0].axhline(0, color="#999999", lw=0.8, ls="--")
    axes[0].set_xlim(x_low, x_high)
    axes[0].set_xticks([x_low, x_high])
    axes[0].set_ylim(y_low, y_high)
    axes[0].set_yticks([y_low, 0, y_high])
    axes[0].set_xlabel("Judge ECI")
    axes[0].set_ylabel("Coefficient cosine")
    axes[0].set_title("Overall", loc="left", color="#666666", fontsize=7, pad=1)
    _ordinary(axes[0])

    cosine_judges = coefficient_judges
    cosine_matrix = (cosine_summary[cosine_summary.benchmark != "all"]
                     .pivot(index="judge", columns="benchmark", values="estimate")
                     .reindex(index=cosine_judges, columns=benches))
    cosine_image = axes[1].imshow(
        cosine_matrix, cmap=coefficient_cmap,
        norm=TwoSlopeNorm(vmin=min(-0.1, y_low), vcenter=0, vmax=y_high), aspect="auto",
    )
    cosine_support = (cosine_summary[cosine_summary.benchmark != "all"]
                      .pivot(index="judge", columns="benchmark", values="observed")
                      .reindex(index=cosine_judges, columns=benches))
    axes[1].set_xticks(range(len(benches)), [BENCHMARK_LABELS[value] for value in benches],
                       rotation=90, ha="center", fontsize=7)
    axes[1].set_yticks(range(len(cosine_judges)), [
        SHORT_JUDGE_LABELS.get(value, value)
        + ("†" if not bool(cosine_support.loc[value].all()) else "")
        for value in cosine_judges
    ], fontsize=7)
    axes[1].set_title("By benchmark", loc="left", color="#666666", fontsize=7, pad=1)
    for row, judge in enumerate(cosine_judges):
        for column, benchmark in enumerate(benches):
            if not bool(cosine_support.loc[judge, benchmark]):
                axes[1].text(column + 0.35, row - 0.22, "†", ha="center", va="center",
                             fontsize=7, color="#555555")
    colorbar = fig.colorbar(cosine_image, ax=axes[1], fraction=0.04, pad=0.03,
                            ticks=[min(-0.1, y_low), 0, y_high])
    colorbar.set_label("Cosine", fontsize=7)
    outputs.append(_save(fig, output_dir, "cosine-vs-eci"))

    contribution_summary = (contributions.groupby(["judge", "feature"], as_index=False)
                            .contribution.mean())
    extremes = contribution_summary.groupby("feature").contribution.agg(["min", "max"])
    positive = extremes.nlargest(min(6, len(extremes)), "max").index.tolist()
    negative = extremes.nsmallest(min(6, len(extremes)), "min").index.tolist()
    features = positive + [feature for feature in negative if feature not in positive]
    matrix = (contribution_summary.pivot(index="feature", columns="judge", values="contribution")
              .reindex(index=features, columns=coefficient_judges).fillna(0))
    scale = max(1e-6, float(np.nanmax(np.abs(matrix.to_numpy(float)))))
    driver_cmap = LinearSegmentedColormap.from_list("drivers", ["#2a78d6", "#f7f7f3", "#eb6834"])
    fig, ax = plt.subplots(figsize=(2.2, 2.8), layout="constrained")
    image = ax.imshow(matrix, cmap=driver_cmap,
                      norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale), aspect="auto")
    ax.set_xticks(range(len(matrix.columns)),
                  [SHORT_JUDGE_LABELS.get(value, value) for value in matrix.columns],
                  rotation=90, ha="center", fontsize=7)
    ax.set_yticks(range(len(matrix.index)),
                  [textwrap.fill(value.replace("_", " "), 13) for value in matrix.index],
                  fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03,
                            ticks=[-scale, 0, scale])
    colorbar.set_label("Cosine contribution", fontsize=7)
    outputs.append(_save(fig, output_dir, "cosine-feature-drivers"))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=CONFIG.input_root)
    parser.add_argument(
        "--capability-coefficients", type=Path,
        default=ROOT / "judge_results/capability_predictions_glm/interpret-v1/full/coefficients.parquet",
    )
    parser.add_argument(
        "--judge-coefficient-root", type=Path,
        default=ROOT / "judge_results/capability_predictions_glm/judge-comparison-v1/source/judge-folds",
    )
    parser.add_argument("--eci", type=Path, default=ROOT / "external_data/epoch-eci/eci_scores.csv")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "judge_results/capability_predictions_glm/judge-comparison-v1",
    )
    args = parser.parse_args(argv)

    validate_manifest(args.input_root / "manifest.json")
    dictionary = pd.read_parquet(args.input_root / "feature-dictionary.parquet")
    responses = pd.read_parquet(args.input_root / "responses.parquet")
    pairs = pd.read_parquet(args.input_root / "pair.parquet")
    capability = pd.read_parquet(args.capability_coefficients)
    coefficient_paths = sorted(args.judge_coefficient_root.glob(
        "*/reference-coefficients/full.parquet"
    ))
    if len(coefficient_paths) != 16:
        raise ValueError(f"expected 16 judge coefficient folds, found {len(coefficient_paths)}")
    judge_coefficients = pd.concat([
        pd.read_parquet(path).assign(fold=path.parents[1].name)
        for path in coefficient_paths
    ], ignore_index=True)
    accuracy, points, cosines, contributions = build_tables(
        pairs, responses, capability, judge_coefficients, dictionary, CONFIG.model_folds,
    )

    eci_raw = pd.read_csv(args.eci)
    reverse_eci = {model: judge for judge, model in ECI_MODELS.items()}
    eci = (eci_raw[eci_raw.Model.isin(reverse_eci)]
           .assign(judge=lambda frame: frame.Model.map(reverse_eci))
           [["judge", "eci", "eci_ci_low", "eci_ci_high"]])
    tables = args.output / "tables"
    figures = args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(tables / "accuracy-by-fold.csv", index=False)
    points.to_csv(tables / "coefficient-points.csv", index=False)
    coefficient_statistics(points).to_csv(tables / "coefficient-statistics.csv", index=False)
    cosines.to_csv(tables / "cosine-by-fold.csv", index=False)
    cosine_summary = _crossed_cosine_summary(cosines)
    cosine_summary.to_csv(tables / "cosine-summary.csv", index=False)
    driver_summary = (contributions.groupby(["judge", "feature"], as_index=False)
                      .contribution.mean())
    driver_summary["aligning_rank"] = driver_summary.groupby("judge").contribution.rank(
        ascending=False, method="first",
    ).astype(int)
    driver_summary["opposing_rank"] = driver_summary.groupby("judge").contribution.rank(
        ascending=True, method="first",
    ).astype(int)
    driver_summary.to_csv(tables / "cosine-feature-contributions.csv", index=False)
    eci.to_csv(tables / "judge-eci.csv", index=False)
    correlation_rows = []
    for benchmark, frame in cosine_summary.merge(eci, on="judge").groupby("benchmark", sort=False):
        for metric, function in (("pearson", pearsonr), ("spearman", spearmanr)):
            correlation_rows.append({
                "benchmark": benchmark, "metric": metric,
                "correlation": float(function(frame.eci, frame.estimate).statistic),
                "n_judges": len(frame),
            })
    pd.DataFrame(correlation_rows).to_csv(tables / "cosine-vs-eci-correlations.csv", index=False)
    outputs = plot_comparisons(accuracy, points, cosines, contributions, eci, figures)
    print(f"wrote {len(outputs)} figures and comparison tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
