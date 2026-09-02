"""Compare Q1-derived rankings with held-out harder-item rankings."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from .config import MODEL_FOLDS, SOURCE_MANIFEST_SHA256, SOURCE_NAME
from .evaluate import ability_table
from .preprocessing import _model_item_rates


TARGETS = ("Q2", "Q3", "Q4")
BENCHMARK_LABELS = {
    "aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
    "swe_bench_mini": "SWE",
}
JUDGE_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v4-flash-0731": "DeepSeek V4",
    "gemma-4-31b-it": "Gemma 4",
    "gpt-oss-120b": "GPT-OSS",
    "ling-2.6-flash": "Ling",
    "o4-mini": "o4-mini",
}
JUDGE_COLORS = {
    "Llama-3.1-8B-Instruct": "#9ecae1",
    "deepseek-r1-0528": "#6baed6",
    "deepseek-v4-flash-0731": "#4292c6",
    "gemma-4-31b-it": "#2171b5",
    "gpt-oss-120b": "#08519c",
    "ling-2.6-flash": "#08306b",
    "o4-mini": "#2f7fb8",
}
JUDGE_BAR_LABELS = {
    "Llama-3.1-8B-Instruct": "L8", "deepseek-r1-0528": "R1",
    "deepseek-v4-flash-0731": "V4", "gemma-4-31b-it": "G4",
    "gpt-oss-120b": "OSS", "ling-2.6-flash": "Ling", "o4-mini": "o4",
}
BLUE_SHADES = tuple(JUDGE_COLORS.values())
CONDITION_COLORS = {
    "Pooled judges": "#082f5a",
    "Judge GLM L1": "#20a4b8",
    "Judge GLM L2": "#c05a9d",
    "Length only": "#eb6834",
    "Length + surface": "#1baf7a",
    "Deterministic GLM": "#e87ba4",
    "Full pairwise GLM": "#4a3aa7",
    "Q1 accuracy": "#eda100",
}
BAR_LABELS = {
    "Pooled judges": "Pooled", "Judge GLM L1": "J-GLM L1",
    "Judge GLM L2": "J-GLM L2", "Length only": "Length",
    "Length + surface": "Surface", "Full pairwise GLM": "Full GLM",
    "Deterministic GLM": "Det. GLM",
    "Q1 accuracy": "Q1 acc.",
}


def _rank(values: pd.Series) -> pd.Series:
    if len(values) < 2:
        return pd.Series(np.nan, index=values.index)
    return (values.rank(method="average") - 1) / (len(values) - 1)


def _tau_a(left: pd.Series, right: pd.Series) -> float:
    """Kendall tau-a: ties contribute zero and remain in the denominator."""
    if len(left) < 2:
        return np.nan
    x, y = left.to_numpy(float), right.to_numpy(float)
    upper = np.triu_indices(len(x), 1)
    return float(np.sign(x[:, None] - x)[upper] @ np.sign(y[:, None] - y)[upper]
                 / len(upper[0]))


def target_accuracies(
    accuracies: pd.DataFrame, item_quartiles: pd.DataFrame,
) -> pd.DataFrame:
    """Return Q2-Q4 accuracies plus exact item-count-weighted pooled accuracy."""
    selected = accuracies[accuracies.quartile.isin(TARGETS)].copy()
    counts = (item_quartiles[item_quartiles.quartile.isin(TARGETS)]
              .groupby(["bench", "quartile"], as_index=False).size()
              .rename(columns={"size": "item_count"}))
    selected = selected.merge(counts, on=["bench", "quartile"],
                              how="left", validate="many_to_one")
    selected["weighted"] = selected.accuracy * selected.item_count
    pooled = (selected.groupby(["bench", "model"], as_index=False, sort=False)
              .agg(weighted=("weighted", "sum"), item_count=("item_count", "sum")))
    pooled["observed_accuracy"] = pooled.weighted / pooled.item_count
    pooled["target_quartile"] = "Q2-Q4"
    separate = selected.rename(columns={
        "quartile": "target_quartile", "accuracy": "observed_accuracy",
    })
    columns = ["bench", "target_quartile", "model", "observed_accuracy"]
    return pd.concat([separate[columns], pooled[columns]], ignore_index=True)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
        "axes.linewidth": 0.7, "scatter.edgecolors": "none",
        "lines.markeredgewidth": 0,
    })


def _ordinary(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_position(("outward", 5))
        ax.spines[name].set_bounds(0, 1)


def _condition_label(condition: str) -> str:
    if condition.startswith("Judge: "):
        judge = condition.removeprefix("Judge: ")
        return JUDGE_LABELS.get(judge, judge)
    if condition in {"Judge GLM L1", "Judge GLM L2"}:
        return f"{condition}\n(o4 + V4)"
    return "Full GLM" if condition == "Full pairwise GLM" else condition


def _bar_label(condition: str) -> str:
    if condition.startswith("Judge: "):
        judge = condition.removeprefix("Judge: ")
        return JUDGE_BAR_LABELS.get(judge, JUDGE_LABELS.get(judge, judge))
    return BAR_LABELS.get(condition, condition)


def _condition_color(condition: str) -> str:
    if condition.startswith("Judge: "):
        judge = condition.removeprefix("Judge: ")
        return JUDGE_COLORS.get(judge, BLUE_SHADES[sum(judge.encode()) % len(BLUE_SHADES)])
    return CONDITION_COLORS.get(condition, "#777777")


def bar_summary(frame: pd.DataFrame, intervals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return condition taus ordered best to worst with optional bootstrap bounds."""
    rows = []
    for condition, values in frame.groupby("condition", sort=False):
        tau = _tau_a(values.predicted_rank, values.true_rank)
        rows.append({"condition": condition, "tau": tau,
                     "lower": tau, "upper": tau})
    output = pd.DataFrame(rows)
    if intervals is not None:
        output = output.drop(columns=["lower", "upper"]).merge(
            intervals[["condition", "lower", "upper"]],
            on="condition", how="left", validate="one_to_one",
        )
    output["color"] = output.condition.map(_condition_color)
    output["label"] = output.condition.map(_bar_label)
    return output.sort_values(["tau", "condition"], ascending=[False, True],
                              kind="stable").reset_index(drop=True)


def bootstrap_target_taus(
    points: pd.DataFrame,
    responses: pd.DataFrame,
    item_quartiles: pd.DataFrame,
    *,
    n_resamples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Percentile intervals from paired resampling of held-out target items."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rates = _model_item_rates(responses)
    rng = np.random.default_rng(seed)
    rows = []
    for (bench, target), frame in points.groupby(["bench", "target_quartile"], sort=False):
        quartiles = TARGETS if target == "Q2-Q4" else (target,)
        items = item_quartiles[
            (item_quartiles.bench == bench) & item_quartiles.quartile.isin(quartiles)
        ].item.drop_duplicates().tolist()
        if not items:
            raise ValueError(f"no target items for {bench} {target}")
        available = (rates[(rates.bench == bench) & rates.item.isin(items)]
                     .pivot(index="model", columns="item", values="correct_rate"))
        draws = rng.integers(len(items), size=(n_resamples, len(items)))
        for condition, values in frame.groupby("condition", sort=False):
            models = values.model.drop_duplicates().tolist()
            matrix = available.reindex(index=models, columns=items).to_numpy(float)
            sampled = matrix[:, draws]
            counts = np.isfinite(sampled).sum(axis=2)
            observed = np.nansum(sampled, axis=2) / np.where(counts, counts, 1)
            valid = (counts > 0).all(axis=0)
            scores = values.set_index("model").score.reindex(models)
            samples = [_tau_a(scores, pd.Series(observed[:, draw]))
                       for draw in np.flatnonzero(valid)]
            rows.append({
                "bench": bench, "target_quartile": target, "condition": condition,
                "lower": float(np.quantile(samples, 0.025)),
                "upper": float(np.quantile(samples, 0.975)),
                "n_resamples": len(samples),
            })
    return pd.DataFrame(rows)


def plot_rank_scatter(
    points: pd.DataFrame, bench: str, target: str, output_dir: str | Path,
    intervals: pd.DataFrame | None = None,
) -> list[Path]:
    """Plot one compact rank scatter facet per concrete condition."""
    frame = points[(points.bench == bench) & (points.target_quartile == target)].copy()
    available = frame.condition.drop_duplicates().tolist()
    judges = [condition for condition in available if condition.startswith("Judge: ")]
    preferred = ["Pooled judges", "Judge GLM L1", "Judge GLM L2",
                 "Length only", "Length + surface",
                 "Deterministic GLM", "Full pairwise GLM", "Q1 accuracy"]
    conditions = judges + [condition for condition in preferred if condition in available]
    conditions += [condition for condition in available if condition not in conditions]
    if not conditions:
        raise ValueError(f"no rank points for {bench} {target}")
    order = {condition: index for index, condition in enumerate(conditions)}
    frame = (frame.assign(_order=frame.condition.map(order))
             .sort_values("_order", kind="stable").drop(columns="_order"))
    bars = bar_summary(frame, intervals)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    columns, rows = min(3, len(conditions)), math.ceil(len(conditions) / 3)
    fig = plt.figure(figsize=(3.4, min(7.5, 0.95 * rows + 2.7)), layout="constrained")
    scatter_figure, bar_figure = fig.subfigures(2, 1, height_ratios=(5, 2.6))
    axes = scatter_figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    for ax, condition in zip(axes.flat, conditions):
        values = frame[frame.condition == condition]
        color = _condition_color(condition)
        ax.scatter(values.predicted_rank, values.true_rank, s=9, color=color,
                   edgecolors="none", linewidths=0, alpha=0.78, clip_on=False)
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", zorder=0)
        tau = _tau_a(values.predicted_rank, values.true_rank)
        ax.text(0, 1.03, _condition_label(condition), transform=ax.transAxes, color=color,
                fontsize=7, ha="left", va="bottom")
        ax.text(0.04, 0.94, f"τ={tau:.2f}", transform=ax.transAxes,
                color="#555555", fontsize=7, ha="left", va="top")
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 1], ["0", "1"]); ax.set_yticks([0, 1], ["0", "1"])
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_aspect("equal", adjustable="box")
        _ordinary(ax)
    for ax in axes.flat[len(conditions):]:
        ax.remove()
    scatter_figure.supxlabel("Predicted rank")
    scatter_figure.supylabel("True rank")

    grid = bar_figure.add_gridspec(1, 12)
    ax = bar_figure.add_subplot(grid[0, 4:8])
    y = np.arange(len(bars))
    widths = bars.tau.clip(lower=0).to_numpy(float)
    ax.barh(y, widths, color=bars.color, edgecolor="none", linewidth=0)
    lower = bars.lower.fillna(bars.tau).clip(lower=0).to_numpy(float)
    upper = bars.upper.fillna(bars.tau).clip(lower=0).to_numpy(float)
    ax.errorbar(widths, y,
                xerr=np.vstack((widths - np.minimum(lower, widths),
                                np.maximum(upper, widths) - widths)),
                fmt="none", ecolor="#444444", lw=0.8, capsize=0, zorder=3)
    negative = bars.tau.to_numpy(float) < 0
    ax.scatter(np.zeros(negative.sum()), y[negative], marker="<", s=12,
               color=bars.loc[negative, "color"], edgecolors="none", linewidths=0,
               clip_on=False, zorder=4)
    maximum = max(0.2, float(np.nanmax(np.r_[widths, upper])))
    maximum = min(1.0, math.ceil(maximum * 1.18 * 10) / 10)
    ax.set_xlim(0, maximum); ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.set_xticks([0, maximum], ["0", f"{maximum:g}"])
    ax.set_yticks(y, bars.label)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.set_xlabel("Kendall τ-a")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["left"].set_bounds(0, len(bars) - 1)
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines["bottom"].set_bounds(0, maximum)
    for position, value, bound in zip(y, bars.tau, upper):
        x = min(maximum * 0.98, max(0, value, bound) + 0.025 * maximum)
        ax.text(x, position, f"{value:.2f}", ha="right" if x == maximum * 0.98 else "left",
                va="center", fontsize=6, color="#222222")
    stem = f"{bench}-{target.lower()}-ranks"
    paths = [output_dir / f"{stem}{suffix}" for suffix in (".png", ".svg", ".csv")]
    frame.to_csv(paths[2], index=False)
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def load_updated_predictions(root: str | Path) -> pd.DataFrame:
    """Load the 12 complete, pinned cells from the expanded Q1 capability run."""
    root = Path(root)
    expected = {f"{fold}-{target}" for fold, _ in MODEL_FOLDS for target in TARGETS}
    paths = sorted(root.glob("model-fold-*-Q*/predictions.csv"))
    if {path.parent.name for path in paths} != expected:
        raise ValueError("updated GLM root must contain 12 complete fold-target cells")
    frames = []
    expected_source = {
        "name": SOURCE_NAME,
        "design_matrix_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    for path in paths:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest.get("source_manifest", {}).get("source") != expected_source:
            raise ValueError(f"updated GLM cell has wrong source provenance: {path.parent}")
        frame = pd.read_csv(path)
        fold, target = path.parent.name.rsplit("-", 1)
        if set(frame.fold) != {fold} or set(frame.target_quartile) != {target}:
            raise ValueError(f"updated GLM cell identity disagrees with its directory: {path.parent}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def crossfit_glm_rankings(
    q1_pairs: pd.DataFrame,
    coefficients: pd.DataFrame,
    model_folds: tuple[tuple[str, tuple[str, ...]], ...],
    condition: str,
) -> pd.DataFrame:
    """Average eligible cross-fold pair probabilities, then fit one global BT rank."""
    pairs = q1_pairs.reset_index(drop=True).assign(_row=lambda frame: np.arange(len(frame)))
    predictions = []
    if "task" in coefficients:
        selected = coefficients[coefficients.task == "pairwise"]
    elif "model" in coefficients:
        selected = coefficients[coefficients.model == "pairwise_glm"]
    else:
        raise ValueError("coefficient table needs task or model")
    for (fold, target), weights in selected.groupby(["fold", "target_quartile"], sort=False):
        held_out = set(dict(model_folds)[fold])
        eligible = pairs[
            pairs.model_left.isin(held_out) | pairs.model_right.isin(held_out)
        ]
        for bench, frame in eligible.groupby("bench", sort=False):
            local = weights[weights.benchmark == bench]
            features = local.feature.astype(str).tolist()
            if not features:
                continue
            columns = [f"d_{feature}" for feature in features]
            if missing := sorted(set(columns) - set(frame)):
                raise ValueError(f"Q1 pair table is missing GLM features: {missing}")
            predictions.append(pd.DataFrame({
                "_row": frame._row.to_numpy(),
                "target_quartile": target,
                "probability": expit(
                    frame[columns].to_numpy(float) @ local.effective.to_numpy(float)
                ),
            }))
    probabilities = (pd.concat(predictions, ignore_index=True)
                     .groupby(["_row", "target_quartile"], as_index=False).probability.mean())
    pooled = (probabilities.groupby("_row", as_index=False).probability.mean()
              .assign(target_quartile="Q2-Q4"))
    probabilities = pd.concat([probabilities, pooled], ignore_index=True)
    predicted_pairs = probabilities.merge(
        pairs.drop(columns=[column for column in pairs if column.startswith("d_")]),
        on="_row", how="left", validate="many_to_one",
    )
    rankings = []
    for target, frame in predicted_pairs.groupby("target_quartile", sort=False):
        rankings.append(ability_table(frame, "probability", "pairwise_glm").assign(
            target_quartile=target, condition=condition, kind="baseline",
        ))
    return pd.concat(rankings, ignore_index=True)[
        ["bench", "target_quartile", "condition", "kind", "model", "score"]
    ]


def judge_rankings(q1_pairs: pd.DataFrame) -> pd.DataFrame:
    """Fit one BT ranking per judge and one from all Q1 verdicts pooled."""
    rankings = []
    groups = [(f"Judge: {judge}", frame, "judge")
              for judge, frame in q1_pairs.groupby("judge", sort=False)]
    groups.append(("Pooled judges", q1_pairs, "pooled_judge"))
    for condition, frame, kind in groups:
        rankings.append(ability_table(frame, "pick_left", "llm_judge").assign(
            condition=condition, kind=kind,
        ))
    return pd.concat(rankings, ignore_index=True)[
        ["bench", "condition", "kind", "model", "score"]
    ]


def saved_pair_rankings(predictions: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Collapse duplicate held-out pair predictions and fit one BT ranking."""
    keys = ["bench", "item", "model_left", "model_right"] + [
        column for column in (
            "epoch_left", "epoch_right", "log_left", "log_right",
            "judge", "judge_sample", "judge_epoch", "judge_log",
        )
        if column in predictions
    ]
    if missing := sorted({*keys, "probability"} - set(predictions)):
        raise ValueError(f"saved pair predictions missing columns: {missing}")
    pairs = predictions.groupby(keys, as_index=False, sort=False).probability.mean()
    return (ability_table(pairs, "probability", "pairwise_glm")
            .assign(condition=condition, kind="baseline")
            [["bench", "condition", "kind", "model", "score"]])


def linear_ensemble_rankings(
    pairs: pd.DataFrame,
    coefficient_tables: list[pd.DataFrame],
    condition: str,
) -> pd.DataFrame:
    """Apply every linear fit to every pair, average probabilities, then BT-rank."""
    if not coefficient_tables:
        raise ValueError("coefficient_tables must not be empty")
    pairs = pairs.reset_index(drop=True)
    probability = np.zeros(len(pairs), dtype=float)
    for coefficients in coefficient_tables:
        predicted = np.full(len(pairs), np.nan)
        for bench, indices in pairs.groupby("bench", sort=False).groups.items():
            weights = coefficients[coefficients.benchmark == bench]
            features = weights.feature.astype(str).tolist()
            columns = [f"d_{feature}" for feature in features]
            if missing := sorted(set(columns) - set(pairs)):
                raise ValueError(f"pair table is missing linear-model features: {missing}")
            predicted[np.asarray(indices)] = expit(
                pairs.loc[indices, columns].to_numpy(float)
                @ weights.effective.to_numpy(float)
            )
        if not np.isfinite(predicted).all():
            raise ValueError("linear model did not predict every pair")
        probability += predicted
    output = pairs.copy()
    output["probability"] = probability / len(coefficient_tables)
    return saved_pair_rankings(output, condition)


def global_rank_points(
    rankings: pd.DataFrame, target_accuracies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match model coverage and score one global ranking per benchmark and target."""
    points = rankings.merge(
        target_accuracies,
        on=["bench", "target_quartile", "model"], how="inner", validate="many_to_one",
    )
    matched = []
    for _, frame in points.groupby(["bench", "target_quartile"], sort=False):
        groups = list(frame.groupby("condition", sort=False))
        core = [(condition, group) for condition, group in groups
                if not condition.startswith("Judge GLM ")]
        model_sets = [set(group.model) for _, group in (core or groups)]
        common = set.intersection(*model_sets) if model_sets else set()
        matched.append(frame[frame.model.isin(common)])
    points = pd.concat(matched, ignore_index=True) if matched else points.iloc[0:0].copy()
    group = ["bench", "target_quartile", "condition", "kind"]
    points["predicted_rank"] = points.groupby(group, sort=False).score.transform(_rank)
    points["true_rank"] = points.groupby(group, sort=False).observed_accuracy.transform(_rank)
    taus = []
    for values, frame in points.groupby(group, sort=False):
        taus.append(dict(zip(group, values)) | {
            "tau": _tau_a(frame.score, frame.observed_accuracy),
            "n_models": len(frame),
        })
    return points, pd.DataFrame(taus)


def summarize_global_taus(
    taus: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add mean and pooled-target-fixed judge rows to global tau estimates."""
    pooled = taus[(taus.kind == "judge") & (taus.target_quartile == "Q2-Q4")]
    selections = []
    for bench, frame in pooled.groupby("bench", sort=False):
        ordered = frame.dropna(subset=["tau"]).sort_values(["tau", "condition"])
        if not ordered.empty:
            selections.append({
                "bench": bench,
                "best_judge": ordered.iloc[-1].condition,
                "worst_judge": ordered.iloc[0].condition,
            })
    selections = pd.DataFrame(selections, columns=["bench", "best_judge", "worst_judge"])
    judges = taus[taus.kind == "judge"]
    mean = (judges.groupby(["bench", "target_quartile"], as_index=False, sort=False)
            .agg(tau=("tau", "mean"), n_models=("n_models", "mean"))
            .assign(condition="Mean judge", kind="judge_summary"))
    selected = []
    for selection in selections.itertuples(index=False):
        for label, judge in (("Best judge", selection.best_judge),
                             ("Worst judge", selection.worst_judge)):
            rows = judges[(judges.bench == selection.bench) & (judges.condition == judge)].copy()
            rows["selected_judge"] = judge
            rows["condition"] = label
            rows["kind"] = "judge_summary"
            selected.append(rows)
    summary = pd.concat([taus, mean, *selected], ignore_index=True, sort=False)
    wide = (summary.pivot(index=["target_quartile", "condition"], columns="bench", values="tau")
            .rename_axis(columns=None).reset_index())
    return summary, selections, wide


def _sort_wide(wide: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "Best judge", "Pooled judges", "Mean judge", "Worst judge",
        "Judge GLM L1", "Judge GLM L2", "Length only", "Length + surface", "Deterministic GLM",
        "Full pairwise GLM", "Q1 accuracy",
    ]
    conditions = preferred + sorted(set(wide.condition) - set(preferred))
    order = {(target, condition): target_index * len(conditions) + condition_index
             for target_index, target in enumerate((*TARGETS, "Q2-Q4"))
             for condition_index, condition in enumerate(conditions)}
    return (wide.assign(_order=[order.get((row.target_quartile, row.condition), len(order))
                               for row in wide.itertuples()])
            .sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True))


def write_global_report(
    rankings: pd.DataFrame,
    target_accuracy: pd.DataFrame,
    output_dir: str | Path,
    intervals: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write global model-rank points, tau tables, and benchmark-target scatters."""
    output_dir = Path(output_dir)
    tables, figures = output_dir / "tables", output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    points, taus = global_rank_points(rankings, target_accuracy)
    summary, selections, wide = summarize_global_taus(taus)
    wide = _sort_wide(wide.rename(columns=BENCHMARK_LABELS))
    outputs = {
        "rank_points": tables / "rank-points.csv",
        "taus": tables / "taus.csv",
        "tau_summary": tables / "tau-summary.csv",
        "selected_judges": tables / "selected-judges.csv",
        "tau_wide": tables / "tau-wide.csv",
    }
    if intervals is not None:
        outputs["bootstrap_intervals"] = tables / "bootstrap-tau-intervals.csv"
    for table, path in (
        (points, outputs["rank_points"]), (taus, outputs["taus"]),
        (summary, outputs["tau_summary"]), (selections, outputs["selected_judges"]),
        (wide, outputs["tau_wide"]),
    ):
        table.to_csv(path, index=False)
    if intervals is not None:
        intervals.to_csv(outputs["bootstrap_intervals"], index=False)
    for bench in points.bench.drop_duplicates():
        for target in (*TARGETS, "Q2-Q4"):
            if ((points.bench == bench) & (points.target_quartile == target)).any():
                selected = (None if intervals is None else intervals[
                    (intervals.bench == bench) & (intervals.target_quartile == target)
                ])
                plot_rank_scatter(points, bench, target, figures, selected)
    return outputs
