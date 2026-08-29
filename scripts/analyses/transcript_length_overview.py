"""Transcript-length diagnostics across the current judging benchmarks.

Usage:
  uv run python scripts/analyses/transcript_length_overview.py
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import value_to_float
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import t, ttest_rel

from plots import INK, INK2, MUTED, PALETTE, plt, style
from stats import wilson

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "interactive_dashboards" / "transcript_viewer" / "data"
OUT = ROOT / "judge_results" / "transcript-length-overview" / "overview"
GUIDANCE_OUT = OUT.with_name("length_guidance")
CANONICAL_DATA = Path("/Volumes/T7/small-judge/derived/transcript-length-canonical")
PAIR_CONFIG = ROOT / "configs" / "experiments" / "pairs"
BINS = np.arange(0, 101, 10)
CONDITIONS = ("Stronger longer", "Stronger shorter")
COND_COLOR = {"Stronger longer": PALETTE[0], "Stronger shorter": PALETTE[1]}
COND_STYLE = {"Stronger longer": "-", "Stronger shorter": "--"}
DELTA_COLOR, MODEL_COLOR, LENGTH_COLOR = PALETTE[6], PALETTE[2], PALETTE[3]
P1_COLOR, RESIDUAL_COLOR = PALETTE[4], PALETTE[6]
SPECS = {
    "GPQA": {
        "datasets": ("data-expansion-gpqa", "hf-judging-2026-08-17-gpqa"),
        "scores": (ROOT / "data/gpqa/63793eb6",
                   ROOT / "data/hf-judging-2026-08-17/benchmark-scores/gpqa/b0a0d9c9"),
    },
    "SWE-bench Mini": {
        "datasets": ("swe-mini-pilot",),
        "scores": (ROOT / "data/swe_bench_mini/477dc96c",),
    },
    "Intercode": {
        "datasets": ("intercode-pilot",),
        "scores": (ROOT / "data/intercode/463eb861",),
    },
}
JUDGE_ALIASES = {
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
    "gemma4-31b": "gemma-4-31b-it",
    "ling": "ling-2.6-flash",
    "llama3.1-8b": "Llama-3.1-8B-Instruct",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "svg.fonttype": "none", "lines.linewidth": 1,
})


def mean_ci(values):
    """Mean and t-based 95% CI across replication units."""
    mean = statistics.fmean(values)
    if len(values) < 3:
        return mean, math.nan, math.nan
    half = t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def bounded_ci(low, high):
    """Clip a probability CI without turning a missing CI into a fake one."""
    if not np.isfinite(low):
        return math.nan, math.nan
    return max(0, low), min(1, high)


def symmetric_limit(values):
    """Round a per-panel symmetric limit up to the next quarter."""
    maximum = max(abs(float(value)) for value in values)
    return min(1, max(0.25, math.ceil(maximum * 4) / 4))


def log_axis_limits(low, high):
    """Readable per-panel log bounds and ticks, including both endpoints."""
    lower = 10 ** math.floor(math.log10(low))
    decade = 10 ** math.floor(math.log10(high))
    upper = next(multiplier * decade for multiplier in (1, 2, 3, 5, 10)
                 if multiplier * decade >= high)
    ticks = tuple(10 ** exponent for exponent in range(
        math.floor(math.log10(lower)), math.floor(math.log10(upper)) + 1))
    if not math.isclose(ticks[-1], upper):
        ticks += (upper,)
    return lower, upper, ticks


def length_accuracy_ci(values):
    """Mean and Wilson interval for a length heuristic with 0.5-credit ties."""
    values = np.asarray(values, float)
    low, high = wilson(values.sum(), len(values))
    return float(values.mean()), low, high


def median_ci(values, clusters, n_boot=2000, seed=0):
    """Median and cluster-bootstrap 95% CI over benchmark items."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    groups = [values[clusters == group] for group in np.unique(clusters)]
    median = float(np.median(values))
    if len(groups) < 3:
        return median, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
             for _ in range(n_boot)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return median, float(low), float(high)


def extract_pair_cells(items, lengths, strong, weak, gap, pair):
    """Counterbalance each judged item and classify which model is longer."""
    rows = []
    for item, judges in items.items():
        len_strong = lengths.get(strong, {}).get(item)
        len_weak = lengths.get(weak, {}).get(item)
        if len_strong is None or len_weak is None:
            continue
        condition = ("Stronger longer" if len_strong > len_weak else
                     "Stronger shorter" if len_strong < len_weak else "Tie")
        for judge, cell in judges.items():
            picks = [order["picked_major"] for order in cell["orders"]
                     if order is not None and order["picked_major"] is not None]
            if picks:
                rows.append({
                    "judge": judge, "pair": pair, "item": item, "gap": gap,
                    "condition": condition,
                    "judge_accuracy": statistics.fmean(picks),
                    "length_accuracy": (float(len_strong > len_weak)
                                        if len_strong != len_weak else 0.5),
                    "chose_longer": (statistics.fmean(picks) if len_strong > len_weak else
                                     1 - statistics.fmean(picks) if len_strong < len_weak else None),
                })
    return rows


def _score_counts(roots):
    """Model -> (correct, scored) from stored benchmark runs."""
    totals = defaultdict(lambda: [0.0, 0])
    to_float = value_to_float()
    epochs = [epoch for root in roots for epoch in root.glob("*/*") if epoch.is_dir()]
    rescored = {epoch.parent.name.removesuffix("-rescored")
                for epoch in epochs if epoch.parent.name.endswith("-rescored")}
    for epoch in epochs:
        source_model = epoch.parent.name
        model = source_model.removesuffix("-rescored")
        if model in rescored and source_model == model:
            continue
        logs_path = epoch / "logs.json"
        if logs_path.exists():
            for log in json.loads(logs_path.read_text()).values():
                if log.get("status") != "success" or log.get("invalidated") or not log.get("results"):
                    continue
                score = next((s for s in log["results"]["scores"]
                              if {"accuracy", "mean"} & set(s["metrics"])), None)
                if score is None:
                    continue
                metric = score["metrics"].get("accuracy") or score["metrics"].get("mean")
                n = int(log["results"].get("total_samples") or score["scored_samples"])
                totals[model][0] += float(metric["value"]) * n
                totals[model][1] += n
            continue
        candidates = []
        for path in epoch.glob("*.eval"):
            if path.name.startswith("._"):
                continue
            log = read_eval_log(str(path))
            values = []
            for sample in log.samples or []:
                if not sample.scores:
                    continue
                converted = []
                for score in sample.scores.values():
                    value = score.value.get("score", 0) if isinstance(score.value, dict) else score.value
                    converted.append(to_float(value))
                values.append(float(any(value >= 1 for value in converted)))
            if values:
                candidates.append(values)
        if candidates:
            values = max(candidates, key=len)
            totals[model][0] += sum(values)
            totals[model][1] += len(values)
    return totals


def _pair_model(name):
    return (name.lower().replace("nvidia-", "").replace("-30b-a3b-bf16", "")
            .replace("-bf16", "").replace("gemma-4-", "gemma4-"))


def _canonical_pairs():
    pairs = set()
    for path in PAIR_CONFIG.glob("*.yml"):
        models = re.findall(r"models/[^/]+/([^/]+)\.yml", path.read_text())
        if len(models) == 2:
            pairs.add(frozenset(map(_pair_model, models)))
    return pairs


def _filter_canonical(bench):
    canonical = _canonical_pairs()
    keep = bench["cells"].pair.map(
        lambda pair: frozenset(_pair_model(model) for model in pair.split(" vs ")) in canonical)
    cells = bench["cells"][keep].copy()
    models = {model for pair in cells.pair.unique() for model in pair.split(" vs ")}
    return {"cells": cells,
            "lengths": bench["lengths"][bench["lengths"].model.isin(models)].copy(),
            "models": bench["models"][bench["models"].model.isin(models)].copy()}


def _load_canonical(name):
    if not CANONICAL_DATA.exists():
        raise FileNotFoundError(f"canonical HF tables not found: {CANONICAL_DATA}")
    cells = pd.read_parquet(CANONICAL_DATA / "cells.parquet")
    lengths = pd.read_parquet(CANONICAL_DATA / "lengths.parquet")
    models = pd.read_csv(CANONICAL_DATA / "models.csv")
    cells = cells[cells.benchmark == name].copy()
    lengths = lengths[lengths.benchmark == name].copy()
    models = models[models.benchmark == name].copy()
    intervals = [wilson(int(round(row.accuracy * row.n_trials)), int(row.n_trials))
                 for row in models.itertuples()]
    models["accuracy_low"] = [low for low, _ in intervals]
    models["accuracy_high"] = [high for _, high in intervals]
    return {"cells": cells, "lengths": lengths, "models": models}


def load_benchmarks(viewer=VIEWER):
    """Load the all-pair GPQA row and canonical comparison rows."""
    benchmarks = {}
    for benchmark, spec in SPECS.items():
        cells, length_rows, labels, fallback = [], [], {}, {}
        for dataset_id in spec["datasets"]:
            data = viewer / dataset_id
            index = json.loads((data / "index.json").read_text())
            for model, meta in index["models"].items():
                if meta.get("accuracy") is not None:
                    # Epoch GPQA accuracies pool four 198-item epochs; those
                    # source score logs are external to this repository.
                    n = 792 if dataset_id == "data-expansion-gpqa" and meta["arm"] == "B" else None
                    fallback[model] = (meta["accuracy"], n)
            lengths = {}
            for path in (data / "transcripts").glob("*.json"):
                texts = json.loads(path.read_text())
                lengths[path.stem] = {item: len(text) for item, text in texts.items()}
                model = path.stem.split("__ep", 1)[0]
                labels[model] = index["models"].get(model, {}).get("label", model)
                for item, text in texts.items():
                    base_item = item.split("__ep", 1)[0]
                    cluster_prefix = dataset_id if benchmark == "AIME" else benchmark
                    length_rows.append({
                        "model": model,
                        "item": f"{dataset_id}:{path.stem}:{item}",
                        "cluster": f"{cluster_prefix}:{base_item}",
                        "length": len(text),
                    })
            for path in (data / "pairs").glob("*.json"):
                strong, weak, _ = path.stem.rsplit("__", 2)
                acc_strong = index["models"].get(strong, {}).get("accuracy")
                acc_weak = index["models"].get(weak, {}).get("accuracy")
                if acc_strong is None or acc_weak is None or acc_strong <= acc_weak:
                    continue
                rows = extract_pair_cells(json.loads(path.read_text()), lengths, strong, weak,
                                          acc_strong - acc_weak, f"{strong} vs {weak}")
                for row in rows:
                    row["judge"] = JUDGE_ALIASES.get(row["judge"], row["judge"])
                    row["item"] = f"{dataset_id}:{row['item']}"
                cells.extend(rows)
        counts = _score_counts(spec["scores"])
        models = []
        for model, group in pd.DataFrame(length_rows).groupby("model"):
            correct, n = counts.get(model, (0, 0))
            if not n:
                accuracy, n = fallback.get(model, (math.nan, None))
                if n is None:
                    continue
                correct = accuracy * n
            accuracy = correct / n
            low, high = wilson(int(round(correct)), n)
            models.append({
                "model": model, "label": labels.get(model, model),
                "accuracy": accuracy, "accuracy_low": low, "accuracy_high": high,
                "n_trials": n,
            })
        benchmarks[benchmark] = {
            "cells": pd.DataFrame(cells),
            "lengths": pd.DataFrame(length_rows),
            "models": pd.DataFrame(models),
        }
    return {
        "GPQA (all)": benchmarks["GPQA"],
        "GPQA (canonical)": _load_canonical("GPQA (canonical)"),
        "SWE-bench Mini": _filter_canonical(benchmarks["SWE-bench Mini"]),
        "Intercode": _filter_canonical(benchmarks["Intercode"]),
        "AIME (canonical)": _load_canonical("AIME (canonical)"),
    }


def filter_judges(benchmarks, excluded=("llama", "ling")):
    """Copy benchmark tables while dropping named judge families only."""
    filtered = {}
    for name, bench in benchmarks.items():
        drop = bench["cells"].judge.str.lower().map(
            lambda judge: any(token in judge for token in excluded))
        filtered[name] = {**bench, "cells": bench["cells"][~drop].copy()}
    return filtered


def _paired_curve(cells):
    data = cells.copy()
    data["bin"] = pd.cut(data.gap * 100, BINS, labels=(BINS[:-1] + BINS[1:]) / 2,
                         right=False, include_lowest=True)
    pair_rates = (data.dropna(subset=["bin"])
                  .groupby(["judge", "bin", "condition", "pair"], observed=True)
                  .judge_accuracy.mean().reset_index())
    judge_rates = (pair_rates.groupby(["judge", "bin", "condition"], observed=True)
                   .judge_accuracy.mean().unstack("condition"))
    paired = judge_rates.dropna(subset=list(CONDITIONS)).reset_index()
    return paired[paired.groupby("bin", observed=True).judge.transform("nunique") >= 3]


def _paired_summary(cells):
    pair_rates = (cells[cells.gap > 0.05]
                  .groupby(["judge", "condition", "pair"]).judge_accuracy.mean().reset_index())
    judge_rates = (pair_rates.groupby(["judge", "condition"]).judge_accuracy.mean()
                   .unstack("condition"))
    return judge_rates.dropna(subset=list(CONDITIONS))


def paired_significance(summary):
    """Two-sided paired t-test across judges, rendered as significance stars."""
    if len(summary) < 3:
        return ""
    p = ttest_rel(summary[CONDITIONS[0]], summary[CONDITIONS[1]]).pvalue
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _model_stats(bench, n_boot):
    rows = []
    for seed, model in enumerate(bench["models"].itertuples(index=False)):
        lengths = bench["lengths"][bench["lengths"].model == model.model]
        median, low, high = median_ci(lengths.length, lengths.cluster, n_boot, seed)
        rows.append({**model._asdict(), "median": median / 1000,
                     "median_low": low / 1000, "median_high": high / 1000})
    return pd.DataFrame(rows)


def length_only_pair_stats(cells, min_gap=0.05):
    """Length-only versus real-judge accuracy for pairs above min_gap."""
    cells = cells[cells.gap > min_gap]
    item_length = cells.groupby(["pair", "item"]).length_accuracy.first().reset_index()
    judged = cells.groupby(["pair", "judge"]).judge_accuracy.mean().reset_index()
    rows = []
    for pair, items in item_length.groupby("pair"):
        judge_values = judged[judged.pair == pair].judge_accuracy.to_numpy()
        if not len(judge_values):
            continue
        x, xlow, xhigh = length_accuracy_ci(items.length_accuracy)
        y, ylow, yhigh = mean_ci(judge_values)
        rows.append({"pair": pair, "length_accuracy": x, "length_low": xlow,
                     "length_high": xhigh, "judge_accuracy": y,
                     "judge_low": ylow, "judge_high": yhigh})
    return pd.DataFrame(rows)


def length_residual_stats(cells, min_gap=0.05):
    """Judge×pair preference for model 1 and residual from length-only choice."""
    cells = cells[cells.gap > min_gap]
    length = (cells.groupby(["pair", "item"]).length_accuracy.first()
              .groupby("pair").mean().rename("length_accuracy"))
    judged = cells.groupby(["pair", "judge"]).judge_accuracy.mean().rename("judge_accuracy")
    gap = cells.groupby("pair").gap.first()
    rows = judged.reset_index().join(length, on="pair").join(gap, on="pair")
    rows["residual"] = rows.judge_accuracy - rows.length_accuracy
    return rows


def judge_color_map(benchmarks):
    """Stable project-palette colour per judge across all benchmark panels."""
    judges = list(dict.fromkeys(
        judge for bench in benchmarks.values() for judge in bench["cells"].judge.unique()))
    colors = list(PALETTE)
    for color in plt.get_cmap("tab20").colors:
        color = to_hex(color)
        if color not in colors:
            colors.append(color)
    if len(judges) > len(colors):
        raise ValueError("not enough distinct judge colours")
    return dict(zip(judges, colors))


def render_overview(benchmarks, out=OUT, n_boot=2000):
    """Render the requested benchmark-row by analysis-column overview."""
    curves = {name: _paired_curve(bench["cells"]) for name, bench in benchmarks.items()}
    summaries = {name: _paired_summary(bench["cells"]) for name, bench in benchmarks.items()}
    model_stats = {name: _model_stats(bench, n_boot) for name, bench in benchmarks.items()}
    residual_stats = {name: length_residual_stats(bench["cells"])
                      for name, bench in benchmarks.items()}
    judge_colors = judge_color_map(benchmarks)

    fig = plt.figure(figsize=(8.2, 10.2))
    outer = fig.add_gridspec(6, len(benchmarks),
                             height_ratios=(1.25, 1, 1, 1, 1, 1),
                             hspace=0.62, wspace=0.48)
    top_axes = []
    label_col = len(benchmarks) // 2
    for col, (name, bench) in enumerate(benchmarks.items()):
        inner = outer[0, col].subgridspec(2, 1, height_ratios=(2, 1), hspace=0.22)
        ax_gap = fig.add_subplot(inner[0])
        ax_delta = fig.add_subplot(inner[1], sharex=ax_gap)
        ax_bar = fig.add_subplot(outer[1, col])
        ax_models = fig.add_subplot(outer[2, col])
        ax_length = fig.add_subplot(outer[3, col])
        ax_p1 = fig.add_subplot(outer[4, col])
        ax_residual = fig.add_subplot(outer[5, col])
        top_axes.append(ax_gap)

        paired = curves[name]
        delta_values = paired[CONDITIONS[0]] - paired[CONDITIONS[1]]
        delta_lim = symmetric_limit(delta_values)
        for condition in CONDITIONS:
            xs, means, lows, highs = [], [], [], []
            for x, group in paired.groupby("bin", observed=True):
                mean, low, high = mean_ci(group[condition].to_numpy())
                low, high = bounded_ci(low, high)
                xs.append(float(x)); means.append(mean); lows.append(low); highs.append(high)
            ax_gap.plot(xs, means, color=COND_COLOR[condition], ls=COND_STYLE[condition],
                        marker="o", ms=3.2, lw=1, zorder=3,
                        mfc=(COND_COLOR[condition] if condition == CONDITIONS[0] else "white"),
                        mec=COND_COLOR[condition], mew=0.8)
            valid = np.isfinite(lows)
            if valid.any():
                ax_gap.fill_between(np.asarray(xs)[valid], np.asarray(lows)[valid],
                                    np.asarray(highs)[valid], color=COND_COLOR[condition],
                                    alpha=0.14, lw=0)
        for _judge, group in paired.groupby("judge"):
            ax_delta.plot(group.bin.astype(float), group[CONDITIONS[0]] - group[CONDITIONS[1]],
                          color=DELTA_COLOR, alpha=0.24, lw=0.55, marker="o", ms=1.5)
        xs, means, lows, highs = [], [], [], []
        for x, group in paired.groupby("bin", observed=True):
            mean, low, high = mean_ci((group[CONDITIONS[0]] - group[CONDITIONS[1]]).to_numpy())
            xs.append(float(x)); means.append(mean); lows.append(low); highs.append(high)
        ax_delta.plot(xs, means, color=DELTA_COLOR, lw=1, marker="o", ms=2.8, zorder=3)
        valid = np.isfinite(lows)
        if valid.any():
            ax_delta.fill_between(np.asarray(xs)[valid], np.asarray(lows)[valid],
                                  np.asarray(highs)[valid], color=DELTA_COLOR, alpha=0.14, lw=0)
        ax_delta.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax_gap.set_ylim(0, 1); ax_gap.set_yticks((0, 0.5, 1))
        ax_delta.set_ylim(-delta_lim, delta_lim); ax_delta.set_yticks((-delta_lim, 0, delta_lim))
        ax_delta.set_xlim(0, 100); ax_delta.set_xticks((0, 50, 100))
        ax_gap.tick_params(labelbottom=False)
        if col == 0:
            ax_gap.set_ylabel("Judge acc."); ax_delta.set_ylabel("Δ acc.")
        if col == label_col:
            ax_delta.set_xlabel("Benchmark gap (pp)")
        style(ax_gap); style(ax_delta)

        summary = summaries[name]
        for x, condition in enumerate(CONDITIONS):
            values = summary[condition].to_numpy()
            mean, low, high = mean_ci(values)
            filled = condition == CONDITIONS[0]
            ax_bar.bar(x, mean, width=0.62,
                       facecolor=COND_COLOR[condition] if filled else "white",
                       edgecolor=COND_COLOR[condition], linewidth=0.9, zorder=2)
            if np.isfinite(low):
                ax_bar.errorbar(x, mean, yerr=[[mean - max(0, low)], [min(1, high) - mean]],
                                fmt="none", color=COND_COLOR[condition], lw=0.8,
                                capsize=0, zorder=4)
        jitter = np.linspace(-0.10, 0.10, len(summary)) if len(summary) else []
        for offset, (_, values) in zip(jitter, summary.iterrows()):
            ax_bar.plot([offset, 1 + offset], values[list(CONDITIONS)], color=DELTA_COLOR,
                        alpha=0.22, lw=0.45, zorder=3)
            ax_bar.scatter(offset, values[CONDITIONS[0]], s=8,
                           facecolor=COND_COLOR[CONDITIONS[0]], edgecolor="white",
                           linewidth=0.4, zorder=4)
            ax_bar.scatter(1 + offset, values[CONDITIONS[1]], s=8,
                           facecolor="white", edgecolor=COND_COLOR[CONDITIONS[1]],
                           linewidth=0.7, zorder=4)
        if marker := paired_significance(summary):
            transform = ax_bar.get_xaxis_transform()
            ax_bar.plot([0, 0, 1, 1], [0.90, 0.93, 0.93, 0.90], color=MUTED,
                        lw=0.7, transform=transform, clip_on=False)
            ax_bar.text(0.5, 0.945, marker, transform=transform, ha="center",
                        va="bottom", color=INK2, fontsize=6.5)
        ax_bar.set_ylim(0, 1); ax_bar.set_yticks((0, 0.5, 1))
        if col == 0:
            ax_bar.set_ylabel("Judge acc.")
        ax_bar.set_xticks((0, 1), ("Strong\nlonger", "Strong\nshorter"))
        style(ax_bar)

        models = model_stats[name]
        length_low = models.median_low.dropna()
        length_high = models.median_high.dropna()
        ymin, ymax, length_ticks = log_axis_limits(
            min(length_low.min(), models["median"].min()),
            max(length_high.max(), models["median"].max()))
        for model in models.itertuples(index=False):
            ax_models.errorbar(model.accuracy, model.median,
                               xerr=[[model.accuracy - model.accuracy_low],
                                     [model.accuracy_high - model.accuracy]],
                               yerr=[[model.median - model.median_low],
                                     [model.median_high - model.median]],
                               fmt="o", ms=3.2, color=MODEL_COLOR, ecolor=MODEL_COLOR,
                               mfc=MODEL_COLOR, mec="white", mew=0.45,
                               elinewidth=0.6, capsize=0, alpha=0.88, zorder=3)
        ax_models.set_xlim(0, 1); ax_models.set_xticks((0, 0.5, 1))
        ax_models.set_yscale("log"); ax_models.set_ylim(ymin, ymax)
        ax_models.set_yticks(length_ticks)
        ax_models.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        if col == 0:
            ax_models.set_ylabel("Median length, correct\nitems (k chars)")
        if col == label_col:
            ax_models.set_xlabel("Benchmark acc.")
        style(ax_models)

        for point in length_only_pair_stats(bench["cells"]).itertuples(index=False):
            x, xlow, xhigh = point.length_accuracy, point.length_low, point.length_high
            y, ylow, yhigh = point.judge_accuracy, point.judge_low, point.judge_high
            yerr = None if not np.isfinite(ylow) else [
                [max(0, y - max(0, ylow))], [max(0, min(1, yhigh) - y)]]
            ax_length.errorbar(x, y,
                               xerr=[[max(0, x - xlow)], [max(0, xhigh - x)]], yerr=yerr,
                               fmt="o", ms=3.0, color=LENGTH_COLOR, ecolor=LENGTH_COLOR,
                               mfc=LENGTH_COLOR, mec="white", mew=0.4,
                               elinewidth=0.5, capsize=0, alpha=0.62, zorder=3)
        ax_length.plot((0, 1), (0, 1), color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax_length.set_xlim(0, 1); ax_length.set_ylim(0, 1)
        ax_length.set_xticks((0, 0.5, 1)); ax_length.set_yticks((0, 0.5, 1))
        if col == 0:
            ax_length.set_ylabel("Judge acc.")
        if col == label_col:
            ax_length.set_xlabel("Length-only acc.")
        style(ax_length)

        points = residual_stats[name]
        residual_lim = symmetric_limit(points.residual)
        for judge, group in points.groupby("judge"):
            color = judge_colors[judge]
            ax_p1.scatter(group.length_accuracy, group.judge_accuracy, s=7,
                          color=color, edgecolor="white", linewidth=0.3,
                          alpha=0.55, zorder=3)
            ax_residual.scatter(group.gap * 100, group.residual, s=7,
                                color=color, edgecolor="white", linewidth=0.3,
                                alpha=0.55, zorder=3)
        ax_p1.plot((0, 1), (0, 1), color=MUTED, lw=0.8,
                   ls=(0, (3, 2)), zorder=1)
        ax_p1.set_xlim(0, 1); ax_p1.set_ylim(0, 1)
        ax_p1.set_xticks((0, 0.5, 1)); ax_p1.set_yticks((0, 0.5, 1))
        if col == 0:
            ax_p1.set_ylabel("Judge P(stronger model)")
        if col == label_col:
            ax_p1.set_xlabel("Length-only P(stronger model)")
        style(ax_p1)

        ax_residual.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax_residual.set_xlim(0, 100); ax_residual.set_xticks((0, 50, 100))
        ax_residual.set_ylim(-residual_lim, residual_lim)
        ax_residual.set_yticks((-residual_lim, 0, residual_lim))
        if col == 0:
            ax_residual.set_ylabel("Δ P(stronger model)")
        if col == label_col:
            ax_residual.set_xlabel("Benchmark gap (pp)")
        style(ax_residual)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.955, bottom=0.15)
    headings = ("GPQA\nall", "GPQA\ncanonical", "SWE-bench\nMini",
                "Intercode", "AIME\ncanonical")
    for ax, heading in zip(top_axes, headings):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.985, heading, ha="center", va="top",
                 color=INK2, fontsize=8, fontweight="bold")
    handles = [
        Line2D([], [], color=COND_COLOR[c], ls=COND_STYLE[c], lw=1, marker="o", ms=3,
               mfc=COND_COLOR[c] if c == CONDITIONS[0] else "white",
               mec=COND_COLOR[c], label=c)
        for c in CONDITIONS
    ] + [
        Line2D([], [], color=DELTA_COLOR, alpha=0.35, lw=0.55, marker="o", ms=2,
               label="Per judge"),
        Line2D([], [], color=DELTA_COLOR, lw=1, marker="o", ms=2.5,
               label="Mean difference"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
               frameon=False, ncol=4, handlelength=1.8, columnspacing=1.2)
    judge_handles = [Line2D([], [], marker="o", ls="", ms=3.5, color=color, label=judge)
                     for judge, color in judge_colors.items()]
    fig.legend(handles=judge_handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
               frameon=False, ncol=8, fontsize=5.5, handletextpad=0.2,
               columnspacing=0.65, labelspacing=0.25)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def render_length_guidance(benchmarks, out=GUIDANCE_OUT):
    """Per-judge length preference and gap-matched accuracy contrast."""
    stats = {}
    all_deltas = []
    for name, bench in benchmarks.items():
        cells = bench["cells"][bench["cells"].gap > 0.05]
        pair_rates = (cells.dropna(subset=["chose_longer"])
                      .groupby(["judge", "pair"]).chose_longer.mean().reset_index())
        paired = _paired_curve(cells)
        paired["delta"] = paired[CONDITIONS[0]] - paired[CONDITIONS[1]]
        all_deltas.extend(paired.delta)
        stats[name] = pair_rates, paired
    delta_lim = min(1, max(0.25, math.ceil(max(abs(v) for v in all_deltas) * 4) / 4))

    fig, axes = plt.subplots(len(benchmarks), 2, figsize=(4.4, 7.6))
    fig.subplots_adjust(left=0.31, right=0.98, top=0.94, bottom=0.105,
                        hspace=0.58, wspace=0.25)
    for row, name in enumerate(benchmarks):
        ax_pref, ax_delta = axes[row]
        pair_rates, paired = stats[name]
        judges = sorted(set(pair_rates.judge) | set(paired.judge))
        for y, judge in enumerate(judges):
            values = pair_rates[pair_rates.judge == judge].chose_longer.to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
            ax_pref.scatter(values, y + jitter, s=5, color=PALETTE[0], alpha=0.25,
                            linewidth=0, zorder=2)
            if len(values):
                mean, low, high = mean_ci(values)
                low, high = bounded_ci(low, high)
                if np.isfinite(low):
                    ax_pref.plot([low, high], [y, y], color=PALETTE[0], lw=1, zorder=3)
                ax_pref.plot(mean, y, "o", ms=3.8, color=PALETTE[0],
                             mec="white", mew=0.5, zorder=4)

            deltas = paired[paired.judge == judge].delta.to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(deltas)) if len(deltas) else []
            ax_delta.scatter(deltas, y + jitter, s=5, color=PALETTE[1], alpha=0.25,
                             linewidth=0, zorder=2)
            if len(deltas):
                mean, low, high = mean_ci(deltas)
                if np.isfinite(low):
                    ax_delta.plot([low, high], [y, y], color=PALETTE[1], lw=1, zorder=3)
                ax_delta.plot(mean, y, "o", ms=3.8, color=PALETTE[1],
                              mec="white", mew=0.5, zorder=4)

        labels = [judge.replace("-Instruct", "").replace("-0731", "") for judge in judges]
        for ax in (ax_pref, ax_delta):
            ax.set_ylim(len(judges) - 0.5, -0.5)
            ax.set_yticks(range(len(judges)))
            style(ax, grid_axis="x")
        ax_pref.set_yticklabels(labels, fontsize=5.5)
        ax_delta.tick_params(labelleft=False)
        ax_pref.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax_pref.set_xlim(0, 1); ax_pref.set_xticks((0, 0.5, 1))
        ax_delta.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax_delta.set_xlim(-delta_lim, delta_lim)
        ax_delta.set_xticks((-delta_lim, 0, delta_lim))
        ax_pref.text(0, 1.05, name, transform=ax_pref.transAxes, color=MUTED,
                     fontsize=7, fontweight="bold", ha="left")
        if row == len(benchmarks) - 1:
            ax_pref.set_xlabel("P(choose longer)")
            ax_delta.set_xlabel("Δ accuracy")

    for ax, heading in zip(axes[0], ("Length preference", "Gap-matched advantage")):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.98, heading, ha="center", va="top",
                 color=INK2, fontsize=8, fontweight="bold")
    handles = [
        Line2D([], [], marker="o", ls="", ms=2.5, color=PALETTE[6], alpha=0.35,
               label="Pair / gap bin"),
        Line2D([], [], marker="o", ls="-", ms=4, color=PALETTE[6],
               label="Judge mean ± 95% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.63, 0.012),
               frameon=False, ncol=2, handlelength=1.5, columnspacing=1.2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    data = load_benchmarks()
    render_overview(data, args.out, args.bootstrap)
    guidance_out = args.out.with_name("length_guidance")
    render_length_guidance(data, guidance_out)
    strong = filter_judges(data)
    strong_out = args.out.with_name(f"{args.out.name}_strong_judges")
    strong_guidance_out = args.out.with_name("length_guidance_strong_judges")
    render_overview(strong, strong_out, args.bootstrap)
    render_length_guidance(strong, strong_guidance_out)
    for name, bench in data.items():
        print(f"{name}: {len(bench['cells']):,} item-judge cells, "
              f"{bench['cells'].judge.nunique()} judges, {len(bench['models'])} transcript models")
    print(f"figures -> {args.out}.png/.svg, {guidance_out}.png/.svg, "
          f"{strong_out}.png/.svg, {strong_guidance_out}.png/.svg")
