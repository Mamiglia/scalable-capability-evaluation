#!/usr/bin/env python3
"""Reproducible, item-level analysis of the pilot judging runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
INK = "#17202A"
GRID = "#D9DEE3"
MODEL_ALIASES = {
    "fireworks/accounts/fireworks/models/deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "moonshotai/kimi-k2.6": "moonshotai/kimi-k2.6",
    "openai-api/runpod/Qwen/Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "openai-api/runpod/google/gemma-4-E2B-it": "google/gemma-4-E2B-it",
}


def canonical_model(name: str) -> str:
    """Map provider-qualified runtime names to stable collected model IDs."""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    for prefix in ("openrouter/", "openai-api/runpod/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    aliases = {
        "deepseek/deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
        "deepseek/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
        "z-ai/glm-5.2": "zai-org/GLM-5.2",
        "qwen/qwen3-32b": "Qwen/Qwen3-32B",
        "anthropic/claude-opus-4.7": "anthropic/claude-opus-4.7",
    }
    return aliases.get(name, name)


def short(name: str) -> str:
    return canonical_model(name).rsplit("/", 1)[-1].replace("claude-", "")


def header(path: Path) -> dict:
    result = subprocess.run(
        ["bsdtar", "-xOf", str(path), "header.json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def header_accuracy(data: dict) -> float:
    for score in (data.get("results") or {}).get("scores", []):
        metric = score.get("metrics", {}).get("accuracy")
        if metric is not None:
            return float(metric["value"])
    raise ValueError("no accuracy metric")


def verdict_answers(path: Path) -> list[tuple[str, str | None]]:
    """Stream only sample ID and parsed answer from journal summaries."""
    archive = subprocess.Popen(
        ["bsdtar", "-xOf", str(path), "_journal/summaries/*"], stdout=subprocess.PIPE
    )
    selected = subprocess.run(
        ["jq", "-c", ".[] | [.id, .scores.judge_scorer.answer]"],
        stdin=archive.stdout,
        capture_output=True,
        text=True,
        check=True,
    )
    if archive.stdout:
        archive.stdout.close()
    if archive.wait() != 0:
        raise subprocess.SubprocessError(f"could not read summaries from {path}")
    return [tuple(json.loads(line)) for line in selected.stdout.splitlines()]


@lru_cache(maxsize=None)
def collection_accuracy(source_path: str) -> float:
    model_dir = Path(source_path).parent.parent
    values = []
    for path in model_dir.glob("[0-9]*/*.eval"):
        try:
            data = header(path)
            if data.get("status") == "success":
                values.append(header_accuracy(data))
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
            continue
    if not values:
        raise ValueError(f"no successful collection logs under {model_dir}")
    return float(np.mean(values))


def inventory_and_verdicts(roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = []
    rows = []
    for condition, root in roots.items():
        for index, path in enumerate(sorted(root.rglob("*.eval")), 1):
            try:
                data = header(path)
                status = data.get("status", "unknown")
            except Exception as exc:
                inventory.append(
                    {"condition": condition, "path": str(path), "status": "unreadable", "detail": str(exc)}
                )
                continue
            metadata = data.get("eval", {}).get("metadata") or {}
            inventory.append(
                {
                    "condition": condition,
                    "path": str(path),
                    "status": status,
                    "detail": data.get("error", {}).get("message", "") if data.get("error") else "",
                }
            )
            required = ("name_major", "name_minor", "major_path", "minor_path")
            if status != "success" or not all(metadata.get(key) for key in required):
                continue
            major = canonical_model(metadata["name_major"])
            minor = canonical_model(metadata["name_minor"])
            pair = " vs ".join(sorted((major, minor)))
            major_acc = collection_accuracy(metadata["major_path"])
            minor_acc = collection_accuracy(metadata["minor_path"])
            if major_acc == minor_acc:
                better = None
            else:
                better = major if major_acc > minor_acc else minor
            benchmark = metadata.get("benchmark_name", "unknown").rsplit("/", 1)[-1]
            benchmark = {"gpqa_diamond": "GPQA", "gpqa": "GPQA", "mbpp": "MBPP"}.get(
                benchmark, benchmark.upper()
            )
            source_repeat = f"{Path(metadata['major_path']).parent.name}:{Path(metadata['minor_path']).parent.name}"
            for sample_id, raw_answer in verdict_answers(path):
                match = re.match(r"^(.*)_([12])$", sample_id)
                if match is None:
                    raise ValueError(f"unexpected sample id {sample_id!r} in {path}")
                item, target = match.groups()
                answer = str(raw_answer) if raw_answer in ("1", "2") else None
                first = major if target == "1" else minor
                second = minor if target == "1" else major
                chosen = first if answer == "1" else second if answer == "2" else None
                rows.append(
                    {
                        "condition": condition,
                        "benchmark": benchmark,
                        "judge": data["eval"]["model"],
                        "experiment": metadata.get("experiment_name", ""),
                        "pair": pair,
                        "major": major,
                        "minor": minor,
                        "major_accuracy": major_acc,
                        "minor_accuracy": minor_acc,
                        "better_model": better,
                        "reference_tie": better is None,
                        "source_repeat": source_repeat,
                        "item": item,
                        "order": int(target),
                        "first_model": first,
                        "second_model": second,
                        "answer": answer,
                        "chosen_model": chosen,
                        "correct": np.nan if chosen is None or better is None else int(chosen == better),
                        "log_path": str(path),
                    }
                )
            if index % 100 == 0:
                print(f"read {index} archives from {root}", file=sys.stderr)
    return pd.DataFrame(rows), pd.DataFrame(inventory)


def balanced_accuracy(frame: pd.DataFrame) -> float:
    valid = frame.dropna(subset=["correct"])
    return float(valid.groupby("pair")["correct"].mean().mean()) if len(valid) else math.nan


def bootstrap_balanced(frame: pd.DataFrame, rng: np.random.Generator, n: int = 500) -> tuple[float, float]:
    valid = frame.dropna(subset=["correct"])
    pairs = valid["pair"].unique()
    if len(pairs) == 0:
        return math.nan, math.nan
    item_means = {
        pair: valid[valid.pair == pair].groupby("item").correct.mean().to_numpy()
        for pair in pairs
    }
    values = []
    for _ in range(n):
        means = [rng.choice(item_means[pair], len(item_means[pair]), replace=True).mean()
                 for pair in rng.choice(pairs, len(pairs), replace=True)]
        values.append(np.mean(means))
    return tuple(np.quantile(values, [0.025, 0.975]))


def spearman_pairs(frame: pd.DataFrame, score: str) -> tuple[float, int]:
    cells = frame.dropna(subset=["correct", score]).groupby("pair").agg(
        accuracy=("correct", "mean"), gap=(score, "first")
    )
    if len(cells) < 3 or cells.gap.nunique() < 2:
        return math.nan, len(cells)
    return float(spearmanr(cells.gap.abs(), cells.accuracy).statistic), len(cells)


def bootstrap_spearman(frame: pd.DataFrame, score: str, rng: np.random.Generator,
                       n: int = 500) -> tuple[float, float]:
    cells = frame.dropna(subset=["correct", score]).groupby("pair").agg(
        accuracy=("correct", "mean"), gap=(score, "first")
    )
    if len(cells) < 3 or cells.gap.nunique() < 2:
        return math.nan, math.nan
    draws = []
    for _ in range(n):
        sample = cells.iloc[rng.integers(0, len(cells), len(cells))]
        if sample.gap.nunique() > 1 and sample.accuracy.nunique() > 1:
            draws.append(spearmanr(sample.gap.abs(), sample.accuracy).statistic)
    return tuple(np.quantile(draws, [0.025, 0.975])) if draws else (math.nan, math.nan)


def add_reference_scores(verdicts: pd.DataFrame, aa_path: Path) -> pd.DataFrame:
    aa = pd.read_csv(aa_path).set_index("model")
    for metric in ("intelligence_index", "math_index", "coding_index"):
        verdicts[f"major_{metric}"] = verdicts.major.map(aa[metric])
        verdicts[f"minor_{metric}"] = verdicts.minor.map(aa[metric])
        verdicts[metric] = verdicts.apply(
            lambda row: abs(aa.at[row.major, metric] - aa.at[row.minor, metric])
            if row.major in aa.index and row.minor in aa.index
            else np.nan,
            axis=1,
        )
    verdicts["collection_accuracy"] = (verdicts.major_accuracy - verdicts.minor_accuracy).abs() * 100
    return verdicts


def deduplicate(verdicts: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    key = ["condition", "judge", "pair", "source_repeat", "item", "order"]
    duplicate = verdicts[verdicts.duplicated(key, keep=False)]
    conflicts = duplicate.groupby(key, dropna=False).chosen_model.nunique(dropna=False)
    conflicting_keys = set(conflicts[conflicts > 1].index)
    conflict_mask = verdicts.set_index(key).index.isin(conflicting_keys)
    conflicting_rows = int(conflict_mask.sum())
    verdicts = verdicts[~conflict_mask]
    count = int(verdicts.duplicated(key).sum())
    message = f"collapsed {count:,} identical duplicate rows; excluded {conflicting_rows:,} conflicting duplicate rows"
    return verdicts.drop_duplicates(key, keep="first").copy(), message


def validate(verdicts: pd.DataFrame, inventory: pd.DataFrame, aa_path: Path) -> list[str]:
    messages = []
    messages.append(f"{len(verdicts):,} item rows; {verdicts.answer.notna().sum():,} parseable verdicts")
    counts = inventory.groupby(["condition", "status"]).size().to_dict()
    messages.append("archive inventory: " + ", ".join(f"{a}/{b}={n}" for (a, b), n in sorted(counts.items())))
    messages.append(f"reference ties: {verdicts.reference_tie.sum():,} rows")
    repeats = verdicts.groupby(["condition", "judge", "pair"])["source_repeat"].nunique()
    messages.append(f"cells missing a second repeat: {(repeats < 2).sum():,}/{len(repeats):,}")
    off = verdicts[verdicts.condition == "reasoning_off"]
    on = verdicts[verdicts.condition == "reasoning_on"]
    match = ["judge", "pair", "source_repeat", "item", "order"]
    messages.append(f"matched reasoning rows: {len(off.merge(on, on=match)):,}")
    aa = pd.read_csv(aa_path)
    candidates = set(verdicts.major) | set(verdicts.minor)
    messages.append(f"AA model coverage: {len(candidates & set(aa.model))}/{len(candidates)} mapped")
    for bench, group in verdicts.groupby("benchmark"):
        graph = defaultdict(set)
        for a, b in group[["major", "minor"]].drop_duplicates().itertuples(index=False):
            graph[a].add(b)
            graph[b].add(a)
        components = []
        unseen = set(graph)
        while unseen:
            todo = [unseen.pop()]
            component = set(todo)
            while todo:
                for model in graph[todo.pop()] - component:
                    component.add(model)
                    unseen.discard(model)
                    todo.append(model)
            components.append(component)
        messages.append(f"{bench} comparison components: " + ", ".join(str(len(c)) for c in components))
    return messages


def matched_reasoning(verdicts: pd.DataFrame) -> pd.DataFrame:
    keys = ["judge", "benchmark", "pair", "source_repeat", "item", "order"]
    cols = keys + ["correct", "chosen_model"]
    off = verdicts[verdicts.condition == "reasoning_off"][cols]
    on = verdicts[verdicts.condition == "reasoning_on"][cols]
    matched = off.merge(on, on=keys, suffixes=("_off", "_on"))
    matched["delta"] = matched.correct_on - matched.correct_off
    matched["flipped"] = matched.chosen_model_on != matched.chosen_model_off
    return matched


def variability(verdicts: pd.DataFrame) -> pd.DataFrame:
    off = verdicts[(verdicts.condition == "reasoning_off") & verdicts.answer.notna()]
    base = ["judge", "benchmark", "pair", "source_repeat", "item"]
    order = off.pivot_table(index=base, columns="order", values="chosen_model", aggfunc="first").dropna()
    order_rows = []
    for (judge, bench), group in order.groupby(level=[0, 1]):
        flips = int((group[1] != group[2]).sum())
        order_rows.append({"kind": "presentation_order", "judge": judge, "benchmark": bench,
                           "flips": flips, "eligible": len(group), "consistency": 1 - flips / len(group)})
    repeat_rows = []
    repeat_counts = defaultdict(lambda: [0, 0])
    repeat_base = ["judge", "benchmark", "pair", "item", "order"]
    for key, group in off.groupby(repeat_base):
        choices = group.groupby("source_repeat").chosen_model.first().tolist()
        for first, second in combinations(choices, 2):
            repeat_counts[key[:2]][1] += 1
            repeat_counts[key[:2]][0] += first != second
    for (judge, bench), (flips, eligible) in repeat_counts.items():
        repeat_rows.append({"kind": "transcript_repeat", "judge": judge, "benchmark": bench,
                            "flips": flips, "eligible": eligible,
                            "consistency": 1 - flips / eligible if eligible else np.nan})
    accuracy_rows = []
    repeat_accuracy = off.dropna(subset=["correct"]).groupby(
        ["judge", "benchmark", "pair", "source_repeat"]
    ).correct.mean().reset_index()
    accuracy_differences = defaultdict(list)
    for (judge, bench, _pair), group in repeat_accuracy.groupby(["judge", "benchmark", "pair"]):
        values = group.sort_values("source_repeat").correct.to_numpy()
        accuracy_differences[(judge, bench)].extend(second - first for first, second in combinations(values, 2))
    for (judge, bench), values in accuracy_differences.items():
        differences = pd.Series(values)
        for kind, value in (("repeat_accuracy_signed", differences.mean()),
                            ("repeat_accuracy_absolute", differences.abs().mean())):
            accuracy_rows.append({"kind": kind, "judge": judge, "benchmark": bench,
                                  "flips": np.nan, "eligible": len(differences), "consistency": value})
    return pd.DataFrame(order_rows + repeat_rows + accuracy_rows)


def btl_fit(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame.chosen_model.notna()]
    output = []
    for bench, bench_frame in valid.groupby("benchmark"):
        edges = bench_frame[["first_model", "second_model"]].drop_duplicates()
        graph = defaultdict(set)
        for a, b in edges.itertuples(index=False):
            graph[a].add(b)
            graph[b].add(a)
        unseen = set(graph)
        component_id = 0
        while unseen:
            root = unseen.pop()
            component = {root}
            todo = [root]
            while todo:
                for model in graph[todo.pop()] - component:
                    component.add(model)
                    unseen.discard(model)
                    todo.append(model)
            component_id += 1
            cell = bench_frame[bench_frame.first_model.isin(component)].copy()
            cell["chosen_first"] = cell.chosen_model == cell.first_model
            if "_weight" not in cell:
                cell["_weight"] = 1
            cell = cell.groupby(
                ["first_model", "second_model", "judge", "chosen_first"], as_index=False
            )["_weight"].sum()
            models = sorted(component)
            judges = sorted(cell.judge.unique())
            model_cols = {model: i for i, model in enumerate(models[1:])}
            judge_cols = {judge: len(model_cols) + i for i, judge in enumerate(judges)}
            x = np.zeros((len(cell), len(model_cols) + len(judge_cols)))
            for row_i, row in enumerate(cell.itertuples()):
                if row.first_model in model_cols:
                    x[row_i, model_cols[row.first_model]] += 1
                if row.second_model in model_cols:
                    x[row_i, model_cols[row.second_model]] -= 1
                x[row_i, judge_cols[row.judge]] = 1
            y = cell.chosen_first.to_numpy(float)
            weights = cell._weight.to_numpy(float)

            def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
                z = x @ beta
                loss = weights @ (np.logaddexp(0, z) - y * z) + 1e-6 * (beta @ beta)
                residual = weights * (1 / (1 + np.exp(-np.clip(z, -30, 30))) - y)
                gradient = x.T @ residual + 2e-6 * beta
                return float(loss), gradient

            fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
            if not fit.success:
                raise RuntimeError(f"BTL fit failed: {fit.message}")
            scores = {models[0]: 0.0, **{model: fit.x[col] for model, col in model_cols.items()}}
            centered = np.mean(list(scores.values()))
            for model in models:
                output.append({"benchmark": bench, "component": component_id, "model": model,
                               "btl_score": scores[model] - centered, "btl_low": np.nan, "btl_high": np.nan})
    return pd.DataFrame(output)


def btl_bootstrap(verdicts: pd.DataFrame, point: pd.DataFrame, rng: np.random.Generator, n: int = 50) -> pd.DataFrame:
    draws = defaultdict(list)
    for _ in range(n):
        parts = []
        for bench, group in verdicts.groupby("benchmark"):
            items = group.item.unique()
            sampled = rng.choice(items, len(items), replace=True)
            counts = pd.Series(sampled).value_counts()
            part = group[group.item.isin(counts.index)].copy()
            part["_weight"] = part.item.map(counts)
            parts.append(part)
        fitted = btl_fit(pd.concat(parts, ignore_index=True))
        for row in fitted.itertuples():
            draws[(row.benchmark, row.component, row.model)].append(row.btl_score)
    for index, row in point.iterrows():
        values = draws[(row.benchmark, row.component, row.model)]
        point.loc[index, ["btl_low", "btl_high"]] = np.quantile(values, [0.025, 0.975])
    return point


def subset_stability(verdicts: pd.DataFrame, rng: np.random.Generator, replicates: int) -> pd.DataFrame:
    base = verdicts[(verdicts.condition == "reasoning_off") & verdicts.correct.notna()]
    rows = []
    for bench, group in base.groupby("benchmark"):
        counts = group.groupby(["item", "pair"]).correct.agg(["sum", "count"]).reset_index()
        sums = counts.pivot(index="item", columns="pair", values="sum").fillna(0)
        ns = counts.pivot(index="item", columns="pair", values="count").fillna(0)
        items = sums.index.to_numpy()
        full_pairs = sums.sum().div(ns.sum()).dropna()
        full_acc = full_pairs.mean()
        gaps = group.groupby("pair").collection_accuracy.first().reindex(full_pairs.index)
        full_corr = spearmanr(gaps, full_pairs).statistic
        for fraction in (0.05, 0.1, 0.2, 0.5, 1.0):
            size = max(1, round(len(items) * fraction))
            for replicate in range(replicates):
                selected = rng.choice(items, size, replace=False)
                pair_acc = sums.loc[selected].sum().div(ns.loc[selected].sum()).reindex(full_pairs.index)
                ok = pair_acc.notna()
                ordering = spearmanr(full_pairs[ok], pair_acc[ok]).statistic if ok.sum() >= 3 else np.nan
                gap_ok = ok & gaps.notna()
                correlation = spearmanr(gaps[gap_ok], pair_acc[gap_ok]).statistic if gap_ok.sum() >= 3 else np.nan
                coverage = int(gap_ok.sum())
                rows.append({"benchmark": bench, "fraction": fraction, "replicate": replicate,
                             "n_items": size, "accuracy_error": pair_acc.mean() - full_acc,
                             "pair_order_spearman": ordering,
                             "gap_correlation": correlation,
                             "gap_correlation_error": correlation - full_corr,
                             "gap_pairs": coverage})
    return pd.DataFrame(rows)


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.set_size_inches(16, 9)
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(verdicts: pd.DataFrame, matched: pd.DataFrame, variation: pd.DataFrame,
             btl: pd.DataFrame, stability: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 25, "axes.labelsize": 18, "text.color": INK,
                         "axes.labelcolor": INK, "axes.edgecolor": INK})
    off = verdicts[verdicts.condition == "reasoning_off"]
    quality = []
    for (judge, bench), group in off.groupby(["judge", "benchmark"]):
        mean = balanced_accuracy(group)
        low, high = bootstrap_balanced(group, rng)
        quality.append((judge, bench, mean, low, high))
    q = pd.DataFrame(quality, columns=["judge", "benchmark", "mean", "low", "high"])
    fig, ax = plt.subplots()
    judges = sorted(q.judge.unique())
    benches = sorted(q.benchmark.unique())
    pair_points = off.dropna(subset=["correct"]).groupby(
        ["judge", "benchmark", "pair"]
    ).correct.mean().mul(100)
    available = set(pair_points.index.droplevel("pair"))
    for bench_i, bench in enumerate(benches):
        positions = np.arange(len(judges)) + 1 + (bench_i - .5) * .28
        values = [pair_points.xs((judge, bench), level=("judge", "benchmark")).to_numpy()
                  if (judge, bench) in available else np.array([])
                  for judge in judges]
        boxes = ax.boxplot(values, positions=positions, widths=.24, patch_artist=True,
                           showfliers=False, manage_ticks=False)
        for box in boxes["boxes"]:
            box.set(facecolor=COLORS[bench_i], alpha=.18, edgecolor=COLORS[bench_i], linewidth=1.5)
        for x, points in zip(positions, values):
            ax.scatter(x + rng.uniform(-.045, .045, len(points)), points, s=20,
                       color=COLORS[bench_i], alpha=.65, zorder=3)
        ax.scatter([], [], color=COLORS[bench_i], label=bench)
    ax.axhline(50, color="#777777", linestyle="--")
    ax.set_xticks(range(1, len(judges) + 1), [short(judge) for judge in judges], rotation=20, ha="right")
    ax.set_ylabel("Pair-balanced judgment accuracy (%)")
    ax.set_title("Judge quality by benchmark — points are model pairs", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style(ax)
    save(fig, out, "01_average_judge_quality")

    for bench, group in off[~off.reference_tie].groupby("benchmark"):
        judge_points = group.groupby(["pair", "judge"]).correct.mean().unstack("judge") * 100
        order = judge_points.mean(axis=1).sort_values().index
        judge_points = judge_points.loc[order]
        fig, ax = plt.subplots()
        values = [row.dropna().to_numpy() for _, row in judge_points.iterrows()]
        boxes = ax.boxplot(values, orientation="horizontal", tick_labels=[" vs ".join(map(short, pair.split(" vs ")))
                            for pair in judge_points.index], widths=.55, patch_artist=True, showfliers=False)
        for box in boxes["boxes"]:
            box.set(facecolor=COLORS[0], alpha=.15, edgecolor=COLORS[0], linewidth=1.5)
        judge_colors = {judge: plt.get_cmap("tab10")(i) for i, judge in enumerate(judge_points.columns)}
        for y, (_, row) in enumerate(judge_points.iterrows(), 1):
            for judge, point in row.dropna().items():
                ax.scatter(point, y + rng.uniform(-.11, .11), color=judge_colors[judge], s=30,
                           alpha=.85, edgecolor="white", linewidth=.4, zorder=3)
        for judge, color in judge_colors.items():
            ax.scatter([], [], color=color, label=short(judge))
        ax.axvline(50, color="#777777", linestyle="--")
        ax.set_xlabel("Judge accuracy (%)")
        ax.set_title(f"{bench}: pair difficulty — points are individual judges", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
        style(ax)
        save(fig, out, f"02_pair_difficulty_{bench.lower()}")

    effect = matched.dropna(subset=["delta"])
    fig, (ax, key_ax) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    pair_effect = effect.groupby(["judge", "benchmark", "pair"]).delta.mean().mul(100)
    judges = sorted(effect.judge.unique())
    benches = sorted(effect.benchmark.unique())
    available = set(pair_effect.index.droplevel("pair"))
    pairs = sorted(pair_effect.index.get_level_values("pair").unique())
    numbers = {pair: number for number, pair in enumerate(pairs, 1)}
    codes = {(bench, pair): f"{bench[0]}{numbers[pair]:02d}" for bench, pair in pair_effect.index.droplevel("judge").unique()}
    for bench_i, bench in enumerate(benches):
        positions = np.arange(len(judges)) + 1 + (bench_i - .5) * .28
        cells = [pair_effect.xs((judge, bench), level=("judge", "benchmark"))
                 if (judge, bench) in available else pd.Series(dtype=float) for judge in judges]
        values = [cell.to_numpy() for cell in cells]
        boxes = ax.boxplot(values, positions=positions, widths=.24, patch_artist=True,
                           showfliers=False, manage_ticks=False)
        for box in boxes["boxes"]:
            box.set(facecolor=COLORS[bench_i], alpha=.18, edgecolor=COLORS[bench_i])
        for x, cell in zip(positions, cells):
            points = cell.to_numpy()
            point_x = x + rng.uniform(-.045, .045, len(points))
            ax.scatter(point_x, points, s=36, color=COLORS[bench_i], alpha=.8, zorder=3)
            for label_x, value, pair in zip(point_x, points, cell.index):
                ax.annotate(codes[(bench, pair)], (label_x, value), xytext=(3, 3), textcoords="offset points",
                            fontsize=8, color=INK, fontweight="bold", zorder=4,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .15})
        ax.scatter([], [], color=COLORS[bench_i], label=bench)
    ax.axhline(0, color="#777777", linestyle="--")
    ax.set_xticks(range(1, len(judges) + 1), [short(j) for j in judges], rotation=20, ha="right")
    ax.set_ylabel("Accuracy change when reasoning is shown (percentage points)")
    ax.set_title("Reasoning effect (above zero = helps)", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style(ax)
    key_ax.axis("off")
    key_ax.set_title("Shared point key", loc="left", fontweight="bold", fontsize=16)
    for row, pair in enumerate(pairs):
        labels = "/".join(codes[(bench, pair)] for bench in benches if (bench, pair) in codes)
        key_ax.text(0, 1 - (row + 2) / (len(pairs) + 3), f"{labels}  {' vs '.join(short(model) for model in pair.split(' vs '))}",
                    color=INK, fontsize=10, va="top", transform=key_ax.transAxes)
    save(fig, out, "03_reasoning_effect")

    for kind, title, name in (("presentation_order", "04 · Verdicts changed after swapping transcript order", "04_order_flips"),
                              ("transcript_repeat", "05 · Verdicts changed with a new transcript repeat", "05_repeat_flips")):
        cell = variation[variation.kind == kind].copy()
        cell["flip_rate"] = 1 - cell.consistency
        cell = cell.sort_values("flip_rate")
        fig, ax = plt.subplots()
        labels = [f"{short(r.judge)} · {r.benchmark}" for r in cell.itertuples()]
        colors = [COLORS[0] if bench == "GPQA" else COLORS[1] for bench in cell.benchmark]
        bars = ax.barh(range(len(cell)), cell.flip_rate * 100, color=colors, alpha=.85)
        for bar, row in zip(bars, cell.itertuples()):
            ax.text(bar.get_width() + .5, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.0f}%  ({int(row.flips)}/{int(row.eligible)})", va="center", fontsize=11)
        ax.set_yticks(range(len(cell)), labels)
        ax.set_xlim(0, max(50, cell.flip_rate.max() * 120))
        ax.set_xlabel("Verdicts that changed (%) — raw count in parentheses")
        ax.set_title(title, loc="left", fontweight="bold")
        style(ax)
        save(fig, out, name)

    gaps = variation[variation.kind.isin(["repeat_accuracy_signed", "repeat_accuracy_absolute"])]
    gaps = gaps.pivot(index=["judge", "benchmark"], columns="kind", values="consistency").reset_index()
    gaps = gaps.sort_values("repeat_accuracy_absolute")
    fig, ax = plt.subplots()
    y = np.arange(len(gaps))
    colors = [COLORS[0] if bench == "GPQA" else COLORS[1] for bench in gaps.benchmark]
    bars = ax.barh(y, gaps.repeat_accuracy_absolute * 100, color=colors, alpha=.45,
                   label="mean absolute movement")
    ax.scatter(gaps.repeat_accuracy_signed * 100, y, color=INK, s=45, zorder=3,
               label="signed mean (direction)")
    for bar in bars:
        ax.text(bar.get_width() + .15, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f} pp", va="center", fontsize=11)
    ax.axvline(0, color="#777777", linestyle="--")
    ax.set_yticks(y, [f"{short(r.judge)} · {r.benchmark}" for r in gaps.itertuples()])
    ax.set_xlabel("Change in model-pair judge accuracy between transcript repeats")
    ax.set_title("06 · How much does pair accuracy move between repeats?", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    save(fig, out, "06_repeat_accuracy_differences")

    metrics = [("collection_accuracy", "Collection accuracy"), ("intelligence_index", "AA Intelligence"),
               ("math_index", "AA Math"), ("coding_index", "AA Coding")]
    fig, axes = plt.subplots(2, 2, layout="constrained")
    for ax, (metric, label) in zip(axes.flat, metrics):
        cells = off.dropna(subset=["correct", metric]).groupby(["benchmark", "pair"]).agg(
            gap=(metric, "first"), accuracy=("correct", "mean")).reset_index()
        for i, (bench, group) in enumerate(cells.groupby("benchmark")):
            rho = spearmanr(group.gap, group.accuracy).statistic if len(group) >= 3 else np.nan
            ax.scatter(group.gap, group.accuracy * 100, color=COLORS[i], s=55,
                       label=f"{bench}: ρ={rho:.2f}, n={len(group)}")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Absolute reference-score gap")
        ax.set_ylabel("Judge accuracy (%)")
        ax.legend(frameon=False, fontsize=11)
        style(ax)
    fig.suptitle("Reference gaps only weakly predict pair difficulty", x=0.06, ha="left", fontweight="bold")
    save(fig, out, "07_reference_relationships")

    for bench, group in btl.groupby("benchmark"):
        ordered = group.sort_values(["component", "btl_score"])
        fig, ax = plt.subplots()
        y = np.arange(len(ordered))
        ax.errorbar(ordered.btl_score, y, xerr=[ordered.btl_score - ordered.btl_low,
                    ordered.btl_high - ordered.btl_score], fmt="o", color=COLORS[0], capsize=3)
        labels = [f"C{r.component} · {short(r.model)}" for r in ordered.itertuples()]
        ax.set_yticks(y, labels)
        ax.axvline(0, color="#777777", linestyle="--")
        ax.set_xlabel("BTL score (centered separately within component)")
        ax.set_title(f"{bench}: inferred candidate ranks within disconnected components", loc="left", fontweight="bold")
        style(ax)
        save(fig, out, f"08_btl_{bench.lower()}")

    fig, axes = plt.subplots(1, 3, layout="constrained")
    measures = [("accuracy_error", "Absolute accuracy error (pp)"),
                ("pair_order_spearman", "Pair-order Spearman ρ"),
                ("gap_correlation_error", "Absolute gap-correlation error")]
    for ax, (metric, label) in zip(axes, measures):
        data = stability.copy()
        if "error" in metric:
            data[metric] = data[metric].abs() * (100 if metric == "accuracy_error" else 1)
        summary = data.groupby(["benchmark", "fraction"])[metric].agg(["median", lambda x: x.quantile(.1), lambda x: x.quantile(.9)]).reset_index()
        summary.columns = ["benchmark", "fraction", "median", "low", "high"]
        for i, (bench, group) in enumerate(summary.groupby("benchmark")):
            ax.plot(group.fraction * 100, group["median"], marker="o", color=COLORS[i], label=bench)
            ax.fill_between(group.fraction * 100, group.low, group.high, color=COLORS[i], alpha=.18)
        ax.set_xlabel("Items retained (%)")
        ax.set_ylabel(label)
        style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("How much data is enough? Median and 10–90% subsampling range", x=.06, ha="left", fontweight="bold")
    save(fig, out, "09_sample_size_stability")


def summary_rows(verdicts: pd.DataFrame, matched: pd.DataFrame, variation: pd.DataFrame,
                 btl: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    off = verdicts[verdicts.condition == "reasoning_off"]
    for (judge, bench), group in off.groupby(["judge", "benchmark"]):
        value = balanced_accuracy(group)
        low, high = bootstrap_balanced(group, rng)
        rows.append({"section": "judge_quality", "benchmark": bench, "judge": judge,
                     "metric": "pair_balanced_accuracy", "value": value, "low": low, "high": high,
                     "n": group.correct.notna().sum(), "note": f"{group.pair.nunique()} pairs"})
    for row in variation.itertuples():
        rows.append({"section": "variability", "benchmark": row.benchmark, "judge": row.judge,
                     "metric": row.kind + "_consistency", "value": row.consistency, "low": np.nan,
                     "high": np.nan, "n": row.eligible, "note": f"{row.flips}/{row.eligible} flips"})
    for (judge, bench), group in matched.dropna(subset=["delta"]).groupby(["judge", "benchmark"]):
        rows.append({"section": "reasoning", "benchmark": bench, "judge": judge,
                     "metric": "accuracy_change", "value": group.groupby("pair").delta.mean().mean(),
                     "low": np.nan, "high": np.nan, "n": len(group), "note": f"{group.flipped.mean():.1%} verdict flips"})
    for bench, group in off.groupby("benchmark"):
        for metric in ("collection_accuracy", "intelligence_index", "math_index", "coding_index"):
            rho, n = spearman_pairs(group, metric)
            low, high = bootstrap_spearman(group, metric, rng)
            rows.append({"section": "reference", "benchmark": bench, "judge": "pooled",
                         "metric": metric + "_spearman", "value": rho, "low": low,
                         "high": high, "n": n, "note": "pair-bootstrap interval; complete pairs"})
            for judge, cell in group.groupby("judge"):
                judge_rho, judge_n = spearman_pairs(cell, metric)
                if judge_n >= 5:
                    judge_low, judge_high = bootstrap_spearman(cell, metric, rng)
                    rows.append({"section": "reference", "benchmark": bench, "judge": judge,
                                 "metric": metric + "_spearman", "value": judge_rho, "low": judge_low,
                                 "high": judge_high, "n": judge_n, "note": "pair-bootstrap; judge-specific"})
    reference_columns = ["collection_accuracy", "intelligence_index", "math_index", "coding_index"]
    major_refs = verdicts[["benchmark", "major", "major_accuracy", "major_intelligence_index",
                           "major_math_index", "major_coding_index"]].copy()
    minor_refs = verdicts[["benchmark", "minor", "minor_accuracy", "minor_intelligence_index",
                           "minor_math_index", "minor_coding_index"]].copy()
    major_refs.columns = ["benchmark", "model", *reference_columns]
    minor_refs.columns = ["benchmark", "model", *reference_columns]
    references = pd.concat([major_refs, minor_refs]).drop_duplicates(["benchmark", "model"]).set_index(
        ["benchmark", "model"]
    )
    for (bench, component), scores in btl.groupby(["benchmark", "component"]):
        cell = scores.set_index("model").join(references.loc[bench], how="left")
        for metric in ("collection_accuracy", "intelligence_index", "math_index", "coding_index"):
            complete = cell.dropna(subset=[metric])
            rho = spearmanr(complete.btl_score, complete[metric]).statistic if len(complete) >= 3 else np.nan
            rows.append({"section": "btl_alignment", "benchmark": bench, "judge": f"component_{component}",
                         "metric": metric + "_rank_spearman", "value": rho, "low": np.nan,
                         "high": np.nan, "n": len(complete), "note": "within-component ranks only"})
    return pd.DataFrame(rows)


def write_readme(out: Path, messages: list[str], summary: pd.DataFrame, inventory: pd.DataFrame,
                 matched: pd.DataFrame) -> None:
    quality = summary[summary.section == "judge_quality"]
    best = quality.loc[quality.groupby("benchmark").value.idxmax()]
    takeaways = "\n".join(
        f"- {row.benchmark}: highest pair-balanced judge accuracy was {short(row.judge)} at {row.value:.1%}."
        for row in best.itertuples()
    )
    text = f"""# Pilot judge results

Generated from item-level Inspect judgments in `logs/pilot/` and `logs/pilot_reasoning/`.

## Coverage and validation

{chr(10).join(f'- {message}' for message in messages)}
- Reasoning analysis uses {len(matched):,} exact matches on judge, pair, source repeat, item, and order.
- Full archive inventory, including failures and unreadable logs, is in `archive_inventory.csv`.

## Definitions

- **Better model:** higher mean collection accuracy on that benchmark, computed from successful source logs. Config `major`/`minor` labels are not ground truth.
- **Judge quality:** mean of pair accuracies, so large pairs do not dominate. Intervals resample pairs and then items.
- **Reasoning effect:** reasoning-on minus reasoning-off accuracy for exact matched judgments.
- **Flips:** changed chosen candidate across presentation order or transcript repeat, divided by eligible complete comparisons.
- **BTL:** Bradley–Terry–Luce candidate scores with judge-specific first-position intercepts. Scores are centered and comparable only within each benchmark/component.
- **Reference relationships:** pair-level Spearman correlations; missing AA scores stay missing and exact collection-score ties are excluded from correctness.

## Numerical takeaways

{takeaways}
- Parse failures remain in `verdicts.csv` with blank choices; reference ties remain with blank correctness.
- AA snapshot fetched 2026-08-16. Data © [Artificial Analysis](https://artificialanalysis.ai/); [methodology](https://artificialanalysis.ai/methodology).

SVG and 220-dpi PNG figures use identical data. `summary.csv`, `btl_scores.csv`, and `subset_stability.csv` contain the plotted estimates.
"""
    (out / "README.md").write_text(text)
    inventory.to_csv(out / "archive_inventory.csv", index=False)


def self_check() -> None:
    assert canonical_model("openrouter/deepseek/deepseek-v3.2") == "deepseek-ai/DeepSeek-V3.2"
    toy = pd.DataFrame([
        {"condition": c, "judge": "j", "benchmark": "GPQA", "pair": "a vs b",
         "source_repeat": "0:0", "item": item, "order": order, "chosen_model": chosen,
         "answer": "1", "correct": int(chosen == "b"), "first_model": first, "second_model": second}
        for c, chosen in (("reasoning_off", "a"), ("reasoning_on", "b"))
        for item in ("i1", "i2") for order, first, second in ((1, "a", "b"), (2, "b", "a"))
    ])
    matched = matched_reasoning(toy)
    assert len(matched) == 4 and matched.flipped.all() and (matched.delta == 1).all()
    variation = variability(toy)
    assert variation.query("kind == 'presentation_order'").flips.sum() == 0
    btl = btl_fit(pd.concat([toy.assign(chosen_model="b")] * 10, ignore_index=True))
    scores = btl.set_index("model").btl_score
    assert scores["b"] > scores["a"]
    print("self-check passed: reorientation, matching, flips, and BTL ordering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=Path("logs/pilot"))
    parser.add_argument("--reasoning-root", type=Path, default=Path("logs/pilot_reasoning"))
    parser.add_argument("--aa-scores", type=Path, default=Path("configs/comparisons/pilot_aa_scores.csv"))
    parser.add_argument("--out", type=Path, default=Path("analysis/pilot-results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample-replicates", type=int, default=500)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    for path in (args.pilot_root, args.reasoning_root, args.aa_scores):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    verdicts, inventory = inventory_and_verdicts(
        {"reasoning_off": args.pilot_root, "reasoning_on": args.reasoning_root}
    )
    verdicts, duplicate_message = deduplicate(verdicts)
    verdicts = add_reference_scores(verdicts, args.aa_scores)
    messages = validate(verdicts, inventory, args.aa_scores)
    messages.insert(1, duplicate_message)
    for message in messages:
        print(message)
    matched = matched_reasoning(verdicts)
    variation = variability(verdicts)
    btl = btl_bootstrap(verdicts[verdicts.condition == "reasoning_off"],
                        btl_fit(verdicts[verdicts.condition == "reasoning_off"]), rng)
    stability = subset_stability(verdicts, rng, args.subsample_replicates)
    summary = summary_rows(verdicts, matched, variation, btl, rng)
    verdicts.to_csv(args.out / "verdicts.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    btl.to_csv(args.out / "btl_scores.csv", index=False)
    stability.to_csv(args.out / "subset_stability.csv", index=False)
    plot_all(verdicts, matched, variation, btl, stability, args.out, rng)
    write_readme(args.out, messages, summary, inventory, matched)
    print(f"wrote analysis to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
