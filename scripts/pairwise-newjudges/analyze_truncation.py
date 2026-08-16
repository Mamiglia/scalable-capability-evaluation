"""Digest and compare the paired transcript-truncation campaign."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log
from scipy.stats import t


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
