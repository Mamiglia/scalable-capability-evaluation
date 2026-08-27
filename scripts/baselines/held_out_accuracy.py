"""Pair accuracy of every judge, GLM wave and baseline on the same held-out pairs, per benchmark.

    uv run python scripts/baselines/held_out_accuracy.py <inputs_dir> <fits_dir> [<fits_dir> ...] --folds <design> ... [--bootstrap 1000]

Judges are scored from <inputs_dir>/<design>/judge_verdict_from_pair.parquet (test rows: pick_left against
the sign of accuracy_difference); fitted models from <fits_dir>/stronger_model_from_pair/<wave>/<design>/
predictions_stronger_model_from_pair.csv. Accuracy is the micro-average over the union of the given designs'
held-out pairs (so with the four folds of one family every model is scored while unseen); ties in benchmark
accuracy are excluded; the mean column is unweighted over benchmarks.
--bootstrap resamples the held-out items with replacement within each benchmark and writes 5th/95th
percentiles of every accuracy (and of the mean over benchmarks) to <output stem>-intervals.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["ranking", "bench", "item"]


def judges(inputs_dir, folds):
    """Long table (ranking, bench, item, correct) for every judge and for all judges pooled."""
    rows = []
    for fold in folds:
        table = pd.read_parquet(inputs_dir / fold / "judge_verdict_from_pair.parquet",
                                columns=["bench", "item", "judge", "pick_left", "accuracy_difference", "test"])
        table = table[table.test & (table.accuracy_difference != 0)]
        rows.append(table.assign(correct=(table.pick_left > 0.5) == (table.accuracy_difference > 0)))
    table = pd.concat(rows).assign(ranking=lambda t: "judge " + t.judge)
    pooled = table.assign(ranking="all judges pooled")
    return pd.concat([table, pooled])[KEYS + ["correct"]]


def fitted(fits_dirs, folds):
    """Long table (ranking, bench, item, correct) for every wave found under the fits dirs."""
    rows = []
    for fits_dir in fits_dirs:
        for wave_dir in sorted(p for p in (fits_dir / "stronger_model_from_pair").glob("*") if p.is_dir()):
            for fold in folds:
                path = wave_dir / fold / "predictions_stronger_model_from_pair.csv"
                if path.exists():
                    rows.append(pd.read_csv(path).assign(ranking=display_name(wave_dir.name)))
    table = pd.concat(rows)
    table["correct"] = (table.predicted > 0.5) == (table.observed > 0.5)
    return table[KEYS + ["correct"]]


def display_name(wave):
    """Row names matching the rankings table: GLM waves, GBM waves, deterministic rules."""
    if wave.startswith("gbm-"):
        return "GBM, " + wave[4:]
    if wave in ("tfidf", "pick-shorter", "pick-longer"):
        return wave
    return "length GLM" if wave == "length" else "GLM, " + wave


def accuracy(table):
    """Wide table ranking x benchmark of micro accuracy, plus the unweighted mean over benchmarks."""
    wide = table.groupby(["ranking", "bench"]).correct.mean().unstack("bench")
    wide["mean"] = wide.mean(axis=1)
    return wide.sort_values("mean", ascending=False)


def intervals(table, n_resamples, seed=0):
    """Item bootstrap within each benchmark: long table (ranking, bench) with p5, p95 of the accuracy.

    A resample is a multinomial count per item; the accuracy of a ranking is then the count-weighted
    mean of its per-item correct counts, so every resample is one matrix product.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for bench, frame in table.groupby("bench"):
        counts = frame.groupby(["ranking", "item"]).correct.agg(["sum", "size"])
        correct = counts["sum"].unstack("item", fill_value=0)     # ranking x item
        total = counts["size"].unstack("item", fill_value=0)
        n_items = correct.shape[1]
        weights = rng.multinomial(n_items, np.full(n_items, 1 / n_items), size=n_resamples)  # resample x item
        with np.errstate(invalid="ignore"):
            resampled = (correct.to_numpy() @ weights.T) / (total.to_numpy() @ weights.T)  # ranking x resample
        draws.append(pd.DataFrame(resampled, index=correct.index).assign(bench=bench))
    draws = pd.concat(draws)
    draws = pd.concat([draws, draws.drop(columns="bench").groupby("ranking").mean().assign(bench="mean")])
    values = draws.set_index("bench", append=True)
    return pd.DataFrame({"p5": values.quantile(0.05, axis=1), "p95": values.quantile(0.95, axis=1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("fits_dirs", type=Path, nargs="+")
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=0, help="item-bootstrap resamples for 5–95%% intervals")
    args = parser.parse_args()
    table = pd.concat([judges(args.inputs_dir, args.folds), fitted(args.fits_dirs, args.folds)])
    wide = accuracy(table)
    if args.output:
        wide.round(3).to_csv(args.output)
    print(wide.round(3).to_string())
    if args.bootstrap:
        bands = intervals(table, args.bootstrap).round(3).loc[list(wide.index)]
        if args.output:
            bands.to_csv(args.output.with_name(args.output.stem + "-intervals.csv"))
        print(bands.to_string())


if __name__ == "__main__":
    main()
