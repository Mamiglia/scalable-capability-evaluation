"""Pair accuracy of every judge, GLM wave and baseline on the same held-out pairs, per benchmark.

    uv run python scripts/baselines/held_out_accuracy.py <inputs_dir> <fits_dir> [<fits_dir> ...] --folds <design> ...

Judges are scored from <inputs_dir>/<design>/judge_verdict_from_pair.parquet (test rows: pick_left against
the sign of accuracy_difference); fitted models from <fits_dir>/stronger_model_from_pair/<wave>/<design>/
predictions_stronger_model_from_pair.csv. Accuracy is the micro-average over the union of the given designs'
held-out pairs (so with the four folds of one family every model is scored while unseen); ties in benchmark
accuracy are excluded; the mean column is unweighted over benchmarks.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def judges(inputs_dir, folds):
    rows = []
    for fold in folds:
        table = pd.read_parquet(inputs_dir / fold / "judge_verdict_from_pair.parquet",
                                columns=["bench", "judge", "pick_left", "accuracy_difference", "test"])
        table = table[table.test & (table.accuracy_difference != 0)]
        rows.append(table.assign(correct=(table.pick_left > 0.5) == (table.accuracy_difference > 0)))
    table = pd.concat(rows)
    pooled = table.assign(judge="all judges pooled")
    return pd.concat([table, pooled]).groupby(["judge", "bench"]).correct.mean().unstack("bench").rename(
        index=lambda j: f"judge {j}" if j != "all judges pooled" else j)


def fitted(fits_dirs, folds):
    rows = []
    for fits_dir in fits_dirs:
        for wave_dir in sorted(p for p in (fits_dir / "stronger_model_from_pair").glob("*") if p.is_dir()):
            for fold in folds:
                path = wave_dir / fold / "predictions_stronger_model_from_pair.csv"
                if path.exists():
                    rows.append(pd.read_csv(path).assign(wave=wave_dir.name))
    table = pd.concat(rows)
    table["correct"] = (table.predicted > 0.5) == (table.observed > 0.5)
    return table.groupby(["wave", "bench"]).correct.mean().unstack("bench")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("fits_dirs", type=Path, nargs="+")
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = pd.concat([judges(args.inputs_dir, args.folds), fitted(args.fits_dirs, args.folds)])
    table["mean"] = table.mean(axis=1)
    table = table.sort_values("mean", ascending=False)
    if args.output:
        table.round(3).to_csv(args.output)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
