#!/usr/bin/env python3
"""Compare two saved hibayes fits on out-of-sample predictive accuracy.

Reads each run's pointwise PSIS-LOO (elpd per observation) and reports the
difference with its standard error. Pointwise is the point: the two fits share
every observation, so the paired difference has a far smaller standard error
than either total's own, which is what makes a small gap readable at all.

Both runs must be fit on the same rows in the same order - checked against the
data each run saved, not against the elpd index, which is only 0..n-1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check_same_rows(candidate: Path, baseline: Path) -> int:
    """Fail unless both runs were fit on identical rows in identical order.

    The pointwise index is a bare 0..n-1 range in every run, so comparing it
    only compares row counts - two fits on completely different data would
    pass. The saved frames are what actually has to match.
    """
    frames = [pd.read_parquet(run / "data.parquet") for run in (candidate, baseline)]
    shared = [column for column in frames[0].columns if column in frames[1].columns]
    if not shared:
        raise ValueError("The two runs share no columns; they are not the same data")
    if not frames[0][shared].equals(frames[1][shared]):
        differing = [column for column in shared if not frames[0][column].equals(frames[1][column])]
        raise ValueError(
            f"The two runs were fit on different rows: {len(differing)} of {len(shared)} shared "
            f"columns differ, e.g. {differing[:3]}. A paired elpd difference would be meaningless."
        )
    return len(shared)


def load_fit(run: Path) -> tuple[str, list[str], pd.Series]:
    """Model name, predictor list and pointwise elpd for the one model in a run."""
    models = sorted((run / "models").iterdir())
    if len(models) != 1:
        raise ValueError(f"{run} holds {len(models)} models; expected exactly one")
    model = models[0]
    config = json.loads((model / "model_config.json").read_text())
    pointwise = pd.read_csv(model / "diagnostics" / "inference_data_loo_i.csv")
    return model.name, config["extra_kwargs"]["predictors"], pointwise.set_index("obs_dim_0")["loo_i"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Run directory for the model being tested")
    parser.add_argument("baseline", type=Path, help="Run directory for the model to beat")
    args = parser.parse_args()

    shared = check_same_rows(args.candidate, args.baseline)
    name_a, predictors_a, elpd_a = load_fit(args.candidate)
    name_b, predictors_b, elpd_b = load_fit(args.baseline)
    if len(elpd_a) != len(elpd_b):
        raise ValueError(f"{len(elpd_a)} vs {len(elpd_b)} pointwise elpd values")

    added = [predictor for predictor in predictors_a if predictor not in predictors_b]
    dropped = [predictor for predictor in predictors_b if predictor not in predictors_a]
    difference = elpd_a.to_numpy() - elpd_b.to_numpy()
    n = len(difference)
    # Vehtari, Gelman & Gabry (2017): the paired standard error of the elpd
    # difference is sqrt(n) times the standard deviation of the pointwise gaps.
    elpd_difference = difference.sum()
    standard_error = np.sqrt(n) * difference.std(ddof=1)

    print(f"candidate {name_a}: {len(predictors_a)} predictors, elpd_loo {elpd_a.sum():,.1f}")
    print(f"baseline  {name_b}: {len(predictors_b)} predictors, elpd_loo {elpd_b.sum():,.1f}")
    print(f"  added:   {added or 'none'}")
    print(f"  dropped: {dropped or 'none'}")
    print(f"observations: {n:,} (identical across both runs on {shared} shared columns)")
    print(f"elpd difference (candidate - baseline): {elpd_difference:+,.2f} +/- {standard_error:,.2f}")
    print(f"  ratio to its standard error: {elpd_difference / standard_error:+.2f}")


if __name__ == "__main__":
    main()
