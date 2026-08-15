#!/usr/bin/env python3
"""Compare two saved hibayes fits on out-of-sample predictive accuracy.

Reads each run's pointwise PSIS-LOO (elpd per observation) and reports the
difference with its standard error. Pointwise is the point: the two fits share
every observation, so the paired difference has a far smaller standard error
than either total's own, which is what makes a small gap readable at all.

Both runs must be fit on the same rows in the same order - checked here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


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

    name_a, predictors_a, elpd_a = load_fit(args.candidate)
    name_b, predictors_b, elpd_b = load_fit(args.baseline)
    if not elpd_a.index.equals(elpd_b.index):
        raise ValueError("The two fits do not cover the same observations")

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
    print(f"observations: {n:,}")
    print(f"elpd difference (candidate - baseline): {elpd_difference:+,.2f} +/- {standard_error:,.2f}")
    print(f"  ratio to its standard error: {elpd_difference / standard_error:+.2f}")


if __name__ == "__main__":
    main()
