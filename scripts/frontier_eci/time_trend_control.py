"""Fit the deliberately dumb pre-release hard-accuracy time trend."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.frontier_eci.canonical_coverage import _load_config, _path  # noqa: E402
from scripts.frontier_eci.rank_regression import (  # noqa: E402
    fit_time_trend_controls, write_time_trend_output,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = _load_config(args.config)
    rank_root = _path(config["results"]) / "analysis/rank_regression"
    source = pd.concat([
        pd.read_csv(rank_root / "training_fitted.csv"),
        pd.read_csv(rank_root / "frontier_fitted.csv"),
    ], ignore_index=True)
    actuals = (source[["bench", "model", "actual", "n_items"]]
               .dropna().drop_duplicates(["bench", "model"]))
    judges = [judge.split("/")[-1] for group in config["judges"].values() for judge in group]
    fits, points = fit_time_trend_controls(
        actuals, list(config["candidates"]), judges, config["releases"],
        resamples=args.resamples, seed=args.seed,
    )

    output = rank_root / "time_trend_control"
    print(json.dumps(write_time_trend_output(fits, points, output, args.resamples), indent=2))


if __name__ == "__main__":
    main()
