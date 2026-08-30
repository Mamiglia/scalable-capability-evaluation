"""Compare held-out ability-to-accuracy calibrators on the frontier experiment."""

import argparse
import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ranking import rank  # noqa: E402

ANALYSIS = run_path(ROOT / "scripts/frontier_eci/analyze.py")


def unit_sigmoid(x, slope, offset):
    return 1 / (1 + np.exp(-(slope * x + offset)))


def fit(name, x, y, x0):
    if name == "scaled_sigmoid":
        params, _ = curve_fit(ANALYSIS["sigmoid"], x, y,
                              p0=(max(float(y.max()), 0.05), 1, 0),
                              bounds=([0.01, 0, -20], [1, 50, 20]), maxfev=20_000)
        return float(ANALYSIS["sigmoid"](x0, *params)), ANALYSIS["sigmoid"](x, *params), params
    if name == "unit_sigmoid":
        params, _ = curve_fit(unit_sigmoid, x, y, p0=(1, 0),
                              bounds=([0, -20], [50, 20]), maxfev=20_000)
        return float(unit_sigmoid(x0, *params)), unit_sigmoid(x, *params), params
    if name == "linear":
        params = np.polyfit(x, y, 1)
        return float(np.clip(np.polyval(params, x0), 0, 1)), np.clip(np.polyval(params, x), 0, 1), params
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tables", required=True, type=Path)
    parser.add_argument("--bench", default="gpqa")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    experiment = yaml.safe_load((ROOT / "configs/frontier_eci/experiment.yaml").read_text())
    candidates = experiment["candidates"]
    releases = ANALYSIS["load_releases"](experiment)
    judges = [slug.split("/")[-1] for group in ("checkpoint", "bridge")
              for slug in experiment["judges"][group]]
    hf = pd.read_parquet(ROOT / "judge_results/frontier/eci-semiannual/hf/verdicts.parquet")
    new = pd.read_parquet(args.new_tables / "verdicts.parquet")
    verdicts = ANALYSIS["combine_verdicts"](hf, new, judges)
    quartiles = pd.read_csv(ROOT / "configs/frontier_eci/item_quartiles.csv", dtype={"item": str})
    quartiles = quartiles[quartiles.bench == args.bench]
    q1 = set(quartiles[quartiles.item_quartile == 1].item)
    verdicts = verdicts[(verdicts.bench == args.bench) & verdicts.item.astype(str).isin(q1)]
    canonical = pd.read_parquet(
        ROOT / "judge_results/frontier/eci-semiannual/canonical-source/identity.parquet",
        columns=["bench", "model", "item", "epoch", "correct", "is_canonical"])
    frontier = pd.read_parquet(args.new_tables / "rollouts.parquet")
    rollouts = ANALYSIS["merge_rollouts"](canonical, frontier)
    rollouts = rollouts.merge(quartiles[["bench", "item", "item_quartile"]],
                              on=["bench", "item"], how="inner")
    accuracy = rollouts.groupby(["item_quartile", "model"]).correct.mean()
    rows = []
    for judge in judges:
        judged = verdicts[verdicts.judge == judge]
        for target in candidates:
            prefix = ANALYSIS["prefix_verdicts"](judged, target, releases, judge)
            ability = rank(prefix).set_index("model").ability
            cutoff = ANALYSIS["prediction_cutoff"](target, judge, releases)
            possible = [model for model in ability.index
                        if model != target and ANALYSIS["release_date"](releases, model) < cutoff]
            for quartile in (2, 3, 4):
                actual = accuracy.loc[quartile]
                anchors = ability.index.intersection(actual.index).intersection(possible)
                x, y = ability[anchors].to_numpy(), actual[anchors].to_numpy()
                if target not in ability or len(anchors) < 4:
                    continue
                for calibrator in ("scaled_sigmoid", "unit_sigmoid", "linear"):
                    predicted, fitted, params = fit(calibrator, x, y, float(ability[target]))
                    rows.append({"judge": judge, "target": target, "quartile": quartile,
                                 "calibrator": calibrator, "actual": actual[target],
                                 "predicted": predicted, "error": predicted - actual[target],
                                 "anchor_mae": float(np.abs(fitted - y).mean()),
                                 "anchor_bias": float((fitted - y).mean()),
                                 "outside_anchor_range": bool(ability[target] < x.min() or ability[target] > x.max()),
                                 "n_anchors": len(anchors), "parameters": repr(np.asarray(params).tolist())})
    rows = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "fits.csv", index=False)
    metrics = (rows.groupby(["judge", "quartile", "calibrator"])
               .agg(mae=("error", lambda values: values.abs().mean()), bias=("error", "mean"),
                    anchor_mae=("anchor_mae", "mean"), anchor_bias=("anchor_bias", "mean"),
                    outside_rate=("outside_anchor_range", "mean"), n_targets=("target", "size"))
               .reset_index())
    metrics.to_csv(args.out / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
