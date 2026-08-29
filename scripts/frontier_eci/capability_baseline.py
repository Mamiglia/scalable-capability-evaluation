"""Rolling Epoch-ECI regression baseline for the frontier accuracy figures."""

import argparse
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = run_path(ROOT / "scripts/frontier_eci/analyze.py")
EPOCH_NAMES = {
    "gpt-4o-1120": "GPT-4o (Nov 2024)",
    "claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "o3": "o3",
    "gpt-5": "GPT-5",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "claude-opus-4.6": "Claude Opus 4.6",
    "claude-opus-4.7": "Claude Opus 4.7",
    "kimi-k3": "Kimi K3",
}


def fit_rolling(actual, eci, candidates):
    rows = []
    for index, model in enumerate(candidates):
        train = [name for name in candidates[:index]
                 if pd.notna(actual.get(name)) and pd.notna(eci.get(name))]
        predicted = low = high = np.nan
        if len(train) >= 2:
            x = eci.reindex(train).to_numpy(float)
            y = actual.reindex(train).to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            x0 = float(eci[model])
            predicted = float(np.clip(slope * x0 + intercept, 0, 1))
            if len(train) >= 3:
                residual = y - (slope * x + intercept)
                sigma = np.sqrt(np.sum(residual ** 2) / (len(train) - 2))
                se = sigma * np.sqrt(1 / len(train) + (x0 - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
                width = t.ppf(0.975, len(train) - 2) * se
                low, high = max(0, predicted - width), min(1, predicted + width)
        rows.append({"model": model, "actual": actual.get(model, np.nan),
                     "predicted": predicted, "predicted_low": low,
                     "predicted_high": high, "n_train": len(train)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    experiment = yaml.safe_load((ROOT / "configs/frontier_eci/experiment.yaml").read_text())
    candidates = experiment["candidates"]
    epoch = pd.read_csv(ROOT / "external_data/epoch-eci/eci_scores.csv").set_index("Model").eci
    eci = pd.Series({model: float(epoch[EPOCH_NAMES[model]]) for model in candidates})
    source = pd.read_csv(args.analysis_dir / "predictions.csv")
    releases = experiment["releases"]
    judge_releases = ANALYSIS["load_releases"](experiment)
    judges = [slug.split("/")[-1] for group in ("checkpoint", "bridge")
              for slug in experiment["judges"][group]]
    outputs = []
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for (bench, quartile), table in source.groupby(["bench", "target_quartile"]):
        actual = table.groupby("model").actual.first()
        baseline = fit_rolling(actual, eci, candidates)
        baseline.insert(0, "target_quartile", quartile)
        baseline.insert(0, "bench", bench)
        baseline["eci"] = baseline.model.map(eci)
        baseline["target_release"] = baseline.model.map(releases)
        outputs.append(baseline)
        for judge in judges:
            plotted = baseline.copy()
            plotted["judge_release"] = ANALYSIS["release_date"](judge_releases, judge)
            for style in ("band", "points"):
                ANALYSIS["plot_frontier_style"](
                    plotted, judge, bench, quartile, figures, style,
                    prediction_label="rolling Epoch-ECI baseline",
                    filename_prefix="eci-baseline_")
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = pd.concat(outputs, ignore_index=True)
    outputs.to_csv(args.out / "predictions.csv", index=False)
    scored = outputs.dropna(subset=["actual", "predicted"]).assign(
        error=lambda frame: frame.predicted - frame.actual)
    metrics = (scored.groupby(["bench", "target_quartile"])
               .agg(mae=("error", lambda values: values.abs().mean()),
                    bias=("error", "mean"),
                    spearman=("predicted", lambda values:
                              spearmanr(values, scored.loc[values.index, "actual"]).statistic),
                    n_models=("model", "size")).reset_index())
    metrics.to_csv(args.out / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
