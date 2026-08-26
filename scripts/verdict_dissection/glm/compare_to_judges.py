"""Rank models from the feature GLMs' out-of-sample predictions and compare with judge-derived ranks.

    uv run python scripts/verdict_dissection/glm/compare_to_judges.py <capability_fits> <pair_fits> <output_csv>

Stitches the four model folds so every model is predicted while unseen, then per benchmark:
- capability GLM: Spearman between each model's mean predicted logit accuracy and its accuracy;
- stronger-model GLM: Bradley-Terry ability fitted to the GLM's held-out pair predictions (the
  GLM treated as a judge), Spearman against accuracy;
- each LLM judge and all judges pooled: Bradley-Terry ability from their verdicts, same score.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
MODEL_FOLDS = ["model-fold-0", "model-fold-1", "model-fold-2", "model-fold-3"]


def bradley_terry(pairs, pick_column):
    """Model abilities from left picks (0/1 or probabilities), with a shared left-position bonus and a light ridge."""
    models = sorted(set(pairs.model_left) | set(pairs.model_right))
    index = {m: i for i, m in enumerate(models)}
    left, right = pairs.model_left.map(index).to_numpy(), pairs.model_right.map(index).to_numpy()
    y = pairs[pick_column].to_numpy(float)

    def loss(theta):
        logits = theta[left] - theta[right] + theta[-1]
        return np.sum(y * np.logaddexp(0, -logits) + (1 - y) * np.logaddexp(0, logits)) + 0.01 * np.sum(theta[:-1] ** 2)

    theta = minimize(loss, np.zeros(len(models) + 1), method="L-BFGS-B").x
    return pd.Series(theta[:-1], index=models)


def stitched(fits_dir, model):
    """Held-out predictions from the four model folds, concatenated (every model held out once)."""
    frames = []
    for wave_dir in sorted(p for p in (fits_dir / model).iterdir() if p.is_dir()):
        for fold in MODEL_FOLDS:
            path = wave_dir / fold / f"predictions_{model}.csv"
            if path.exists():
                frames.append(pd.read_csv(path).assign(wave=wave_dir.name))
    return pd.concat(frames) if frames else pd.DataFrame()


def score(ability_by_bench, accuracy, label):
    rows = []
    for bench, ability in ability_by_bench.items():
        truth = accuracy[accuracy.bench == bench].set_index("model").accuracy
        common = ability.index.intersection(truth.index)
        rows.append({"ranking": label, "bench": bench, "n_models": len(common),
                     "spearman": spearmanr(ability[common], truth[common]).statistic})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capability_fits", type=Path)
    parser.add_argument("pair_fits", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    accuracy = pd.read_parquet(Path(config["features_root"]) / "raw/model-benchmark-accuracy.parquet")
    rows = []

    capability = stitched(args.capability_fits, "capability_from_transcript")
    for wave, frame in capability.groupby("wave"):
        means = frame.groupby(["bench", "model"]).predicted.mean()
        rows += score({b: means[b] for b in means.index.levels[0]}, accuracy, f"capability GLM, {wave}")

    stronger = stitched(args.pair_fits, "stronger_model_from_pair")
    for wave, frame in stronger.groupby("wave"):
        rows += score({b: bradley_terry(g, "predicted") for b, g in frame.groupby("bench")}, accuracy,
                      f"stronger-model GLM as judge, {wave}")

    verdicts = pd.read_parquet(Path(config["output_root"]) / "pair.parquet",
                               columns=["bench", "judge", "model_left", "model_right", "pick_left"])
    for judge, frame in list(verdicts.groupby("judge")) + [("all judges pooled", verdicts)]:
        rows += score({b: bradley_terry(g, "pick_left") for b, g in frame.groupby("bench")}, accuracy, f"judge {judge}")

    table = pd.DataFrame(rows)
    table.to_csv(args.output_csv, index=False)
    wide = table.pivot(index="ranking", columns="bench", values="spearman")
    wide["mean"] = wide.mean(axis=1)
    print(wide.sort_values("mean", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
