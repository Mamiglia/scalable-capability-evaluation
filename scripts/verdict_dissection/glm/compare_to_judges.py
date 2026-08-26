"""Rank models from the feature GLMs' out-of-sample predictions and compare with judge-derived ranks.

    uv run python scripts/verdict_dissection/glm/compare_to_judges.py <capability_fits> <pair_fits> [<pair_fits> ...] --output <csv>

Stitches the four model folds so every model is predicted while unseen, then per benchmark scores
each ranking by Spearman against the models' benchmark accuracy (src/ranking.score):
- capability GLM: each model's mean predicted logit accuracy over its transcripts;
- stronger-model GLM, and any baseline written in the same layout (scripts/baselines/): Bradley-Terry
  ability (src/ranking.rank) fitted to the held-out pair predictions, i.e. the model treated as a judge;
- each LLM judge, and all judges pooled: Bradley-Terry ability from their verdicts.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from ranking import rank, score

ROOT = Path(__file__).resolve().parents[3]
MODEL_FOLDS = ["model-fold-0", "model-fold-1", "model-fold-2", "model-fold-3"]


def stitched(fits_dir, model):
    """Held-out predictions from the four model folds, concatenated (every model held out once)."""
    frames = []
    for wave_dir in sorted(p for p in (fits_dir / model).iterdir() if p.is_dir()):
        for fold in MODEL_FOLDS:
            path = wave_dir / fold / f"predictions_{model}.csv"
            if path.exists():
                frames.append(pd.read_csv(path).assign(wave=wave_dir.name))
    return pd.concat(frames) if frames else pd.DataFrame()


def score_by_bench(ranking_by_bench, validation, label):
    """One row per benchmark: Spearman between the ranking's ability and accuracy on that benchmark."""
    return [{"ranking": label, "bench": bench, **score(ranking, validation[validation.bench == bench], "spearman")}
            for bench, ranking in ranking_by_bench.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capability_fits", type=Path)
    parser.add_argument("pair_fits", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    identity = pd.read_parquet(Path(config["features_root"]) / "raw/identity.parquet",
                               columns=["bench", "model", "item", "correct", "is_canonical"])
    validation = identity[identity.is_canonical].drop(columns="is_canonical")   # every canonical rollout
    rows = []

    capability = stitched(args.capability_fits, "capability_from_transcript")
    for wave, frame in capability.groupby("wave"):
        means = frame.groupby(["bench", "model"], as_index=False).predicted.mean().rename(columns={"predicted": "ability"})
        rows += score_by_bench({b: g for b, g in means.groupby("bench")}, validation, f"capability GLM, {wave}")

    stronger = pd.concat([stitched(fits, "stronger_model_from_pair") for fits in args.pair_fits])
    for wave, frame in stronger.groupby("wave"):
        verdicts = frame.rename(columns={"predicted": "pick_left"}).assign(judge=wave)
        rows += score_by_bench({b: rank(g) for b, g in verdicts.groupby("bench")}, validation,
                               f"stronger-model {wave} as judge")

    verdicts = pd.read_parquet(Path(config["output_root"]) / "pair.parquet",
                               columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
    for judge, frame in list(verdicts.groupby("judge")) + [("all judges pooled", verdicts)]:
        rows += score_by_bench({b: rank(g) for b, g in frame.groupby("bench")}, validation, f"judge {judge}")

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    wide = table.pivot(index="ranking", columns="bench", values="correlation")
    wide["mean"] = wide.mean(axis=1)
    print(wide.sort_values("mean", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
