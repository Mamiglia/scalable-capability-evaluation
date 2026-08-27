"""Rank models from the feature GLMs' out-of-sample predictions and compare with judge-derived ranks.

    uv run python scripts/verdict_dissection/glm/compare_to_judges.py <capability_fits> <pair_fits> [<pair_fits> ...] --output <csv> [--bootstrap 200]

Stitches the four folds named by --folds (default the model folds; the new-models-and-items folds give
every model unseen on unseen items) so every model is predicted while unseen, then per benchmark scores
each ranking by Spearman against the models' benchmark accuracy (src/ranking.score):
- capability GLM: each model's mean predicted logit accuracy over its transcripts;
- stronger-model GLM, and any baseline written in the same layout (scripts/baselines/): Bradley-Terry
  ability (src/ranking.rank) fitted to the held-out pair predictions, i.e. the model treated as a judge;
- each LLM judge, and all judges pooled: Bradley-Terry ability from their verdicts on the same
  pairs the GLM was scored on.
--bootstrap resamples the held-out items with replacement within each benchmark (the validation set stays
fixed) and adds p5/p95 columns for every correlation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ranking import rank, score
from ranking.scoring import resample_items

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from build_design_matrices import pool_aime  # noqa: E402
MODEL_FOLDS = ["model-fold-0", "model-fold-1", "model-fold-2", "model-fold-3"]


def stitched(fits_dir, model, folds=MODEL_FOLDS):
    """Held-out predictions from the given folds, concatenated (every model held out once)."""
    frames = []
    for wave_dir in sorted(p for p in (fits_dir / model).glob("*") if p.is_dir()):
        for fold in folds:
            path = wave_dir / fold / f"predictions_{model}.csv"
            if path.exists():
                frames.append(pd.read_csv(path).assign(wave=wave_dir.name))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["wave"])


def score_by_bench(frames_by_bench, ranking_of, validation, label, n_resamples=0, seed=0):
    """One row per benchmark: Spearman between ranking_of(frame)'s ability and accuracy on that benchmark,
    with p5/p95 from n_resamples item bootstraps of the frame when asked."""
    rng = np.random.default_rng(seed)
    rows = []
    for bench, frame in frames_by_bench.items():
        held_out = validation[validation.bench == bench]
        row = {"ranking": label, "bench": bench, **score(ranking_of(frame), held_out, "spearman")}
        if n_resamples:
            draws = [score(ranking_of(resample_items(frame, rng)), held_out, "spearman")["correlation"]
                     for _ in range(n_resamples)]
            row.update(p5=np.quantile(draws, 0.05), p95=np.quantile(draws, 0.95))
        rows.append(row)
    return rows


def mean_prediction(frame):
    """Capability GLM ranking: each model's mean predicted logit accuracy over its transcripts."""
    return frame.groupby("model", as_index=False).predicted.mean().rename(columns={"predicted": "ability"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capability_fits", type=Path)
    parser.add_argument("pair_fits", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=MODEL_FOLDS, help="designs to stitch")
    parser.add_argument("--bootstrap", type=int, default=0, help="item-bootstrap resamples for 5–95%% intervals")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    identity = pool_aime(pd.read_parquet(Path(config["features_root"]) / "raw/identity.parquet",
                                         columns=["bench", "model", "item", "correct", "is_canonical"]))
    validation = identity[identity.is_canonical].drop(columns="is_canonical")   # every canonical rollout
    rows = []

    capability = stitched(args.capability_fits, "capability_from_transcript", args.folds)
    for wave, frame in capability.groupby("wave"):
        rows += score_by_bench(dict(iter(frame.groupby("bench"))), mean_prediction, validation,
                               f"capability GLM, {wave}", args.bootstrap)

    stronger = pd.concat([stitched(fits, "stronger_model_from_pair", args.folds) for fits in args.pair_fits])
    for wave, frame in stronger.groupby("wave"):
        verdicts = frame.rename(columns={"predicted": "pick_left"}).assign(judge=wave)
        rows += score_by_bench(dict(iter(verdicts.groupby("bench"))), rank, validation,
                               f"stronger-model {wave} as judge", args.bootstrap)

    # judges are ranked from the same pairs the GLM was scored on (both shown orders of each stitched pair)
    verdicts = pd.read_parquet(Path(config["output_root"]) / "pair.parquet",
                               columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
    pair_keys = ["bench", "item", "model_left", "model_right"]
    stitched_pairs = pd.concat([stronger[pair_keys], stronger[pair_keys].rename(
        columns={"model_left": "model_right", "model_right": "model_left"})]).drop_duplicates()
    verdicts = verdicts.merge(stitched_pairs, on=pair_keys)
    for judge, frame in list(verdicts.groupby("judge")) + [("all judges pooled", verdicts)]:
        rows += score_by_bench(dict(iter(frame.groupby("bench"))), rank, validation, f"judge {judge}", args.bootstrap)

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    wide = table.pivot(index="ranking", columns="bench", values="correlation")
    wide["mean"] = wide.mean(axis=1)
    print(wide.sort_values("mean", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
