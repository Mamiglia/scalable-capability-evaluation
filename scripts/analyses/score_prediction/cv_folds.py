"""Cross-validation folds for the score model.

rows.parquet -> cv_folds.parquet (row_id, regime, fold). Three regimes:
held-out items, held-out pairs, held-out judges. Rows move between train
and test in groups (all rows of an item / pair together) so shared
transcripts never leak across the boundary; the file freezes one fold
assignment shared by every downstream model.

Usage:
  uv run python scripts/analyses/score_prediction/cv_folds.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def grouped_folds(rows: pd.DataFrame, group_col: str, n_folds: int,
                  seed: int) -> pd.Series:
    """Fold per row; groups stay whole, dealt largest-first to the
    currently smallest fold so fold sizes balance."""
    sizes = rows.groupby(group_col).size().sample(frac=1, random_state=seed)
    fold_of_group = {}
    fold_loads = np.zeros(n_folds, dtype=int)
    for group, size in sizes.sort_values(ascending=False, kind="stable").items():
        fold = int(fold_loads.argmin())
        fold_of_group[group] = fold
        fold_loads[fold] += size
    return rows[group_col].map(fold_of_group)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    seed = cfg["seeds"][0]

    rows = pd.read_parquet(out / "rows.parquet")
    judges = sorted(rows["judge"].unique())
    regimes = {
        "items": ("item", grouped_folds(rows, "item", cfg["folds"]["items"], seed)),
        "pairs": ("pair_key", grouped_folds(rows, "pair_key", cfg["folds"]["pairs"], seed)),
        "judges": ("judge", rows["judge"].map({j: i for i, j in enumerate(judges)})),
    }

    frames = []
    for regime, (group_col, fold) in regimes.items():
        assigned = rows.assign(fold=fold)
        assert assigned["fold"].notna().all(), f"{regime}: unassigned rows"
        assert (assigned.groupby(group_col)["fold"].nunique() == 1).all(), \
            f"{regime}: a {group_col} spans folds"
        assert (assigned.groupby(["judge", "major", "minor", "item"])["fold"]
                .nunique() == 1).all(), f"{regime}: orders of an item split"
        summary = assigned.groupby("fold").agg(
            n_rows=("row_id", "size"), n_groups=(group_col, "nunique"))
        print(f"\n{regime} regime (group = {group_col}):")
        print(summary.to_string())
        if regime == "pairs":
            for fold_idx, fold_rows in assigned.groupby("fold"):
                print(f"  fold {fold_idx} holds out: "
                      f"{', '.join(sorted(fold_rows['pair_key'].unique()))}")
        frames.append(assigned.assign(regime=regime)[["row_id", "regime", "fold"]])

    cv = pd.concat(frames, ignore_index=True)
    assert len(cv) == 3 * len(rows), "rows must appear once per regime"
    cv.to_parquet(out / "cv_folds.parquet", index=False)
    print(f"\njudges fold order: {', '.join(f'{i}={j}' for i, j in enumerate(judges))}")
    print(f"{len(cv)} assignments -> {out}/cv_folds.parquet")


if __name__ == "__main__":
    main()
