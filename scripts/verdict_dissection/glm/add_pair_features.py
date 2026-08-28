"""Attach pair-level features (the pairwise rubric comparisons) to prepared judge inputs.

    uv run python scripts/verdict_dissection/glm/add_pair_features.py <inputs_dir> <out_dir> <pairwise.parquet>

For every design in <inputs_dir>, joins the signed comparisons (+1 = the left-shown transcript is
better) onto judge_verdict_from_pair.parquet by the rollout-pair keys, scales each column by its SD
over the design's training rows within each benchmark (not centred, so a reversed pair stays
negated, as the design matrices do) and writes the table to <out_dir>/<design>/. Rows whose pair was
not annotated get 0 (a tie) and a `pairwise_missing` flag; benchmarks that were not annotated at all
are entirely 0.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["bench", "item", "model_left", "model_right", "epoch_left", "epoch_right", "log_left", "log_right"]
MODEL = "judge_verdict_from_pair"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("pairwise", type=Path)
    args = parser.parse_args()
    pairwise = pd.read_parquet(args.pairwise)
    features = [c for c in pairwise.columns if c not in KEYS]
    for design_dir in sorted(p for p in args.inputs_dir.iterdir() if p.is_dir()):
        table = pd.read_parquet(design_dir / f"{MODEL}.parquet")
        merged = table.merge(pairwise, on=KEYS, how="left", validate="many_to_one")
        merged["pairwise_missing"] = merged[features[0]].isna().astype(int)
        for f in features:
            merged[f] = merged[f].fillna(0.0).astype(np.float32)
            train = merged[~merged.test]
            sd = train.groupby("bench")[f].std().replace(0, np.nan).fillna(1.0)
            merged[f] = merged[f] / merged.bench.map(sd).astype(np.float32)
        out = args.out_dir / design_dir.name
        out.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out / f"{MODEL}.parquet", index=False)
        covered = 1 - merged.groupby("bench").pairwise_missing.mean()
        print(f"{design_dir.name}: {len(merged)} rows; annotated share per bench {covered.round(2).to_dict()}", flush=True)


if __name__ == "__main__":
    main()
