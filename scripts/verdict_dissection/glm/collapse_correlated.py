"""Collapse near-duplicate predictors: keep one representative per cluster of |r| >= threshold.

    uv run python scripts/verdict_dissection/glm/collapse_correlated.py predictors-all.yaml predictors-all-dedup.yaml [--threshold 0.9]

Correlations are over representation_train transcripts of the design matrices (all benchmarks,
z-scaled per benchmark). Average-linkage clustering on 1 - |r|; each cluster keeps the member
with the shortest name (the plainest), listed with the members it stands for.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[3]


def representatives(values, names, threshold):
    correlation = np.nan_to_num(np.corrcoef(values.T))
    distance = 1 - np.abs(correlation)
    np.fill_diagonal(distance, 0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"), 1 - threshold, criterion="distance")
    groups = {}
    for name, label in zip(names, labels):
        groups.setdefault(label, []).append(name)
    return {min(members, key=len): sorted(members) for members in groups.values()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    names = yaml.safe_load(args.predictors.read_text())
    table = pd.read_parquet(Path(config["output_root"]) / "transcript.parquet", columns=["split"] + names)
    train = table[table.split == "representation_train"]
    kept = representatives(train[names].to_numpy(np.float64), names, args.threshold)
    lines = [f"# {args.predictors.name} collapsed at |r| >= {args.threshold}: {len(kept)} of {len(names)} kept; each line notes what it stands for"]
    for name in names:                               # keep the original order
        if name in kept:
            others = [m for m in kept[name] if m != name]
            lines.append(f"- {name}" + (f"    # also {', '.join(others)}" if others else ""))
    args.output.write_text("\n".join(lines) + "\n")
    print(f"{len(kept)} of {len(names)} predictors kept -> {args.output}")
    merged = {k: v for k, v in kept.items() if len(v) > 1}
    for k, v in sorted(merged.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {k}: {', '.join(m for m in v if m != k)}")


if __name__ == "__main__":
    main()
