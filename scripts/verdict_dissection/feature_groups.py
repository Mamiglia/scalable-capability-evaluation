"""Merge near-duplicate features into groups and write grouped design matrices.

    uv run python scripts/verdict_dissection/feature_groups.py [--threshold 0.8] [--dense-components 20]

Nameable features (all blocks except the dense embedding and the per-benchmark SAE, which
stay as they are) are clustered on 1 − |correlation| over representation_train transcripts
(average linkage); clusters with |r| above the threshold become one group. Each group is
replaced by its first principal component, sign-aligned to and named after the group's most
central feature. The dense embedding block becomes its top principal components. Pair
matrices get the same loadings applied to the left-minus-right columns (the projection is
linear, so this equals the difference of projections). Everything is re-scaled per benchmark
on training rows. Outputs go to <output_root>/grouped/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from build_design_matrices import ROOT, scale

UNGROUPED_BLOCKS = {"embedding-dense", "embedding-sae-benchmark"}


def cluster_features(train_values, names, threshold):
    """Group labels (1-based) from average-linkage clustering on 1 − |r|."""
    correlation = np.nan_to_num(np.corrcoef(train_values.T))
    tree = linkage(squareform(1 - np.abs(correlation), checks=False), method="average")
    return fcluster(tree, t=1 - threshold, criterion="distance"), correlation


def group_loadings(train_values, correlation):
    """First principal component of one group, signed so its most central feature loads positively."""
    if train_values.shape[1] == 1:
        return np.ones(1)
    centred = train_values - train_values.mean(axis=0)
    loadings = np.linalg.svd(centred, full_matrices=False)[2][0]
    central = np.abs(correlation).mean(axis=0).argmax()
    return loadings * np.sign(loadings[central]), central


def build_groups(transcript, names, threshold):
    """Table feature → group name and loading, plus the loading matrix (features × groups)."""
    train = transcript.loc[transcript.split == "representation_train", names].to_numpy(np.float64)
    labels, correlation = cluster_features(train, names, threshold)
    rows, columns, loading_matrix = [], [], np.zeros((len(names), labels.max()))
    for group_id in range(1, labels.max() + 1):
        members = np.flatnonzero(labels == group_id)
        if len(members) == 1:
            loadings, central = np.ones(1), 0
        else:
            loadings, central = group_loadings(train[:, members], correlation[np.ix_(members, members)])
        group_name = names[members[central]]
        columns.append(group_name)
        loading_matrix[members, group_id - 1] = loadings
        rows.extend({"feature": names[m], "group": group_name, "loading": float(w), "group_size": len(members)}
                    for m, w in zip(members, loadings))
    return pd.DataFrame(rows), pd.DataFrame(loading_matrix, index=names, columns=columns)


def dense_components(transcript, names, n_components):
    """Loadings (features × components) of the dense block's top principal components on training rows."""
    train = transcript.loc[transcript.split == "representation_train", names].to_numpy(np.float64)
    loadings = np.linalg.svd(train - train.mean(axis=0), full_matrices=False)[2][:n_components].T
    return pd.DataFrame(loadings, index=names, columns=[f"dense_component_{i}" for i in range(n_components)])


def project(frame, loadings, prefix=""):
    """Apply loadings to frame's columns (pair frames use the d_ prefix), returning the new columns."""
    values = frame[[f"{prefix}{name}" for name in loadings.index]].to_numpy(np.float32)
    return pd.DataFrame(values @ loadings.to_numpy(np.float32), columns=[f"{prefix}{c}" for c in loadings.columns],
                        index=frame.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/design_matrices.yaml")
    parser.add_argument("--threshold", type=float, default=0.8, help="|correlation| at which features merge")
    parser.add_argument("--dense-components", type=int, default=20)
    args = parser.parse_args()
    root = Path(yaml.safe_load(args.config.read_text())["output_root"])
    out = root / "grouped"
    out.mkdir(exist_ok=True)

    dictionary = pd.read_parquet(root / "feature-dictionary.parquet")
    nameable = dictionary[~dictionary.block.isin(UNGROUPED_BLOCKS)].feature.tolist()
    dense = dictionary[dictionary.block == "embedding-dense"].feature.tolist()
    sae_benchmark = dictionary[dictionary.block == "embedding-sae-benchmark"].feature.tolist()
    transcript = pd.read_parquet(root / "transcript.parquet")
    pair = pd.read_parquet(root / "pair.parquet")

    groups, group_loading = build_groups(transcript, nameable, args.threshold)
    dense_loading = dense_components(transcript, dense, args.dense_components)
    print(f"{len(nameable)} nameable features -> {group_loading.shape[1]} groups at |r| >= {args.threshold}; "
          f"dense {len(dense)} -> {args.dense_components} components")

    meta = [c for c in transcript.columns if c not in dictionary.feature.values]
    grouped_transcript = pd.concat([transcript[meta], project(transcript, group_loading),
                                    project(transcript, dense_loading), transcript[sae_benchmark]], axis=1)
    pair_meta = [c for c in pair.columns if not c.startswith("d_")]
    grouped_pair = pd.concat([pair[pair_meta], project(pair, group_loading, "d_"), project(pair, dense_loading, "d_"),
                              pair[[f"d_{c}" for c in sae_benchmark]]], axis=1)
    features = list(group_loading.columns) + list(dense_loading.columns) + sae_benchmark
    grouped_transcript, transcript_scaling = scale(grouped_transcript, features, centre=True)
    grouped_pair, pair_scaling = scale(grouped_pair, [f"d_{f}" for f in features], centre=False)

    block = dictionary.set_index("feature").block
    grouped_dictionary = pd.DataFrame({
        "feature": features,
        "block": [block[f] for f in group_loading.columns] + ["embedding-dense"] * len(dense_loading.columns)
                 + ["embedding-sae-benchmark"] * len(sae_benchmark),
        "group_size": [int((groups.group == g).sum()) for g in group_loading.columns]
                      + [len(dense)] * len(dense_loading.columns) + [1] * len(sae_benchmark),
    })
    for name, table in {"transcript": grouped_transcript, "pair": grouped_pair, "feature-groups": groups,
                        "feature-dictionary": grouped_dictionary, "scaling-transcript": transcript_scaling,
                        "scaling-pair": pair_scaling}.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
    dense_loading.to_parquet(out / "dense-loadings.parquet")
    print(grouped_dictionary.groupby("block", sort=False).size().to_string())


if __name__ == "__main__":
    main()
