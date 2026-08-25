"""Flat transcript x feature and pair x feature tables from the dense and SAE artifacts.

Rows are keyed like workstream A's tables (bench, model, item, epoch, log,
sample_epoch, content_hash); the pair table joins verdicts the way
aggregate_surprisal.build_pairs does and holds left - right differences.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcript_embeddings import verify_artifact_files

TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
KEYS = TRANSCRIPT_KEYS + ["sample_epoch", "content_hash"]
ROW_INFO = ["split_role", "chunk_count", "transcript_tokens"]


def verified(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_artifact_files(directory, manifest)
    return manifest


def transcript_table(dense_dir: Path, sae_dir: Path, family: str) -> pd.DataFrame:
    verified(dense_dir)
    verified(sae_dir)
    rows = pd.read_parquet(dense_dir / "rows.parquet")
    audit = pd.read_parquet(dense_dir / "audit_rows.parquet")
    if not pd.read_parquet(sae_dir / "rows.parquet")["transcript_key"].equals(rows["transcript_key"]):
        raise ValueError("SAE rows are not in dense row order")
    keys = rows[["transcript_key", *ROW_INFO]].merge(
        audit[["transcript_key", *KEYS]], on="transcript_key", how="left", validate="one_to_one")
    if keys[KEYS].isna().any().any():
        raise ValueError("audit map does not cover every dense row")
    dense = np.load(dense_dir / "transcript_only__final.npy", allow_pickle=False)
    activations = np.load(sae_dir / "activations.npy", allow_pickle=False)
    labels = np.load(sae_dir / "feature_labels.npy", allow_pickle=False)
    if len(dense) != len(rows) or len(activations) != len(rows) or activations.shape[1] != len(labels):
        raise ValueError("dense/SAE arrays do not align with rows")
    return pd.concat([
        keys[KEYS + ROW_INFO],
        pd.DataFrame(dense.astype(np.float32),
                     columns=[f"{family}_dim_{i:04d}" for i in range(dense.shape[1])]),
        pd.DataFrame(activations.astype(np.float32),
                     columns=[f"sae_{str(label).replace(':', '_')}" for label in labels]),
    ], axis=1)


def build_pairs(verdicts: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS).reset_index(drop=True)
    features = [column for column in raw if column not in KEYS and column != "split_role"]
    values = raw[features].to_numpy(np.float32)
    keyed = raw[KEYS + ["split_role"]].assign(row=np.arange(len(raw)))
    pairs = verdicts.drop(columns=["judge_input", "judge_sample_metadata_json"], errors="ignore")
    for side in ("left", "right"):
        side_keys = keyed.rename(columns={
            "model": f"model_{side}", "epoch": f"epoch_{side}", "log": f"log_{side}",
            "sample_epoch": f"sample_epoch_{side[0]}", "content_hash": f"content_hash_{side[0]}",
            "row": f"row_{side}"})
        if side == "right":
            side_keys = side_keys.drop(columns="split_role")
        pairs = pairs.merge(side_keys, on=["bench", "item", f"model_{side}", f"epoch_{side}", f"log_{side}"],
                            how="inner", validate="many_to_one")
    diffs = values[pairs.pop("row_left").to_numpy()] - values[pairs.pop("row_right").to_numpy()]
    return pd.concat([pairs.reset_index(drop=True),
                      pd.DataFrame(diffs, columns=[f"d_{feature}" for feature in features])], axis=1)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    frame.to_parquet(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--family", default="qwen")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transcripts = transcript_table(args.dense_dir, args.sae_dir, args.family)
    pairs = build_pairs(pd.read_parquet(args.verdicts), transcripts)
    write_parquet(transcripts, args.output_dir / f"transcript_features_{args.family}.parquet")
    write_parquet(pairs, args.output_dir / f"pair_features_{args.family}.parquet")
    n_sae = sum(column.startswith("sae_") for column in transcripts)
    print(f"transcripts={len(transcripts)} pairs={len(pairs)} "
          f"dense_dims={sum(column.startswith(f'{args.family}_dim_') for column in transcripts)} sae_features={n_sae}")
