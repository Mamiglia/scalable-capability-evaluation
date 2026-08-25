"""Aggregate frozen content-addressed surprisal caches into raw transcript and pair tables."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from features_basic import basic_features
from llm_surprisal import surprisal_features

TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
KEYS = TRANSCRIPT_KEYS + ["sample_epoch", "content_hash"]


def aggregate_features(store, cache_dir):
    store = store[store.is_canonical & store.correct].reset_index(drop=True)
    rows = []
    for index, row in enumerate(store.itertuples(index=False), 1):
        parts = json.loads(row.transcript_parts_json)
        assistant_text = "\n\n".join(part["content"] for part in parts if part["role"] == "assistant")
        features = basic_features(assistant_text)
        features.update(surprisal_features(
            row.transcript, parts, row.bench, row.model, row.item, row.epoch,
            row.log, row.content_hash, cache_dir))
        rows.append({key: getattr(row, key) for key in KEYS} | features)
        if index % 500 == 0:
            print(f"features {index}/{len(store)}", flush=True)
    return pd.DataFrame(rows)


def build_pairs(verdicts, raw):
    raw = raw.sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    features = [column for column in raw if column not in KEYS]
    left = raw.rename(columns={"model": "model_left", "epoch": "epoch_left", "log": "log_left"})
    right = raw.rename(columns={"model": "model_right", "epoch": "epoch_right", "log": "log_right"})
    pairs = verdicts.drop(columns=["judge_input", "judge_sample_metadata_json"], errors="ignore")
    pairs = pairs.merge(left, on=["bench", "item", "model_left", "epoch_left", "log_left"],
                        how="inner", validate="many_to_one")
    pairs = pairs.merge(right, on=["bench", "item", "model_right", "epoch_right", "log_right"],
                        how="inner", suffixes=("_l", "_r"), validate="many_to_one")
    for feature in features:
        pairs[f"d_{feature}"] = pairs[f"{feature}_l"] - pairs[f"{feature}_r"]
    return pairs.drop(columns=[f"{feature}_{side}" for feature in features for side in ("l", "r")])


def write_parquet(frame, path):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = aggregate_features(pd.read_parquet(args.store), args.cache_dir)
    pairs = build_pairs(pd.read_parquet(args.verdicts), raw)
    write_parquet(raw, args.output_dir / "raw_features_with_surprisal.parquet")
    write_parquet(pairs, args.output_dir / "pairs_raw_with_surprisal.parquet")
    print(f"raw={len(raw)} pairs={len(pairs)} features={len(raw.columns) - len(KEYS)}")
