"""Audit provisional surprisal caches against a frozen content-addressed store."""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["bench", "model", "item", "epoch", "log"]
SCORERS = ("llama", "qwen", "gemma")


def match_legacy_rows(legacy, frozen):
    """Match frozen rows to the first content that could have populated each old cache key."""
    old = (legacy.drop_duplicates(KEYS, keep="first")[KEYS + ["question", "transcript"]]
           .rename(columns={"question": "legacy_question", "transcript": "legacy_transcript"}))
    matched = frozen.merge(old, on=KEYS, how="left", validate="many_to_one", indicator=True)
    matched["status"] = np.where(
        matched._merge == "left_only", "legacy_key_absent",
        np.where((matched.question == matched.legacy_question)
                 & (matched.transcript == matched.legacy_transcript), "exact", "text_mismatch"))
    return matched.drop(columns="_merge")


def legacy_cache_file(cache_dir, scorer, row):
    safe = lambda value: re.sub(r"[^\w.-]", "_", str(value))
    name = f"{safe(row.model)}__{safe(row.item)}__ep{row.epoch}__{safe(row.log)}.npz"
    return Path(cache_dir) / scorer / row.bench / name


def content_cache_file(cache_dir, row):
    safe = lambda value: re.sub(r"[^\w.-]", "_", str(value))
    name = (f"{safe(row.model)}__{safe(row.item)}__ep{row.epoch}__{safe(row.log)}"
            f"__{row.content_hash[:16]}.npz")
    return Path(cache_dir) / row.scorer / row.bench / name


def cache_is_valid(path, transcript):
    try:
        with np.load(path) as cache:
            arrays = [cache[name] for name in ("full", "noq", "local")]
            offsets = cache["offsets"]
            return (all(len(array) == len(offsets) and np.isfinite(array).all() for array in arrays)
                    and offsets.shape == (len(offsets), 2)
                    and (not len(offsets) or offsets[-1, 1] <= len(transcript)))
    except (OSError, KeyError, ValueError):
        return False


def audit(legacy_store, frozen_store, cache_dir):
    legacy = pd.read_parquet(legacy_store)
    frozen = pd.read_parquet(frozen_store)
    frozen = frozen[frozen.is_canonical & frozen.correct].reset_index(drop=True)
    matched = match_legacy_rows(legacy, frozen)
    rows = []
    for scorer in SCORERS:
        for row in matched.itertuples(index=False):
            status = row.status
            path = legacy_cache_file(cache_dir, scorer, row)
            if status == "exact":
                status = ("reusable" if path.exists() and cache_is_valid(path, row.transcript)
                          else "cache_invalid" if path.exists() else "cache_missing")
            rows.append({key: getattr(row, key) for key in KEYS + ["sample_epoch", "content_hash"]}
                        | {"scorer": scorer, "legacy_cache": str(path), "status": status})
    return pd.DataFrame(rows)


def migrate_reusable(audit_rows, target_dir):
    """Copy validated legacy arrays into new content-addressed files; never alter legacy caches."""
    migrated = []
    for row in audit_rows[audit_rows.status == "reusable"].itertuples(index=False):
        target = content_cache_file(target_dir, row)
        if target.exists():
            if np.load(target)["content_hash"].item() != row.content_hash:
                raise ValueError(f"content hash mismatch: {target}")
        else:
            with np.load(row.legacy_cache) as source:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp.npz")
                np.savez(temporary, full=source["full"], noq=source["noq"],
                         local=source["local"], offsets=source["offsets"],
                         content_hash=row.content_hash)
                os.replace(temporary, target)
        migrated.append(row._asdict() | {"new_cache": str(target)})
    return pd.DataFrame(migrated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_store", type=Path)
    parser.add_argument("frozen_store", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--migrate-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = audit(args.legacy_store, args.frozen_store, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby(["scorer", "status"]).size().unstack(fill_value=0).to_string())
    if args.migrate_dir:
        migrated = migrate_reusable(result, args.migrate_dir)
        print(f"migrated {len(migrated)} reusable cache files -> {args.migrate_dir}")
