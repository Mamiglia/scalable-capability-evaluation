"""Export cached per-token surprisal passes as a partitioned Parquet dataset."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from llm_surprisal import SCORERS, assistant_token_mask, cache_file
from llm_multiscale import SETTINGS, _matches


KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
SCHEMA = pa.schema([
    ("bench", pa.string()), ("model", pa.string()), ("item", pa.string()),
    ("epoch", pa.int64()), ("log", pa.string()), ("sample_epoch", pa.int64()),
    ("content_hash", pa.string()), ("scorer", pa.string()),
    ("token_index", pa.int64()), ("char_start", pa.int64()), ("char_end", pa.int64()),
    ("assistant", pa.bool_()), ("long_context_eligible", pa.bool_()),
    ("full_bits", pa.float64()), ("noq_bits", pa.float64()), ("local_bits", pa.float64()),
    ("question_gain_bits", pa.float64()), ("total_context_gain_bits", pa.float64()),
    ("long_context_gain_bits", pa.float64()),
])
ROLLING_SCHEMA = pa.schema([
    ("bench", pa.string()), ("model", pa.string()), ("item", pa.string()),
    ("epoch", pa.int64()), ("log", pa.string()), ("sample_epoch", pa.int64()),
    ("content_hash", pa.string()), ("scorer", pa.string()),
    ("token_index", pa.int64()), ("char_start", pa.int64()), ("char_end", pa.int64()),
    ("assistant", pa.bool_()), ("context_eligible", pa.bool_()),
    ("tail_eligible", pa.bool_()), ("short_bits", pa.float64()),
    ("long_bits", pa.float64()), ("long_question_bits", pa.float64()),
    ("context_gain_bits", pa.float64()), ("question_gain_bits", pa.float64()),
])


def _bits(cache, name, n, path):
    values = cache[name]
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{path}: {name} length does not match offsets")
    return values


def _number(value):
    return float(value) if np.isfinite(value) else None


def export_token_predictability(store_path, cache_dir, output_dir, batch_rows=100_000,
                                rolling=False):
    """Write every available cached scorer/token pair for canonical-correct rows."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    cache_dir = Path(cache_dir)
    store = pd.read_parquet(store_path)
    selected = store[store["is_canonical"].eq(True) & store["correct"].eq(True)]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    batch, written, part = [], 0, 0

    def flush():
        nonlocal batch, part
        if batch:
            schema = ROLLING_SCHEMA if rolling else SCHEMA
            pq.write_to_dataset(pa.Table.from_pylist(batch, schema=schema), root_path=stage,
                                partition_cols=["scorer", "bench"],
                                basename_template=f"part-{part}-{{i}}.parquet")
            batch, part = [], part + 1

    try:
        for row in selected.itertuples(index=False):
            parts = json.loads(row.transcript_parts_json)
            scorer_dirs = ([cache_dir / scorer for scorer in SCORERS] if rolling else
                           sorted(p for p in cache_dir.iterdir() if p.is_dir()))
            for scorer_dir in scorer_dirs:
                scorer = scorer_dir.name
                path = cache_file(cache_dir, scorer, row.bench, row.model, row.item,
                                  row.epoch, row.log, row.content_hash)
                if not path.exists():
                    if rolling:
                        raise FileNotFoundError(f"missing rolling cache: {path}")
                    continue
                with np.load(path, allow_pickle=False) as cache:
                    if cache["content_hash"].item() != row.content_hash:
                        raise ValueError(f"cache content hash mismatch: {path}")
                    offsets = cache["offsets"]
                    if offsets.ndim != 2 or offsets.shape[1] != 2:
                        raise ValueError(f"{path}: offsets must have shape (tokens, 2)")
                    n = len(offsets)
                    assistant = assistant_token_mask(row.transcript, offsets, parts)
                    if rolling:
                        if not _matches(cache, row.content_hash, offsets, assistant):
                            raise ValueError(f"rolling cache settings mismatch: {path}")
                        short, long, long_question = (
                            _bits(cache, name, n, path)
                            for name in ("short", "long", "long_question"))
                    else:
                        full, noq, local = (_bits(cache, name, n, path)
                                            for name in ("full", "noq", "local"))
                base = {key: getattr(row, key) for key in KEYS} | {"scorer": scorer}
                if rolling:
                    for token_index, ((char_start, char_end), is_assistant, short_bit,
                                      long_bit, question_bit) in enumerate(zip(
                                          offsets, assistant, short, long, long_question)):
                        batch.append(base | {
                            "token_index": token_index, "char_start": int(char_start),
                            "char_end": int(char_end), "assistant": bool(is_assistant),
                            "context_eligible": token_index >= 64,
                            "tail_eligible": token_index >= 8192,
                            "short_bits": _number(short_bit), "long_bits": _number(long_bit),
                            "long_question_bits": _number(question_bit),
                            "context_gain_bits": (_number(short_bit - long_bit)
                                                  if token_index >= 64 else None),
                            "question_gain_bits": _number(long_bit - question_bit),
                        })
                        written += 1
                        if len(batch) == batch_rows:
                            flush()
                    continue
                for token_index, ((char_start, char_end), is_assistant, f, q, l) in enumerate(
                        zip(offsets, assistant, full, noq, local)):
                    eligible = token_index >= 64
                    batch.append(base | {
                        "token_index": token_index, "char_start": int(char_start),
                        "char_end": int(char_end), "assistant": bool(is_assistant),
                        "long_context_eligible": eligible, "full_bits": float(f),
                        "noq_bits": float(q), "local_bits": float(l),
                        "question_gain_bits": float(q - f),
                        "total_context_gain_bits": float(l - f),
                        "long_context_gain_bits": float(l - q) if eligible else None,
                    })
                    written += 1
                    if len(batch) == batch_rows:
                        flush()
        flush()
        if not written:
            raise ValueError("no scored canonical-correct transcripts found")
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        os.rename(stage, output_dir)
        return written
    except Exception:
        print(f"export failed; staging output remains at {stage}", file=sys.stderr)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store")
    parser.add_argument("cache_dir")
    parser.add_argument("output_dir", nargs="?", default="raw/token-predictability")
    parser.add_argument("--rolling", action="store_true")
    args = parser.parse_args()
    print(export_token_predictability(args.store, args.cache_dir, args.output_dir,
                                      rolling=args.rolling))
