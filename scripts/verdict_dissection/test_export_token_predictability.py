"""Token-predictability export checks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from export_token_predictability import export_token_predictability


def test_exports_structured_tokens_to_partitioned_parquet(tmp_path):
    """Changing spans, gains, or partitions makes this real export fail."""
    text = "[Assistant]: marker [Tool]: is prose\n\n[Tool]: actual tool"
    assistant = "marker [Tool]: is prose"
    parts = [
        {"role": "assistant", "content": assistant, "start": 13,
         "end": 13 + len(assistant), "bare": False},
        {"role": "tool", "content": "actual tool", "start": 45,
         "end": len(text), "bare": False},
    ]
    content_hash = "a" * 64
    store = pd.DataFrame([{
        "bench": "toy", "model": "model", "item": "item", "epoch": 0,
        "log": "run", "sample_epoch": 2, "content_hash": content_hash,
        "transcript": text, "transcript_parts_json": json.dumps(parts),
        "correct": True, "is_canonical": True,
    }, {
        "bench": "skip", "model": "model", "item": "item", "epoch": 0,
        "log": "run", "sample_epoch": 2, "content_hash": "b" * 64,
        "transcript": "unused", "transcript_parts_json": "[]",
        "correct": False, "is_canonical": True,
    }])
    store_path, cache_dir = tmp_path / "store.parquet", tmp_path / "cache"
    store.to_parquet(store_path)
    cache = cache_dir / "tiny" / "toy" / f"model__item__ep0__run__{content_hash[:16]}.npz"
    cache.parent.mkdir(parents=True)
    offsets = np.array([[13, 19]] * 65, dtype=np.int32)
    offsets[1] = [40, 44]
    offsets[2] = [28, 35]  # after the marker-like text, still inside structured assistant prose
    np.savez(cache, full=np.arange(65, dtype=float), noq=np.arange(65, dtype=float) + 1,
             local=np.arange(65, dtype=float) + 3, offsets=offsets, content_hash=content_hash)

    target = tmp_path / "raw" / "token-predictability"
    export_token_predictability(store_path, cache_dir, target, batch_rows=10)

    assert (target / "scorer=tiny" / "bench=toy").is_dir()
    assert len(list(target.rglob("*.parquet"))) == 7
    rows = pq.read_table(target).to_pandas().sort_values("token_index").reset_index(drop=True)
    assert len(rows) == 65
    assert rows.loc[0, ["sample_epoch", "full_bits", "question_gain_bits",
                        "total_context_gain_bits"]].tolist() == [2, 0.0, 1.0, 3.0]
    assert bool(rows.loc[0, "assistant"])
    assert not bool(rows.loc[1, "assistant"])
    assert bool(rows.loc[2, "assistant"])
    assert not bool(rows.loc[63, "long_context_eligible"])
    assert pd.isna(rows.loc[63, "long_context_gain_bits"])
    assert bool(rows.loc[64, "long_context_eligible"])
    assert rows.loc[64, "long_context_gain_bits"] == 2.0


def test_leaves_failed_staging_output_for_ceph_inspection(tmp_path, capsys):
    """Cache validation failures preserve the staged artifact for inspection."""
    content_hash = "a" * 64
    store = pd.DataFrame([{
        "bench": "toy", "model": "model", "item": "item", "epoch": 0,
        "log": "run", "sample_epoch": 1, "content_hash": content_hash,
        "transcript": "answer", "transcript_parts_json": json.dumps([
            {"role": "assistant", "content": "answer", "start": 0, "end": 6, "bare": True}]),
        "correct": True, "is_canonical": True,
    }])
    store_path, cache_dir = tmp_path / "store.parquet", tmp_path / "cache"
    store.to_parquet(store_path)
    cache = cache_dir / "tiny" / "toy" / f"model__item__ep0__run__{content_hash[:16]}.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, full=[1.0], noq=[2.0], local=[3.0], offsets=[[0, 6]], content_hash="wrong")

    target = tmp_path / "raw" / "token-predictability"
    with pytest.raises(ValueError, match="content hash mismatch"):
        export_token_predictability(store_path, cache_dir, target)

    stages = list(target.parent.glob(".token-predictability.*"))
    assert len(stages) == 1 and stages[0].is_dir()
    assert str(stages[0]) in capsys.readouterr().err


def test_exports_whole_transcript_short_and_long_tokens(tmp_path):
    text, content_hash = "answer", "a" * 64
    parts = [{"role": "assistant", "content": text, "start": 0,
              "end": len(text), "bare": True}]
    store = pd.DataFrame([{
        "bench": "toy", "model": "model", "item": "item", "epoch": 0,
        "log": "run", "sample_epoch": 2, "content_hash": content_hash,
        "transcript": text, "transcript_parts_json": json.dumps(parts),
        "correct": True, "is_canonical": True,
    }])
    store_path, cache_dir = tmp_path / "store.parquet", tmp_path / "cache"
    store.to_parquet(store_path)
    cache = cache_dir / "tiny" / "toy" / f"model__item__ep0__run__{content_hash[:16]}.npz"
    cache.parent.mkdir(parents=True)
    offsets = np.array([[0, 6]] * 65, dtype=np.int32)
    np.savez(cache, short=np.arange(65, dtype=float) + 5,
             long=np.arange(65, dtype=float) + 3,
             long_question=np.arange(65, dtype=float) + 2,
             offsets=offsets, assistant_mask=np.ones(65, dtype=bool),
             content_hash=content_hash, short_history=32, short_block=32,
             long_history=7168, long_block=1024)

    target = tmp_path / "raw" / "token-predictability-rolling"
    export_token_predictability(store_path, cache_dir, target, rolling=True)
    rows = pq.read_table(target).to_pandas().sort_values("token_index")
    assert rows.iloc[0][["short_bits", "long_bits", "long_question_bits",
                         "context_gain_bits", "question_gain_bits"]].tolist() == [
                             5.0, 3.0, 2.0, 2.0, 1.0]
    assert not bool(rows.iloc[63].context_eligible)
    assert bool(rows.iloc[64].context_eligible)
    assert not bool(rows.iloc[64].tail_eligible)
