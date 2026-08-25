import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from transcript_embeddings import write_dense_artifacts
from transcript_sae import SparseAutoencoder, write_sae_artifacts
from transcript_sae_tables import KEYS, build_pairs, transcript_table


def artifacts(tmp_path):
    keys = ["k1" * 32, "k2" * 32, "k3" * 32]
    rows = pd.DataFrame({"transcript_key": keys, "item_group_key": ["i"] * 3, "benchmark": ["gpqa"] * 3,
                         "split_role": ["representation_train"] * 3, "chunk_count": [1, 1, 2],
                         "transcript_tokens": [10, 20, 40]})
    audit = pd.DataFrame({"transcript_key": keys, "item_group_key": ["i"] * 3,
                          "bench": ["gpqa"] * 3, "model": ["a", "b", "c"], "item": ["q1"] * 3,
                          "epoch": [0, 0, 1], "log": ["la", "lb", "lc"], "sample_epoch": [0, 0, 0],
                          "content_hash": ["ha", "hb", "hc"]})
    dense = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    write_dense_artifacts(tmp_path / "dense", rows, {"transcript_only__final": dense},
                          {"family": "qwen"}, audit_rows=audit)
    activations = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]], dtype=np.float32)
    write_sae_artifacts(tmp_path / "sae", {"gpqa": SparseAutoencoder(2, 2, 1)},
                        {"activations": activations, "feature_labels": np.array(["gpqa:0", "gpqa:5"])},
                        {"report": {}}, rows=rows)
    return tmp_path / "dense", tmp_path / "sae"


def test_transcript_table_carries_a_keys_dense_dims_and_named_sae_columns(tmp_path):
    table = transcript_table(*artifacts(tmp_path), "qwen")
    assert list(table.columns[:len(KEYS)]) == KEYS
    assert list(table.columns[-4:]) == ["qwen_dim_0000", "qwen_dim_0001", "sae_gpqa_0", "sae_gpqa_5"]
    assert table.model.tolist() == ["a", "b", "c"] and table.sae_gpqa_5.tolist() == [0.0, 3.0, 0.0]
    assert table.chunk_count.tolist() == [1, 1, 2]


def test_pairs_are_left_minus_right_on_verdict_rows(tmp_path):
    table = transcript_table(*artifacts(tmp_path), "qwen")
    verdicts = pd.DataFrame({
        "bench": ["gpqa", "gpqa", "gpqa"], "judge": ["j"] * 3, "item": ["q1"] * 3,
        "model_left": ["a", "c", "a"], "model_right": ["b", "a", "zz"],
        "epoch_left": [0, 1, 0], "epoch_right": [0, 0, 0],
        "log_left": ["la", "lc", "la"], "log_right": ["lb", "la", "lz"],
        "pick_left": [True, False, True], "judge_input": ["long text"] * 3,
    })
    pairs = build_pairs(verdicts, table)
    assert len(pairs) == 2 and "judge_input" not in pairs and "d_split_role" not in pairs
    assert pairs.split_role.tolist() == ["representation_train"] * 2
    assert pairs.d_qwen_dim_0000.tolist() == pytest.approx([1.0, -0.4])
    assert pairs.d_sae_gpqa_0.tolist() == [2.0, -1.0] and pairs.d_chunk_count.tolist() == [0, 1]
    assert {"sample_epoch_l", "content_hash_r"} <= set(pairs)
