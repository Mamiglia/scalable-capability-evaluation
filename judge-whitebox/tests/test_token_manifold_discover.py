import sys

import numpy as np
import pandas as pd
import pytest
import torch

import token_manifold_discover as discover
import token_manifold_record as record


def planted_token_data(n_items, prompts_per_item, tokens, layers, d, signal_layer, seed):
    rng = np.random.default_rng(seed)
    n_prompts = n_items * prompts_per_item
    gap = rng.normal(size=n_prompts)
    item = np.repeat([f"it{i}" for i in range(n_items)], prompts_per_item)
    prompt_idx = np.repeat(np.arange(n_prompts), tokens)
    X = rng.normal(size=(n_prompts * tokens, layers, d))
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    X[:, signal_layer] += 4 * gap[prompt_idx, None] * direction
    return {"X": X.astype(np.float32), "gap": gap, "prompt_idx": prompt_idx,
            "item": item, "group_id": np.ones(len(X), dtype=np.int8),
            "position": np.tile(np.arange(tokens), n_prompts)}


def fake_recorded_row(uid, item, n_tokens):
    return {"row_idx": int(uid[1:]), "pair_uid": uid, "item": item, "order": 0,
            "resid": torch.zeros(n_tokens, 2, 4, dtype=torch.float16),
            "input_ids": list(range(n_tokens)),
            "offsets": [(i, i + 1) for i in range(n_tokens)],
            "group_id": np.ones(n_tokens, dtype=np.int8),
            "l1": 1.0, "l2": 0.0, "gap": 1.0}


def test_prompt_weights_sum_to_one_per_prompt():
    prompt_idx = np.array([0, 0, 1, 1, 1, 1])

    weights = discover.prompt_weights(prompt_idx)

    assert weights[prompt_idx == 0].sum() == pytest.approx(1.0)
    assert weights[prompt_idx == 1].sum() == pytest.approx(1.0)


def test_ridge_timeline_finds_planted_layer_and_uses_item_groups():
    data = planted_token_data(n_items=20, prompts_per_item=2, tokens=6,
                              layers=4, d=10, signal_layer=2, seed=0)

    summary, oof = discover.ridge_timeline(data, alphas=(0.1, 1.0, 10.0), cv_folds=5)

    peak = summary.groupby("layer").r2.mean().idxmax()
    assert peak == 2
    assert summary.loc[summary.layer == 2, "r2"].mean() > 0.7
    assert set(oof["item"]) == set(data["item"])


def test_load_manifest_verifies_and_offsets_prompt_indices(tmp_path):
    rows = [fake_recorded_row("p0", "it0", 3), fake_recorded_row("p1", "it1", 2)]
    manifest = record.write_shards(iter(rows), tmp_path / "recording", 1,
                                   {"model": "dev", "layers": [0, 2]})

    data = discover.load_manifest(manifest)

    assert data["X"].shape == (5, 2, 4)
    assert data["prompt_idx"].tolist() == [0, 0, 0, 1, 1]
    assert data["item"].tolist() == ["it0", "it1"]


def test_load_manifest_rejects_tampered_shard(tmp_path):
    manifest = record.write_shards(
        iter([fake_recorded_row("p0", "it0", 2)]), tmp_path / "recording", 1,
        {"model": "dev", "layers": [0, 2]})
    shard = manifest.parent / "shard_00000.pt"
    with shard.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        discover.load_manifest(manifest)


def test_main_timeline_writes_r2_and_shuffled_null(tmp_path, monkeypatch):
    rows = []
    for prompt in range(12):
        row = fake_recorded_row(f"p{prompt}", f"it{prompt}", 2)
        row["gap"] = row["l1"] = float(prompt)
        row["resid"][:, 1, 0] = float(prompt)
        rows.append(row)
    manifest = record.write_shards(iter(rows), tmp_path / "recording", 4,
                                   {"model": "dev", "layers": [0, 2]})
    out = tmp_path / "timeline.csv"
    monkeypatch.setattr(sys, "argv", ["token_manifold_discover.py", "timeline",
                                       "--manifest", str(manifest), "--out", str(out),
                                       "--alphas", "0.1,1", "--cv-folds", "3", "--seed", "0"])

    discover.main()

    table = pd.read_csv(out)
    assert {"r2", "null_r2"} <= set(table)
    assert table.loc[table.layer == 1, "r2"].mean() > 0.9
