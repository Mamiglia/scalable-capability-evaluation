import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

import mask_regression
import token_manifold_record
import token_mediation


def test_sentence_features_are_unit_means_of_layer_minus_embedding():
    # 4 tokens, layers (0, 2): layer-0 slice is the embedding
    states = torch.zeros(4, 2, 3, dtype=torch.float16)
    states[:, 0, :] = torch.tensor([[1.0, 0, 0]] * 4, dtype=torch.float16)
    states[:, 1, :] = torch.tensor([[2.0, 0, 0], [4.0, 0, 0],
                                    [6.0, 0, 0], [8.0, 0, 0]], dtype=torch.float16)
    unit_ids = np.array([0, 0, 1, 1])

    out = token_mediation.sentence_features(states, unit_ids, layers=[0, 2])

    assert np.allclose(out[2][:, 0], [2.0, 6.0])   # mean of (h_L − h_0)
    assert np.allclose(out["h0"][:, 0], [1.0, 1.0])
    assert out[2].shape == (2, 3)


def test_extract_writes_aligned_features_and_index(monkeypatch, tmp_path):
    d, layers = 3, [0, 2]
    meta = {"prompt": "ab cd. ef", "input_ids": [5, 6, 7],
            "offsets": [(0, 2), (3, 6), (7, 9)],
            "group_id": np.array([0, 0, 3], dtype=np.int8),
            "group_names": token_manifold_record.STRUCTURAL_GROUPS}
    monkeypatch.setattr(token_manifold_record, "prompt_token_meta",
                        lambda row, tok, c: meta)
    monkeypatch.setattr(token_mediation.wbpairs, "load_pairs",
                        lambda c, split: pd.DataFrame(
                            [{"pair_uid": "p0", "order": 0}]))

    states = torch.randn(3, 2, d).half()
    shard = tmp_path / "shard_00000.pt"
    torch.save({"resid": states, "prompt_idx": torch.zeros(3, dtype=torch.int32),
                "input_ids": torch.tensor([5, 6, 7], dtype=torch.int32),
                "prompts": [{"pair_uid": "p0", "order": 0}]}, shard)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"layers": layers, "split": "train",
                                    "shards": [{"path": "shard_00000.pt"}]}))

    token_mediation.extract(manifest, {}, None, tmp_path / "feat")

    feats = np.load(tmp_path / "feat" / "feat_L2.npy")
    h0 = np.load(tmp_path / "feat" / "feat_h0.npy")
    index = pd.read_parquet(tmp_path / "feat" / "index.parquet")
    # sentence split: "ab cd." | group change -> "ef" = 2 units
    assert feats.shape == (2, d) and h0.shape == (2, d)
    assert index.n_tokens.tolist() == [2, 1]
    assert index.group.tolist() == [token_manifold_record.STRUCTURAL_GROUPS[0],
                                    token_manifold_record.STRUCTURAL_GROUPS[3]]
    expect = (states[:2, 1, :].float() - states[:2, 0, :].float()).mean(0).numpy()
    assert np.allclose(feats[0], expect, atol=1e-2)
    assert np.isclose(index.rel_end.iloc[-1], 1.0)


def test_layer_fit_recovers_planted_linear_signal_and_grouped_floor():
    rng = np.random.default_rng(0)
    n, d = 600, 12
    items = np.repeat([f"i{k}" for k in range(30)], 20)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = X @ w + 1.0 * rng.normal(size=n)

    fit = token_mediation.layer_fit(X, y, items)
    shuffled = token_mediation.layer_fit(X, rng.permutation(y), items)

    assert fit["heldout_r2"] > 0.5
    assert not fit["endpoint"]
    assert shuffled["heldout_r2"] < 0.05


def test_layer_fit_folds_never_split_an_item():
    folds = token_mediation._grouped_folds(np.repeat(["a", "b", "c", "d", "e"], 4), 2)
    all_items = np.concatenate(folds)
    assert sorted(all_items) == ["a", "b", "c", "d", "e"]


def test_control_features_shapes():
    index = pd.DataFrame({"group": ["question", "transcript_1", "transcript_1"],
                          "rel_start": [0.0, 0.3, 0.6], "rel_end": [0.3, 0.6, 1.0],
                          "n_tokens": [10, 20, 30]})
    X = token_mediation.control_features(index)
    assert X.shape == (3, 4 + 2)


def test_role_ids_split_transcripts_by_decile_and_keep_scaffolding_whole():
    index = pd.DataFrame({
        "pair_uid": "p0", "order": 0,
        "unit": range(6),
        "group": ["question", "transcript_1", "transcript_1", "transcript_1",
                  "transcript_1", "answer_readout"],
        "rel_start": np.linspace(0, 0.8, 6), "rel_end": np.linspace(0.2, 1.0, 6),
        "n_tokens": [5] * 6})

    roles, names = token_mediation.role_ids(index, deciles=2)

    assert names.count("question") == 1 and "transcript_1_d0" in names
    t1 = roles[1:5]
    assert (t1[:2] == t1[0]).all() and (t1[2:] == t1[2]).all() and t1[0] != t1[2]


def test_rank_curve_separates_planted_two_direction_structure():
    rng = np.random.default_rng(0)
    n, d = 2400, 16
    items = np.repeat([f"i{k}" for k in range(40)], 60)
    roles = rng.integers(0, 6, size=n)
    X = rng.normal(size=(n, d)).astype(np.float32)
    u1, u2 = np.zeros(d), np.zeros(d)
    u1[0], u2[1] = 1.0, 1.0
    # roles 0-2 read direction 1, roles 3-5 read direction 2
    signal = np.where(roles < 3, X @ u1, X @ u2)
    y = signal + 0.3 * rng.normal(size=n)

    curve = token_mediation.rank_curve(X, y, roles, items, ranks=(1, 2, 4))

    assert curve[2] > curve[1] + 0.1
    assert abs(curve[4] - curve[2]) < 0.05

    shared = X @ u1 + 0.3 * rng.normal(size=n)
    flat = token_mediation.rank_curve(X, shared, roles, items, ranks=(1, 2))
    assert abs(flat[2] - flat[1]) < 0.03
