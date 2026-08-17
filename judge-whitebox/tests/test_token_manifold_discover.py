import sys
from pathlib import Path

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


def curved_gap_data(n, d, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    intrinsic = np.column_stack([np.cos(theta), np.sin(theta), np.cos(2 * theta)])
    projection = rng.normal(size=(3, d))
    X = intrinsic @ projection + 0.02 * rng.normal(size=(n, d))
    return X.astype(np.float32), theta


def sparse_manifold_mix(n, d, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    active = rng.integers(0, 2, n)
    z = np.column_stack([np.cos(theta), np.sin(theta), active, active * theta])
    return (z @ rng.normal(size=(4, d))).astype(np.float32), z


def planted_binary_feature_groups(n, groups, seed):
    rng = np.random.default_rng(seed)
    codes = np.zeros((n, 4), dtype=np.float32)
    choice = rng.integers(0, len(groups), n)
    for row, selected in enumerate(choice):
        codes[row, list(groups[selected])] = 1
    return codes


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


def test_fit_cebra_returns_requested_dimension_and_seed_metadata():
    pytest.importorskip("cebra")
    X, y = curved_gap_data(n=240, d=12, seed=0)

    a = discover.fit_cebra(X[:180], y[:180], X[180:], 4, 3, 200, device="cpu")

    assert a.train_code.shape == (180, 4) and a.test_code.shape == (60, 4)
    assert np.isfinite(a.test_code).all()
    assert a.test_reconstruction is None
    assert a.metadata == {"output_dim": 4, "seed": 3}


def test_code_gap_r2_uses_train_only_regression():
    y_train = np.arange(20, dtype=np.float32)
    y_test = np.arange(20, 30, dtype=np.float32)
    result = discover.MethodResult("exact", y_train[:, None], y_test[:, None], None, {})

    score = discover.code_gap_r2(result, y_train, y_test)

    assert score > 0.99


@pytest.mark.parametrize("method", ["bsf", "sae"])
def test_reconstructive_method_returns_codes_and_test_reconstruction(method):
    pytest.importorskip("bsf" if method == "bsf" else "saes")
    X, _ = sparse_manifold_mix(n=256, d=12, seed=0)

    result = discover.fit_method(method, X[:192], None, X[192:], seed=2, quick=True)

    assert result.train_code.shape[0] == 192
    assert result.test_code.shape[0] == 64
    assert result.test_reconstruction.shape == X[192:].shape
    assert np.isfinite(result.test_reconstruction).all()


def test_ising_clusters_groups_planted_feature_pairs():
    pytest.importorskip("unsupervised_clustering")
    codes = planted_binary_feature_groups(n=400, groups=((0, 1), (2, 3)), seed=0)

    labels = discover.ising_clusters(codes, alpha=0.01, method="spectral", n_clusters=2)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def write_fake_manifest(out):
    rows = []
    for prompt in range(8):
        generator = torch.Generator().manual_seed(prompt)
        rows.append({"row_idx": prompt, "pair_uid": f"p{prompt}",
                     "item": f"it{prompt // 2}", "order": prompt % 2,
                     "resid": torch.randn(4, 2, 6, generator=generator),
                     "input_ids": [1, 2, 3, 4],
                     "offsets": [(0, 1), (1, 2), (2, 3), (3, 4)],
                     "group_id": np.array([0, 1, 1, 5], dtype=np.int8),
                     "l1": float(prompt), "l2": 0.0, "gap": float(prompt)})
    return record.write_shards(iter(rows), out, shard_prompts=4,
                               manifest_meta={"model": "dev", "layers": [0, 1]})


def fake_fit_method(name, X_train, gap_train, X_test, seed, quick):
    width = 2 if name == "cebra" else 4
    train = np.asarray(X_train)[:, :width]
    test = np.asarray(X_test)[:, :width]
    reconstruction = None if name == "cebra" else np.asarray(X_test)
    return discover.MethodResult(name, train, test, reconstruction, {"seed": seed})


def test_bakeoff_writes_one_row_per_method(tmp_path, monkeypatch):
    manifest = write_fake_manifest(tmp_path / "recording")
    monkeypatch.setattr(discover, "fit_method", fake_fit_method)
    out = tmp_path / "bakeoff"

    discover.run_bakeoff(manifest, layer=1, methods=("cebra", "bsf", "sae"),
                         out_dir=out, seeds=(0,), quick=True)

    table = pd.read_csv(out / "bakeoff.csv")
    assert set(table.method) == {"cebra", "bsf", "sae"}
    assert {"test_gap_r2", "test_recon_r2", "runtime_s"} <= set(table)
    assert (out / "bakeoff.pt").exists()


def test_main_bakeoff_dispatches_cli_arguments(tmp_path, monkeypatch):
    seen = {}

    def fake_run(manifest, layer, methods, out_dir, seeds, quick):
        seen.update(manifest=manifest, layer=layer, methods=methods, out_dir=out_dir,
                    seeds=seeds, quick=quick)

    monkeypatch.setattr(discover, "run_bakeoff", fake_run)
    monkeypatch.setattr(sys, "argv", ["token_manifold_discover.py", "bakeoff",
                                       "--manifest", "recording/manifest.json", "--layer", "12",
                                       "--methods", "cebra,bsf", "--seeds", "1,3",
                                       "--out", str(tmp_path / "out"), "--quick"])

    discover.main()

    assert seen == {"manifest": "recording/manifest.json", "layer": 12,
                    "methods": ("cebra", "bsf"), "out_dir": str(tmp_path / "out"),
                    "seeds": (1, 3), "quick": True}


def test_sbatch_has_explicit_stages_and_never_installs_packages():
    text = (Path(__file__).parents[1] / "sbatch" / "token_manifold.sbatch").read_text()

    assert "WB_STAGE" in text
    for stage in ("record", "timeline", "bakeoff"):
        assert stage in text
    assert "WB_MANIFOLD_VENV" in text
    assert "pip install" not in text
