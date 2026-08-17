import sys
import types
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


def test_training_token_indices_caps_each_prompt_group():
    prompt_idx = np.repeat(np.arange(2), 100)
    group_id = np.tile(np.repeat([1, 2], 50), 2)
    position = np.tile(np.arange(100), 2)

    selected = discover.training_token_indices(prompt_idx, group_id, position, max_per_group=10)

    assert len(selected) == 2 * 2 * 10
    for prompt in range(2):
        for group in (1, 2):
            mask = (prompt_idx[selected] == prompt) & (group_id[selected] == group)
            assert mask.sum() == 10


def test_relative_token_indices_are_prompt_balanced_and_include_ends():
    prompt_idx = np.repeat(np.arange(2), [5, 7])
    position = np.concatenate([np.arange(5), np.arange(7)])

    selected = discover.relative_token_indices(prompt_idx, position, n_per_prompt=3)

    assert position[selected[prompt_idx[selected] == 0]].tolist() == [0, 2, 4]
    assert position[selected[prompt_idx[selected] == 1]].tolist() == [0, 3, 6]


def test_code_deltas_never_cross_prompt_boundaries():
    code = np.array([[1.0], [4.0], [10.0], [12.0], [17.0]])
    prompt_idx = np.array([0, 0, 1, 1, 1])
    position = np.array([0, 1, 0, 1, 2])

    delta, rows = discover.code_deltas(code, prompt_idx, position)

    assert rows.tolist() == [1, 3, 4]
    assert delta[:, 0].tolist() == [3.0, 2.0, 5.0]


def test_ridge_timeline_finds_planted_layer_and_uses_item_groups():
    data = planted_token_data(n_items=20, prompts_per_item=2, tokens=6,
                              layers=4, d=10, signal_layer=2, seed=0)

    summary, oof = discover.ridge_timeline(data, alphas=(0.1, 1.0, 10.0), cv_folds=5)

    peak = summary.groupby("layer").r2.mean().idxmax()
    assert peak == 2
    assert summary.loc[summary.layer == 2, "r2"].mean() > 0.7
    assert set(oof["item"]) == set(data["item"])


def test_ridge_timeline_is_invariant_to_feature_scales():
    data = planted_token_data(n_items=20, prompts_per_item=2, tokens=6,
                              layers=4, d=10, signal_layer=2, seed=1)
    rng = np.random.default_rng(2)
    y = data["gap"][data["prompt_idx"]]
    data["X"][:, 2] = rng.normal(size=data["X"][:, 2].shape)
    data["X"][:, 2, 0] = (4 * y + rng.normal(size=len(y))) * 1e-4
    data["X"][:, 2, 1:] *= 1e4

    summary, _ = discover.ridge_timeline(data, alphas=(0.1, 1.0, 10.0, 100.0), cv_folds=5)

    assert summary.loc[summary.layer == 2, "r2"].mean() > 0.7


def test_load_manifest_verifies_and_offsets_prompt_indices(tmp_path):
    rows = [fake_recorded_row("p0", "it0", 3), fake_recorded_row("p1", "it1", 2)]
    manifest = record.write_shards(iter(rows), tmp_path / "recording", 1,
                                   {"model": "dev", "layers": [0, 2]})

    data = discover.load_manifest(manifest)

    assert data["X"].shape == (5, 2, 4)
    assert data["prompt_idx"].tolist() == [0, 0, 0, 1, 1]
    assert data["item"].tolist() == ["it0", "it1"]


def test_load_manifest_can_select_one_physical_layer(tmp_path):
    rows = [fake_recorded_row("p0", "it0", 3), fake_recorded_row("p1", "it1", 2)]
    rows[0]["resid"][:, 1] = 7
    rows[1]["resid"][:, 1] = 7
    manifest = record.write_shards(iter(rows), tmp_path / "recording", 1,
                                   {"model": "dev", "layers": [0, 2]})

    data = discover.load_manifest(manifest, layer=2)

    assert data["X"].shape == (5, 1, 4)
    assert np.all(data["X"] == 7)


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


def test_fit_cebra_standardises_continuous_labels_for_delta_sampling(monkeypatch):
    seen = {}

    class FakeCEBRA:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def fit(self, X, y):
            seen["labels"] = y

        def transform(self, X):
            return np.zeros((len(X), 3), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "cebra", types.SimpleNamespace(CEBRA=FakeCEBRA))
    X = np.arange(40, dtype=np.float32).reshape(10, 4)

    discover.fit_cebra(X[:8], np.arange(8, dtype=np.float32), X[8:], 3, 0, 10,
                       conditional="delta_normal", delta=0.25, num_hidden_units=128)

    assert seen["kwargs"]["conditional"] == "delta_normal"
    assert seen["kwargs"]["num_hidden_units"] == 128
    assert seen["labels"].mean() == pytest.approx(0, abs=1e-6)
    assert seen["labels"].std() == pytest.approx(1, abs=1e-6)


def test_code_gap_r2_uses_train_only_regression():
    y_train = np.arange(20, dtype=np.float32)
    y_test = np.arange(20, 30, dtype=np.float32)
    result = discover.MethodResult("exact", y_train[:, None], y_test[:, None], None, {})

    score = discover.code_gap_r2(result, y_train, y_test)

    assert score > 0.99


def test_grouped_code_decoder_regularises_scale_stressed_p_greater_than_n_codes():
    rng = np.random.default_rng(7)
    n_items, prompts_per_item, width = 80, 2, 512
    n = n_items * prompts_per_item
    item = np.repeat(np.arange(n_items), prompts_per_item)
    y = rng.normal(size=n)
    code = rng.normal(scale=0.4, size=(n, width))
    code[:, 0] = 1e-3 * (y + rng.normal(scale=0.25, size=n))
    train, test = next(discover.GroupKFold(2).split(code, y, item))

    pred, alpha = discover.grouped_code_predictions(
        code[train], y[train], item[train], code[test],
        alphas=(0.01, 0.1, 1, 10, 100, 1000), cv_folds=4)

    assert alpha in (0.01, 0.1, 1, 10, 100, 1000)
    assert discover.r2_score(y[test], pred) > 0.1


def test_grouped_knn_decoder_recovers_curved_code_geometry():
    rng = np.random.default_rng(9)
    theta = rng.uniform(-2.5, 2.5, 160)
    code = np.column_stack([np.cos(theta), np.sin(theta)])
    item = np.repeat(np.arange(80), 2)
    train, test = next(discover.GroupKFold(2).split(code, theta, item))

    pred, neighbors = discover.grouped_knn_predictions(
        code[train], theta[train], item[train], code[test],
        neighbors=(3, 5, 10, 20), cv_folds=4)

    assert neighbors in (3, 5, 10, 20)
    assert discover.r2_score(theta[test], pred) > 0.8


@pytest.mark.parametrize("method", ["bsf", "sae"])
def test_reconstructive_method_returns_codes_and_test_reconstruction(method):
    pytest.importorskip("bsf" if method == "bsf" else "saes")
    X, _ = sparse_manifold_mix(n=256, d=12, seed=0)

    result = discover.fit_method(method, X[:192], None, X[192:], seed=2, quick=True)

    assert result.train_code.shape[0] == 192
    assert result.test_code.shape[0] == 64
    assert result.test_reconstruction.shape == X[192:].shape
    assert np.isfinite(result.test_reconstruction).all()


@pytest.mark.parametrize("method", ["raw", "pca", "random"])
def test_control_method_returns_finite_train_and_test_codes(method):
    X, _ = sparse_manifold_mix(n=80, d=12, seed=0)
    config = discover.trajectory_config(method, d_in=12, quick=True)

    result = discover.fit_method(method, X[:60], None, X[60:], seed=2,
                                 quick=True, config=config)

    assert result.train_code.shape[0] == 60 and result.test_code.shape[0] == 20
    assert np.isfinite(result.train_code).all() and np.isfinite(result.test_code).all()
    assert result.train_code.shape[1] <= 12


def test_ising_clusters_groups_planted_feature_pairs():
    pytest.importorskip("unsupervised_clustering")
    codes = planted_binary_feature_groups(n=400, groups=((0, 1), (2, 3)), seed=0)

    labels = discover.ising_clusters(codes, alpha=0.01, method="spectral", n_clusters=2)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_method_configs_expand_scientific_grid_but_not_quick_grid():
    assert len(discover.method_configs("cebra", d_in=4096, quick=True)) == 1
    assert [c["output_dim"] for c in discover.method_configs(
        "cebra", d_in=4096, quick=False)] == [1, 2, 4, 8, 16]
    assert [c["group_size"] for c in discover.method_configs(
        "bsf", d_in=4096, quick=False)] == [1, 2, 4, 8]
    assert discover.method_configs("sae", d_in=4096, quick=False)[0]["width"] == 4096


def test_trajectory_configs_use_non_temporal_cebra_and_reference_bsf():
    cebra = discover.trajectory_config("cebra", d_in=3072, quick=True)
    cebra_full = discover.trajectory_config("cebra", d_in=3072, quick=False)
    bsf = discover.trajectory_config("bsf", d_in=3072, quick=True)

    assert cebra["conditional"] == "delta_normal" and cebra["output_dim"] == 3
    assert cebra_full["output_dim"] == 8 and cebra_full["num_hidden_units"] == 128
    assert bsf["group_size"] == 3 and bsf["l0"] == 16 and bsf["snr"] == 0.1
    small = discover.trajectory_config("bsf", d_in=8, quick=True)
    assert small["l0"] <= small["n_groups"]


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


def fake_fit_method(name, X_train, gap_train, X_test, seed, quick, config=None):
    width = 2 if name == "cebra" else 4
    train = np.asarray(X_train)[:, :width]
    test = np.asarray(X_test)[:, :width]
    reconstruction = None if name == "cebra" else np.asarray(X_test)
    return discover.MethodResult(name, train, test, reconstruction, {"seed": seed})


def write_trajectory_manifest(out):
    rows = []
    for prompt in range(40):
        gap = float(prompt % 10)
        resid = torch.randn(6, 2, 8, generator=torch.Generator().manual_seed(prompt))
        resid[:, 1, 0] = gap
        rows.append({"row_idx": prompt, "pair_uid": f"p{prompt}",
                     "item": f"it{prompt // 2}", "order": prompt % 2,
                     "resid": resid, "input_ids": list(range(6)),
                     "offsets": [(i, i + 1) for i in range(6)],
                     "group_id": np.array([0, 2, 3, 3, 4, 5], dtype=np.int8),
                     "l1": gap, "l2": 0.0, "gap": gap})
    return record.write_shards(iter(rows), out, shard_prompts=10,
                               manifest_meta={"model": "dev", "layers": [0, 2]})


def test_run_trajectory_writes_state_and_delta_surfaces(tmp_path, monkeypatch):
    manifest = write_trajectory_manifest(tmp_path / "recording")
    monkeypatch.setattr(discover, "fit_method", fake_fit_method)
    out = tmp_path / "trajectory"

    table = discover.run_trajectory(
        manifest, layers=(2,), method="cebra", out_dir=out, seeds=(0,),
        positions_per_prompt=6, cv_folds=2, quick=True)

    assert set(table.kind) == {"state", "delta"}
    assert set(table.decoder) == {"ridge", "knn"}
    assert set(table.layer) == {2}
    assert set(table.method) == {"cebra"}
    assert (out / "trajectory.csv").exists()
    assert table.loc[table.kind == "state", "r2"].max() > 0.9


def test_bakeoff_writes_one_row_per_method(tmp_path, monkeypatch):
    manifest = write_fake_manifest(tmp_path / "recording")
    monkeypatch.setattr(discover, "fit_method", fake_fit_method)
    out = tmp_path / "bakeoff"

    discover.run_bakeoff(manifest, layer=1, methods=("cebra", "bsf", "sae"),
                         out_dir=out, seeds=(0,), quick=True)

    table = pd.read_csv(out / "bakeoff.csv")
    assert set(table.method) == {"cebra", "bsf", "sae"}
    assert "config" in table
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


def test_main_trajectory_dispatches_cli_arguments(tmp_path, monkeypatch):
    seen = {}

    def fake_run(manifest, layers, method, out_dir, seeds, positions_per_prompt,
                 cv_folds, quick):
        seen.update(manifest=manifest, layers=layers, method=method, out_dir=out_dir,
                    seeds=seeds, positions_per_prompt=positions_per_prompt,
                    cv_folds=cv_folds, quick=quick)

    monkeypatch.setattr(discover, "run_trajectory", fake_run)
    monkeypatch.setattr(sys, "argv", ["token_manifold_discover.py", "trajectory",
                                       "--manifest", "recording/manifest.json",
                                       "--layers", "3,15,25", "--method", "bsf",
                                       "--seeds", "0,2", "--positions-per-prompt", "32",
                                       "--cv-folds", "2", "--out", str(tmp_path / "out"),
                                       "--quick"])

    discover.main()

    assert seen == {"manifest": "recording/manifest.json", "layers": (3, 15, 25),
                    "method": "bsf", "out_dir": str(tmp_path / "out"),
                    "seeds": (0, 2), "positions_per_prompt": 32,
                    "cv_folds": 2, "quick": True}


def test_sbatch_has_explicit_stages_and_never_installs_packages():
    text = (Path(__file__).parents[1] / "sbatch" / "token_manifold.sbatch").read_text()

    assert "WB_STAGE" in text
    for stage in ("record", "timeline", "bakeoff", "trajectory"):
        assert stage in text
    assert "WB_MANIFOLD_VENV" in text
    assert "WB_RUN" in text
    assert "timeline_${RUN}.csv" in text
    assert "bakeoff_${RUN}_L${LAYER}" in text
    assert "trajectory_${RUN}_${METHOD}" in text
    assert "pip install" not in text
