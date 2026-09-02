import json

import safetensors.torch
import torch

import delta_decomp as dd
import screen
import svd_components as svdc
import wbconfig
import wbfig

ARCH = {"n_heads": 4, "n_kv_heads": 2, "head_dim": 8}
D_MODEL = 32


def toy_state_dict(seed=0):
    """Synthetic Qwen3-shaped state dict, same layout as tests/test_svd.py."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    p = "model.layers.0."
    sd[p + "self_attn.q_proj.weight"] = torch.randn(4 * 8, D_MODEL, generator=g)
    sd[p + "self_attn.k_proj.weight"] = torch.randn(2 * 8, D_MODEL, generator=g)
    sd[p + "self_attn.v_proj.weight"] = torch.randn(2 * 8, D_MODEL, generator=g)
    sd[p + "self_attn.o_proj.weight"] = torch.randn(D_MODEL, 4 * 8, generator=g)
    sd[p + "mlp.gate_proj.weight"] = torch.randn(64, D_MODEL, generator=g)
    sd[p + "mlp.up_proj.weight"] = torch.randn(64, D_MODEL, generator=g)
    sd[p + "mlp.down_proj.weight"] = torch.randn(D_MODEL, 64, generator=g)
    return sd


def toy_decomp():
    sd = toy_state_dict()
    return sd, svdc.decompose(sd, ARCH)


def _embed(key, decomp, sd, delta_slice, by_columns):
    """Zero-filled full-parent-shaped delta with delta_slice placed at this key's registry slice
    (screen._slice_bounds), keyed by the HF weight name delta_in_base_basis looks up. Parent
    shape comes from the actual state dict, not recomputed from ARCH."""
    meta = decomp.meta[key]
    U, S, V = decomp.entries[key]
    lo, hi = screen._slice_bounds(by_columns, meta["head"], U, V)
    hf_name = dd._hf_weight_name(meta)
    full = torch.zeros(sd[hf_name].shape)
    if by_columns:
        full[:, lo:hi] = delta_slice
    else:
        full[lo:hi, :] = delta_slice
    return {hf_name: full}


def _orth_vector(basis, dim, generator):
    """Unit vector in R^dim orthogonal to basis's (orthonormal) columns."""
    rand = torch.randn(dim, 1, generator=generator)
    q, _ = torch.linalg.qr(torch.cat([basis, rand], dim=1))
    return q[:, -1]


def test_diagonal_component_lands_entirely_on_diag():
    sd, decomp = toy_decomp()
    key = "L0.q.h0"
    U, S, V = decomp.entries[key]
    c = 3.0
    delta_slice = c * torch.outer(U[:, 2], V[:, 2])
    deltas = _embed(key, decomp, sd, delta_slice, by_columns=False)

    df = dd.delta_in_base_basis(decomp, deltas)
    row = df[df["key"] == key].iloc[0]

    assert abs(row["energy_diag"] - c ** 2) < 1e-3
    assert abs(row["energy_offdiag"]) < 1e-3
    assert abs(row["energy_outside"]) < 1e-3
    i, j, val = row["top5"][0]
    assert (i, j) == (2, 2)
    assert abs(val - c) < 1e-3


def test_offdiagonal_component_lands_off_diag():
    sd, decomp = toy_decomp()
    key = "L0.q.h0"
    U, S, V = decomp.entries[key]
    c = 2.5
    delta_slice = c * torch.outer(U[:, 1], V[:, 3])
    deltas = _embed(key, decomp, sd, delta_slice, by_columns=False)

    df = dd.delta_in_base_basis(decomp, deltas)
    row = df[df["key"] == key].iloc[0]

    assert abs(row["energy_offdiag"] - c ** 2) < 1e-3
    assert abs(row["energy_diag"]) < 1e-3
    assert abs(row["energy_outside"]) < 1e-3
    i, j, val = row["top5"][0]
    assert (i, j) == (1, 3)


def test_component_outside_retained_topk_is_all_outside(monkeypatch):
    sd, decomp = toy_decomp()
    key = "L0.q.h0"
    U, S, V = decomp.entries[key]
    rank = S.shape[0]
    k = 3
    assert k < rank, "test requires a strict subset of this key's rank"
    monkeypatch.setattr(dd.wbconfig, "cfg", lambda path=None: {"screen": {"store_top_k": k}})

    g = torch.Generator().manual_seed(1)
    u_orth = _orth_vector(U[:, :k], U.shape[0], g)
    v_orth = _orth_vector(V[:, :k], V.shape[0], g)
    delta_slice = 4.0 * torch.outer(u_orth, v_orth)
    deltas = _embed(key, decomp, sd, delta_slice, by_columns=False)

    df = dd.delta_in_base_basis(decomp, deltas)
    row = df[df["key"] == key].iloc[0]

    assert row["topk_k"] == k
    frac_outside = row["energy_outside"] / row["energy_total_delta"]
    assert frac_outside > 0.999


def test_delta_confined_to_one_head_only_scores_that_head():
    sd, decomp = toy_decomp()
    key = "L0.q.h1"
    U, S, V = decomp.entries[key]
    g = torch.Generator().manual_seed(2)
    delta_slice = torch.randn(U.shape[0], D_MODEL, generator=g)
    deltas = _embed(key, decomp, sd, delta_slice, by_columns=False)

    df = dd.delta_in_base_basis(decomp, deltas)
    q_rows = df[df["kind"] == "q"].set_index("head")

    assert q_rows.loc[1, "energy_total_delta"] > 1e-6
    for h in (0, 2, 3):
        assert q_rows.loc[h, "energy_total_delta"] < 1e-9
        assert q_rows.loc[h, "energy_diag"] < 1e-9
        assert q_rows.loc[h, "energy_offdiag"] < 1e-9
        assert q_rows.loc[h, "energy_outside"] < 1e-9


def test_load_adapter_deltas_matches_alpha_over_r_times_ba(tmp_path):
    alpha, r, d_model, d_out = 16, 4, 8, 6
    g = torch.Generator().manual_seed(3)
    a = torch.randn(r, d_model, generator=g)
    b = torch.randn(d_out, r, generator=g)

    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "lora_alpha": alpha, "r": r, "target_modules": ["q_proj"], "peft_type": "LORA",
    }))
    safetensors.torch.save_file({
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    }, tmp_path / "adapter_model.safetensors")

    deltas = dd.load_adapter_deltas(tmp_path)

    expected_name = "model.layers.0.self_attn.q_proj.weight"
    assert set(deltas) == {expected_name}
    expected = (alpha / r) * (b @ a)
    assert torch.allclose(deltas[expected_name], expected, atol=1e-5)


def test_fig_energy_writes_png_and_svg(tmp_path, monkeypatch):
    monkeypatch.setattr(wbfig.wbconfig, "cfg", lambda path=None: {"paths": {"out": str(tmp_path)}})

    df = dd.pd.DataFrame([
        {"key": "L0.q.h0", "layer": 0, "kind": "q", "head": 0,
         "energy_diag": 3.0, "energy_offdiag": 1.0, "energy_outside": 0.5, "energy_total_delta": 4.5},
        {"key": "L1.q.h0", "layer": 1, "kind": "q", "head": 0,
         "energy_diag": 1.0, "energy_offdiag": 0.5, "energy_outside": 2.0, "energy_total_delta": 3.5},
    ])
    dd.fig_energy(df, "toydelta")

    figures_dir = tmp_path / "figures"
    assert (figures_dir / "delta_toydelta_energy.png").exists()
    assert (figures_dir / "delta_toydelta_energy.svg").exists()
