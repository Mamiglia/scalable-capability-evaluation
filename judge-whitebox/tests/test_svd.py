import pathlib
import tempfile

import torch
import svd_components as svdc

ARCH = {"n_heads": 4, "n_kv_heads": 2, "head_dim": 8}
D_MODEL = 32
INTERMEDIATE = 64
N_LAYERS = 2


def toy_state_dict():
    g = torch.Generator().manual_seed(0)
    sd = {}
    for n in range(N_LAYERS):
        p = f"model.layers.{n}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(4 * 8, D_MODEL, generator=g)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(2 * 8, D_MODEL, generator=g)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(2 * 8, D_MODEL, generator=g)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(D_MODEL, 4 * 8, generator=g)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(INTERMEDIATE, D_MODEL, generator=g)
        sd[p + "mlp.up_proj.weight"] = torch.randn(INTERMEDIATE, D_MODEL, generator=g)
        sd[p + "mlp.down_proj.weight"] = torch.randn(D_MODEL, INTERMEDIATE, generator=g)
        sd[p + "input_layernorm.weight"] = torch.randn(D_MODEL, generator=g)
        sd[p + "self_attn.q_proj.bias"] = torch.randn(4 * 8, generator=g)
    return sd


def sliced_weight(sd, layer, kind, head, head_dim=8):
    lo, hi = head * head_dim, (head + 1) * head_dim
    if kind == "q":
        return sd[f"model.layers.{layer}.self_attn.q_proj.weight"][lo:hi, :]
    if kind == "o":
        return sd[f"model.layers.{layer}.self_attn.o_proj.weight"][:, lo:hi]
    if kind == "k":
        return sd[f"model.layers.{layer}.self_attn.k_proj.weight"][lo:hi, :]
    if kind == "v":
        return sd[f"model.layers.{layer}.self_attn.v_proj.weight"][lo:hi, :]
    raise ValueError(kind)


def test_reconstruction():
    sd = toy_state_dict()
    decomp = svdc.decompose(sd, ARCH)
    for key, (u, s, v) in decomp.entries.items():
        recon = u @ torch.diag(s) @ v.T
        meta = decomp.meta[key]
        if meta["kind"] in ("q", "k", "v", "o"):
            expected = sliced_weight(sd, meta["layer"], meta["kind"], meta["head"])
        else:
            expected = sd[f"model.layers.{meta['layer']}.mlp.{meta['kind']}_proj.weight"]
        err = (recon - expected).norm().item()
        assert err < 1e-4, f"{key}: reconstruction error {err}"


def test_shapes():
    sd = toy_state_dict()
    decomp = svdc.decompose(sd, ARCH)

    u, s, v = decomp.entries["L0.q.h0"]
    assert u.shape == (8, 8) and s.shape == (8,) and v.shape == (32, 8)

    u, s, v = decomp.entries["L0.o.h0"]
    assert u.shape == (32, 8) and v.shape == (8, 8)

    u, s, v = decomp.entries["L0.down"]
    rank = min(D_MODEL, INTERMEDIATE)
    assert u.shape == (D_MODEL, rank) and s.shape == (rank,) and v.shape == (INTERMEDIATE, rank)


def test_kv_mapping():
    sd = toy_state_dict()
    decomp = svdc.decompose(sd, ARCH)
    expected = {0: 0, 1: 0, 2: 1, 3: 1}
    for h, kv in expected.items():
        assert decomp.meta[f"L0.q.h{h}"]["kv_head"] == kv
        assert decomp.meta[f"L0.o.h{h}"]["kv_head"] == kv
    assert decomp.meta["L0.k.h0"]["kv_head"] == 0
    assert decomp.meta["L0.k.h1"]["kv_head"] == 1


def test_save_load_roundtrip():
    sd = toy_state_dict()
    decomp = svdc.decompose(sd, ARCH)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "decomp.pt"
        svdc.save(decomp, path)
        loaded = svdc.load(path)

    assert set(loaded.entries.keys()) == set(decomp.entries.keys())
    assert loaded.meta == decomp.meta
    for key in decomp.entries:
        for a, b in zip(decomp.entries[key], loaded.entries[key]):
            assert torch.allclose(a, b)


def test_ignores_non_weight_keys():
    sd = toy_state_dict()
    decomp = svdc.decompose(sd, ARCH)
    keys = "\n".join(decomp.entries.keys())
    assert "input_layernorm" not in keys
    assert "bias" not in keys
    n_per_layer = 4 + 2 + 2 + 4 + 3  # q + k + v + o heads, plus gate/up/down
    assert len(decomp.entries) == N_LAYERS * n_per_layer
