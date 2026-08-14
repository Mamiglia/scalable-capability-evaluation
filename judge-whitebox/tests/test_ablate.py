import json

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

import ablate
import svd_components as svdc
import wbconfig
import wbfig


class FakeAttn(torch.nn.Module):
    def __init__(self, d_model, n_heads, head_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, d_model, bias=False)


class FakeMLP(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = torch.nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = torch.nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = torch.nn.Linear(d_ff, d_model, bias=False)


class FakeLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_ff):
        super().__init__()
        self.self_attn = FakeAttn(d_model, n_heads, head_dim)
        self.mlp = FakeMLP(d_model, d_ff)


class FakeQwen(torch.nn.Module):
    """model.model.layers[n].{self_attn,mlp}.{...}_proj — the layout screen.qwen_entry_map reads,
    with weight names matching svd_components' "model.layers.{n}.X" parsing via state_dict()."""

    def __init__(self, n_layers, d_model, n_heads, head_dim, d_ff):
        super().__init__()
        layers = [FakeLayer(d_model, n_heads, head_dim, d_ff) for _ in range(n_layers)]
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(layers)


def _build_model_and_decomp(seed=0, n_heads=2, head_dim=3, d_model=6, d_ff=8):
    torch.manual_seed(seed)
    model = FakeQwen(n_layers=1, d_model=d_model, n_heads=n_heads, head_dim=head_dim, d_ff=d_ff)
    arch = {"n_heads": n_heads, "n_kv_heads": n_heads, "head_dim": head_dim}
    decomp = svdc.decompose(model.state_dict(), arch)
    return model, decomp


def test_ablate_only_component_of_rank1_entry_zeros_slice():
    """A 1-component entry's weight slice IS S[0]*u@v.T; ablating (scale=0) zeros it exactly."""
    model, decomp = _build_model_and_decomp()
    key = "L0.down"  # whole MLP matrix, head=None
    U, S, V = decomp.entries[key]
    r = 0
    rank1 = (S[r] * torch.outer(U[:, r], V[:, r]))
    # rebuild the entry as an exact 1-component factorisation and overwrite the module weight
    decomp.entries[key] = (U[:, r:r + 1], S[r:r + 1], V[:, r:r + 1])
    model.model.layers[0].mlp.down_proj.weight.data.copy_(rank1)

    ablate.apply_ablation(model, decomp, [(key, 0)], scale=0.0)
    assert torch.allclose(model.model.layers[0].mlp.down_proj.weight.data,
                           torch.zeros_like(rank1), atol=1e-5)


def test_ablate_scale_one_is_identity():
    model, decomp = _build_model_and_decomp(seed=1)
    key = "L0.q.h0"
    before = model.model.layers[0].self_attn.q_proj.weight.data.clone()
    ablate.apply_ablation(model, decomp, [(key, 0)], scale=1.0)
    after = model.model.layers[0].self_attn.q_proj.weight.data
    assert torch.equal(before, after)


def test_head_sliced_writeback_touches_only_target_head():
    """2-head toy: ablating a head-1 component must leave head-0's rows untouched and change
    head-1's rows by exactly S[c]*u@v.T."""
    model, decomp = _build_model_and_decomp(seed=2, n_heads=2, head_dim=3, d_model=6)
    key = "L0.q.h1"
    U, S, V = decomp.entries[key]
    c = 0
    q_weight = model.model.layers[0].self_attn.q_proj.weight
    before = q_weight.data.clone()

    ablate.apply_ablation(model, decomp, [(key, c)], scale=0.0)
    after = q_weight.data

    head_dim = 3
    assert torch.equal(before[:head_dim, :], after[:head_dim, :])  # head 0 untouched
    delta = (S[c] * torch.outer(U[:, c], V[:, c])).to(after.dtype)
    assert torch.allclose(after[head_dim:, :], before[head_dim:, :] - delta, atol=1e-5)


def test_restore_returns_bit_identical_weights():
    model, decomp = _build_model_and_decomp(seed=3)
    comps = [("L0.q.h0", 0), ("L0.o.h1", 1), ("L0.gate", 2)]
    before = {k: p.weight.data.clone() for k, p in
              [("q", model.model.layers[0].self_attn.q_proj),
               ("o", model.model.layers[0].self_attn.o_proj),
               ("gate", model.model.layers[0].mlp.gate_proj)]}

    saved = ablate.apply_ablation(model, decomp, comps, scale=0.0)
    ablate.restore(model, saved)

    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight.data, before["q"])
    assert torch.equal(model.model.layers[0].self_attn.o_proj.weight.data, before["o"])
    assert torch.equal(model.model.layers[0].mlp.gate_proj.weight.data, before["gate"])


def _synthetic_rows(mode):
    rows = [{"mode": mode, "comps": [], "key": None, "comp": None,
             "acc": 0.60, "position_term": 0.10, "flip_rate": None, "control_nll": 2.0, "n_pairs": 8}]
    comps_so_far = []
    specs = [("L0.q.h0", 3, 0.55, 0.05, 0.30, 2.1),
             ("L0.o.h1", 1, 0.50, -0.02, 0.20, 2.3)]
    for key, comp, acc, pos, flip, nll in specs:
        comps_so_far = comps_so_far + [[key, comp]] if mode == "cumulative" else [[key, comp]]
        rows.append({"mode": mode, "comps": comps_so_far, "key": key, "comp": comp,
                     "acc": acc, "position_term": pos, "flip_rate": flip,
                     "control_nll": nll, "n_pairs": 8})
    return rows


def test_fig_deltas_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: {"paths": {"out": str(tmp_path)}})
    df = pd.DataFrame(_synthetic_rows("single"))
    ablate.fig_deltas(df, "smoke")
    figures_dir = tmp_path / "figures"
    assert (figures_dir / "ablate_smoke_deltas.png").exists()
    assert (figures_dir / "ablate_smoke_deltas.svg").exists()


def test_fig_dose_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: {"paths": {"out": str(tmp_path)}})
    df = pd.DataFrame(_synthetic_rows("cumulative"))
    ablate.fig_dose(df, "smoke")
    figures_dir = tmp_path / "figures"
    assert (figures_dir / "ablate_smoke_dose.png").exists()
    assert (figures_dir / "ablate_smoke_dose.svg").exists()


def test_load_reference_from_jsonl(tmp_path):
    path = tmp_path / "ref.jsonl"
    path.write_text(json.dumps({"pair_uid": "p1", "order": 0, "pick": 1}) + "\n")
    ref = ablate._load_reference(str(path))
    assert ref == {("p1", 0): 1}


def test_load_reference_passthrough_dict_and_none():
    d = {("p1", 0): 2}
    assert ablate._load_reference(d) is d
    assert ablate._load_reference(None) is None
