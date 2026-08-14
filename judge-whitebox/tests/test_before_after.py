import math
import sys

import matplotlib
matplotlib.use("Agg")
import torch

import before_after as ba
import svd_components as svdc
import wbconfig

R, D_OUT, D_IN = 8, 12, 10
KEY = "L3.gate"
CFG = {"before_after": {"min_sim": 0.5}, "screen": {"act_eps": 1e-9}}


def _toy_decomp(seed):
    """Orthonormal U/V (via SVD of a random matrix) with a controlled, distinct, descending
    singular spectrum: overriding S doesn't touch orthonormality of U/V."""
    g = torch.Generator().manual_seed(seed)
    u, _, vh = torch.linalg.svd(torch.randn(D_OUT, D_IN, generator=g), full_matrices=False)
    s = torch.linspace(10.0, 3.0, R)
    decomp = svdc.Decomp()
    decomp.entries[KEY] = (u[:, :R].clone(), s, vh.T[:, :R].clone())
    decomp.meta[KEY] = {"layer": 3, "kind": "gate", "head": None, "kv_head": None}
    return decomp


def _rotate_pair(mat, i, j, alpha_deg):
    """Rotate columns i, j of an orthonormal matrix within their span; stays orthonormal."""
    out = mat.clone()
    ci, cj = mat[:, i].clone(), mat[:, j].clone()
    a = math.radians(alpha_deg)
    out[:, i] = math.cos(a) * ci + math.sin(a) * cj
    out[:, j] = -math.sin(a) * ci + math.cos(a) * cj
    return out


def _perturbed_pair(seed=42, alpha_u=70.0, alpha_v=10.0):
    """decompB = decompA with comps 3,4 rotated by DIFFERENT angles in U and V — a rank-2
    perturbation confined to span(u3,u4) x span(v3,v4). Distinct U/V angles are needed: a
    2x2 orthogonal block's |.| entries are always symmetric (sim[3,3]==sim[4,4] and
    sim[3,4]==sim[4,3]), so equal angles can never push the diagonal below min_sim while
    keeping it the arg-max — independent angles can.
    """
    decompA = _toy_decomp(seed)
    u, s, v = decompA.entries[KEY]
    u_b = _rotate_pair(u, 3, 4, alpha_u)
    v_b = _rotate_pair(v, 3, 4, alpha_v)
    decompB = svdc.Decomp()
    decompB.entries[KEY] = (u_b, s.clone(), v_b)
    decompB.meta[KEY] = dict(decompA.meta[KEY])
    return decompA, decompB


def test_match_identity_away_from_perturbed_comps(monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: CFG)
    decompA, decompB = _perturbed_pair()
    pairs, summary = ba.match(decompA, decompB)

    assert len(pairs) == R
    by_comp = {r.comp_a: r for r in pairs.itertuples()}
    for i in range(R):
        assert by_comp[i].comp_b == i, f"comp {i} should self-match under identity"
        if i in (3, 4):
            assert by_comp[i].sim < 0.5, f"comp {i} sim should be depressed below min_sim"
            assert by_comp[i].low_sim
        else:
            assert by_comp[i].sim > 0.95, f"comp {i} sim should stay near 1"
            assert not by_comp[i].low_sim

    row = summary[summary["key"] == KEY].iloc[0]
    expected_mass = sum((pairs.loc[pairs["comp_a"] == i, "sv_a"].item() ** 2 +
                          pairs.loc[pairs["comp_a"] == i, "sv_b"].item() ** 2) / 2
                         for i in (3, 4))
    assert math.isclose(row["unmatched_mass"], expected_mass, rel_tol=1e-5)
    assert 0.0 < row["frac_unmatched"] < 1.0
    assert math.isclose(row["frac_unmatched"], row["unmatched_mass"] / row["total_energy"])


def test_sign_invariance(monkeypatch):
    """Flipping the sign of a whole (u,v) column pair is a valid alternate SVD (u v^T == (-u)(-v)^T)
    and must not change sim, since sim uses abs()."""
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: CFG)
    decompA, decompB = _perturbed_pair()
    pairs_before, _ = ba.match(decompA, decompB)

    u, s, v = decompB.entries[KEY]
    u2, v2 = u.clone(), v.clone()
    u2[:, 0] *= -1
    v2[:, 0] *= -1
    decompB2 = svdc.Decomp()
    decompB2.entries[KEY] = (u2, s, v2)
    decompB2.meta[KEY] = dict(decompB.meta[KEY])

    pairs_after, _ = ba.match(decompA, decompB2)
    assert torch.allclose(torch.tensor(pairs_before["sim"].to_numpy()),
                           torch.tensor(pairs_after["sim"].to_numpy()), atol=1e-6)


def _screen(attr_abs_vec, n_prompts=5, model=None, adapter=None):
    return {"attr_abs": {KEY: torch.tensor(attr_abs_vec)}, "n_prompts": n_prompts,
            "model": model, "adapter": adapter}


def test_importance_diff_aligns_and_flags_low_sim(monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: CFG)
    decompA, decompB = _perturbed_pair()
    pairs, _ = ba.match(decompA, decompB)

    attr_a = [1.0, 2.0, 3.0, 40.0, 5.0, 6.0, 7.0, 8.0]  # comp 3 planted as the big mover
    attr_b = [1.1, 2.1, 3.1, 4.0, 5.1, 6.1, 7.1, 8.1]
    # A and B are meant to differ (before/after fine-tuning) — importance_diff must record both
    # sides' identity, not assert they match (that's merge_table's job for judging vs control).
    screenA = _screen(attr_a, model="Qwen3-8B", adapter=None)
    screenB = _screen(attr_b, model="Qwen3-8B", adapter="/path/to/adapter")

    diff = ba.importance_diff(pairs, screenA, screenB)
    assert len(diff) == R
    assert (diff["model_a"] == "Qwen3-8B").all() and diff["adapter_a"].isna().all()
    assert (diff["model_b"] == "Qwen3-8B").all() and (diff["adapter_b"] == "/path/to/adapter").all()
    # sorted by |delta| descending
    deltas = diff["delta"].abs().to_numpy()
    assert (deltas[:-1] >= deltas[1:]).all()
    # the planted comp-3 swing (|40/5 - 4.0/5| = 7.2) is the biggest and correctly aligned
    top = diff.iloc[0]
    assert top["key"] == KEY and top["comp_a"] == 3 and top["comp_b"] == 3
    assert math.isclose(top["attr_abs_a"], 40.0 / 5)
    assert math.isclose(top["attr_abs_b"], 4.0 / 5)
    assert math.isclose(top["delta"], 4.0 / 5 - 40.0 / 5)
    assert top["low_sim"]
    # a well-matched comp (not 3 or 4) is not flagged
    comp0 = diff[diff["comp_a"] == 0].iloc[0]
    assert not comp0["low_sim"]


def test_real_decompose_self_match_is_identity(monkeypatch):
    """Full pipeline sanity check: decompose a synthetic Qwen-shaped state_dict and match it
    against itself — every component must self-match at sim ~= 1, no unmatched mass anywhere."""
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: CFG)
    g = torch.Generator().manual_seed(7)
    arch = {"n_heads": 4, "n_kv_heads": 2, "head_dim": 8}
    sd = {}
    for n in range(2):
        p = f"model.layers.{n}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(32, 32, generator=g)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(16, 32, generator=g)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(16, 32, generator=g)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(32, 32, generator=g)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(64, 32, generator=g)
        sd[p + "mlp.up_proj.weight"] = torch.randn(64, 32, generator=g)
        sd[p + "mlp.down_proj.weight"] = torch.randn(32, 64, generator=g)
    decomp = svdc.decompose(sd, arch)

    pairs, summary = ba.match(decomp, decomp)
    assert (pairs["comp_a"] == pairs["comp_b"]).all()
    assert (pairs["sim"] > 0.999).all()
    assert not pairs["low_sim"].any()
    assert (summary["frac_unmatched"] == 0).all()


def test_figures_written(tmp_path, monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: {**CFG, "paths": {"out": str(tmp_path)}})
    decompA, decompB = _perturbed_pair()
    pairs, summary = ba.match(decompA, decompB)
    diff = ba.importance_diff(pairs, _screen([1, 2, 3, 4, 5, 6, 7, 8]), _screen([2, 1, 5, 9, 1, 6, 8, 7]))

    ba.fig_scatter(diff, "smoke", eps=1e-9)
    ba.fig_unmatched(summary, "smoke")

    figures_dir = tmp_path / "figures"
    for stem in ("before_after_smoke_scatter", "before_after_smoke_unmatched"):
        assert (figures_dir / f"{stem}.png").exists()
        assert (figures_dir / f"{stem}.svg").exists()


def test_main_writes_csvs_and_figures(tmp_path, monkeypatch):
    decompA, decompB = _perturbed_pair()
    decomp_a_path, decomp_b_path = tmp_path / "a.pt", tmp_path / "b.pt"
    svdc.save(decompA, decomp_a_path)
    svdc.save(decompB, decomp_b_path)
    screen_a_path, screen_b_path = tmp_path / "sa.pt", tmp_path / "sb.pt"
    torch.save(_screen([1, 2, 3, 40, 5, 6, 7, 8]), screen_a_path)
    torch.save(_screen([1, 2, 3, 4, 5, 6, 7, 8]), screen_b_path)

    cfg = {**CFG, "paths": {"out": str(tmp_path)}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv", ["before_after.py", "--decomp-a", str(decomp_a_path),
                                       "--decomp-b", str(decomp_b_path), "--screen-a", str(screen_a_path),
                                       "--screen-b", str(screen_b_path), "--tag", "t"])
    ba.main()

    assert (tmp_path / "before_after_t.csv").exists()
    assert (tmp_path / "before_after_t_summary.csv").exists()
    figures_dir = tmp_path / "figures"
    assert (figures_dir / "before_after_t_scatter.png").exists()
    assert (figures_dir / "before_after_t_unmatched.png").exists()
