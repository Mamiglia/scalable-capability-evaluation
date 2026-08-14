import math
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch

import order_polarity as op
import svd_components as svdc
import wbconfig

CFG = {"order_polarity": {"flip_r": 0.5, "n_outliers": 3, "boot_n": 50, "boot_seed": 0, "min_effect": 0.2}}


def _pairs_df(pair_uids, winner_slot=None):
    """order-0/1 rows per pair_uid, interleaved -- matches how a judging screen's kept_row_idx
    positions line up when pairs_df has exactly this shape. winner_slot: optional dict pair_uid ->
    slot (1 or 2) of the winning transcript at order 0 (defaults to 1 for every pair); order-1's
    label is the flip (3 - slot), matching the real pairs.parquet convention (swapping order also
    swaps text_1/text_2, so the winner's slot number flips too)."""
    winner_slot = winner_slot or {}
    rows = []
    for uid in pair_uids:
        w = winner_slot.get(uid, 1)
        rows.append({"pair_uid": uid, "order": 0, "label": w})
        rows.append({"pair_uid": uid, "order": 1, "label": 3 - w})
    return pd.DataFrame(rows)


def _screen(top_k_components, n_prompts, kept_row_idx=None, per_prompt_attr=None,
            per_prompt_proj=None, model="dev", adapter=None, attr_signed=None, attr_abs=None):
    return {
        "set_name": "judging", "n_prompts": n_prompts,
        "kept_row_idx": kept_row_idx if kept_row_idx is not None else list(range(n_prompts)),
        "top_k_components": top_k_components, "model": model, "adapter": adapter,
        "per_prompt_attr": per_prompt_attr, "per_prompt_proj": per_prompt_proj,
        "attr_signed": attr_signed, "attr_abs": attr_abs,
    }


def test_order_split_recovers_planted_sym_and_anti():
    """comp0 is pure-symmetric (same value both orders), comp1 is pure-anti (flips sign)."""
    pairs_df = _pairs_df(["p0", "p1", "p2"])
    attr = torch.tensor([
        [5.0, 5.0], [5.0, -5.0],   # p0 order0, order1
        [5.0, 5.0], [5.0, -5.0],   # p1
        [5.0, 5.0], [5.0, -5.0],   # p2
    ])
    screen = _screen([("L0.q", 0), ("L0.q", 1)], n_prompts=6, per_prompt_attr=attr)

    df = op.order_split(screen, pairs_df)
    comp0 = df[df["comp"] == 0].iloc[0]
    comp1 = df[df["comp"] == 1].iloc[0]

    assert math.isclose(comp0["sym"], 5.0) and math.isclose(comp0["anti"], 0.0)
    assert math.isclose(comp0["abs_sym"], 5.0) and math.isclose(comp0["abs_anti"], 0.0)
    assert math.isclose(comp1["sym"], 0.0) and math.isclose(comp1["anti"], 5.0)
    assert math.isclose(comp1["abs_sym"], 0.0) and math.isclose(comp1["abs_anti"], 5.0)
    assert comp0["n_pairs"] == 3 and comp0["n_dropped_pairs"] == 0


def test_order_split_drops_pairs_missing_an_order():
    """p1's order-1 row was dropped upstream by screen.py (over-length) -- order_split must
    exclude p1 entirely (not just impute a missing side) and count it as dropped."""
    pairs_df = _pairs_df(["p0", "p1"])  # 4 rows: p0/o0, p0/o1, p1/o0, p1/o1
    attr = torch.tensor([[1.0], [1.0], [9.0]])  # kept rows: p0/o0, p0/o1, p1/o0 (p1/o1 missing)
    screen = _screen([("L0.q", 0)], n_prompts=3, kept_row_idx=[0, 1, 2], per_prompt_attr=attr)

    df = op.order_split(screen, pairs_df)
    row = df.iloc[0]
    assert math.isclose(row["sym"], 1.0) and math.isclose(row["anti"], 0.0)
    assert row["n_pairs"] == 1 and row["n_dropped_pairs"] == 1


def test_order_split_requires_per_prompt_attr():
    screen = _screen([("L0.q", 0)], n_prompts=1)  # per_prompt_attr left as None
    with pytest.raises(AssertionError, match="per_prompt_attr"):
        op.order_split(screen, _pairs_df(["p0"]))


def test_coherence_works_on_legacy_screen():
    """coherence() needs only the aggregate attr_signed/attr_abs dicts -- no per_prompt_attr,
    no pairs_df -- so it works on a legacy (pre-task-14) screen dict."""
    legacy = {"attr_signed": {"L0.q": torch.tensor([3.0, -1.0])},
              "attr_abs": {"L0.q": torch.tensor([5.0, 4.0])},
              "top_k_components": [("L0.q", 0), ("L0.q", 1)]}
    assert "per_prompt_attr" not in legacy

    df = op.coherence(legacy)
    assert math.isclose(df[df["comp"] == 0]["coherence"].iloc[0], 3.0 / 5.0)
    assert math.isclose(df[df["comp"] == 1]["coherence"].iloc[0], 1.0 / 4.0)


def _polarity_screens(model="dev", adapter=None):
    """3 components x 8 rows (4 pairs x 2 orders), same model+adapter both sides:
    comp0 = assessment (proj r=+1, attr r=-1), comp1 = decision-state (proj r=-1),
    comp2 = bypass (proj r=+1, attr r=+1)."""
    pairs_df = _pairs_df(["p0", "p1", "p2", "p3"])
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    top_k = [("L0.q", 0), ("L0.q", 1), ("L0.q", 2)]

    attr_norm = torch.stack([v, v, v], dim=1)
    attr_inv = torch.stack([-v, v, v], dim=1)
    proj_norm = torch.stack([v, v, v], dim=1)
    proj_inv = torch.stack([v, -v, v], dim=1)

    screen_norm = _screen(top_k, n_prompts=8, per_prompt_attr=attr_norm, per_prompt_proj=proj_norm,
                           model=model, adapter=adapter)
    screen_inv = _screen(top_k, n_prompts=8, per_prompt_attr=attr_inv, per_prompt_proj=proj_inv,
                          model=model, adapter=adapter)
    return pairs_df, screen_norm, screen_inv


def test_polarity_compare_classifies_assessment_decision_bypass(monkeypatch):
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: CFG)
    pairs_df, screen_norm, screen_inv = _polarity_screens()

    classes = op.polarity_compare(screen_norm, screen_inv, pairs_df)
    by_comp = classes.set_index("comp")

    assert by_comp.loc[0, "class"] == "assessment"
    assert math.isclose(by_comp.loc[0, "proj_r"], 1.0, abs_tol=1e-6)
    assert math.isclose(by_comp.loc[0, "attr_r"], -1.0, abs_tol=1e-6)

    assert by_comp.loc[1, "class"] == "decision-state"
    assert math.isclose(by_comp.loc[1, "proj_r"], -1.0, abs_tol=1e-6)

    assert by_comp.loc[2, "class"] == "bypass"
    assert math.isclose(by_comp.loc[2, "proj_r"], 1.0, abs_tol=1e-6)
    assert math.isclose(by_comp.loc[2, "attr_r"], 1.0, abs_tol=1e-6)


def test_polarity_compare_row_alignment_drops_non_shared_pairs(monkeypatch):
    """p4 exists only in screen_norm; polarity_compare must align on shared (pair_uid, order)
    rows only, so p4's deliberately-poisoned values never enter the r/flip-rate computation."""
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: CFG)
    pairs_df = _pairs_df(["p0", "p1", "p2", "p3", "p4"])  # 10 rows
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    top_k = [("L0.q", 0)]

    attr_norm = torch.cat([-v, torch.tensor([-999.0, 5.0])]).unsqueeze(1)
    proj_norm = torch.cat([v, torch.tensor([100.0, -100.0])]).unsqueeze(1)
    screen_norm = _screen(top_k, n_prompts=10, per_prompt_attr=attr_norm, per_prompt_proj=proj_norm)

    # screen_inv never saw p4 at all (kept_row_idx stops at row 7 -> p0..p3 only)
    screen_inv = _screen(top_k, n_prompts=8, kept_row_idx=list(range(8)),
                          per_prompt_attr=v.unsqueeze(1), per_prompt_proj=v.unsqueeze(1))

    classes = op.polarity_compare(screen_norm, screen_inv, pairs_df)
    row = classes.iloc[0]
    assert row["n_rows"] == 8
    assert math.isclose(row["proj_r"], 1.0, abs_tol=1e-6)
    assert math.isclose(row["attr_r"], -1.0, abs_tol=1e-6)
    assert row["class"] == "assessment"


def test_polarity_compare_requires_matching_model_adapter():
    pairs_df, screen_norm, screen_inv = _polarity_screens()
    screen_inv["model"] = "other-model"
    with pytest.raises(AssertionError, match="same model\\+adapter"):
        op.polarity_compare(screen_norm, screen_inv, pairs_df)


def test_class_distribution_side_by_side():
    classes_base = pd.DataFrame({"class": ["assessment", "assessment", "bypass"]})
    classes_ft = pd.DataFrame({"class": ["bypass", "bypass", "mixed"]})
    dist = op.class_distribution(classes_base, classes_ft)

    def frac(side, cls):
        return dist[(dist["side"] == side) & (dist["class"] == cls)]["frac"].iloc[0]

    assert math.isclose(frac("base", "assessment"), 2 / 3)
    assert math.isclose(frac("base", "bypass"), 1 / 3)
    assert math.isclose(frac("ft", "bypass"), 2 / 3)
    assert math.isclose(frac("ft", "mixed"), 1 / 3)


def _reclass_fixture():
    """4 pairs (mixed winner_slot so slot/winner-side are orthogonal, balanced 2x2), 4 components:
    L0.q = planted mean-offset (+1.5, identical shape/proj otherwise) -- instruction-sensitive,
    legacy class is "bypass" (this is the real-world regression case: Pearson r alone is
    offset-blind). L1.k.h0 = planted slot component (value = 5*slot, winner-invariant) -- slot.
    L2.v.h0 = planted winner-side/content component (value = 5*winner, slot-invariant) -- content.
    L3.o.h0 = flat noise, tiny attr_abs -- unidentified. attr_abs deliberately spread (15/25/30/
    0.5, median 20) so the median gate cleanly separates identified from noise components.

    winner is built directly from the row's own raw label (label == 1), independently of
    order_regression's internals -- NOT via a reconstructed "pair-anchored" label. That anchor
    reconstruction (label if order==0 else 3-label) is pair-CONSTANT and is exactly the
    interaction column (slot*winner), not winner_side itself -- planting the content component
    off that formula would tautologically validate whatever order_regression happens to compute,
    including the beta_winner/beta_inter swap bug this fixture caught (see task-18-report.md,
    fix round 1): under the pre-fix code this same 5*winner array loads entirely onto beta_inter
    (content=False, wrongly unidentified), not beta_winner."""
    winner_slot = {"p0": 1, "p1": 2, "p2": 1, "p3": 2}
    pairs_df = _pairs_df(["p0", "p1", "p2", "p3"], winner_slot=winner_slot)
    order = pairs_df["order"].to_numpy()
    label = pairs_df["label"].to_numpy()
    slot = np.where(order == 0, 1.0, -1.0)
    winner = np.where(label == 1, 1.0, -1.0)  # row-wise: flips across a pair's two order rows

    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    delta = 1.5
    comp_slot = torch.tensor(5.0 * slot)
    comp_content = torch.tensor(5.0 * winner)
    comp_noise = torch.zeros(8)

    attr_norm = torch.stack([v, comp_slot, comp_content, comp_noise], dim=1)
    attr_inv = torch.stack([v + delta, comp_slot, comp_content, comp_noise], dim=1)
    proj_norm = attr_norm.clone()  # projections unchanged across polarity for every component here
    top_k = [("L0.q", 0), ("L1.k.h0", 0), ("L2.v.h0", 0), ("L3.o.h0", 0)]
    attr_abs = {"L0.q": torch.tensor([15.0]), "L1.k.h0": torch.tensor([25.0]),
                "L2.v.h0": torch.tensor([30.0]), "L3.o.h0": torch.tensor([0.5])}
    attr_signed = {k: v.clone() for k, v in attr_abs.items()}

    screen_norm = _screen(top_k, n_prompts=8, per_prompt_attr=attr_norm, per_prompt_proj=proj_norm,
                           attr_signed=attr_signed, attr_abs=attr_abs)
    screen_inv = _screen(top_k, n_prompts=8, per_prompt_attr=attr_inv, per_prompt_proj=proj_norm,
                          attr_signed=attr_signed, attr_abs=attr_abs)
    return pairs_df, screen_norm, screen_inv


def test_affine_stats_recovers_mean_offset_pearson_r_misses():
    """y = x + constant: Pearson r is 1 (offset-blind) but affine_stats' mean_shift/effect catch
    the shift -- the exact failure mode this replaces."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    y = x + 1.5
    stats = op.affine_stats(x, y)
    assert math.isclose(stats["mean_shift"], 1.5)
    assert math.isclose(stats["scale"], 1.0)
    assert math.isclose(stats["r_centred"], 1.0, abs_tol=1e-9)
    assert stats["effect"] > 1.0
    assert math.isnan(stats["ci_lo"]) and math.isnan(stats["ci_hi"])  # no boot_idx given


def test_boot_pair_idx_resamples_pairs_not_rows():
    """Each draw's row set is built from whole resampled pairs -- both rows of a resampled pair
    show up together, never just one."""
    pair_uid = np.array(["p0", "p0", "p1", "p1", "p2", "p2"])
    draws = op._boot_pair_idx(pair_uid, boot_n=20, seed=0)
    assert len(draws) == 20
    for idx in draws:
        drawn_uids = pair_uid[idx]
        for uid in np.unique(drawn_uids):
            assert (drawn_uids == uid).sum() % 2 == 0  # whole pairs, never a lone row


def test_boot_pair_idx_deterministic_via_seed():
    pair_uid = np.array(["p0", "p0", "p1", "p1"])
    a = op._boot_pair_idx(pair_uid, boot_n=10, seed=0)
    b = op._boot_pair_idx(pair_uid, boot_n=10, seed=0)
    for xa, xb in zip(a, b):
        assert np.array_equal(xa, xb)


def test_order_regression_recovers_planted_slot_and_winner_effects():
    _, screen_norm, _ = _reclass_fixture()
    reg = op.order_regression(screen_norm, _pairs_df(["p0", "p1", "p2", "p3"],
                                                       winner_slot={"p0": 1, "p1": 2, "p2": 1, "p3": 2}))
    by_key = reg.set_index("key")
    assert math.isclose(by_key.loc["L1.k.h0", "beta_slot"], 5.0)
    assert math.isclose(by_key.loc["L1.k.h0", "beta_winner"], 0.0, abs_tol=1e-9)
    assert math.isclose(by_key.loc["L2.v.h0", "beta_winner"], 5.0)
    assert math.isclose(by_key.loc["L2.v.h0", "beta_slot"], 0.0, abs_tol=1e-9)


def test_reclassify_planted_signatures(monkeypatch):
    """End-to-end: order_regression + polarity_affine -> reclassify recovers all four required
    signatures from _reclass_fixture, and the mean-offset component's legacy class (still
    computed by polarity_compare, unchanged) is "bypass" -- the regression this whole rewrite
    fixes."""
    pairs_df, screen_norm, screen_inv = _reclass_fixture()
    order_reg = op.order_regression(screen_norm, pairs_df)
    affine = op.polarity_affine(screen_norm, screen_inv, pairs_df, boot_n=50, seed=0)
    reclass = op.reclassify(order_reg, affine, screen_norm, min_effect=0.2)
    by_key = reclass.set_index("key")

    assert by_key.loc["L0.q", "instruction_sensitive"] and not by_key.loc["L0.q", "unidentified"]
    assert not by_key.loc["L0.q", "slot"] and not by_key.loc["L0.q", "content"]

    assert by_key.loc["L1.k.h0", "slot"] and not by_key.loc["L1.k.h0", "unidentified"]
    assert not by_key.loc["L1.k.h0", "content"] and not by_key.loc["L1.k.h0", "instruction_sensitive"]

    assert by_key.loc["L2.v.h0", "content"] and not by_key.loc["L2.v.h0", "unidentified"]
    assert not by_key.loc["L2.v.h0", "slot"] and not by_key.loc["L2.v.h0", "instruction_sensitive"]

    assert by_key.loc["L3.o.h0", "unidentified"]
    assert not (by_key.loc["L3.o.h0", "slot"] or by_key.loc["L3.o.h0", "content"]
                or by_key.loc["L3.o.h0", "instruction_sensitive"])

    legacy = op.polarity_compare(screen_norm, screen_inv, pairs_df).set_index("key")
    assert legacy.loc["L0.q", "class"] == "bypass"


def test_main_writes_affine_csv_and_boolean_figure(tmp_path, monkeypatch):
    """CLI with --screen-inv: order_polarity_{tag}.csv gains beta columns (order_regression),
    and order_polarity_{tag}_affine.csv + its figure are written with the boolean labels."""
    pairs_df, screen_norm, screen_inv = _reclass_fixture()
    for s in (screen_norm, screen_inv):
        s["split"], s["limit"] = None, None
    norm_path, inv_path = tmp_path / "norm.pt", tmp_path / "inv.pt"
    torch.save(screen_norm, norm_path)
    torch.save(screen_inv, inv_path)

    cfg = {**CFG, "paths": {"out": str(tmp_path)}}
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(op.wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)
    monkeypatch.setattr(sys, "argv", ["order_polarity.py", "--screen-norm", str(norm_path),
                                       "--screen-inv", str(inv_path), "--tag", "t"])

    op.main()

    main_df = pd.read_csv(tmp_path / "order_polarity_t.csv")
    assert {"beta_slot", "se_slot", "beta_winner", "se_winner", "coherence"} <= set(main_df.columns)

    affine_path = tmp_path / "order_polarity_t_affine.csv"
    assert affine_path.exists()
    affine_df = pd.read_csv(affine_path)
    assert {"instruction_sensitive", "slot", "content", "unidentified", "class_legacy"} <= set(affine_df.columns)
    assert bool(affine_df.set_index("key").loc["L0.q", "instruction_sensitive"])

    figures_dir = tmp_path / "figures"
    assert (figures_dir / "order_polarity_t_class_by_kind.png").exists()
    assert (figures_dir / "order_polarity_t_class_by_kind.svg").exists()


def test_figures_written(tmp_path, monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: {"paths": {"out": str(tmp_path)}})

    main_df = pd.DataFrame({
        "key": ["L0.q", "L1.k.h0"], "comp": [0, 1],
        "abs_sym": [4.0, 1.0], "abs_anti": [1.0, 4.0], "coherence": [0.8, 0.2],
    })
    classes = pd.DataFrame({
        "key": ["L0.q", "L1.k.h0", "L2.gate"], "comp": [0, 1, 0],
        "attr_r": [-1.0, 1.0, 1.0], "proj_r": [1.0, -1.0, 1.0],
        "class": ["assessment", "decision-state", "bypass"],
    })
    dist = op.class_distribution(classes, classes)
    reclass = pd.DataFrame({
        "key": ["L0.q", "L1.k.h0", "L2.gate"], "comp": [0, 1, 0],
        "instruction_sensitive": [True, False, False], "slot": [False, True, False],
        "content": [False, False, False], "unidentified": [False, False, True],
    })

    op.fig_coherence(main_df, "smoke", n_outliers=2)
    op.fig_polarity(classes, "smoke", thresh=0.5)
    op.fig_class_composition(classes, "smoke")
    op.fig_bft_distribution(dist, "smoke")
    op.fig_boolean_class_by_kind(reclass, "smoke")

    figures_dir = tmp_path / "figures"
    for stem in ("order_polarity_smoke_coherence", "order_polarity_smoke_polarity",
                 "order_polarity_smoke_class_by_layer", "order_polarity_smoke_bft_distribution",
                 "order_polarity_smoke_class_by_kind"):
        assert (figures_dir / f"{stem}.png").exists()
        assert (figures_dir / f"{stem}.svg").exists()


def test_main_writes_norm_only_outputs(tmp_path, monkeypatch):
    """CLI smoke test with only --screen-norm: order split + coherence csv, no classes file."""
    pairs_df = _pairs_df(["p0", "p1"])
    attr = torch.tensor([[5.0], [5.0], [3.0], [3.0]])
    screen = _screen([("L0.q", 0)], n_prompts=4, per_prompt_attr=attr,
                      attr_signed={"L0.q": torch.tensor([2.0])}, attr_abs={"L0.q": torch.tensor([4.0])})
    screen["split"], screen["limit"] = None, None
    screen_path = tmp_path / "screen.pt"
    torch.save(screen, screen_path)

    cfg = {**CFG, "paths": {"out": str(tmp_path)}}
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(op.wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)
    monkeypatch.setattr(sys, "argv", ["order_polarity.py", "--screen-norm", str(screen_path), "--tag", "t"])

    op.main()

    assert (tmp_path / "order_polarity_t.csv").exists()
    assert not (tmp_path / "order_polarity_t_classes.csv").exists()
    df = pd.read_csv(tmp_path / "order_polarity_t.csv")
    assert set(df.columns) >= {"key", "comp", "sym", "anti", "abs_sym", "abs_anti", "coherence"}


def test_classify_ambiguous_r_is_mixed():
    """Neither correlation clears the threshold in either direction -- not stable, not flipped."""
    assert op._classify(attr_r=0.2, proj_r=0.2, thresh=0.5) == "mixed"


def test_polarity_compare_constant_column_is_mixed(monkeypatch):
    """A component whose per_prompt_attr is constant across the shared rows has an undefined
    (nan) Pearson r -- must classify as mixed, not fall through to bypass."""
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: CFG)
    pairs_df, screen_norm, screen_inv = _polarity_screens()
    n = screen_norm["per_prompt_attr"].shape[0]
    const_col = torch.full((n, 1), 3.0)  # zero variance -> attr_r is nan
    varying_col = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]).unsqueeze(1)

    screen_norm["top_k_components"] = screen_norm["top_k_components"] + [("L0.q", 3)]
    screen_inv["top_k_components"] = screen_inv["top_k_components"] + [("L0.q", 3)]
    screen_norm["per_prompt_attr"] = torch.cat([screen_norm["per_prompt_attr"], const_col], dim=1)
    screen_inv["per_prompt_attr"] = torch.cat([screen_inv["per_prompt_attr"], const_col], dim=1)
    screen_norm["per_prompt_proj"] = torch.cat([screen_norm["per_prompt_proj"], varying_col], dim=1)
    screen_inv["per_prompt_proj"] = torch.cat([screen_inv["per_prompt_proj"], varying_col], dim=1)

    classes = op.polarity_compare(screen_norm, screen_inv, pairs_df)
    row = classes[classes["comp"] == 3].iloc[0]
    assert math.isnan(row["attr_r"])
    assert row["class"] == "mixed"


def _swap01(decomp, key):
    """Same decomp, with the first two components of `key` swapped -- simulates comp index j
    pointing at a different direction across two decomps of the same architecture."""
    u, s, v = decomp.entries[key]
    perm = torch.tensor([1, 0] + list(range(2, u.shape[1])))
    d2 = svdc.Decomp()
    d2.entries[key] = (u[:, perm].clone(), s[perm].clone(), v[:, perm].clone())
    d2.meta[key] = dict(decomp.meta[key])
    return d2


def _toy_decomp(seed, key="L0.q"):
    g = torch.Generator().manual_seed(seed)
    u, _, vh = torch.linalg.svd(torch.randn(8, 6, generator=g), full_matrices=False)
    s = torch.linspace(10.0, 3.0, 6)
    decomp = svdc.Decomp()
    decomp.entries[key] = (u.clone(), s.clone(), vh.T.clone())
    decomp.meta[key] = {"layer": 0, "kind": "q", "head": None, "kv_head": None}
    return decomp


def test_migration_matched_pairs_swapped_components_correctly(monkeypatch):
    """decompB's key has comps 0/1 swapped relative to decompA -- migration through
    before_after.match() must pair base-comp0 with ft-comp1 (the same physical direction), not
    base-comp0 with ft-comp0 (what a naive (key, comp) index join would do, and which would be
    wrong here)."""
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: {"before_after": {"min_sim": 0.5}})
    decompA = _toy_decomp(seed=1)
    decompB = _swap01(decompA, "L0.q")

    classes_base = pd.DataFrame({"key": ["L0.q", "L0.q"], "comp": [0, 1],
                                  "class": ["assessment", "bypass"]})
    classes_ft = pd.DataFrame({"key": ["L0.q", "L0.q"], "comp": [0, 1],
                                "class": ["decision-state", "mixed"]})

    migration = op.migration_matched(classes_base, classes_ft, decompA, decompB)

    row_a0 = migration[migration["comp_a"] == 0].iloc[0]
    assert row_a0["comp_b"] == 1  # A's direction0 sits at B's index1
    assert row_a0["class_base"] == "assessment" and row_a0["class_ft"] == "mixed"
    assert row_a0["sim"] > 0.99 and not row_a0["low_sim"]

    row_a1 = migration[migration["comp_a"] == 1].iloc[0]
    assert row_a1["comp_b"] == 0  # A's direction1 sits at B's index0
    assert row_a1["class_base"] == "bypass" and row_a1["class_ft"] == "decision-state"

    # what a naive index join would have wrongly produced for comp 0:
    assert classes_ft.set_index("comp").loc[0, "class"] != row_a0["class_ft"]


def test_main_requires_decomps_when_bft_adapters_differ(tmp_path, monkeypatch):
    """base and ft share `model` but differ in `adapter` -- (key, comp) is NOT the same SVD
    basis, so main() must refuse the migration step (with a clear message) unless
    --decomp-a/--decomp-b are given, rather than silently naive-index-joining wrong pairs."""
    pairs_df, screen_norm, screen_inv = _polarity_screens(model="dev", adapter=None)
    _, screen_b_norm, screen_b_inv = _polarity_screens(model="dev", adapter="/path/to/ft")
    # main() computes the order_split+coherence csv (screen-norm only) before ever reaching the
    # migration step -- coherence() needs attr_signed/attr_abs, so screen_norm must carry them.
    screen_norm["attr_signed"] = {"L0.q": torch.tensor([1.0, 1.0, 1.0])}
    screen_norm["attr_abs"] = {"L0.q": torch.tensor([2.0, 2.0, 2.0])}

    paths = {}
    for name, screen in (("norm", screen_norm), ("inv", screen_inv),
                          ("b_norm", screen_b_norm), ("b_inv", screen_b_inv)):
        p = tmp_path / f"{name}.pt"
        torch.save(screen, p)
        paths[name] = str(p)

    cfg = {"order_polarity": {"flip_r": 0.5, "n_outliers": 3, "boot_n": 50, "boot_seed": 0,
                               "min_effect": 0.2},
           "before_after": {"min_sim": 0.5}, "paths": {"out": str(tmp_path)}}
    monkeypatch.setattr(op.wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(op.wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)
    monkeypatch.setattr(sys, "argv", [
        "order_polarity.py", "--screen-norm", paths["norm"], "--screen-inv", paths["inv"],
        "--screen-b-norm", paths["b_norm"], "--screen-b-inv", paths["b_inv"], "--tag", "t"])

    with pytest.raises(AssertionError, match="--decomp-a"):
        op.main()
