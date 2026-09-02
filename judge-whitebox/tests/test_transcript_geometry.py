import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch

import transcript_geometry
import wbconfig

PAIRS_PATH = "/fake/pairs.parquet"


def _rec(resid, item, cap, text_hash=None, pairs_path=PAIRS_PATH):
    # cap/item cast to native Python types -- torch.save/load (weights_only=True default since
    # torch 2.6) can't round-trip bare numpy scalars (np.float64/np.int64) inside a plain list.
    n = resid.shape[0]
    d = {"mode": "single_transcript", "resid": torch.tensor(resid, dtype=torch.float32),
         "item": [x.item() if hasattr(x, "item") else x for x in item],
         "cap": [float(x) for x in cap], "pairs_path": pairs_path}
    if text_hash is not None:
        d["text_hash"] = list(text_hash)
    else:
        d["text_hash"] = [f"h{i}" for i in range(n)]
    return d


def test_strong_weak_basic_two_per_item():
    item = np.array([0, 0, 1, 1])
    cap = np.array([0.5, 0.8, 0.3, 0.3])  # item 1 has a single distinct cap value -> dropped
    label, keep = transcript_geometry._strong_weak(item, cap)
    assert keep.tolist() == [True, True, False, False]
    assert label[0] == 0 and label[1] == 1  # lower cap -> weak, higher cap -> strong


def test_strong_weak_odd_item_puts_extra_in_strong_half():
    item = np.array([0, 0, 0])
    cap = np.array([0.1, 0.5, 0.9])
    label, keep = transcript_geometry._strong_weak(item, cap)
    assert keep.all()
    assert label.tolist() == [0, 1, 1]  # half = 3//2 = 1 weak, 2 strong


def test_center_within_item_removes_item_mean():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0], [20.0, 20.0]])
    item = np.array([0, 0, 1, 1])
    Xc = transcript_geometry._center_within_item(X, item)
    for it in (0, 1):
        assert np.allclose(Xc[item == it].mean(axis=0), 0.0)


def test_strength_axis_planted_axis_separates_strong_weak():
    """Per item: a strength axis w whose displacement scales with the item's own |cap gap| --
    within-item centring should isolate it cleanly, so the strong/weak classifier and RSA should
    both pick it up at the signal layer and see nothing at the others."""
    rng = np.random.default_rng(0)
    n_items, d, n_layers, signal_layer = 40, 8, 3, 1
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    gaps = rng.uniform(0.2, 1.0, size=n_items)
    base_cap = rng.uniform(0, 3, size=n_items)

    item, cap, resid = [], [], []
    for i in range(n_items):
        r = rng.normal(scale=0.2, size=(2, n_layers, d))
        off = rng.normal(size=d) * 5.0  # item-level nuisance, same for both transcripts
        r[:, signal_layer, :] += off
        r[0, signal_layer, :] += -3.0 * w                     # weak
        r[1, signal_layer, :] += (-3.0 + 6.0 * gaps[i]) * w    # strong, gap-scaled
        resid.append(r)
        item += [i, i]
        cap += [base_cap[i], base_cap[i] + gaps[i]]

    rec = _rec(np.concatenate(resid, axis=0), item, cap)
    summary, directions = transcript_geometry.strength_axis(
        rec, alphas=[0.1, 1, 10, 100], cv_folds=4, rsa_min_items=5)

    sig = summary.loc[summary["layer"] == signal_layer].iloc[0]
    null = summary.loc[summary["layer"] == 0].iloc[0]
    assert sig["accuracy"] > 0.9
    assert sig["rsa_r"] > 0
    assert abs(null["accuracy"] - 0.5) < 0.2
    assert directions.shape == (n_layers, d)


def test_strength_axis_drops_items_without_distinct_caps():
    """Item 0 has no distinct cap values -- excluded from the strong/weak classifier (would break
    GroupKFold with fewer than 2 qualifying items) but RSA's n_items still counts every item with
    >=2 transcripts, distinct caps or not (a zero gap is still a valid data point)."""
    resid = np.zeros((6, 2, 3))
    item = [0, 0, 1, 1, 2, 2]
    cap = [0.5, 0.5, 0.2, 0.5, 0.3, 0.5]
    rec = _rec(resid, item, cap)
    summary, _ = transcript_geometry.strength_axis(rec, alphas=[1.0], cv_folds=2, rsa_min_items=1)
    assert (summary["n_items"] == 3).all()


def test_strength_axis_raises_when_no_item_has_distinct_caps():
    resid = np.zeros((4, 2, 3))
    item = [0, 0, 1, 1]
    cap = [0.5, 0.5, 0.5, 0.5]
    rec = _rec(resid, item, cap)
    with pytest.raises(AssertionError, match="distinct cap"):
        transcript_geometry.strength_axis(rec, alphas=[1.0], cv_folds=2, rsa_min_items=1)


def test_strength_axis_requires_single_transcript_mode():
    rec = {"mode": "pairwise", "resid": torch.zeros(2, 1, 3), "item": [0, 1], "cap": [0.1, 0.5]}
    with pytest.raises(AssertionError, match="single_transcript"):
        transcript_geometry.strength_axis(rec, alphas=[1.0], cv_folds=2, rsa_min_items=1)


def test_stratified_cka_equals_plain_when_strata_identical():
    """Weak and strong copies of each item share the EXACT SAME underlying resid (only their cap
    label differs) -- duplicating every row uniformly across a Gram-matrix ratio leaves linear CKA
    unchanged, so cka_all/cka_strong/cka_weak must come out identical (not just close)."""
    rng = np.random.default_rng(2)
    n_items, d, n_layers = 20, 5, 3
    v = rng.normal(size=(n_items, n_layers, d))       # rec_a's per-item content
    u = v + rng.normal(scale=0.3, size=v.shape)        # rec_b's version of the same content

    item, cap, resid_a, resid_b, text_hash = [], [], [], [], []
    for i in range(n_items):
        for label, c in ((0, 0.0), (1, 1.0)):  # weak, strong -- identical underlying resid
            item.append(i)
            cap.append(c)
            resid_a.append(v[i])
            resid_b.append(u[i])
            text_hash.append(f"h{i}_{label}")

    rec_a = _rec(np.stack(resid_a), item, cap, text_hash)
    rec_b = _rec(np.stack(resid_b), item, cap, text_hash)

    cka_df = transcript_geometry.stratified_cka(rec_a, rec_b)
    for row in cka_df.itertuples():
        assert row.cka_strong == pytest.approx(row.cka_all, abs=1e-6)
        assert row.cka_weak == pytest.approx(row.cka_all, abs=1e-6)


def test_stratified_cka_aligns_by_text_hash_not_position():
    """rec_b's rows are in a different order (and has one extra transcript rec_a never kept) --
    alignment must go by text_hash, not row position."""
    rng = np.random.default_rng(3)
    n_items, d, n_layers = 10, 4, 2
    item = [i // 2 for i in range(2 * n_items)]
    cap = [0.0, 1.0] * n_items
    text_hash = [f"h{i}" for i in range(2 * n_items)]
    resid_a = rng.normal(size=(2 * n_items, n_layers, d))

    # rec_b: same transcripts, shuffled order, plus one extra row rec_a never has.
    perm = rng.permutation(2 * n_items)
    resid_b = resid_a[perm] + rng.normal(scale=0.01, size=resid_a.shape)
    text_hash_b = [text_hash[i] for i in perm] + ["h_extra"]
    resid_b = np.concatenate([resid_b, rng.normal(size=(1, n_layers, d))], axis=0)

    rec_a = _rec(resid_a, item, cap, text_hash)
    rec_b = _rec(resid_b, [0] * (2 * n_items + 1), [0.0] * (2 * n_items + 1), text_hash_b)

    cka_df = transcript_geometry.stratified_cka(rec_a, rec_b)
    # rec_b is essentially rec_a (reordered, tiny noise) restricted to shared rows -> CKA ~ 1.
    assert (cka_df["cka_all"] > 0.9).all()


def test_stratified_cka_raises_on_pairs_path_mismatch():
    rec_a = _rec(np.zeros((2, 2, 3)), [0, 0], [0.1, 0.5], pairs_path="/a.parquet")
    rec_b = _rec(np.zeros((2, 2, 3)), [0, 0], [0.1, 0.5], pairs_path="/b.parquet")
    with pytest.raises(AssertionError, match="pairs-file mismatch"):
        transcript_geometry.stratified_cka(rec_a, rec_b)


def test_stratified_cka_requires_single_transcript_mode():
    rec_a = {"mode": "pairwise"}
    rec_b = _rec(np.zeros((2, 2, 3)), [0, 0], [0.1, 0.5])
    with pytest.raises(AssertionError, match="single_transcript"):
        transcript_geometry.stratified_cka(rec_a, rec_b)


def _make_full_rec(n_items, n_layers, d, seed, pairs_path=PAIRS_PATH):
    rng = np.random.default_rng(seed)
    item = np.repeat(np.arange(n_items), 2)
    cap = np.tile([0.0, 1.0], n_items) + rng.uniform(0, 0.3, size=2 * n_items)  # varied gaps, no ties
    resid = rng.normal(size=(2 * n_items, n_layers, d))
    text_hash = [f"h{i}" for i in range(2 * n_items)]
    return _rec(resid, item, cap, text_hash, pairs_path=pairs_path)


def test_main_single_rec_writes_strength_outputs_no_cka(tmp_path, monkeypatch):
    rec_a = _make_full_rec(n_items=10, n_layers=2, d=4, seed=0)
    p = tmp_path / "a.pt"
    torch.save(rec_a, p)

    cfg = {"paths": {"out": str(tmp_path), "pairs": PAIRS_PATH},
           "resid": {"alphas": [1.0], "cv_folds": 2},
           "transcript_geometry": {"rsa_min_items": 1}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv", ["transcript_geometry.py", "--rec-a", str(p), "--tag", "solo"])

    transcript_geometry.main()

    out_dir = tmp_path
    assert (out_dir / "transcript_geometry_solo.pt").exists()
    csv = pd.read_csv(out_dir / "transcript_geometry_solo_strength.csv")
    assert set(csv["model"]) == {"a"}
    figs = out_dir / "figures"
    assert (figs / "transcript_geometry_solo_accuracy.png").exists()
    assert (figs / "transcript_geometry_solo_rsa.png").exists()
    assert not (out_dir / "transcript_geometry_solo_cka.csv").exists()
    assert not (figs / "transcript_geometry_solo_cka.png").exists()


def test_main_two_recs_writes_cka(tmp_path, monkeypatch):
    rec_a = _make_full_rec(n_items=10, n_layers=2, d=4, seed=0)
    rec_b = _make_full_rec(n_items=10, n_layers=2, d=4, seed=1)
    rec_b["text_hash"] = rec_a["text_hash"]  # same underlying transcripts across both recordings
    pa, pb = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(rec_a, pa)
    torch.save(rec_b, pb)

    cfg = {"paths": {"out": str(tmp_path), "pairs": PAIRS_PATH},
           "resid": {"alphas": [1.0], "cv_folds": 2},
           "transcript_geometry": {"rsa_min_items": 1}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv",
                         ["transcript_geometry.py", "--rec-a", str(pa), "--rec-b", str(pb), "--tag", "both"])

    transcript_geometry.main()

    out_dir = tmp_path
    csv = pd.read_csv(out_dir / "transcript_geometry_both_strength.csv")
    assert set(csv["model"]) == {"a", "b"}
    cka_csv = pd.read_csv(out_dir / "transcript_geometry_both_cka.csv")
    assert set(cka_csv.columns) >= {"layer", "cka_all", "cka_strong", "cka_weak"}
    figs = out_dir / "figures"
    assert (figs / "transcript_geometry_both_cka.png").exists()
