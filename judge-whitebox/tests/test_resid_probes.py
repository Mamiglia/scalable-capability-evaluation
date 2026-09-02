import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch

import resid_probes
import wbconfig
import wbpairs


def _planted_rec(n_items, n_per_item, n_layers, d, signal_layer, seed, scale=6.0):
    """n = n_items * n_per_item prompts, grouped by item (n_per_item rows share an item). A
    random direction w linearly separates a random binary y at signal_layer only; every other
    layer is pure noise. Returns (rec dict, pairs_df, y)."""
    rng = np.random.default_rng(seed)
    n = n_items * n_per_item
    items = np.repeat(np.arange(n_items), n_per_item)
    y = rng.integers(0, 2, size=n)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    resid = rng.normal(size=(n, n_layers, d))
    resid[:, signal_layer, :] += scale * (2 * y - 1)[:, None] * w[None, :]

    rec = {"resid": torch.tensor(resid, dtype=torch.float32),
           "l1": torch.zeros(n), "l2": torch.zeros(n), "kept_row_idx": list(range(n)),
           "split": None, "limit": None, "invert": False, "model": "base", "adapter": None}
    pairs_df = pd.DataFrame({"item": items, "label": np.where(y == 1, 1, 2)})
    return rec, pairs_df, y


def test_fit_probes_winner_planted_decodable_and_null_layer():
    rec, pairs_df, _ = _planted_rec(n_items=20, n_per_item=2, n_layers=4, d=10, signal_layer=2, seed=0)

    probes = resid_probes.fit_probes(rec, pairs_df, ["winner"], alphas=[0.1, 1, 10, 100], cv_folds=4)
    summary, directions = probes["winner"]

    signal_acc = summary.loc[summary["layer"] == 2, "accuracy"].iloc[0]
    null_acc = summary.loc[summary["layer"] == 0, "accuracy"].iloc[0]
    assert signal_acc > 0.9
    assert abs(null_acc - 0.5) < 0.2
    assert directions.shape == (4, 10)


def test_fit_probes_verdict_target_uses_l1_l2_sign():
    n, n_layers, d, signal_layer = 30, 3, 6, 1
    rng = np.random.default_rng(3)
    items = np.repeat(np.arange(15), 2)
    y = rng.integers(0, 2, size=n)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    resid = rng.normal(size=(n, n_layers, d))
    resid[:, signal_layer, :] += 6.0 * (2 * y - 1)[:, None] * w[None, :]
    l1 = np.where(y == 1, 1.0, -1.0)
    l2 = -l1  # l1 > l2 exactly when y == 1

    rec = {"resid": torch.tensor(resid, dtype=torch.float32),
           "l1": torch.tensor(l1, dtype=torch.float32), "l2": torch.tensor(l2, dtype=torch.float32),
           "kept_row_idx": list(range(n)), "split": None, "limit": None,
           "invert": False, "model": "base", "adapter": None}
    pairs_df = pd.DataFrame({"item": items, "label": [1] * n})

    probes = resid_probes.fit_probes(rec, pairs_df, ["verdict"], alphas=[0.1, 1, 10], cv_folds=3)
    summary, _ = probes["verdict"]
    assert summary.loc[summary["layer"] == signal_layer, "accuracy"].iloc[0] > 0.9


def test_fit_probes_polarity_target_needs_rec_inv():
    rec = {"resid": torch.zeros(2, 1, 3), "l1": torch.zeros(2), "l2": torch.zeros(2),
           "kept_row_idx": [0, 1], "split": None, "limit": None, "invert": False,
           "model": "base", "adapter": None}
    pairs_df = pd.DataFrame({"item": [0, 1], "label": [1, 2]})
    with pytest.raises(AssertionError, match="rec_inv"):
        resid_probes.fit_probes(rec, pairs_df, ["polarity"], alphas=[1], cv_folds=2)


def test_fit_probes_polarity_target_decodes_planted_shift():
    n, n_layers, d, signal_layer = 20, 3, 6, 0
    rng = np.random.default_rng(4)
    items = np.repeat(np.arange(10), 2)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    resid_norm = rng.normal(size=(n, n_layers, d))
    resid_inv = rng.normal(size=(n, n_layers, d))
    resid_inv[:, signal_layer, :] += 6.0 * w[None, :]  # shifted at signal_layer only

    base = dict(kept_row_idx=list(range(n)), split=None, limit=None, model="base", adapter=None)
    rec = {"resid": torch.tensor(resid_norm, dtype=torch.float32), "l1": torch.zeros(n),
           "l2": torch.zeros(n), "invert": False, **base}
    rec_inv = {"resid": torch.tensor(resid_inv, dtype=torch.float32), "l1": torch.zeros(n),
               "l2": torch.zeros(n), "invert": True, **base}
    pairs_df = pd.DataFrame({"item": items, "label": [1] * n})

    probes = resid_probes.fit_probes(rec, pairs_df, ["polarity"], alphas=[0.1, 1, 10], cv_folds=4,
                                      rec_inv=rec_inv)
    summary, directions = probes["polarity"]
    assert summary.loc[summary["layer"] == signal_layer, "accuracy"].iloc[0] > 0.9
    assert directions.shape == (n_layers, d)


def test_principal_angle_identical_is_zero():
    v = np.array([1.0, 2.0, -1.0])
    assert resid_probes.principal_angle_deg(v, v) == pytest.approx(0.0, abs=1e-4)


def test_principal_angle_orthogonal_is_90():
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert resid_probes.principal_angle_deg(u, v) == pytest.approx(90.0, abs=1e-4)


def test_principal_angle_sign_invariant():
    v = np.array([1.0, 2.0, -1.0])
    assert resid_probes.principal_angle_deg(v, -v) == pytest.approx(0.0, abs=1e-4)


def test_linear_cka_identical_matrices_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    assert resid_probes.linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_unrelated_matrices_is_low():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    Y = rng.normal(size=(300, 5))
    assert resid_probes.linear_cka(X, Y) < 0.3


PAIRS_PATH = "/fake/pairs.parquet"


def _make_rec(n, n_layers, d, model, adapter, invert, seed, pairs_path=PAIRS_PATH):
    rng = np.random.default_rng(seed)
    return {"resid": torch.tensor(rng.normal(size=(n, n_layers, d)), dtype=torch.float32),
            "l1": torch.tensor(rng.normal(size=n), dtype=torch.float32),
            "l2": torch.tensor(rng.normal(size=n), dtype=torch.float32),
            "kept_row_idx": list(range(n)), "split": "eval_mm", "limit": None,
            "invert": invert, "model": model, "adapter": adapter, "pairs_path": pairs_path}


def test_main_end_to_end_base_and_ft(tmp_path, monkeypatch):
    n, n_layers, d = 12, 3, 6
    recs = {
        "norm": _make_rec(n, n_layers, d, "base", None, False, 0),
        "inv": _make_rec(n, n_layers, d, "base", None, True, 1),
        "bnorm": _make_rec(n, n_layers, d, "ft", "adap", False, 2),
        "binv": _make_rec(n, n_layers, d, "ft", "adap", True, 3),
    }
    paths = {}
    for name, rec in recs.items():
        p = tmp_path / f"{name}.pt"
        torch.save(rec, p)
        paths[name] = p

    pairs_df = pd.DataFrame({"item": [f"it{i // 2}" for i in range(n)], "label": [1, 2] * (n // 2)})
    cfg = {"paths": {"out": str(tmp_path), "pairs": PAIRS_PATH}, "resid": {"alphas": [1.0, 10.0], "cv_folds": 3}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)
    monkeypatch.setattr(sys, "argv", ["resid_probes.py",
                                       "--rec-norm", str(paths["norm"]), "--rec-inv", str(paths["inv"]),
                                       "--rec-b-norm", str(paths["bnorm"]), "--rec-b-inv", str(paths["binv"]),
                                       "--tag", "t"])

    resid_probes.main()

    assert (tmp_path / "resid_probes_t.pt").exists()
    csv = pd.read_csv(tmp_path / "resid_probes_t.csv")
    assert set(csv["model"]) == {"base", "ft"}
    assert set(csv["target"]) == {"verdict", "winner", "polarity"}

    figs = tmp_path / "figures"
    for stem in ("accuracy", "angle", "cka"):
        assert (figs / f"resid_probes_t_{stem}.png").exists()
        assert (figs / f"resid_probes_t_{stem}.svg").exists()


def test_main_base_only_skips_comparison_figures(tmp_path, monkeypatch):
    n, n_layers, d = 8, 2, 4
    rec_norm = _make_rec(n, n_layers, d, "base", None, False, 5)
    p = tmp_path / "norm.pt"
    torch.save(rec_norm, p)

    pairs_df = pd.DataFrame({"item": [f"it{i // 2}" for i in range(n)], "label": [1, 2] * (n // 2)})
    cfg = {"paths": {"out": str(tmp_path), "pairs": PAIRS_PATH}, "resid": {"alphas": [1.0], "cv_folds": 2}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)
    monkeypatch.setattr(sys, "argv", ["resid_probes.py", "--rec-norm", str(p), "--tag", "b"])

    resid_probes.main()

    figs = tmp_path / "figures"
    assert (figs / "resid_probes_b_accuracy.png").exists()
    assert not (figs / "resid_probes_b_angle.png").exists()
    assert not (figs / "resid_probes_b_cka.png").exists()


def test_main_raises_on_split_mismatch_between_recordings(tmp_path, monkeypatch):
    """rec-norm and rec-inv have identical shapes but different `split` -- main() loads ONE
    pairs_df keyed off rec-norm and decodes rec-inv's kept_row_idx against it positionally, so a
    split mismatch would silently mislabel/mispair prompts if not caught up front."""
    n, n_layers, d = 8, 2, 4
    rec_norm = _make_rec(n, n_layers, d, "base", None, False, 0)
    rec_inv = _make_rec(n, n_layers, d, "base", None, True, 1)
    rec_inv["split"] = "eval_other"  # mismatch vs rec_norm's "eval_mm"

    p_norm, p_inv = tmp_path / "norm.pt", tmp_path / "inv.pt"
    torch.save(rec_norm, p_norm)
    torch.save(rec_inv, p_inv)

    pairs_df = pd.DataFrame({"item": [f"it{i // 2}" for i in range(n)], "label": [1, 2] * (n // 2)})
    cfg = {"paths": {"out": str(tmp_path), "pairs": PAIRS_PATH}, "resid": {"alphas": [1.0], "cv_folds": 2}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)
    monkeypatch.setattr(sys, "argv", ["resid_probes.py", "--rec-norm", str(p_norm),
                                       "--rec-inv", str(p_inv), "--tag", "mismatch"])

    with pytest.raises(AssertionError) as exc_info:
        resid_probes.main()
    assert str(p_inv) in str(exc_info.value)
    assert "eval_other" in str(exc_info.value) and "eval_mm" in str(exc_info.value)


def test_main_raises_on_pairs_path_mismatch_vs_cfg(tmp_path, monkeypatch):
    """rec-norm was recorded against a DIFFERENT pairs parquet than what this run's cfg resolves
    to (e.g. the eval_mm split name exists in both the main pairs file and a sweep file) -- same
    split NAME, same row count, so the split/limit check alone would pass while pairs_df is loaded
    from the wrong file and every kept_row_idx is decoded against the wrong rows."""
    n, n_layers, d = 8, 2, 4
    rec_norm = _make_rec(n, n_layers, d, "base", None, False, 0, pairs_path="/other/sweep_pairs.parquet")
    p_norm = tmp_path / "norm.pt"
    torch.save(rec_norm, p_norm)

    cfg = {"paths": {"out": str(tmp_path), "pairs": PAIRS_PATH}, "resid": {"alphas": [1.0], "cv_folds": 2}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv", ["resid_probes.py", "--rec-norm", str(p_norm), "--tag", "pmismatch"])

    with pytest.raises(AssertionError) as exc_info:
        resid_probes.main()
    assert str(p_norm) in str(exc_info.value)
    assert "/other/sweep_pairs.parquet" in str(exc_info.value) and PAIRS_PATH in str(exc_info.value)


def test_assert_pairs_path_warns_not_raises_on_legacy_missing_field(capsys):
    resid_probes._assert_pairs_path(None, PAIRS_PATH, "legacy_rec.pt")  # no raise
    assert "legacy" in capsys.readouterr().out.lower()
