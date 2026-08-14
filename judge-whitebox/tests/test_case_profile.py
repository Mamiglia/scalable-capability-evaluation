import math
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

import case_profile
import wbconfig
import wbpairs


@pytest.mark.parametrize("model_name,expected", [
    ("open-mixtral-8x22b", "open-mixtral"),
    ("claude-3-5-haiku-20241022", "claude"),
    ("DeepSeek-R1-0528", "deepseek-r1"),
    ("qwen3-235b-a22b", "qwen3"),
    ("Llama-3.1-405B-Instruct", "llama"),
])
def test_family_matches_real_roster(model_name, expected):
    assert case_profile.family(model_name) == expected


def _synthetic_meta(n):
    return pd.DataFrame({"order": [i % 2 for i in range(n)], "item": [f"item{i}" for i in range(n)],
                          "model_1": ["gpt-4o"] * n, "model_2": ["claude-3"] * n})


def test_nmf_recovers_block_structure():
    """Two prompt groups x two component groups, block signal + noise: NMF(2 factors) recovers
    which components belong to which factor (ARI on H-argmax > 0.9), and prompt groups separate
    cleanly on W's argmax factor (up to the arbitrary factor-label permutation)."""
    n_prompts, n_components = 20, 30
    rng = np.random.RandomState(0)
    X = rng.uniform(0.0, 0.3, size=(n_prompts, n_components))
    X[:10, :15] += rng.uniform(3.0, 5.0, size=(10, 15))  # prompts 0-9 load on comps 0-14
    X[10:, 15:] += rng.uniform(3.0, 5.0, size=(10, 15))  # prompts 10-19 load on comps 15-29

    top_k_components = [("L0.q", i) for i in range(n_components)]
    prompt_meta = _synthetic_meta(n_prompts)

    W, H, report = case_profile.profile(X, top_k_components, prompt_meta, n_factors=2, seed=0, max_iter=500)
    assert W.shape == (n_prompts, 2)
    assert H.shape == (2, n_components)

    true_comp_group = np.array([0] * 15 + [1] * 15)
    comp_factor = H.argmax(axis=0)
    assert adjusted_rand_score(true_comp_group, comp_factor) > 0.9

    prompt_factor = W.argmax(axis=1)
    assert len(set(prompt_factor[:10])) == 1
    assert len(set(prompt_factor[10:])) == 1
    assert prompt_factor[0] != prompt_factor[10]


def test_context_report_group_means():
    W = np.array([[1.0, 5.0], [3.0, 5.0], [10.0, 0.0], [20.0, 0.0]])
    prompt_meta = pd.DataFrame({
        "order": [0, 1, 0, 1], "item": ["itemA", "itemA", "itemB", "itemB"],
        "model_1": ["gpt-4o-2024", "gpt-4o-2024", "Llama-3-70b", "Llama-3-70b"],
        "model_2": ["claude-3-opus"] * 4,
    })

    report = case_profile._context_report(W, prompt_meta)

    def value(variable, group, factor):
        row = report[(report["variable"] == variable) & (report["group"] == group) & (report["factor"] == factor)]
        assert len(row) == 1
        return row["value"].iloc[0]

    assert math.isclose(value("order", 0, 0), (1.0 + 10.0) / 2)  # rows 0, 2
    assert math.isclose(value("order", 1, 1), (5.0 + 0.0) / 2)  # rows 1, 3
    assert math.isclose(value("item", "itemA", 1), 5.0)  # rows 0, 1
    assert math.isclose(value("model_1_family", "gpt", 0), (1.0 + 3.0) / 2)  # rows 0, 1
    assert math.isclose(value("model_2_family", "claude", 0), W[:, 0].mean())  # all 4 rows


def test_kind_composition_fractions_sum_to_one_per_factor():
    H = np.array([[1.0, 3.0, 0.0], [2.0, 0.0, 4.0]])
    top_k_components = [("L0.q", 0), ("L1.k.h0", 0), ("L2.gate", 0)]
    kinds = case_profile._kind_composition(H, top_k_components)

    for f in (0, 1):
        assert math.isclose(kinds[kinds["factor"] == f]["value"].sum(), 1.0)
    row = kinds[(kinds["group"] == "q") & (kinds["factor"] == 0)]
    assert math.isclose(row["value"].iloc[0], 1.0 / 4.0)


def test_figures_written(tmp_path, monkeypatch):
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: {"paths": {"out": str(tmp_path)}})

    W = np.array([[1.0, 0.5], [0.2, 1.5], [3.0, 0.1], [0.4, 2.0]])
    context = case_profile._context_report(W, _synthetic_meta(4))
    H = np.array([[1.0, 2.0, 0.0], [0.5, 0.0, 3.0]])
    kinds = case_profile._kind_composition(H, [("L0.q", 0), ("L1.k.h0", 0), ("L2.gate", 0)])
    report = pd.concat([context, kinds], ignore_index=True)

    case_profile.fig_context(report, "smoke", max_context_bins=12)
    case_profile.fig_kinds(report, "smoke")

    figures_dir = tmp_path / "figures"
    for stem in ("case_smoke_context", "case_smoke_kinds"):
        assert (figures_dir / f"{stem}.png").exists()
        assert (figures_dir / f"{stem}.svg").exists()


def _fake_pairs_df():
    return pd.DataFrame({
        "pair_uid": ["p0", "p1", "p2", "p3", "p4"], "order": [0, 1, 0, 1, 0],
        "item": ["a", "a", "b", "b", "c"],
        "model_1": ["gpt-4o", "gpt-4o", "claude-3", "claude-3", "Llama-3-70b"],
        "model_2": ["claude-3"] * 5,
    })


def test_main_reconstructs_prompt_meta_via_kept_row_idx(tmp_path, monkeypatch):
    """End-to-end CLI: screen result carries split/limit/kept_row_idx (rows 1, 3 were dropped as
    over-length by screen.py) — case_profile must join only the surviving rows, in order."""
    pairs_df = _fake_pairs_df()
    monkeypatch.setattr(wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)

    n_kept, k = 3, 4
    screen_dict = {
        "n_prompts": n_kept, "per_prompt_activity": torch.rand(n_kept, k),
        "top_k_components": [("L0.q", 0), ("L0.q", 1), ("L1.k.h0", 0), ("L2.gate", 0)],
        "kept_row_idx": [0, 2, 4], "split": "eval_mm", "limit": None, "set_name": "judging",
    }
    screen_path = tmp_path / "screen.pt"
    torch.save(screen_dict, screen_path)

    cfg = {"paths": {"out": str(tmp_path)},
           "case_profile": {"n_factors": 2, "nmf_seed": 0, "nmf_max_iter": 200, "max_context_bins": 12}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv", ["case_profile.py", "--screen", str(screen_path), "--tag", "t"])
    case_profile.main()

    report = pd.read_csv(tmp_path / "case_profile_t.csv")
    # kept rows are positions 0, 2, 4 (items a, b, c) -- row 1 ("a" dup) and 3 ("b" dup) dropped
    assert set(report[report["variable"] == "item"]["group"]) == {"a", "b", "c"}

    assert (tmp_path / "case_profile_t_H.csv").exists()
    figures_dir = tmp_path / "figures"
    for stem in ("case_t_context", "case_t_kinds"):
        assert (figures_dir / f"{stem}.png").exists()


def test_main_warns_and_falls_back_when_kept_row_idx_missing(tmp_path, monkeypatch, capsys):
    """Legacy screen result saved before kept_row_idx/split/limit existed lacks them (but is
    still tagged set_name="judging" — that gate is a separate, always-mandatory concern): main()
    must still run (positional fallback, assuming no skips) and warn loudly rather than fail
    silently."""
    pairs_df = _fake_pairs_df().iloc[:3].reset_index(drop=True)  # pretend 3 rows, no skips
    monkeypatch.setattr(wbpairs, "load_pairs", lambda c, split=None, limit=None: pairs_df)

    n_kept, k = 3, 2
    screen_dict = {"n_prompts": n_kept, "per_prompt_activity": torch.rand(n_kept, k),
                   "top_k_components": [("L0.q", 0), ("L1.k.h0", 0)], "set_name": "judging"}
    screen_path = tmp_path / "screen.pt"
    torch.save(screen_dict, screen_path)

    cfg = {"paths": {"out": str(tmp_path)},
           "case_profile": {"n_factors": 2, "nmf_seed": 0, "nmf_max_iter": 200, "max_context_bins": 12}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv", ["case_profile.py", "--screen", str(screen_path), "--tag", "t"])
    case_profile.main()

    assert "WARNING" in capsys.readouterr().out
    assert (tmp_path / "case_profile_t.csv").exists()


def test_control_set_screen_raises():
    """screen.py's kept_row_idx is always populated (batch positions), even for --set control —
    but control fans one pairs-df row into several batch entries, so those positions do NOT index
    into pairs_df. Pointing case_profile at a control-tagged screen must hard-error, not produce
    a fully-formed but silently-wrong report."""
    screen_dict = {
        "n_prompts": 2, "per_prompt_activity": torch.rand(2, 2),
        "top_k_components": [("L0.q", 0), ("L0.q", 1)],
        "kept_row_idx": [0, 1], "split": "eval_mm", "limit": None, "set_name": "control",
    }
    with pytest.raises(AssertionError, match="judging"):
        case_profile._load_prompt_meta(screen_dict, {})
