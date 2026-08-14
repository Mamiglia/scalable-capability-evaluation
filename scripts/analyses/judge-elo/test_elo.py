# scripts/analyses/judge-elo/test_elo.py
from unittest.mock import patch

import numpy as np
from scipy.stats import kendalltau

from elo import (collect_pair_wins, components, fit_bt, oversight_values,
                 paired_diff, pick_accuracy)


def test_fit_bt_recovers_ranking():
    true = {f"m{i}": t for i, t in enumerate(np.linspace(-2, 2, 8))}
    models = list(true)
    pair_wins = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            p = 1 / (1 + np.exp(-(true[a] - true[b])))
            pair_wins[(a, b)] = [40 * p, 40]  # expected wins at the true strengths
    est = fit_bt(pair_wins)
    tau = kendalltau([est[m] for m in models], [true[m] for m in models]).statistic
    assert tau > 0.999  # tau-b uses sqrt, exact 1.0 not guaranteed
    assert abs(sum(est.values())) < 1e-6


def test_fit_bt_finite_on_sweep():
    # one model wins every pick; l2 must keep its strength finite
    est = fit_bt({("a", "b"): [10, 10], ("b", "c"): [7, 10]})
    assert np.isfinite(est["a"]) and est["a"] > est["b"] > est["c"]


def test_fit_bt_extreme_gaps_converge():
    # chain of pure sweeps; softplus loss must stay finite and ordered
    est = fit_bt({("a", "b"): [50, 50], ("b", "c"): [50, 50], ("c", "d"): [50, 50]})
    assert all(np.isfinite(v) for v in est.values())
    assert est["a"] > est["b"] > est["c"] > est["d"]


def test_components_largest_first():
    comps = components([("a", "b"), ("b", "c"), ("d", "e")])
    assert [len(c) for c in comps] == [3, 2]
    assert comps[0] == {"a", "b", "c"}


def test_pick_accuracy_orientation_and_subsets():
    cfg = {"eci": {"strong": 130.0, "mid": 128.0, "weak": 110.0},
           "judge_eci": {"j": 120.0}}
    # judge always picks the ECI-stronger model: picked_major=1 when major is
    # stronger, =0 when major is weaker. Accuracy must be 1.0 everywhere.
    rows = [
        {"bench": "b", "judge": "j", "major": "strong", "minor": "weak",
         "item_id": "i1", "picked_major": "1"},
        {"bench": "b", "judge": "j", "major": "weak", "minor": "strong",
         "item_id": "i2", "picked_major": "0"},
        {"bench": "b", "judge": "j", "major": "strong", "minor": "mid",
         "item_id": "i3", "picked_major": "1"},
    ]
    with patch("elo.load_config", return_value=cfg):
        acc = pick_accuracy(rows, close_eci=5.0, min_n=1, min_models=1)
    assert acc[("b", "j", "all")][0] == 1.0
    # strong-vs-mid gap is 2 <= 5 -> the only close pair; strong,mid both > judge 120
    assert acc[("b", "j", "close")] == (1.0, 1)
    # strong(130) & mid(128) both above judge 120; weak(110) is below -> its
    # pairs excluded from "stronger". One qualifying pair.
    assert acc[("b", "j", "stronger")] == (1.0, 1)


def test_collect_pair_wins_counterbalances_orders():
    # two presentation orders of item i1 disagree -> averaged to a 0.5 vote;
    # i2 has one order -> its value. Each item is ONE observation.
    rows = [
        {"bench": "b", "condition": "c", "judge": "j", "major": "a", "minor": "x",
         "item_id": "i1", "picked_major": "True"},
        {"bench": "b", "condition": "c", "judge": "j", "major": "a", "minor": "x",
         "item_id": "i1", "picked_major": "False"},
        {"bench": "b", "condition": "c", "judge": "j", "major": "a", "minor": "x",
         "item_id": "i2", "picked_major": "True"},
    ]
    g = collect_pair_wins(rows, ["bench", "condition", "judge"])
    assert g[("b", "c", "j")][("a", "x")] == [1.5, 2]  # 0.5 + 1.0 over 2 items


def test_oversight_values_stronger_subset():
    cfg = {"eci": {"s1": 140.0, "s2": 138.0, "s3": 136.0, "w": 100.0},
           "accuracy": {"s1": 0.9, "s2": 0.8, "s3": 0.7, "w": 0.1},
           "judge_eci": {"j": 120.0}}
    elo_rows, mean_rows = [], []
    for m, e, ms in [("s1", 3.0, 90.0), ("s2", 2.0, 80.0), ("s3", 1.0, 70.0), ("w", 0.0, 60.0)]:
        elo_rows.append({"bench": "b", "source": "pairwise", "condition": "a_baseline",
                         "judge": "j", "model": m, "elo": e, "component": "0"})
        mean_rows.append({"bench": "b", "judge": "j", "model": m, "mean_score": ms})
    with patch("elo.load_config", return_value=cfg):
        out = oversight_values(elo_rows, mean_rows, "a_baseline", floor=3)
    # weak model (eci 100 < judge 120) excluded -> 3 stronger models, perfect rank
    assert out[("b", "eci", "pairwise", "j")] == (1.0, 3)
    assert out[("b", "accuracy", "direct-score", "j")] == (1.0, 3)


def test_paired_diff_matches_hand_calc():
    cfg = {"eci": {"a1": 140.0, "b1": 135.0, "a2": 141.0, "b2": 136.0},
           "judge_eci": {"j": 120.0}}
    # both pairs close (<10) and both models above judge 120. pair1: both sources
    # correct -> diff 0. pair2: pairwise correct, pointwise wrong -> diff 1.
    pairwise = [
        {"bench": "x", "judge": "j", "major": "a1", "minor": "b1", "item_id": "i", "picked_major": "True"},
        {"bench": "x", "judge": "j", "major": "a2", "minor": "b2", "item_id": "i", "picked_major": "True"},
    ]
    direct = [
        {"bench": "x", "judge": "j", "major": "a1", "minor": "b1", "item_id": "i", "picked_major": "1"},
        {"bench": "x", "judge": "j", "major": "a2", "minor": "b2", "item_id": "i", "picked_major": "0"},
    ]
    with patch("elo.load_config", return_value=cfg):
        out = paired_diff(pairwise, direct, close_eci=10.0, min_models=1)
    both = next(r for r in out if r["slice"] == "close, both>judge")
    assert both["diff"] == 0.5 and both["n_pairs"] == 2
    assert both["pairwise_acc"] == 1.0 and both["pointwise_acc"] == 0.5


def test_pick_accuracy_counts_ties_half():
    cfg = {"eci": {"a": 130.0, "b": 128.0}, "judge_eci": {}}
    rows = [{"bench": "x", "judge": "j", "major": "a", "minor": "b",
             "item_id": "i1", "picked_major": "0.5"}]
    with patch("elo.load_config", return_value=cfg):
        acc = pick_accuracy(rows, close_eci=5.0, min_n=1, min_models=1)
    assert acc[("x", "j", "all")][0] == 0.5
