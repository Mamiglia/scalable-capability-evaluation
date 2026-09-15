"""Baselines, judge conversions, scores and judge summaries on toy data (the toy task from test_protocols)."""

import numpy as np
import pandas as pd

from analysis.baselines import AccuracyOracle, Intercept
from test_protocols import toy_task


def test_intercept_and_oracle(monkeypatch):
    t = toy_task()  # references a, b, c with theta 0, 1, 2 on train items "0", "1"; targets t, u
    assert Intercept()(t).eq(1.0).all()
    # a HARD item "9" over two repeats: a 0/2, b 1/2, c 2/2, t 2/2, u 0/2 -> theta = 2 * accuracy on the toy
    rows = [dict(bench="b", item="9", model=m, repeat=r, correct=int(r < k))
            for m, k in {"a": 0, "b": 1, "c": 2, "t": 2, "u": 0}.items() for r in (0, 1)]
    rows += [dict(bench="b", item="0", model=m, repeat=0, correct=1) for m in "abctu"]  # a train item: ignored
    items = pd.DataFrame({"bench": "b", "item": ["0", "1", "9"], "s_set": ["easy", "easy", "hard"]})
    monkeypatch.setattr("analysis.baselines.tables.load_responses", lambda dom: pd.DataFrame(rows))
    monkeypatch.setattr("analysis.baselines.tables.load", lambda dom: (None, items))
    oracle = AccuracyOracle(None)
    assert oracle.name == "accuracy_hard"
    np.testing.assert_allclose(oracle(t).to_numpy(), [2.0, 0.0], atol=1e-9)


from analysis import judges


def toy_ratings():
    """One bench, two items; references a, b, c (theta 0, 1, 2) rated 2/4/6 on an easy item and 1/2/3
    on a hard one; targets t (rated like c) and u (rated like a)."""
    base = {"a": 0, "b": 1, "c": 2, "t": 2, "u": 0}
    rows = []
    for model, level in base.items():
        for item, scale, offset in [("0", 2, 2), ("1", 1, 1)]:
            rows.append(dict(bench="b", model=model, item=item, score=offset + scale * level, semester="s"))
    return pd.DataFrame(rows)


def test_direct_theta_raw_and_item_z():
    ratings, refs = toy_ratings(), pd.Series({"a": 0.0, "b": 1.0, "c": 2.0})
    raw = judges.direct_theta(ratings, refs, ["t", "u"], "raw")
    np.testing.assert_allclose(raw.to_numpy(), [2.0, 0.0], atol=1e-9)
    z = judges.direct_theta(ratings, refs, ["t", "u"], "item_z")
    np.testing.assert_allclose(z.to_numpy(), [2.0, 0.0], atol=1e-9)
    # a constant item contributes z = 0, not NaN
    flat = pd.concat([ratings, pd.DataFrame(dict(bench="b", model=list("abctu"), item="9", score=5.0, semester="s"))])
    assert judges.direct_theta(flat, refs, ["t"], "item_z").notna().all()
    # the leave-one-out error on the references is exact here: each reference is on the others' line
    loo = judges.direct_reference_loo(ratings, refs, "raw")
    np.testing.assert_allclose(loo.to_numpy(), refs.to_numpy(), atol=1e-9)


def test_pairwise_reference_loo_recovers_references():
    rng = np.random.default_rng(0)
    # 8 references as in the real task: a leave-one-out fold keeps 21 pairs, enough for the per-bench slope
    theta = {f"r{i}": float(t) for i, t in enumerate(np.linspace(-2.5, 2.5, 8), 1)}
    refs = pd.Series(theta)
    rows = []
    for a in theta:
        for b in theta:
            if a >= b:
                continue
            for item in range(80):
                p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
                rows.append(("g", "j", str(item), a, b, float(rng.random() < p), "s", False))
    verdicts = pd.DataFrame(rows, columns=["bench", "judge", "item", "model_left", "model_right", "pick_left", "semester", "imported"])
    for conversion in judges.PAIRWISE:
        loo = judges.pairwise_reference_loo(verdicts, refs, conversion)
        assert loo.index.equals(refs.index) and ((loo - refs).abs() < 0.8).all(), conversion
