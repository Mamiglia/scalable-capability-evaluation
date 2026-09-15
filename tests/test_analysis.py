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
