"""Local checks for the cross-domain prep + analysis logic (no cluster, no API)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_cross_domain import pooled_eci, _item_held


def test_pooled_eci_zscore_average():
    logits = pd.DataFrame(
        {"gpqa_logit": [2.0, 0.0, -2.0],
         "aime_logit": [1.0, 0.0, -1.0],
         "chess_logit": [0.0, 0.0, 0.0]},   # zero-variance domain -> 0, must not NaN
        index=["a", "b", "c"])
    eci = pooled_eci(logits)
    assert eci.loc["a"] > eci.loc["b"] > eci.loc["c"]
    assert abs(eci.loc["b"]) < 1e-9
    assert eci.notna().all()


def test_item_held_deterministic_and_fraction():
    ids = [f"item-{i}" for i in range(4000)]
    frac = 0.2
    held = [_item_held(i, frac) for i in ids]
    assert held == [_item_held(i, frac) for i in ids]        # deterministic
    rate = sum(held) / len(held)
    assert abs(rate - frac) < 0.03                           # ~20% held
    # chess-style integer ids and gpqa-style rec ids both work, don't collide
    assert isinstance(_item_held(42, frac), bool)


def test_cell_accuracy_counterbalanced():
    from cross_domain_analysis import cell_accuracy
    # judge always right (picks slot with higher logit == label side)
    d = pd.DataFrame({"pair_uid": ["p", "p"], "order": [0, 1], "label": [1, 2],
                      "logit_1": [2.0, -2.0], "logit_2": [-2.0, 2.0]})
    assert cell_accuracy(d) == 1.0
    # judge always picks slot 1 -> right on order0 (label1), wrong on order1 (label2) -> 0.5
    d2 = pd.DataFrame({"pair_uid": ["p", "p"], "order": [0, 1], "label": [1, 2],
                       "logit_1": [2.0, 2.0], "logit_2": [-2.0, -2.0]})
    assert cell_accuracy(d2) == 0.5
