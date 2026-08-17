import numpy as np
import pandas as pd

import token_bias_probes as probes


def test_probe_table_recovers_capability_length_and_position_without_item_leakage():
    rng = np.random.default_rng(0)
    n_items, per_item = 60, 2
    n = n_items * per_item
    item = np.repeat(np.arange(n_items), per_item)
    capability = rng.normal(size=n)
    length = rng.normal(size=n)
    position = rng.integers(0, 2, size=n)
    code = np.column_stack([
        capability + .1 * rng.normal(size=n),
        length + .1 * rng.normal(size=n),
        2 * position - 1 + .1 * rng.normal(size=n),
        rng.normal(size=n),
    ])
    frame = pd.DataFrame({"item": item, "capability_gap": capability,
                          "token_length_gap": length, "picked_slot_1": position})

    out = probes.probe_table(code, frame, cv_folds=5)

    scores = out.set_index("target").score
    assert scores["capability_gap"] > .8
    assert scores["token_length_gap"] > .8
    assert scores["picked_slot_1"] > .9


def test_probe_table_reports_unavailable_single_class_error_slice():
    frame = pd.DataFrame({"item": np.repeat(np.arange(10), 2),
                          "capability_gap": np.arange(20),
                          "token_length_gap": np.arange(20),
                          "picked_slot_1": np.zeros(20),
                          "picked_longer_weaker": np.zeros(20)})

    out = probes.probe_table(np.ones((20, 2)), frame, cv_folds=2)

    row = out[out.target == "picked_longer_weaker"].iloc[0]
    assert row.metric == "unavailable" and np.isnan(row.score)
