import json

import pandas as pd

from export_sae_feature_viewer import feature_payload, pick_rates


def _frame():
    rows = []
    for i, (item, model, act) in enumerate([("q1", "a", 2.0), ("q1", "b", 0.0), ("q1", "c", -1.0),
                                            ("q2", "a", 1.0), ("q2", "b", 0.0)]):
        rows.append({"bench": "gpqa", "item": item, "model": model, "epoch": 0, "log": f"l{i}",
                     "split_role": "representation_train", "question": "Q", "transcript": f"T{i}",
                     "accuracy": 0.5, "pick_rate": 0.6, "pick_n": 4, "sae_shared_3": act})
    return pd.DataFrame(rows)


def test_feature_payload_examples_and_contrasts():
    described = pd.Series({"description": "does x", "kept": True, "reason": float("nan"), "correlation": 0.3,
                           "p_value": 0.001, "candidates_json": json.dumps([{"description": "does x", "correlation": 0.3, "p_value": 0.001}])})
    payload = feature_payload(3, _frame(), {"shared:3@gpqa": described}, n_examples=10, seed=0)
    gpqa = payload["groups"]["gpqa"]
    assert [e["model"] for e in gpqa["firing"]] == ["a", "a"]
    assert [e["model"] for e in gpqa["negative"]] == ["c"]
    assert [e["item"] for e in gpqa["contrast"]] == ["q1", "q2"]
    assert all(e["activation"] == 0.0 for e in gpqa["contrast"])
    assert gpqa["description"]["kept"] and gpqa["description"]["reason"] is None
    assert payload["groups"]["aime"]["n"] == 0 and payload["groups"]["aime"]["description"] is None


def test_pick_rates_pools_both_sides():
    verdicts = pd.DataFrame({"bench": ["gpqa"] * 2, "item": ["q1"] * 2, "model_left": ["a", "b"],
                             "model_right": ["b", "a"], "epoch_left": [0, 0], "epoch_right": [0, 0],
                             "log_left": ["la", "lb"], "log_right": ["lb", "la"], "pick_left": [True, True]})
    rates = pick_rates(verdicts).set_index("model")
    assert rates.loc["a", "pick_rate"] == 0.5 and rates.loc["a", "pick_n"] == 2
