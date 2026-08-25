import json

import numpy as np
import pandas as pd

from export_sae_feature_viewer import by_model_examples, choose_examples, feature_payload, pick_rates


def _frame():
    rows = []
    activations = [("q1", "a", 2.0), ("q1", "b", 0.0), ("q1", "c", -1.0), ("q2", "a", 1.0), ("q2", "b", 0.0)]
    activations += [("q3", f"m{i}", 0.9 - i * 0.05) for i in range(12)]  # 12 more positives for the quantile buckets
    for i, (item, model, act) in enumerate(activations):
        rows.append({"bench": "gpqa", "item": item, "model": model, "epoch": 0, "log": f"l{i}",
                     "split_role": "representation_train", "question": "Q", "transcript": f"T{i}",
                     "accuracy": 0.5, "pick_rate": 0.6, "pick_n": 4, "sae_shared_3": act})
    return pd.DataFrame(rows)


def test_choose_examples_buckets():
    chosen = choose_examples(_frame(), "sae_shared_3", n_top=2, n_quantile=2, n_negative=5, rng=np.random.default_rng(0))
    buckets = [b for _, b in chosen]
    assert buckets[:2] == ["top", "top"] and [r.model for r, _ in chosen[:2]] == ["a", "a"]
    assert buckets.count("50-90%") == 2 and buckets.count("10-50%") == 2 and buckets.count("0-10%") == 1
    assert [r["item"] for r, b in chosen if b == "contrast"] == ["q1", "q2"]
    assert [r.model for r, b in chosen if b == "negative"] == ["c"]
    assert all(r.sae_shared_3 == 0 for r, b in chosen if b == "contrast")


def test_feature_payload_description_and_empty_group():
    described = pd.Series({"description": "does x", "kept": True, "reason": float("nan"), "correlation": 0.3,
                           "p_value": 0.001, "candidates_json": json.dumps([{"description": "does x", "correlation": 0.3, "p_value": 0.001}])})
    payload = feature_payload(3, _frame(), {"shared:3@gpqa": described}, 2, 2, 5, 1, seed=0)
    gpqa = payload["groups"]["gpqa"]
    assert gpqa["description"]["kept"] and gpqa["description"]["reason"] is None
    assert gpqa["examples"][0]["bucket"] == "top" and gpqa["examples"][0]["log"] == "l0"
    assert payload["groups"]["aime"]["n"] == 0 and payload["groups"]["aime"]["description"] is None
    assert gpqa["by_model"][0]["n"] >= 1 and len(gpqa["pick_by_bin"]) == 7
    assert gpqa["pick_by_bin"][0]["bin"] == "negative" and gpqa["pick_by_bin"][0]["n"] == 1


def test_pick_rates_pools_both_sides():
    verdicts = pd.DataFrame({"bench": ["gpqa"] * 2, "item": ["q1"] * 2, "model_left": ["a", "b"],
                             "model_right": ["b", "a"], "epoch_left": [0, 0], "epoch_right": [0, 0],
                             "log_left": ["la", "lb"], "log_right": ["lb", "la"], "pick_left": [True, True]})
    rates = pick_rates(verdicts).set_index("model")
    assert rates.loc["a", "pick_rate"] == 0.5 and rates.loc["a", "pick_n"] == 2


def test_by_model_examples():
    chosen = by_model_examples(_frame(), "sae_shared_3", n_zero=1, rng=np.random.default_rng(0))
    by_bucket = {}
    for r, b in chosen:
        by_bucket.setdefault(b, []).append(r.model)
    assert by_bucket["model top"][:2] == ["a", "b"] or "a" in by_bucket["model top"]
    assert by_bucket["model negative"] == ["c"]
    assert set(by_bucket["model zero"]) == {"b"}
    tops = [r for r, b in chosen if b == "model top" and r.model == "a"]
    assert len(tops) == 1 and tops[0]["item"] == "q1"
