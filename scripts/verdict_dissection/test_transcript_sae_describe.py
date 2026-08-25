import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from transcript_sae_describe import (balanced_accuracy, choose_examples, describe_features,
                                     parse_numbers, snippet)

SETTINGS = {"describe_examples": 3, "contrast_examples": 2, "fidelity_examples": 2,
            "example_chars": 40, "min_fidelity": 0.75, "seed": 0}


def group_rows(n):
    return pd.DataFrame({"transcript_key": [f"k{i}" for i in range(n)],
                         "item_group_key": [f"item{i // 2}" for i in range(n)],
                         "benchmark": ["gpqa"] * n,
                         "transcript": [f"transcript {i} " + "x" * 100 for i in range(n)]})


def test_examples_are_item_deduplicated_and_disjoint():
    rows = group_rows(20)
    activations = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1] + [0] * 10, dtype=float)
    chosen = choose_examples(activations, rows, n_describe=3, n_contrast=2, n_fidelity=2, seed=0)
    assert chosen["describe"].tolist() == [0, 2, 4]          # one per item
    assert chosen["held_active"].tolist() == [6, 8]
    assert set(chosen["contrast"]).isdisjoint(chosen["held_inactive"])
    assert all(activations[chosen["held_inactive"]] == 0) and len(chosen["held_inactive"]) == 2


def test_fidelity_is_balanced_accuracy_over_parsed_numbers():
    assert parse_numbers("The description applies to [1, 3].") == {1, 3}
    assert parse_numbers("none") == set()
    assert balanced_accuracy({1, 3}, np.array([True, False, True, False])) == 1.0
    assert balanced_accuracy({1, 2}, np.array([True, False, True, False])) == 0.5
    assert snippet("abcdefghij", 4) == "ab\n[...]\nij"


def test_describe_features_keeps_only_faithful_descriptions():
    rows = group_rows(40)
    activations = np.zeros((40, 2))
    activations[:10, 0] = np.arange(10, 0, -1)
    activations[:10, 1] = np.arange(10, 0, -1)
    replies = iter(["long answers", "[]", "short answers", "[1, 2, 3, 4]"])

    def ask(system, user):
        reply = next(replies)
        if reply.startswith("["):
            numbers = [number for number in range(1, 5)
                       if f"=== Transcript {number} ===\ntranscript 6" in user
                       or f"=== Transcript {number} ===\ntranscript 8" in user]
            return str(numbers) if reply == "[]" else "[]"
        return reply

    logged = []
    table = describe_features(activations, ["gpqa:0", "gpqa:5"], rows, {"gpqa": ["gpqa"]},
                              SETTINGS, ask, logged.append)
    assert table.feature.tolist() == ["gpqa:0", "gpqa:5"]
    assert table.fidelity.tolist() == [1.0, 0.5] and table.kept.tolist() == [True, False]
    assert table.description.tolist() == ["long answers", "short answers"]
    assert len(logged) == 2 and len(logged[0]["held_out_keys"]) == 4
