import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from features_only_baseline import (
    LOW_GAP,
    clustered_accuracy,
    cross_validated,
    leave_one_model_out,
    pair_grouped,
    score,
    slices,
)

CONFIG = {"group_column": "pair", "model_columns": ["left_model", "right_model"]}


def toy_comparisons():
    """Ten pairs over five models."""
    return pd.DataFrame(
        {
            "left_model": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
            "right_model": ["b", "c", "c", "d", "d", "e", "e", "a", "a", "b"],
            "pair": [f"p{i}" for i in range(10)],
        }
    )


def test_pair_grouped_keeps_pairs_out_of_their_own_training_folds():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    # Pair identity is the only signal, and each pair's label is a coin flip, so a
    # split that let a pair into its own training folds would score near 1.
    target = rng.integers(0, 2, size=10)[groups]
    features = np.eye(10)[groups]
    comparisons = pd.DataFrame({"pair": groups})
    index, correct, chosen = cross_validated(
        features,
        target,
        pair_grouped(comparisons, CONFIG, {"outer_folds": 5}),
        groups,
        LogisticRegression(),
        {"C": [0.1, 1.0]},
        2,
    )
    assert len(index) == len(target)
    assert correct.mean() < 0.7
    assert len(chosen) == 5 and all("C" in fold for fold in chosen)


def test_leave_one_model_out_holds_out_a_model_pair_grouping_leaves_in():
    comparisons = toy_comparisons()
    folds = list(leave_one_model_out(comparisons, CONFIG, {}))
    assert len(folds) == comparisons[CONFIG["model_columns"]].stack().nunique()
    for train, test in folds:
        held = set(comparisons.loc[test, CONFIG["model_columns"]].stack()) - set(
            comparisons.loc[train, CONFIG["model_columns"]].stack()
        )
        assert len(held) == 1
    # Every comparison is tested, twice over, once against each of its two models.
    assert sorted(np.concatenate([test for _, test in folds])).count(0) == 2
    # Pair grouping does not do this: a held-out pair's models are still in training.
    train, test = next(pair_grouped(comparisons, CONFIG, {"outer_folds": 5}))
    assert not set(comparisons.loc[test, CONFIG["model_columns"]].stack()) - set(
        comparisons.loc[train, CONFIG["model_columns"]].stack()
    )


def test_clustered_accuracy_band_widens_when_the_errors_sit_in_whole_pairs():
    correct = np.tile([1.0] * 8 + [0.0] * 2, 10)
    scattered = clustered_accuracy(correct, np.repeat(np.arange(10), 10), 500, 0)
    clumped = clustered_accuracy(
        np.repeat([1.0] * 8 + [0.0] * 2, 10), np.repeat(np.arange(10), 10), 500, 0
    )
    assert scattered[0] == pytest.approx(clumped[0])
    assert clumped[2] - clumped[1] > scattered[2] - scattered[1]


def test_low_gap_slice_is_strictly_below_the_threshold():
    frame = pd.DataFrame({"capability_gap": [9.99, 10.0, 10.01], "pair": list("abc")})
    assert slices(frame["capability_gap"], 10.0)[LOW_GAP].tolist() == [True, False, False]
    rows = score(frame, pd.Series([1, 1, 0]), "pair", 10.0, 50, 0)
    assert {row["slice"]: row["n_rows"] for row in rows}[LOW_GAP] == 1
