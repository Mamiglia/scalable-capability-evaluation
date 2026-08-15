import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from features_only_baseline import LOW_GAP, clustered_accuracy, out_of_fold, score, slices


def test_out_of_fold_keeps_pairs_out_of_their_own_training_folds():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    # Pair identity is the only signal, and each pair's label is a coin flip, so a
    # split that let a pair into its own training folds would score near 1.
    label = rng.integers(0, 2, size=10)
    target = label[groups]
    features = np.eye(10)[groups]
    prediction, chosen = out_of_fold(
        features, target, groups, LogisticRegression(), {"C": [0.1, 1.0]}, 5, 2
    )
    assert (prediction == target).mean() < 0.7
    assert len(chosen) == 5 and all("C" in fold for fold in chosen)


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
