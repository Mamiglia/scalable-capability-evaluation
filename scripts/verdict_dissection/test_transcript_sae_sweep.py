import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from transcript_sae_sweep import PAIR_KEYS, grouped_accuracy, model_fold_accuracy, score_block, stronger_pairs


def test_grouped_accuracy_learns_a_separable_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    n = 600
    signal = rng.normal(size=n)
    features = np.c_[signal, rng.normal(size=(n, 3))]
    target = signal > 0
    groups = np.arange(n) // 6                     # 100 items, 6 pairs each
    assert grouped_accuracy(features, target, groups) > 0.9
    assert abs(grouped_accuracy(features[:, 1:], target, groups) - 0.5) < 0.1


def test_score_block_reports_each_group_and_a_pooled_accuracy():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    pairs = pd.DataFrame({"bench": ["aime2024"] * 200 + ["gpqa"] * 200, "item": np.arange(n) // 4,
                          "pick_left": x > 0, "d_sae_0": x, "d_sae_1": rng.normal(size=n)})
    out = score_block(pairs, ["d_sae_0", "d_sae_1"], {"aime2024": "aime", "gpqa": "gpqa"})
    assert set(out) == {"aime", "gpqa", "pooled"} and out["pooled"] > 0.9


def test_stronger_pairs_deduplicates_judges_and_drops_ties():
    keys = pd.DataFrame({"bench": ["gpqa"] * 3, "item": ["q"] * 3, "model_left": ["a", "a", "b"],
                         "model_right": ["b", "b", "c"], "epoch_left": [0, 0, 0], "epoch_right": [0, 0, 0],
                         "log_left": ["la", "la", "lb"], "log_right": ["lb", "lb", "lc"]})
    pairs = keys.assign(judge=["j1", "j2", "j1"], pick_left=[True, False, True], d_sae_0=[1.0, 1.0, -1.0])
    labels = keys.drop_duplicates(PAIR_KEYS).assign(left_stronger=[1.0, np.nan])
    out = stronger_pairs(pairs, labels)
    assert len(out) == 1 and out.left_stronger.tolist() == [True] and out.d_sae_0.tolist() == [1.0]


def test_model_folds_cannot_use_model_fingerprints():
    rng = np.random.default_rng(0)
    n, models = 3000, np.array([f"m{i}" for i in range(20)])
    left, right = rng.choice(models, n), rng.choice(models, n)
    keep = left != right
    left, right = left[keep], right[keep]
    strength = {m: i for i, m in enumerate(models)}
    target = np.array([strength[a] > strength[b] for a, b in zip(left, right)])
    fingerprint = np.array([np.eye(20)[strength[a]] - np.eye(20)[strength[b]] for a, b in zip(left, right)])
    # a one-hot model fingerprint predicts perfectly with item folds, but not with model folds
    assert grouped_accuracy(fingerprint, target, np.arange(len(target))) > 0.95
    assert model_fold_accuracy(fingerprint, target, left, right) < 0.7
