import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from transcript_sae_sweep import grouped_accuracy, score_block


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
