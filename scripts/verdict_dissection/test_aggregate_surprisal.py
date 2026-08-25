import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_surprisal import build_pairs


def test_build_pairs_uses_raw_left_minus_right_features():
    raw = pd.DataFrame([
        {"bench": "gpqa", "model": "x", "item": "q", "epoch": 0, "log": "x",
         "sample_epoch": 1, "content_hash": "hx", "feature": 3.0},
        {"bench": "gpqa", "model": "y", "item": "q", "epoch": 0, "log": "y",
         "sample_epoch": 1, "content_hash": "hy", "feature": 1.0},
    ])
    verdicts = pd.DataFrame([{
        "bench": "gpqa", "item": "q", "model_left": "x", "model_right": "y",
        "epoch_left": 0, "epoch_right": 0, "log_left": "x", "log_right": "y",
        "pick_left": 1,
    }])

    pairs = build_pairs(verdicts, raw)
    assert len(pairs) == 1 and pairs.iloc[0].d_feature == 2.0
