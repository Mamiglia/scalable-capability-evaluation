"""main.py checks: feature calculation on a toy store, z-scoring."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from main import KEYS, calculate_raw_features, normalise

STORE = pd.DataFrame([
    {"bench": "gpqa", "model": "x", "item": "q1", "epoch": 0, "question": "Why?",
     "transcript": "Because $n = 2$. Therefore it holds, I think.", "correct": True},
    {"bench": "gpqa", "model": "y", "item": "q1", "epoch": 0, "question": "Why?",
     "transcript": "It just does. No doubt about it.", "correct": True},
])


def test_raw_features_keep_keys_and_have_no_nans(tmp_path):
    raw = calculate_raw_features(STORE, surprisal_cache=tmp_path)   # empty cache -> basic features only
    assert list(raw.columns[:4]) == KEYS
    assert raw.notna().all().all()
    assert raw.loc[0, "personal_pronouns_count"] > raw.loc[1, "personal_pronouns_count"]


def test_normalise_gives_zero_mean_unit_sd(tmp_path):
    raw = calculate_raw_features(STORE, surprisal_cache=tmp_path)
    dm, scaling = normalise(raw)
    feats = [c for c in dm.columns if c not in KEYS]
    assert np.allclose(dm[feats].mean(), 0, atol=1e-12)
    assert set(scaling.index) == set(feats)
    constant = [c for c in feats if raw[c].nunique() == 1]
    assert all((dm[c] == 0).all() for c in constant)     # sd 0 columns stay 0, not NaN
