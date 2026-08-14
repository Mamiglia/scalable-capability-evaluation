"""Focused compatibility check for stored condition-effect layouts."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HIBAYES = Path(__file__).resolve().parents[1] / "scripts" / "hibayes"
sys.path.insert(0, str(HIBAYES))

import communicators


class Variable:
    def __init__(self, values):
        self.values = np.asarray(values)


def model(condition_values):
    posterior = {
        "judge_predictor_effects": Variable(np.zeros((1, 4, 2, 1))),
        "judge_intercepts": Variable(np.zeros((1, 4, 2))),
        "judge_condition_effects": Variable(condition_values),
    }
    return SimpleNamespace(
        inference_data=SimpleNamespace(posterior=posterior),
        coords={
            "judge": ["j1", "j2"],
            "transcript_condition": ["raw", "truncated_10", "truncated_50"],
        },
        features={
            "judge_index": np.array([0, 0, 1, 1]),
            "x": np.array([0.0, 1.0, 0.0, 1.0]),
            "transcript_condition_index": np.array([0, 1, 0, 2]),
        },
    )


free = np.arange(16, dtype=float).reshape(1, 4, 2, 2)
full = np.concatenate([np.zeros((1, 4, 2, 1)), free], axis=-1)
free_table = communicators._effect_table(model(free), ["x"], "baseline")
full_table = communicators._effect_table(model(full), ["x"], "baseline")
columns = ["judge", "feature", "mean_log_odds", "identified"]
assert free_table[columns].equals(full_table[columns])
print("communicator posterior-layout compatibility: OK")
