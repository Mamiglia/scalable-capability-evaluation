from pathlib import Path
from runpy import run_path

import pandas as pd


fit_rolling = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/capability_baseline.py")["fit_rolling"]


def test_rolling_baseline_never_uses_current_or_future_accuracy():
    candidates = ["a", "b", "c", "d"]
    eci = pd.Series({"a": 1, "b": 2, "c": 3, "d": 4})
    actual = pd.Series({"a": 0.1, "b": 0.2, "c": 0.9, "d": 0.8})
    first = fit_rolling(actual, eci, candidates).set_index("model")
    changed = actual.copy()
    changed[["c", "d"]] = [0.0, 0.0]
    second = fit_rolling(changed, eci, candidates).set_index("model")
    assert first.loc["c", "predicted"] == second.loc["c", "predicted"]
