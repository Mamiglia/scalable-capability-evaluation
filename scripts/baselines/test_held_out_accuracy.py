import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from held_out_accuracy import accuracy, intervals  # noqa: E402


def test_intervals_bracket_the_accuracy_and_collapse_for_a_constant_ranking():
    rng = np.random.default_rng(0)
    rows = [{"ranking": ranking, "bench": bench, "item": item, "correct": correct}
            for bench in ("aime", "gpqa") for item in range(30) for pair in range(4)
            for ranking, correct in (("noisy", rng.random() < 0.7), ("always right", True))]
    table = pd.DataFrame(rows)
    point = accuracy(table)
    bands = intervals(table, n_resamples=200)
    for bench in ("aime", "gpqa", "mean"):
        assert bands.loc[("always right", bench), "p5"] == bands.loc[("always right", bench), "p95"] == 1.0
        assert bands.loc[("noisy", bench), "p5"] < point.loc["noisy", bench] < bands.loc[("noisy", bench), "p95"]
