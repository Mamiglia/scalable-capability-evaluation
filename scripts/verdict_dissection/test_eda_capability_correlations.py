import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eda_capability_correlations import capability_correlations, summarise


def test_correlations_are_within_item_and_average_model_repeats():
    rows = []
    for item in ("q1", "q2"):
        for model, accuracy, value in (("weak", 0.2, 1.0),
                                       ("mid", 0.5, 2.0),
                                       ("strong", 0.8, 3.0)):
            rows.extend([
                {"bench": "bench", "item": item, "model": model,
                 "accuracy": accuracy, "positive": value - 0.5,
                 "negative": -value, "constant": 1.0},
                {"bench": "bench", "item": item, "model": model,
                 "accuracy": accuracy, "positive": value + 0.5,
                 "negative": -value, "constant": 1.0},
            ])
    dictionary = pd.DataFrame({
        "feature": ["positive", "negative", "constant"],
        "group": ["surface-textual"] * 3,
    })

    by_item = capability_correlations(pd.DataFrame(rows), dictionary,
                                      min_models=3)
    assert set(by_item.n_models) == {3}
    assert set(by_item.loc[by_item.feature == "positive", "spearman_rho"]) == {1.0}
    assert set(by_item.loc[by_item.feature == "negative", "spearman_rho"]) == {-1.0}
    assert by_item.loc[by_item.feature == "constant", "spearman_rho"].isna().all()

    overall, by_benchmark = summarise(by_item)
    assert overall.loc[overall.feature == "positive", "median_item_rho"].item() == 1.0
    assert overall.loc[overall.feature == "negative", "fraction_positive"].item() == 0.0
    assert by_benchmark.loc[by_benchmark.feature == "positive", "n_items"].item() == 2


def test_too_few_models_returns_missing_correlation():
    transcripts = pd.DataFrame([
        {"bench": "b", "item": "q", "model": "m1", "accuracy": 0.1,
         "feature": 1.0},
        {"bench": "b", "item": "q", "model": "m2", "accuracy": 0.9,
         "feature": 2.0},
    ])
    dictionary = pd.DataFrame({"feature": ["feature"], "group": ["g"]})
    got = capability_correlations(transcripts, dictionary, min_models=3)
    assert got.n_models.item() == 2
    assert np.isnan(got.spearman_rho.item())
