import numpy as np
import pytest
import pandas as pd

from scorer_agreement import family, pairwise, target


def _frame():
    rng = np.random.default_rng(0)
    rows = 40
    base = rng.normal(size=rows)
    frame = pd.DataFrame({"bench": ["gpqa"] * 20 + ["aime"] * 20, "model": [f"m{i % 5}" for i in range(rows)],
                          "split": "representation_train"})
    frame["surprisal_x__llama"] = base
    frame["surprisal_x__gemma"] = base
    frame["surprisal_x__qwen"] = -base
    return frame


def test_pairwise_identical_and_reversed_scorers():
    table = pairwise(_frame(), ["surprisal_x"])
    pooled = table[table.bench == "pooled"].set_index("pair").spearman
    assert pooled["llama-gemma"] == pytest.approx(1) and pooled["llama-qwen"] == pytest.approx(-1)


def test_target_uses_model_means():
    frame = _frame()
    by_model = frame.groupby(["bench", "model"]).surprisal_x__llama.mean()
    accuracy = by_model.rank().rename("accuracy")            # accuracy ordered like the llama mean
    table = target(frame, ["surprisal_x"], accuracy).set_index(["bench", "scorer"]).spearman_with_accuracy
    assert table[("gpqa", "llama")] == pytest.approx(1) and table[("gpqa", "qwen")] == pytest.approx(-1)
    assert table[("gpqa", "mean")] == pytest.approx(1)


def test_family_names():
    assert family("surprisal_rolling_tail_context_gain_mean") == "rolling tail gain"
    assert family("surprisal_question_gain_bpc") == "gain"
    assert family("surprisal_rolling_long_position_adjusted_mean") == "rolling level"
    assert family("surprisal_full_count") == "count"
