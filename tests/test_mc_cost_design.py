import numpy as np
import pandas as pd
import pytest

from mc_cost.design import AutomaticConfig, automatic_comparisons, manual_comparisons


def catalogue():
    return pd.DataFrame([
        {"model_id": "j1", "theta": 0.0, "release": "2024-01-01", "judge_pool": True, "candidate_pool": False,
         "reasoning_class": "non_reasoning"},
        {"model_id": "j2", "theta": 0.5, "release": "2024-06-01", "judge_pool": True, "candidate_pool": False,
         "reasoning_class": "reasoning"},
        {"model_id": "a", "theta": 2.0, "release": "2026-01-01", "judge_pool": False, "candidate_pool": True,
         "reasoning_class": "reasoning"},
        {"model_id": "old-strong", "theta": 0.3, "release": "2025-01-01", "judge_pool": False, "candidate_pool": True,
         "reasoning_class": "non_reasoning"},
        {"model_id": "old-weak", "theta": -1.0, "release": "2023-01-01", "judge_pool": False, "candidate_pool": True,
         "reasoning_class": "non_reasoning"},
    ])


def test_automatic_uses_shared_anchor_and_only_older_comparators():
    config = AutomaticConfig(j=2, m=1, n=2, attempted_items=10, presentations=2,
                             selected_judges=("j1", "j2"), require_stronger=True)
    got = automatic_comparisons(catalogue(), config, np.random.default_rng(4))
    assert set(got["judge"]) == {"j1", "j2"}
    assert got["m1"].nunique() == 1
    assert set(got["m2"]) == {"old-strong", "old-weak"}


def test_automatic_reuses_each_anchor_comparators_for_every_judge():
    config = AutomaticConfig(j=2, m=1, n=2, attempted_items=10, selected_judges=("j1", "j2"))
    got = automatic_comparisons(catalogue(), config, np.random.default_rng(2))
    pairs = got.groupby("judge")[["m1", "m2"]].apply(lambda rows: set(map(tuple, rows.to_numpy())))
    assert pairs.iloc[0] == pairs.iloc[1]


def test_automatic_reports_eligible_count_instead_of_lowering_n():
    config = AutomaticConfig(j=1, m=1, n=9, attempted_items=10, presentations=2,
                             selected_judges=("j1",), require_stronger=True)
    with pytest.raises(ValueError, match="requested N=9.*eligible"):
        automatic_comparisons(catalogue(), config, np.random.default_rng(1))


def test_automatic_uses_only_release_date_for_comparators():
    models = pd.concat([catalogue(), pd.DataFrame([{
        "model_id": "unpooled-old", "theta": np.nan, "release": "2020-01-01",
        "judge_pool": False, "candidate_pool": False, "reasoning_class": pd.NA,
    }])], ignore_index=True)
    config = AutomaticConfig(j=1, m=1, n=3, attempted_items=10, selected_judges=("j1",))
    got = automatic_comparisons(models, config, np.random.default_rng(3))
    assert "unpooled-old" in set(got["m2"])


def test_manual_retains_duplicates_and_warns_without_blocking():
    rows = pd.DataFrame([{"judge": "j2", "m1": "old-weak", "m2": "a"}] * 2)
    comparisons, warnings = manual_comparisons(catalogue(), rows)
    assert len(comparisons) == 2
    assert "automatic_rule_violation" in set(warnings["code"])
    assert set(warnings["row"]) == {0, 1}


def test_manual_keeps_known_identity_with_missing_prediction_metadata():
    models = pd.concat([catalogue(), pd.DataFrame([{
        "model_id": "undated", "theta": np.nan, "release": None,
        "judge_pool": False, "candidate_pool": False, "reasoning_class": pd.NA,
    }])], ignore_index=True)
    rows = pd.DataFrame([{"judge": "j1", "m1": "a", "m2": "undated"}])
    comparisons, warnings = manual_comparisons(models, rows)
    assert comparisons.loc[0, "predictable"] == False
    assert "missing_prediction_metadata" in set(warnings["code"])


def test_manual_uses_reasoning_override_and_rejects_bad_schema_or_identity():
    models = catalogue().drop(columns="reasoning_class")
    rows = pd.DataFrame([{
        "judge": "j1", "m1": "a", "m2": "old-weak",
        "judge_reasoning_class": "reasoning", "m1_reasoning_class": "non_reasoning",
        "m2_reasoning_class": "reasoning",
    }])
    comparisons, _ = manual_comparisons(models, rows)
    assert comparisons.loc[0, "m1_reasoning_class"] == "non_reasoning"
    with pytest.raises(ValueError, match="columns"):
        manual_comparisons(catalogue(), rows.assign(extra="no"))
    with pytest.raises(ValueError, match="unknown model"):
        manual_comparisons(catalogue(), pd.DataFrame([{"judge": "missing", "m1": "a", "m2": "old-weak"}]))


def test_manual_rejects_invalid_reasoning_class():
    rows = pd.DataFrame([{
        "judge": "j1", "m1": "a", "m2": "old-weak", "m1_reasoning_class": "maybe",
    }])
    with pytest.raises(ValueError, match="reasoning_class"):
        manual_comparisons(catalogue(), rows)


def test_manual_allows_nullable_pool_flags():
    models = catalogue()
    models[["judge_pool", "candidate_pool"]] = pd.NA
    comparisons, warnings = manual_comparisons(
        models, pd.DataFrame([{"judge": "j1", "m1": "a", "m2": "old-weak"}])
    )
    assert comparisons.loc[0, "predictable"] == True
    assert "out_of_support" not in set(warnings["code"])
