import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mc_cost.simulation import (
    LatentCache,
    SimulationConfig,
    TRIAL_METRICS,
    cost_rows,
    counterbalance,
    simulate,
    summarize_trials,
)


def catalogue():
    return pd.DataFrame([
        {"model_id": "j", "theta": 0.0, "release": "2024-01-01", "family": "x",
         "reasoning_class": "non_reasoning", "price_in": 1.0, "price_out": 2.0},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01", "family": "x",
         "reasoning_class": "non_reasoning", "price_in": 3.0, "price_out": 4.0},
        {"model_id": "b", "theta": 0.0, "release": "2023-01-01", "family": "y",
         "reasoning_class": "non_reasoning", "price_in": 5.0, "price_out": 6.0},
    ])


def irt():
    return {
        "models": np.array(["a", "b"]),
        "items": np.array(["gpqa:i1", "gpqa:i2"]),
        "theta": np.array([[1.0, 0.0]]),
        "a": np.array([[100.0, 100.0]]),
        "b": np.array([[-100.0, 100.0]]),
        "floor": np.array([0.0, 0.0]),
    }


def judgment_artifact(*, observed_judge=True, judge_scale=0.0):
    judges = {"j": 0} if observed_judge else {}
    posterior = {
        "judge_effects": np.array([[[0.0, 1.0]]]) if observed_judge else np.empty((1, 0, 2)),
        "judge_population": np.array([[[0.0], [1.0]]]),
        "judge_scales": np.array([[judge_scale, 0.0]]),
        "judge_corr": np.array([np.eye(2)]),
        "candidate_effects": np.array([[0.0]]),
        "candidate_scale": np.array([0.0]),
        "item_effects": np.array([[0.0, 0.0]]),
        "item_scale": np.array([0.0]),
    }
    return {
        "posterior": posterior,
        "feature_order": [],
        "category_maps": {
            "judges": judges, "models": {"a": 0}, "items": {"i1": 0, "i2": 1},
        },
        "training_scales": {
            "theta_gap": {"mean": 0.0, "scale": 1.0},
            "judge_theta": {"mean": 0.0, "scale": 1.0},
            "judge_release_days": {"mean": 0.0, "scale": 1.0},
        },
    }


def candidate_artifact():
    return {
        "role": "candidate", "models": ["a"],
        "reasoning_classes": ["non_reasoning"],
        "model_reasoning_classes": ["non_reasoning"],
        "input_features": [], "output_features": [], "length_features": [],
        "posterior": {
            "input_mean": np.array([[np.log1p(10)]]),
            "output_mean": np.array([[np.log1p(5)]]),
            "class_input_mean": np.array([[np.log1p(20)]]),
            "class_output_mean": np.array([[np.log1p(8)]]),
            "input_model_scale": np.array([0.0]), "output_model_scale": np.array([0.0]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
            "length_class_mean": np.array([[np.log1p(22)]]),
            "length_model_scale": np.array([0.0]), "length_beta": np.empty((1, 0)),
            "length_sigma": np.array([0.0]),
        },
    }


def candidate_artifact_with_override_population():
    artifact = candidate_artifact()
    artifact["reasoning_classes"] = ["non_reasoning", "reasoning"]
    artifact["posterior"].update({
        "class_input_mean": np.array([[np.log1p(20), np.log1p(30)]]),
        "class_output_mean": np.array([[np.log1p(8), np.log1p(12)]]),
        "class_reasoning_logit": np.array([[0.0, 0.0]]),
        "reasoning_model_scale": np.array([0.0]), "reasoning_sigma": np.array([0.0]),
        "length_class_mean": np.array([[np.log1p(22), np.log1p(30)]]),
    })
    return artifact


def judge_token_artifact():
    return {
        "role": "judge", "models": ["j"], "reasoning_classes": ["non_reasoning"],
        "model_reasoning_classes": ["non_reasoning"],
        "input_features": ["transcript_chars_log_sd"], "output_features": [],
        "training_scales": {
            "transcript_chars_log_sd": {
                "source": "transcript_chars", "transform": "log1p", "mean": 0.0, "scale": 1.0,
            },
        },
        "posterior": {
            "input_mean": np.array([[0.0]]), "input_beta": np.array([[1.0]]),
            "output_mean": np.array([[np.log1p(10)]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }


def observed_cells():
    return pd.DataFrame([
        {"model": "a", "item": "i1", "correct": 1, "response_chars": 11,
         "visible_reasoning": 0},
        {"model": "a", "item": "i2", "correct": 1, "response_chars": 12,
         "visible_reasoning": 0},
    ])


def run_simulation(*, trials=1, reuse_responses=False, comparisons=None, artifact=None, seed=4):
    comparisons = comparisons if comparisons is not None else pd.DataFrame([
        {"judge": "j", "m1": "a", "m2": "b"},
    ])
    return simulate(
        comparisons,
        SimulationConfig(
            trials=trials, attempted_items=2, presentations=2,
            reuse_responses=reuse_responses, mode="manual",
        ),
        item_ids=["i1", "i2"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=artifact or judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=seed,
    )


def metric(result, name, **keys):
    rows = result.trials[result.trials["metric"].eq(name)]
    if "comparison_id" not in keys:
        rows = rows[rows["comparison_id"].isna()]
    for key, value in keys.items():
        rows = rows[rows[key].eq(value)]
    return rows["value"].tolist()


def test_cost_prices_total_output_once():
    usage = pd.DataFrame([{
        "trial": 0, "role": "judge", "model": "j", "input_tokens": 100,
        "output_tokens": 80, "reasoning_tokens": 30, "ordinary_output_tokens": 50,
    }])
    prices = pd.DataFrame([{"model_id": "j", "price_in": 2.0, "price_out": 10.0}])
    got, warnings = cost_rows(usage, prices)
    assert np.isclose(got.loc[0, "cost"], (100 * 2 + 80 * 10) / 1_000_000)
    assert warnings.empty


def test_cost_derives_ordinary_output_when_only_billed_total_is_supplied():
    usage = pd.DataFrame([{
        "trial": 0, "role": "judge", "model": "j", "input_tokens": 100,
        "output_tokens": 80,
    }])
    got, _ = cost_rows(
        usage, pd.DataFrame([{"model_id": "j", "price_in": 2.0, "price_out": 10.0}]),
    )
    assert got.loc[0, "reasoning_tokens"] == 0
    assert got.loc[0, "ordinary_output_tokens"] == 80


def test_missing_price_keeps_calls_and_tokens_but_suppresses_currency():
    usage = pd.DataFrame([{
        "trial": 0, "role": "candidate", "model": "unpriced", "calls": 2,
        "input_tokens": 100, "output_tokens": 80,
        "reasoning_tokens": 30, "ordinary_output_tokens": 50,
    }])
    prices = pd.DataFrame([{
        "model_id": "unpriced", "price_in": np.nan, "price_out": np.nan,
    }])
    got, warnings = cost_rows(usage, prices)
    assert got.loc[0, "calls"] == 2 and got.loc[0, "output_tokens"] == 80
    assert np.isnan(got.loc[0, "cost"])
    assert warnings.query("code == 'missing_price'").iloc[0].model == "unpriced"


def test_summary_uses_p5_median_p95_and_all_group_keys():
    trials = pd.DataFrame({
        "trial": list(range(5)), "configuration": ["c"] * 5,
        "warning_state": ["ok"] * 5, "metric": ["usable_items"] * 5,
        "value": [0, 1, 2, 3, 4],
    })
    row = summarize_trials(trials).iloc[0]
    assert row["p50"] == 2 and row["p5"] < row["p50"] < row["p95"]
    assert row["n_trials"] == 5 and row["configuration"] == "c" and row["warning_state"] == "ok"


def test_trial_output_contains_every_required_metric():
    assert set(TRIAL_METRICS) <= set(run_simulation().trials["metric"])


def test_latent_cache_reuses_within_trial_and_redraws_when_trial_changes():
    cache = LatentCache(np.random.default_rng(3))
    first = cache.normal((0, "judge", "unseen"), 0.0, 1.0)
    repeated = cache.normal((0, "judge", "unseen"), 0.0, 1.0)
    next_trial_same_posterior = cache.normal((1, "judge", "unseen"), 0.0, 1.0)
    assert first == repeated
    assert next_trial_same_posterior != first


def test_counterbalance_reverses_each_pair_without_changing_attempts():
    rows = pd.DataFrame([
        {"comparison_id": 0, "judge": "j", "m1": "a", "m2": "b", "item": "i1"},
    ])
    got = counterbalance(rows, 2)
    assert got[["presentation", "left_model", "right_model", "m1_is_left"]].to_dict("records") == [
        {"presentation": 0, "left_model": "a", "right_model": "b", "m1_is_left": True},
        {"presentation": 1, "left_model": "b", "right_model": "a", "m1_is_left": False},
    ]
    assert got["item"].tolist() == ["i1", "i1"]


def test_odd_presentations_randomize_starting_side_deterministically():
    rows = pd.DataFrame([
        {"comparison_id": index, "m1": "a", "m2": "b"} for index in range(20)
    ])
    first = counterbalance(rows, 1, np.random.default_rng(5))
    repeated = counterbalance(rows, 1, np.random.default_rng(5))
    assert set(first["m1_is_left"]) == {False, True}
    pd.testing.assert_frame_equal(first, repeated)


def test_simulation_keeps_attempted_budget_filters_calls_and_uses_observed_and_unseen_lengths():
    got = run_simulation()
    assert metric(got, "usable_items") == [1.0]
    assert metric(got, "judgment_calls") == [2.0]
    assert metric(got, "candidate_calls") == [4.0]
    assert metric(got, "judge_input_tokens", model="j", role="judge") == [66.0]
    exact = got.summary.query("metric == 'judgment_call_upper_bound'").iloc[0]
    assert exact["value"] == 4 and np.isnan(exact["p5"])
    assert metric(got, "judgment_calls")[0] <= exact["value"]


def test_prefixed_irt_item_ids_still_use_bare_observed_cells_and_item_effects():
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=2, presentations=2, mode="manual"),
        item_ids=["gpqa:i1", "gpqa:i2"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=4,
    )
    assert metric(got, "usable_items") == [1.0]
    assert metric(got, "judge_input_tokens", model="j", role="judge") == [66.0]


def test_reused_responses_remove_candidate_generation_but_not_judgments():
    generated = run_simulation(reuse_responses=False)
    reused = run_simulation(reuse_responses=True)
    assert metric(generated, "candidate_calls") == [4.0]
    assert metric(reused, "candidate_calls") == [0.0]
    assert metric(generated, "judgment_calls") == metric(reused, "judgment_calls") == [2.0]


def test_per_model_intervals_include_zero_call_trials():
    no_usable = observed_cells().assign(correct=0)
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=3, attempted_items=2, presentations=2, mode="manual"),
        item_ids=["i1", "i2"], observed_cells=no_usable,
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=2,
    )
    row = got.summary.query(
        "metric == 'judge_input_tokens' and model == 'j' and role == 'judge'"
    ).iloc[0]
    assert row["n_trials"] == 3 and row["p50"] == 0


def test_reasoning_override_uses_population_configuration_not_fitted_identity_or_observed_length():
    got = simulate(
        pd.DataFrame([{
            "judge": "j", "m1": "a", "m2": "b", "m1_reasoning_class": "reasoning",
        }]),
        SimulationConfig(trials=1, attempted_items=2, presentations=2, mode="manual",
                         reuse_responses=False),
        item_ids=["i1", "i2"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact_with_override_population(),
        judge_token_artifact=judge_token_artifact(), catalogue=catalogue(), seed=4,
    )
    assert metric(got, "candidate_input_tokens", model="a", role="candidate") == [60.0]
    assert metric(got, "judge_input_tokens", model="j", role="judge") == [104.0]


def test_manual_rows_keep_raw_probability_and_ties_out_of_accuracy():
    got = run_simulation()
    assert np.isclose(metric(got, "p_choose_m1", comparison_id=0)[0], expit(1.0))
    tied = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 1.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=1,
    )
    assert metric(tied, "ties") == [2.0]
    assert np.isnan(metric(tied, "choice_accuracy")[0])
    assert len(metric(tied, "p_choose_m1", comparison_id=0)) == 1


def test_manual_rows_have_their_own_usable_call_tie_and_accuracy_metrics():
    got = simulate(
        pd.DataFrame([
            {"judge": "j", "m1": "a", "m2": "b"},
            {"judge": "j", "m1": "a", "m2": "a"},
        ]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=3,
    )
    assert metric(got, "usable_items", comparison_id=0) == [1.0]
    assert metric(got, "judgment_calls", comparison_id=0) == [2.0]
    assert metric(got, "ties", comparison_id=0) == [0.0]
    assert np.isfinite(metric(got, "choice_accuracy", comparison_id=0)[0])
    assert metric(got, "ties", comparison_id=1) == [2.0]
    assert np.isnan(metric(got, "choice_accuracy", comparison_id=1)[0])


def test_self_comparison_uses_candidate_identity_feature():
    artifact = judgment_artifact()
    artifact["feature_order"] = ["self_comparison"]
    artifact["posterior"]["global_beta"] = np.array([[1.0]])
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "a"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=artifact,
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=4,
    )
    assert np.isclose(metric(got, "p_choose_m1", comparison_id=0)[0], expit(1.0))


def test_unpredictable_manual_row_stays_named_with_nan_probability_and_exact_bound():
    prices = catalogue()
    prices.loc[prices.model_id == "b", "reasoning_class"] = pd.NA
    comparisons = pd.DataFrame([{
        "judge": "j", "m1": "a", "m2": "b", "predictable": False,
        "m2_reasoning_class": pd.NA,
    }])
    got = simulate(
        comparisons,
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a"], capability_draws=np.array([[0.0, 1.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=prices, seed=1,
    )
    assert np.isnan(metric(got, "p_choose_m1", comparison_id=0)[0])
    assert got.summary.query("metric == 'judgment_call_upper_bound'").iloc[0]["value"] == 2


def test_one_cache_reuses_unseen_judge_within_trial_and_redraws_across_cycled_posterior():
    comparisons = pd.DataFrame([
        {"judge": "j", "m1": "a", "m2": "b"},
        {"judge": "j", "m1": "a", "m2": "b"},
    ])
    got = run_simulation(
        trials=2, comparisons=comparisons,
        artifact=judgment_artifact(observed_judge=False, judge_scale=1.0), seed=12,
    )
    probabilities = got.trials.query("metric == 'p_choose_m1'").pivot(
        index="trial", columns="comparison_id", values="value",
    )
    assert probabilities.loc[0, 0] == probabilities.loc[0, 1]
    assert probabilities.loc[1, 0] == probabilities.loc[1, 1]
    assert probabilities.loc[0, 0] != probabilities.loc[1, 0]


def test_shared_graph_counts_unique_candidate_generation_and_seed_is_deterministic():
    comparisons = pd.DataFrame([
        {"judge": "j", "m1": "a", "m2": "b"},
        {"judge": "j", "m1": "a", "m2": "b"},
    ])
    first = run_simulation(comparisons=comparisons, seed=8)
    second = run_simulation(comparisons=comparisons, seed=8)
    assert metric(first, "candidate_calls") == [4.0]
    pd.testing.assert_frame_equal(first.trials, second.trials)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.warnings, second.warnings)


def test_missing_constituent_price_propagates_to_total_cost():
    prices = catalogue()
    prices.loc[prices.model_id == "b", "price_out"] = np.nan
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=2, presentations=2, mode="manual",
                         reuse_responses=False),
        item_ids=["i1", "i2"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=prices, seed=4,
    )
    assert np.isnan(metric(got, "cost", model="__total__", role="all")[0])
    assert got.warnings.query("code == 'missing_price'")["model"].tolist() == ["b"]


def test_required_family_feature_rejects_missing_catalogue_metadata():
    artifact = judgment_artifact()
    artifact["feature_order"] = ["same_family_x_theta_gap_sd"]
    artifact["training_scales"]["same_family_x_theta_gap"] = {"mean": 0.0, "scale": 1.0}
    artifact["posterior"]["global_beta"] = np.array([[1.0]])
    models = catalogue()
    models.loc[models.model_id == "b", "family"] = pd.NA
    with pytest.raises(ValueError, match="family"):
        simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
            item_ids=["i1"], observed_cells=observed_cells(),
            capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
            irt=irt(), judgment_artifact=artifact,
            candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
            catalogue=models, seed=1,
        )


def test_static_design_warnings_keep_source_provenance_without_trial_duplicates():
    comparisons = pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}])
    warnings = pd.DataFrame([{
        "code": "automatic_rule_violation", "row": 0, "source_row": "original",
        "message": "m1 is not newer than judge",
    }])
    got = simulate(
        lambda _rng: (comparisons, warnings),
        SimulationConfig(trials=2, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=1,
    )
    design = got.warnings.query("code == 'automatic_rule_violation'")
    assert len(design) == 1 and design.iloc[0]["source_row"] == "original"
