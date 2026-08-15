import copy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import mc_cost.simulation as simulation
from mc_cost.simulation import (
    LatentCache,
    SimulationConfig,
    TRIAL_METRICS,
    cost_rows,
    counterbalance,
    summarize_trials,
)
from mc_cost.simulation import simulate as core_simulate
from mc_cost.validation import write_decision
from mc_cost.validation import support_distance


DATA_HASHES = {"judgments": "abc", "candidate_tokens": "def", "judge_tokens": "ghi"}
_DECISION_DIRECTORY = tempfile.TemporaryDirectory()
DECISION_PATH = Path(_DECISION_DIRECTORY.name) / "decision.json"


def support_decision(path, names, matrix, *, gate=False, coordinate_system="raw"):
    matrix = np.asarray(matrix, dtype=float)
    threshold = float(support_distance(matrix, matrix[:1]).iloc[0]["threshold"])
    write_decision(path, {
        "data_hashes": DATA_HASHES,
        "judge_covariate_gate": {"enabled": gate},
        "judgment_support": {
            "coordinate_system": coordinate_system,
            "predictor_names": list(names), "training_matrix": matrix.tolist(),
            "training_ranges": np.column_stack([matrix.min(axis=0), matrix.max(axis=0)]).tolist(),
            "nearest_neighbor_threshold": threshold,
        },
    })


support_decision(DECISION_PATH, ["theta_gap"], [[-1_000_000], [0], [1_000_000]])


def simulate(*args, **kwargs):
    artifact = kwargs.get("judgment_artifact")
    if artifact is None and len(args) > 8:
        artifact = args[8]
    if artifact is not None:
        metadata = artifact.metadata if hasattr(artifact, "metadata") else artifact
        posterior = artifact.posterior if hasattr(artifact, "posterior") else artifact["posterior"]
        names = [
            "theta_gap",
            *(simulation.FEATURE_COLUMNS.get(feature, feature) for feature in metadata.get("feature_order", ())),
            *(["length_difference"] if "length_mean" in posterior else []),
        ]
        matrix = np.vstack([
            np.zeros(len(names)), np.eye(len(names)) * 1_000_000,
            -np.eye(len(names)) * 1_000_000,
        ])
        support_decision(DECISION_PATH, names, matrix)
    kwargs.setdefault("decision_path", DECISION_PATH)
    kwargs.setdefault("data_hashes", DATA_HASHES)
    return core_simulate(*args, **kwargs)


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
        "presentation_include_reasoning": False,
        "presentation_max_chars": 65536,
        "presentation_truncate_frac": None,
        "presentation_truncate_keep": "first",
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


def core_simulation_call(comparisons, **decision):
    return core_simulate(
        comparisons,
        SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=1, **decision,
    )


def test_simulation_refuses_absent_decision_before_any_work(tmp_path):
    called = False

    def comparisons(_rng):
        nonlocal called
        called = True
        return pd.DataFrame()

    with pytest.raises(FileNotFoundError, match="decision.json"):
        core_simulation_call(
            comparisons, decision_path=tmp_path / "decision.json", data_hashes=DATA_HASHES,
        )
    assert not called


def test_simulation_refuses_stale_decision_before_any_work(tmp_path):
    decision = tmp_path / "decision.json"
    write_decision(decision, {"data_hashes": {**DATA_HASHES, "judgments": "stale"}})
    called = False

    def comparisons(_rng):
        nonlocal called
        called = True
        return pd.DataFrame()

    with pytest.raises(ValueError, match="data hash mismatch"):
        core_simulation_call(comparisons, decision_path=decision, data_hashes=DATA_HASHES)
    assert not called


def test_simulation_fails_closed_on_missing_or_malformed_support(tmp_path):
    decision = tmp_path / "decision.json"
    write_decision(decision, {"data_hashes": DATA_HASHES})
    with pytest.raises(ValueError, match="judgment support"):
        core_simulation_call(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            decision_path=decision, data_hashes=DATA_HASHES,
        )

    support_decision(decision, ["wrong_name"], [[-1], [0], [1]])
    with pytest.raises(ValueError, match="predictor names"):
        core_simulation_call(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            decision_path=decision, data_hashes=DATA_HASHES,
        )

    support_decision(decision, ["theta_gap"], [[-1], [0], [1]], coordinate_system="standardized")
    with pytest.raises(ValueError, match="coordinate_system.*raw"):
        core_simulation_call(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            decision_path=decision, data_hashes=DATA_HASHES,
        )


def _support_simulation(tmp_path, training):
    decision = tmp_path / "decision.json"
    support_decision(decision, ["theta_gap", "oversight"], training)
    artifact = judgment_artifact()
    artifact["feature_order"] = ["oversight_sd"]
    artifact["training_scales"]["oversight"] = {"mean": 0.0, "scale": 1.0}
    artifact["posterior"]["global_beta"] = np.array([[0.0]])
    return core_simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=artifact,
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), decision_path=decision, data_hashes=DATA_HASHES, seed=1,
    )


def test_simulation_support_warns_for_range_and_joint_extrapolation(tmp_path):
    in_support = _support_simulation(tmp_path, [[-1, .5], [0, 0], [1, .5], [0, 1]])
    assert not set(in_support.warnings["code"]) & {
        "outside_training_range", "outside_joint_support",
    }

    outside_range = _support_simulation(tmp_path, [[-.1, 0], [0, .1], [.1, 0]])
    assert "outside_training_range" in set(outside_range.warnings["code"])
    assert metric(outside_range, "outside_training_range", comparison_id=0) == [1.0]

    diagonal = [[value, value / 2] for value in np.linspace(-2, 2, 9)]
    joint = _support_simulation(tmp_path, diagonal)
    assert "outside_training_range" not in set(joint.warnings["code"])
    assert "outside_joint_support" in set(joint.warnings["code"])
    assert metric(joint, "outside_joint_support", comparison_id=0) == [1.0]


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


def test_summary_collapses_mixed_states_and_counts_finite_trials():
    trials = pd.DataFrame({
        "trial": [0, 1, 2], "configuration": ["c"] * 3, "metric": ["cost"] * 3,
        "value": [1.0, np.nan, 3.0],
        "warning_state": ["ok", "missing_price", "ok"],
        "reason": ["", "missing input or output price for b", ""],
    })

    summary = summarize_trials(trials)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row[["p5", "p50", "p95"]].notna().all() and row.p50 == 2.0
    assert row.n_trials == 3 and row.n_finite_trials == 2
    assert row.warning_state == "partial"
    assert row.reason == "missing input or output price for b"


def test_summary_all_unavailable_retains_null_quantiles_and_named_reason():
    trials = pd.DataFrame({
        "trial": [0, 1], "metric": ["choice_accuracy"] * 2, "value": [np.nan, np.nan],
        "warning_state": ["unavailable", "unavailable"],
        "reason": ["no usable judgments", "all usable judgments tied"],
    })

    row = summarize_trials(trials).iloc[0]

    assert row[["p5", "p50", "p95"]].isna().all()
    assert row.n_trials == 2 and row.n_finite_trials == 0
    assert row.warning_state == "unavailable"
    assert row.reason == "all usable judgments tied; no usable judgments"


def test_trial_output_contains_every_required_metric():
    result = run_simulation()
    assert set(TRIAL_METRICS) <= set(result.trials["metric"])
    assert "reason" in result.trials and "reason" in result.summary


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


def test_configurations_of_one_model_keep_separate_observed_judgment_usage_and_cost_identity():
    judgment = judgment_artifact()
    judgment["posterior"]["candidate_effects"] = np.array([[3.0]])
    got = simulate(
        pd.DataFrame([
            {"judge": "j", "m1": "a", "m2": "b"},
            {"judge": "j", "m1": "a", "m2": "b", "m1_reasoning_class": "reasoning"},
        ]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual",
                         reuse_responses=False),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment,
        candidate_token_artifact=candidate_artifact_with_override_population(),
        judge_token_artifact=judge_token_artifact(), catalogue=catalogue(), seed=4,
    )
    usage = got.trials.query(
        "metric == 'candidate_input_tokens' and model == 'a' and role == 'candidate'"
    )
    assert dict(zip(usage["reasoning_class"], usage["value"])) == {
        "non_reasoning": 10.0, "reasoning": 30.0,
    }
    costs = got.trials.query("metric == 'cost' and model == 'a' and role == 'candidate'")
    assert dict(zip(costs["reasoning_class"], costs["value"])) == {
        "non_reasoning": 50 / 1_000_000, "reasoning": 138 / 1_000_000,
    }
    assert np.isclose(metric(got, "p_choose_m1", comparison_id=1)[0], expit(1.0))


def test_observed_cells_are_sampled_by_exact_reasoning_configuration():
    observed = observed_cells().assign(reasoning_class="non_reasoning")
    observed = pd.concat([observed, pd.DataFrame([{
        "model": "a", "item": "i1", "reasoning_class": "reasoning",
        "correct": 0, "response_chars": 99, "visible_reasoning": 1,
    }])], ignore_index=True)
    got = simulate(
        pd.DataFrame([{
            "judge": "j", "m1": "a", "m2": "b", "m1_reasoning_class": "reasoning",
        }]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed,
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact_with_override_population(),
        judge_token_artifact=judge_token_artifact(), catalogue=catalogue(), seed=4,
    )
    assert metric(got, "usable_items", comparison_id=0) == [0.0]
    assert np.isnan(metric(got, "p_choose_m1", comparison_id=0)[0])


def test_final_only_candidate_lengths_drive_judgment_features_and_judge_input(monkeypatch):
    observed = pd.DataFrame([
        {
            "model": "a", "item": "i1", "correct": 1, "response_chars": 11,
            "full_response_chars": 111, "generated_reasoning": 1, "visible_reasoning": 1,
        },
        {
            "model": "b", "item": "i1", "correct": 1, "response_chars": 13,
            "full_response_chars": 130, "generated_reasoning": 0, "visible_reasoning": 0,
        },
    ])
    presented_rows = []

    def probabilities(_artifact, presented, *_args):
        presented_rows.append(presented.copy())
        return np.full(len(presented), 0.5)

    monkeypatch.setattr(simulation, "_scenario_left_probabilities", probabilities)
    artifact = judgment_artifact()
    artifact["presentation_max_chars"] = 12
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
        item_ids=["i1"], observed_cells=observed,
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=artifact,
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=4,
    )

    presented = presented_rows[0]
    assert sorted(presented[["left_chars", "right_chars"]].iloc[0]) == [11, 12]
    assert presented[["left_visible_reasoning", "right_visible_reasoning"]].eq(0).all(axis=None)
    assert metric(got, "judge_input_tokens", model="j", role="judge") == [23.0]


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
    assert tied.trials.query(
        "metric == 'choice_accuracy' and comparison_id.isnull()"
    )["warning_state"].tolist() == ["unavailable"]
    assert tied.trials.query(
        "metric == 'choice_accuracy' and comparison_id.isnull()"
    )["reason"].tolist() == ["all usable judgments tied"]
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


@pytest.mark.parametrize("feature", ["self_comparison", "reasoning_difference_sd"])
def test_unsupported_prior_only_judgment_features_are_rejected(feature):
    artifact = judgment_artifact()
    artifact["feature_order"] = [feature]
    if feature.endswith("_sd"):
        artifact["training_scales"]["reasoning_difference"] = {"mean": 0.0, "scale": 1.0}
    artifact["posterior"]["global_beta"] = np.array([[1.0]])
    with pytest.raises(ValueError, match="unsupported judgment feature"):
        simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "a"}]),
            SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
            item_ids=["i1"], observed_cells=observed_cells(),
            capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
            irt=irt(), judgment_artifact=artifact,
            candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
            catalogue=catalogue(), seed=4,
        )


@pytest.mark.parametrize("condition", [True, None])
def test_simulation_rejects_non_final_or_undeclared_judgment_presentation(condition):
    artifact = judgment_artifact()
    artifact["presentation_include_reasoning"] = condition

    with pytest.raises(ValueError, match="final-only|include_reasoning=false"):
        simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
            item_ids=["i1"], observed_cells=observed_cells(),
            capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
            irt=irt(), judgment_artifact=artifact,
            candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
            catalogue=catalogue(), seed=4,
        )


def test_empty_presentation_still_rejects_unsupported_judgment_condition():
    artifact = judgment_artifact()
    artifact["presentation_include_reasoning"] = True

    with pytest.raises(ValueError, match="final-only|include_reasoning=false"):
        simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
            item_ids=["i2"], observed_cells=observed_cells().assign(correct=0),
            capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
            irt=irt(), judgment_artifact=artifact,
            candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
            catalogue=catalogue(), seed=4,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("presentation_max_chars", None, "max chars"),
        ("presentation_truncate_frac", 0.5, "paired truncation"),
        ("presentation_truncate_keep", "last", "truncate keep"),
    ],
)
def test_simulation_rejects_undeclared_or_paired_truncation(field, value, message):
    artifact = judgment_artifact()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
            item_ids=["i2"], observed_cells=observed_cells().assign(correct=0),
            capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
            irt=irt(), judgment_artifact=artifact,
            candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
            catalogue=catalogue(), seed=4,
        )


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
    unavailable = got.trials.loc[
        got.trials["metric"].isin(["choice_accuracy", "p_choose_m1"]), "warning_state",
    ]
    assert set(unavailable) == {"unavailable"}
    assert got.summary.query("metric == 'judgment_call_upper_bound'").iloc[0]["value"] == 2


def test_direct_missing_capability_draw_becomes_named_unavailable_comparison():
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a"], capability_draws=np.array([[0.0, 1.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=catalogue(), seed=1,
    )

    unavailable = got.trials.query(
        "comparison_id == 0 and metric in ['choice_accuracy', 'p_choose_m1']"
    )
    assert unavailable["value"].isna().all()
    assert set(unavailable["reason"]) == {"missing capability draw: b"}
    assert got.warnings.query("code == 'missing_capability_draw'")["model"].tolist() == ["b"]


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


def test_independent_posterior_draws_repeat_for_the_same_seed():
    def with_two_draws(artifact):
        result = copy.deepcopy(artifact)
        result["posterior"] = {
            name: np.repeat(np.asarray(values), 2, axis=0)
            for name, values in result["posterior"].items()
        }
        return result

    judgment = with_two_draws(judgment_artifact())
    judgment["posterior"]["judge_effects"][1, 0, 1] = 2.0
    candidate = with_two_draws(candidate_artifact())
    candidate["posterior"]["input_mean"][1, 0] = np.log1p(11)
    judge_tokens = with_two_draws(judge_token_artifact())
    judge_tokens["posterior"]["output_mean"][1, 0] = np.log1p(12)
    irt_draws = copy.deepcopy(irt())
    irt_draws["a"] = np.repeat(irt_draws["a"], 2, axis=0)
    irt_draws["b"] = np.repeat(irt_draws["b"], 2, axis=0)
    irt_draws["a"][1] = [90.0, 90.0]

    def run():
        return simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(trials=3, attempted_items=1, presentations=2, mode="manual"),
            item_ids=["i1"], observed_cells=observed_cells(),
            capability_models=["j", "a", "b"],
            capability_draws=np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
            irt=irt_draws, judgment_artifact=judgment,
            candidate_token_artifact=candidate, judge_token_artifact=judge_tokens,
            catalogue=catalogue(), seed=5,
        )

    first = run()
    allowed = np.array([expit(1.0), expit(2.0), expit(4.0)])
    assert all(np.isclose(value, allowed).any() for value in metric(first, "p_choose_m1", comparison_id=0))
    second = run()
    pd.testing.assert_frame_equal(first.trials, second.trials)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.warnings, second.warnings)


def test_independent_posterior_schedules_use_full_ranges_and_keep_capability_irt_aligned(monkeypatch):
    def repeated(artifact, count):
        result = copy.deepcopy(artifact)
        result["posterior"] = {
            name: np.repeat(np.asarray(values), count, axis=0)
            for name, values in result["posterior"].items()
        }
        return result

    judgment = repeated(judgment_artifact(), 4000)
    candidate = repeated(candidate_artifact(), 4000)
    judge_tokens = repeated(judge_token_artifact(), 4000)
    capability_draws = np.zeros((1000, 3))
    capability_draws[:, 1] = 1.0 + np.arange(1000) / 1000
    irt_draws = copy.deepcopy(irt())
    irt_draws["a"] = np.repeat(irt_draws["a"], 1000, axis=0)
    irt_draws["b"] = np.repeat(irt_draws["b"], 1000, axis=0)
    irt_draws["b"][:, 0] = np.arange(1000) / 1000
    irt_draws["floor"] = np.ones(2)
    empty_observed = pd.DataFrame(columns=["model", "item", "correct"])
    captured = {}
    original_cells = simulation._candidate_cells
    original_probabilities = simulation._scenario_left_probabilities
    original_lengths = simulation.draw_transcript_lengths
    original_usage = simulation.draw_token_usage

    def cells_probe(*args, **kwargs):
        cells = original_cells(*args, **kwargs)
        a = cells.loc[cells.model.eq("a")].iloc[0]
        captured["aligned"].append((a.theta, a.difficulty))
        return cells

    def probability_probe(artifact, presented, metadata, capability, posterior_index, cache, trial):
        captured["judgment"].append(posterior_index)
        return original_probabilities(
            artifact, presented, metadata, capability, posterior_index, cache, trial,
        )

    def length_probe(artifact, rows, rng, posterior_index, *args, **kwargs):
        captured["candidate_length"].append(posterior_index)
        return original_lengths(artifact, rows, rng, posterior_index, *args, **kwargs)

    def usage_probe(artifact, rows, rng, posterior_index, *args, **kwargs):
        captured[artifact["role"]].append(posterior_index)
        return original_usage(artifact, rows, rng, posterior_index, *args, **kwargs)

    monkeypatch.setattr(simulation, "_candidate_cells", cells_probe)
    monkeypatch.setattr(simulation, "_scenario_left_probabilities", probability_probe)
    monkeypatch.setattr(simulation, "draw_transcript_lengths", length_probe)
    monkeypatch.setattr(simulation, "draw_token_usage", usage_probe)

    def run(seed):
        captured.clear()
        captured.update({
            "aligned": [], "judgment": [], "candidate_length": [], "candidate": [], "judge": [],
        })
        result = simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(
                trials=20, attempted_items=1, presentations=1, mode="manual",
                reuse_responses=False,
            ),
            item_ids=["i1"], observed_cells=empty_observed,
            capability_models=["j", "a", "b"], capability_draws=capability_draws,
            irt=irt_draws, judgment_artifact=judgment,
            candidate_token_artifact=candidate, judge_token_artifact=judge_tokens,
            catalogue=catalogue(), seed=seed,
        )
        return result, copy.deepcopy(captured)

    first, first_indexes = run(12)
    second, second_indexes = run(12)

    for name in ("judgment", "candidate_length", "candidate", "judge"):
        assert max(first_indexes[name]) > 999
    assert first_indexes["candidate_length"] == first_indexes["candidate"]
    assert all(np.isclose(theta - 1.0, difficulty) for theta, difficulty in first_indexes["aligned"])
    assert first_indexes == second_indexes
    pd.testing.assert_frame_equal(first.trials, second.trials)


def test_artifact_posterior_arrays_must_have_one_draw_count():
    artifact = judgment_artifact()
    artifact["posterior"]["judge_effects"] = np.repeat(
        artifact["posterior"]["judge_effects"], 2, axis=0,
    )

    with pytest.raises(ValueError, match="judgment.*aligned draw counts"):
        simulate(
            pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
            SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
            item_ids=["i1"], observed_cells=observed_cells(),
            capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
            irt=irt(), judgment_artifact=artifact,
            candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
            catalogue=catalogue(), seed=4,
        )


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
    unavailable = got.trials.query("metric == 'cost' and warning_state == 'missing_price'")
    assert set(unavailable["reason"]) == {"missing input or output price for b"}
    assert set(got.summary.query(
        "metric == 'cost' and warning_state == 'missing_price'"
    )["reason"]) == {"missing input or output price for b"}


def test_zero_use_unpriced_planned_configuration_costs_zero_without_suppressing_total():
    prices = catalogue()
    prices.loc[prices.model_id == "b", ["price_in", "price_out"]] = np.nan
    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=2, mode="manual"),
        item_ids=["i2"], observed_cells=observed_cells().assign(correct=0),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=judgment_artifact(),
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=prices, seed=2,
    )
    assert metric(
        got, "cost", model="b", role="candidate", reasoning_class="non_reasoning",
    ) == [0.0]
    assert metric(got, "cost", model="__total__", role="all") == [0.0]
    assert got.warnings.query("code == 'missing_price'").empty


def test_family_feature_normalizes_catalogue_values_like_training():
    artifact = judgment_artifact()
    artifact["feature_order"] = ["same_family_x_theta_gap_sd"]
    artifact["training_scales"]["same_family_x_theta_gap"] = {"mean": 0.0, "scale": 1.0}
    artifact["posterior"]["global_beta"] = np.array([[1.0]])
    models = catalogue()
    models.loc[models.model_id == "a", "family"] = " x "
    models.loc[models.model_id == "b", "family"] = "x"

    got = simulate(
        pd.DataFrame([{"judge": "j", "m1": "a", "m2": "b"}]),
        SimulationConfig(trials=1, attempted_items=1, presentations=1, mode="manual"),
        item_ids=["i1"], observed_cells=observed_cells(),
        capability_models=["j", "a", "b"], capability_draws=np.array([[0.0, 1.0, 0.0]]),
        irt=irt(), judgment_artifact=artifact,
        candidate_token_artifact=candidate_artifact(), judge_token_artifact=judge_token_artifact(),
        catalogue=models, seed=1,
    )

    assert np.isclose(metric(got, "p_choose_m1", comparison_id=0)[0], expit(2.0))


@pytest.mark.parametrize("family", [pd.NA, "  "])
def test_required_family_feature_rejects_missing_catalogue_metadata(family):
    artifact = judgment_artifact()
    artifact["feature_order"] = ["same_family_x_theta_gap_sd"]
    artifact["training_scales"]["same_family_x_theta_gap"] = {"mean": 0.0, "scale": 1.0}
    artifact["posterior"]["global_beta"] = np.array([[1.0]])
    models = catalogue()
    models.loc[models.model_id == "b", "family"] = family
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
