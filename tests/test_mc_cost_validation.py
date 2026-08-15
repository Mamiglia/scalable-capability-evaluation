import importlib.util
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from pathlib import Path

from mc_cost.model import JUDGMENT_SPECS
from mc_cost.ranking import lomo_predictions as packaged_lomo_predictions
from mc_cost.validation import (
    choose_judge_gate,
    choose_judgment_spec,
    compare_downsamples,
    cross_validate_judgments,
    decision_payload,
    fitted_ranking_components,
    posterior_predictive_replay,
    ranking_recovery,
    require_decision,
    support_distance,
    build_support_index,
    validate_tokens,
    write_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_irt_metrics_reexports_packaged_lomo_implementation():
    spec = importlib.util.spec_from_file_location(
        "irt_proxy_metrics", ROOT / "scripts/irt_proxy/metrics.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.lomo_predictions is packaged_lomo_predictions


def test_judge_covariates_require_improvement_beyond_one_se():
    conditional = pd.DataFrame({"fold": ["a", "b", "c"], "log_loss": [0.49, 0.52, 0.50]})
    pooled = pd.DataFrame({"fold": ["a", "b", "c"], "log_loss": [0.50, 0.51, 0.52]})
    assert not choose_judge_gate(conditional, pooled)["enabled"]

    conditional["log_loss"] = [0.40, 0.41, 0.42]
    assert choose_judge_gate(conditional, pooled)["enabled"]


def test_joint_support_flags_far_points_using_training_loo_threshold():
    train = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    query = np.array([[0.05, 0.05], [10.0, 10.0]])
    got = support_distance(train, query)
    assert not got.iloc[0].outside_support
    assert got.iloc[1].outside_support


def test_ranking_recovery_returns_component_results_and_warning():
    votes = pd.DataFrame([
        {"fit": "gpqa", "major": "a", "minor": "b", "item_id": "i1", "picked_major": 1},
        {"fit": "gpqa", "major": "c", "minor": "d", "item_id": "i1", "picked_major": 1},
    ])
    reference = pd.DataFrame({"model": ["a", "b", "c", "d", "e"], "score": [4, 3, 2, 1, 0]})
    results, warnings = ranking_recovery(votes, reference)
    assert results.loc[results["n_votes"].gt(0), "component"].nunique() == 2
    isolated = results[results["component_models"].eq("e")].iloc[0]
    assert not isolated["valid"]
    assert isolated["reason"] == "reference model has no comparison edge"
    assert "disconnected_graph" in set(warnings["code"])
    assert "isolated_reference_model" in set(warnings["code"])


def test_fitted_ranking_components_have_stable_membership_keys():
    fitted = fitted_ranking_components({
        ("c", "d"): [4, 5],
        ("a", "b"): [3, 5],
    })

    assert [part["component_models"] for part in fitted] == ["a|b", "c|d"]
    assert [part["members"] for part in fitted] == [("a", "b"), ("c", "d")]
    assert all(set(part["strengths"]) == set(part["members"]) for part in fitted)


def test_ranking_recovery_retains_reference_when_no_component_matches():
    votes = pd.DataFrame([
        {"fit": "gpqa", "major": "a", "minor": "b", "item_id": "i1", "picked_major": 1},
    ])
    reference = pd.DataFrame({"model": ["e"], "score": [0.0]})

    results, warnings = ranking_recovery(votes, reference)

    isolated = results[results["component_models"].eq("e")].iloc[0]
    assert not isolated["valid"]
    assert isolated["reason"] == "reference model has no comparison edge"
    assert warnings.loc[warnings["code"].eq("isolated_reference_model"), "fit"].tolist() == ["gpqa"]


def test_judgment_spec_uses_paired_one_se_and_simplest_order():
    folds = pd.DataFrame([
        {"spec": "baseline", "fold": "a", "log_loss": 0.50},
        {"spec": "baseline", "fold": "b", "log_loss": 0.54},
        {"spec": "baseline", "fold": "c", "log_loss": 0.48},
        {"spec": "core", "fold": "a", "log_loss": 0.50},
        {"spec": "core", "fold": "b", "log_loss": 0.50},
        {"spec": "core", "fold": "c", "log_loss": 0.50},
        {"spec": "core_release", "fold": "a", "log_loss": 0.49},
        {"spec": "core_release", "fold": "b", "log_loss": 0.51},
        {"spec": "core_release", "fold": "c", "log_loss": 0.50},
    ])
    assert choose_judgment_spec(folds)["selected"] == "baseline"

    folds.loc[folds["spec"].eq("baseline"), "log_loss"] = [0.60, 0.60, 0.60]
    assert choose_judgment_spec(folds)["selected"] == "core"


def test_paired_gates_reject_mismatched_or_duplicate_folds():
    conditional = pd.DataFrame({"fold": ["a", "b"], "log_loss": [0.4, 0.4]})
    with pytest.raises(ValueError, match="same unique folds"):
        choose_judge_gate(
            conditional,
            pd.DataFrame({"fold": ["a", "c"], "log_loss": [0.5, 0.5]}),
        )
    with pytest.raises(ValueError, match="same unique folds"):
        choose_judge_gate(
            conditional,
            pd.DataFrame({"fold": ["a", "a"], "log_loss": [0.5, 0.5]}),
        )


def test_cross_validation_holds_out_judges_and_candidate_configurations():
    frame = pd.DataFrame([
        {"judge": "j1", "left_model": "a", "right_model": "b", "chosen_left": 1},
        {"judge": "j1", "left_model": "c", "right_model": "a", "chosen_left": 0},
        {"judge": "j2", "left_model": "b", "right_model": "c", "chosen_left": 1},
        {"judge": "j2", "left_model": "a", "right_model": "c", "chosen_left": 1},
    ]).assign(
        length_difference=[-2.0, -1.0, 1.0, 2.0],
        difficulty=[-1.0, 0.0, 1.0, 2.0],
        correct=[0, 1, 0, 1],
        visible_reasoning=[0, 0, 1, 1],
    )
    calls = []

    def fit_predict(train, test, spec, conditional_judges):
        assert "chosen_left" not in test
        calls.append((train.copy(), test.copy(), spec, conditional_judges))
        return np.full(len(test), 0.75 if conditional_judges else 0.5)

    predictions, folds = cross_validate_judgments(frame, fit_predict)

    assert set(folds["spec"]) == set(JUDGMENT_SPECS)
    assert set(folds["validation"]) == {"judge", "candidate"}
    assert set(folds.loc[folds["validation"].eq("judge"), "population"]) == {
        "conditional", "pooled",
    }
    assert set(folds.loc[folds["validation"].eq("candidate"), "population"]) == {
        "conditional",
    }
    assert choose_judgment_spec(folds)["selected"] in JUDGMENT_SPECS
    assert predictions["probability"].between(0, 1).all()
    assert {"length_difference", "difficulty", "correct", "visible_reasoning"} <= set(predictions)
    for train, test, _spec, _conditional in calls:
        if test["judge"].nunique() == 1 and test["judge"].iloc[0] not in set(train["judge"]):
            continue
        held_models = (
            set(test["left_model"]) | set(test["right_model"])
        ) - (set(train["left_model"]) | set(train["right_model"]))
        assert len(held_models) == 1


def test_cross_validation_retains_each_posterior_probability_draw():
    frame = pd.DataFrame([
        {"judge": "j1", "left_model": "a", "right_model": "b", "chosen_left": 1},
        {"judge": "j2", "left_model": "a", "right_model": "b", "chosen_left": 0},
    ])

    def fit_predict(_train, test, _spec, _conditional_judges):
        return np.vstack([np.full(len(test), 0.2), np.full(len(test), 0.8)])

    predictions, folds = cross_validate_judgments(frame, fit_predict)

    assert set(predictions["draw"]) == {0, 1}
    assert set(predictions["probability"]) == {0.2, 0.8}
    assert predictions.groupby([
        "spec", "validation", "population", "fold", "source_row",
    ])["draw"].nunique().eq(2).all()
    assert np.allclose(folds["log_loss"], -np.log(0.5))


def test_token_validation_holds_exact_configuration_and_reports_length_strata():
    frame = pd.DataFrame([
        {
            "role": "candidate", "model": model, "reasoning_class": reasoning_class,
            "input_tokens": 10 + row, "output_tokens": 20 + row,
            "reasoning_tokens": (5 + row) if reasoning_class == "reasoning" else 0,
            "response_chars": 100 + 10 * row, "release_day": 100 * row,
            "difficulty": float(row % 2), "correct": row % 2,
            "visible_reasoning": int(reasoning_class == "reasoning"),
        }
        for row, (model, reasoning_class) in enumerate([
            ("r1", "reasoning"), ("r2", "reasoning"),
            ("n1", "non_reasoning"), ("n2", "non_reasoning"),
        ])
    ])
    held = []
    expected_targets = {
        "r1": {"input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 5, "response_chars": 100},
        "r2": {"input_tokens": 11, "output_tokens": 21, "reasoning_tokens": 6, "response_chars": 110},
        "n1": {"input_tokens": 12, "output_tokens": 22, "reasoning_tokens": 0, "response_chars": 120},
        "n2": {"input_tokens": 13, "output_tokens": 23, "reasoning_tokens": 0, "response_chars": 130},
    }

    def fit_predict(train, test, role):
        assert not set(expected_targets["r1"]) & set(test)
        key = tuple(test[["model", "role", "reasoning_class"]].iloc[0])
        held.append(key)
        assert not train[["model", "role", "reasoning_class"]].eq(key).all(axis=1).any()
        result = {}
        for outcome in ("input_tokens", "output_tokens", "reasoning_tokens", "response_chars"):
            actual = np.log1p([expected_targets[model][outcome] for model in test["model"]])
            result[f"{outcome}_mean"] = actual + 0.1
            result[f"{outcome}_p5"] = actual - 0.2
            result[f"{outcome}_p95"] = actual + 0.2
        return pd.DataFrame(result, index=test.index)

    predictions, summary, length = validate_tokens(frame, "candidate", fit_predict)

    assert set(held) == {
        ("r1", "candidate", "reasoning"), ("r2", "candidate", "reasoning"),
        ("n1", "candidate", "non_reasoning"), ("n2", "candidate", "non_reasoning"),
    }
    assert set(summary["outcome"]) == {
        "input_tokens", "output_tokens", "reasoning_tokens", "response_chars",
    }
    assert np.allclose(summary["log_mae"], 0.1)
    assert set(length["stratum"]) == {
        "release_date_band", "item_difficulty", "correctness", "visible_reasoning",
    }
    assert predictions["covered_90"].all()


def test_support_scaling_and_threshold_do_not_depend_on_queries():
    train = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])
    near = support_distance(train, np.array([[1.5, 2.0]]))
    mixed = support_distance(train, np.array([[1.5, 2.0], [1e9, -1e9]]))
    assert near.iloc[0].distance == mixed.iloc[0].distance
    assert near.iloc[0].threshold == mixed.iloc[0].threshold


def test_support_index_uses_ckdtree_and_matches_public_query_contract():
    train = np.column_stack([np.arange(5000, dtype=float), np.zeros(5000)])
    index = build_support_index(train)

    assert isinstance(index.tree, cKDTree)
    assert support_distance(train, train[[2500]]).iloc[0].distance == pytest.approx(
        index.query(train[[2500]]).iloc[0].distance
    )


def test_support_index_is_duplicate_invariant_for_interpolated_queries():
    distinct = np.array([[0.0], [2.0], [4.0]])
    duplicated = np.repeat(distinct, [20, 1, 1], axis=0)

    expected = support_distance(distinct, np.array([[1.0]])).iloc[0]
    actual = support_distance(duplicated, np.array([[1.0]])).iloc[0]

    assert actual.distance == pytest.approx(expected.distance)
    assert actual.threshold == pytest.approx(expected.threshold)
    assert actual.threshold > 0
    assert not actual.outside_support
    with pytest.raises(ValueError, match="distinct"):
        build_support_index(np.array([[1.0], [1.0]]))


def test_ranking_recovery_labels_theta_and_runs_lomo_on_connected_component():
    votes = pd.DataFrame([
        {"fit": "gpqa", "major": a, "minor": b, "item_id": f"i{i}", "picked_major": 1}
        for i, (a, b) in enumerate([("a", "b"), ("b", "c"), ("c", "d")])
    ])
    reference = pd.DataFrame({
        "model": ["a", "b", "c", "d"],
        "score": [4.0, 3.0, 2.0, 1.0],
        "easy_accuracy": [0.8, 0.7, 0.6, 0.5],
        "hard_accuracy": [0.7, 0.6, 0.5, 0.4],
    })
    results, warnings = ranking_recovery(
        votes, reference, generating_theta=dict(zip(reference["model"], reference["score"])),
    )
    assert warnings.empty
    assert results.loc[0, "recovery_type"] == "generating_theta_consistency"
    assert results.loc[0, "kendall_tau"] == 1.0
    assert results.loc[0, "lomo_reason"] == ""
    assert np.isfinite(results.loc[0, "lomo_mae"])


def test_decision_writer_is_deterministic_and_hash_bound(tmp_path):
    predictors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    threshold = float(support_distance(predictors, predictors[:1]).iloc[0].threshold)
    token_coverage = pd.DataFrame([
        {"role": "candidate", "reasoning_class": "reasoning", "outcome": "output_tokens", "coverage_90": 0.9},
        {"role": "judge", "reasoning_class": "non_reasoning", "outcome": "input_tokens", "coverage_90": 0.8},
    ])
    payload = decision_payload(
        selected_spec="core",
        judge_gate={"enabled": True, "mean_improvement": 0.1, "standard_error": 0.01},
        token_coverage=token_coverage,
        predictors=predictors,
        predictor_names=["theta_gap", "oversight"],
        support_threshold=threshold,
        data_hashes={"judgments": "abc", "candidate_tokens": "def"},
        validation_timestamp="2026-08-15T12:00:00Z",
    )
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_decision(first, payload)
    write_decision(second, payload)

    assert first.read_bytes() == second.read_bytes()
    assert payload["judgment_support"] == {
        "coordinate_system": "raw",
        "predictor_names": ["theta_gap", "oversight"],
        "training_matrix": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "training_ranges": [[0.0, 1.0], [0.0, 1.0]],
        "nearest_neighbor_threshold": threshold,
    }
    assert require_decision(first, payload["data_hashes"]) == payload
    with pytest.raises(ValueError, match="data hash mismatch"):
        require_decision(first, {"judgments": "changed", "candidate_tokens": "def"})
    with pytest.raises(FileNotFoundError, match="decision.json"):
        require_decision(tmp_path / "decision.json", payload["data_hashes"])


def _replay_frame():
    observations = pd.DataFrame({
        "row_id": range(4), "observed_choice": [0, 1, 0, 1],
        "probability": [0.1, 0.9, 0.2, 0.8],
        "judge": ["j1", "j1", "j2", "j2"], "m1_is_left": [0, 1, 0, 1],
        "theta_gap": [-2.0, -1.0, 1.0, 2.0], "oversight": [-1.0, 0.0, 1.0, 2.0],
        "difficulty": [-1.0, 0.0, 1.0, 2.0], "release_gap_days": [-10, -5, 5, 10],
        "length_difference": [-2.0, -1.0, 1.0, 2.0], "correct": [0, 1, 0, 1],
        "visible_reasoning": [0, 0, 1, 1],
    })
    return pd.concat([
        observations.assign(draw=draw, predicted_choice=np.roll([0, 1, 0, 1], draw))
        for draw in range(3)
    ], ignore_index=True)


def test_posterior_replay_keeps_all_declared_slices_and_failed_reasons():
    replay = posterior_predictive_replay(_replay_frame())
    assert {
        "aggregate_choice_rate", "per_judge", "position_effect",
        "capability_gap", "oversight", "item_difficulty",
        "release_date_difference", "transcript_length_difference",
        "length_by_item_difficulty", "length_by_correctness",
        "length_by_visible_reasoning",
    } <= set(replay["check"])
    assert replay["reference"].eq("fitted_posterior").all()
    length_calibration = replay[replay["check"].str.startswith("length_by_")]
    assert set(length_calibration["predicted_mean"].round(1)) == {0.1, 0.2, 0.8, 0.9}
    assert np.allclose(
        length_calibration["calibration_error"],
        length_calibration["observed_choice_rate"] - length_calibration["predicted_mean"],
    )
    failed_input = _replay_frame().drop(columns="oversight")
    failed = posterior_predictive_replay(failed_input)
    row = failed[failed["check"].eq("oversight")].iloc[0]
    assert not row["valid"]
    assert row["reason"] == "missing column: oversight"


def test_configured_grid_compares_empirical_and_posterior_deterministically():
    grid = {
        "judge_counts": [1, 2], "anchor_counts": [2], "comparator_counts": [1],
        "attempted_item_counts": [10, 20],
    }

    def empirical(cell, rng):
        return pd.DataFrame({
            "replicate": [0, 1],
            "interval_width": rng.random(2),
            "ranking_recovery": [cell["J"] / 10, cell["attempted_items"] / 100],
        })

    def posterior(cell, rng):
        return pd.DataFrame({
            "replicate": [0, 1],
            "interval_width": rng.random(2),
            "ranking_recovery": [cell["M"] / 10, cell["N"] / 10],
        })

    first = compare_downsamples(grid, empirical, posterior, seed=7)
    second = compare_downsamples(grid, empirical, posterior, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["reference"]) == {"observed_full_data", "fitted_posterior"}
    assert set(first["metric"]) == {"interval_width", "ranking_recovery"}
    assert len(first) == 2 * 1 * 1 * 2 * 2 * 2 * 2


def test_empty_downsample_result_retains_both_references_and_metrics():
    grid = {
        "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
        "attempted_item_counts": [10],
    }

    def empty(_cell, _rng):
        return pd.DataFrame(columns=["interval_width", "ranking_recovery"])

    def posterior(_cell, _rng):
        return pd.DataFrame({"interval_width": [0.2], "ranking_recovery": [0.8]})

    got = compare_downsamples(grid, empty, posterior)

    assert set(zip(got["reference"], got["metric"])) == {
        ("observed_full_data", "interval_width"),
        ("observed_full_data", "ranking_recovery"),
        ("fitted_posterior", "interval_width"),
        ("fitted_posterior", "ranking_recovery"),
    }
    failed = got[got["reference"].eq("observed_full_data")]
    assert not failed["valid"].any()
    assert set(failed["reason"]) == {"downsample callback returned no rows"}
