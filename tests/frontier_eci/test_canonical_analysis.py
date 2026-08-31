from pathlib import Path
from runpy import run_path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.frontier_eci.canonical_analysis import analyze_blocks_parallel


module = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/canonical_analysis.py")
analyze_blocks = module["analyze_blocks"]
analysis_tasks = module["_analysis_tasks"]
checkpoint_tables = module["_checkpoint_tables"]
clear_checkpoints = module["_clear_checkpoints"]
cap_pair_items = module["cap_pair_items"]
filter_protocol_verdicts = module["filter_protocol_verdicts"]
fit_binomial_logit = module["fit_binomial_logit"]
merge_rollout_sources = module["merge_rollout_sources"]
predict_from_ability = module["predict_from_ability"]
predict_block = module["predict_block"]
select_anchor_ids = module["select_anchor_ids"]
summarize_metrics = module["summarize_metrics"]


def counterbalanced_rows(items):
    rows = []
    for item in items:
        rows.extend([
            {"bench": "gpqa", "judge": "j", "item": str(item),
             "model_left": "a", "model_right": "b", "pick_left": 1},
            {"bench": "gpqa", "judge": "j", "item": str(item),
             "model_left": "b", "model_right": "a", "pick_left": 0},
        ])
    return pd.DataFrame(rows)


def test_pair_cap_keeps_both_orders_for_the_same_eight_items():
    verdicts = counterbalanced_rows(range(12))
    capped = cap_pair_items(verdicts, limit=8)
    assert capped.item.nunique() == 8
    assert capped.groupby("item").size().eq(2).all()


def test_pair_cap_is_invariant_to_input_order():
    verdicts = counterbalanced_rows(range(12))
    pd.testing.assert_frame_equal(cap_pair_items(verdicts),
                                  cap_pair_items(verdicts.sample(frac=1, random_state=7)))


def test_pair_cap_orders_repeated_judge_epochs_deterministically():
    verdicts = pd.concat([counterbalanced_rows(range(3)).assign(judge_epoch=epoch)
                          for epoch in (1, 2)], ignore_index=True)
    pd.testing.assert_frame_equal(cap_pair_items(verdicts),
                                  cap_pair_items(verdicts.sample(frac=1, random_state=7)))


def test_protocol_filter_keeps_only_q1_exact_epoch_both_correct_rows():
    verdicts = pd.DataFrame({
        "bench": ["gpqa"] * 3, "judge": ["j"] * 3,
        "item": ["q1", "q1", "q2"],
        "model_left": ["a", "a", "a"], "model_right": ["b", "c", "b"],
        "epoch_left": [0, 0, 0], "epoch_right": [0, 0, 0],
        "judge_epoch": [1, 1, 1], "pick_left": [1, 1, 1],
    })
    rollouts = pd.DataFrame({
        "bench": ["gpqa"] * 6, "model": ["a", "b", "c"] * 2,
        "item": ["q1"] * 3 + ["q2"] * 3, "epoch": [0] * 6,
        "correct": [True, True, False, True, True, True],
    })
    quartiles = pd.DataFrame({"bench": ["gpqa", "gpqa"],
                              "item": ["q1", "q2"], "item_quartile": [1, 2]})
    filtered = filter_protocol_verdicts(verdicts, rollouts, quartiles)
    assert filtered[["item", "model_left", "model_right"]].to_dict("records") == [
        {"item": "q1", "model_left": "a", "model_right": "b"}
    ]


def test_protocol_filter_normalizes_reviewed_transcript_aliases():
    verdicts = pd.DataFrame({
        "bench": ["gpqa"], "judge": ["j"], "item": ["q1"],
        "model_left": ["a"], "model_right": ["gemma3-12b-it"],
        "epoch_left": [0], "epoch_right": [0], "judge_epoch": [1], "pick_left": [1],
    })
    rollouts = pd.DataFrame({
        "bench": ["gpqa", "gpqa"], "model": ["a", "gemma-3-12b-it"],
        "item": ["q1", "q1"], "epoch": [0, 0], "correct": [True, True],
    })
    quartiles = pd.DataFrame({"bench": ["gpqa"], "item": ["q1"], "item_quartile": [1]})
    filtered = filter_protocol_verdicts(verdicts, rollouts, quartiles)
    assert filtered.iloc[0].model_right == "gemma-3-12b-it"


def test_actual_supplement_fills_only_missing_benchmark_models():
    def rows(model, correct):
        return pd.DataFrame({"bench": ["swe_bench_mini"], "model": [model],
                             "item": ["x"], "epoch": [0], "correct": [correct]})
    merged = merge_rollout_sources(
        rows("canonical", True).assign(is_canonical=True),
        rows("frontier", False), rows("historical", True),
        pd.concat([rows("canonical", False), rows("missing-target", True)]))
    assert merged.set_index("model").correct.to_dict() == {
        "canonical": True, "frontier": False,
        "historical": True, "missing-target": True,
    }


def test_anchor_selection_uses_all_connected_non_frontier_canonical_anchors():
    verdicts = pd.DataFrame({
        "model_left": ["target", "a", "b", "c", "target", "target"],
        "model_right": ["a", "b", "c", "d", "frontier", "historical"],
    })
    canonical, backfill = select_anchor_ids(
        verdicts, "target", ["a", "b", "c", "d", "frontier"],
        {"target", "frontier"}, ["historical"], minimum_anchors=4)
    assert canonical == ["a", "b", "c", "d"]
    assert backfill == []


def test_anchor_selection_adds_approved_backfill_only_below_four_canonical():
    verdicts = pd.DataFrame({
        "model_left": ["target", "a", "b", "target"],
        "model_right": ["a", "b", "c", "historical"],
    })
    canonical, backfill = select_anchor_ids(
        verdicts, "target", ["a", "b", "c"], {"target"},
        ["historical"], minimum_anchors=4)
    assert canonical == ["a", "b", "c"]
    assert backfill == ["historical"]


def test_binomial_logit_has_balanced_weighted_anchor_residuals():
    ability = pd.Series({"a": -1.5, "b": -0.5, "c": 0.5, "d": 1.5})
    outcomes = pd.DataFrame({"model": ability.index,
                             "successes": [2, 6, 14, 18], "total": [20] * 4})
    fit = fit_binomial_logit(ability, outcomes)
    assert fit.converged
    assert abs(fit.gradient_intercept) < 1e-6
    assert abs(fit.gradient_slope) < 1e-6
    assert (fit.anchor_table.residual > 0).any()
    assert (fit.anchor_table.residual < 0).any()


def test_binomial_logit_handles_large_offset_predictors():
    ability = pd.Series({"a": 100.0, "b": 110.0, "c": 120.0, "d": 130.0})
    outcomes = pd.DataFrame({"model": ability.index,
                             "successes": [2, 7, 13, 18], "total": [20] * 4})
    fit = fit_binomial_logit(ability, outcomes)
    assert abs(fit.gradient_intercept) < 1e-6
    assert abs(fit.gradient_slope) < 1e-6


def test_binomial_logit_rejects_fewer_than_four_anchors():
    outcomes = pd.DataFrame({"model": ["a", "b", "c"],
                             "successes": [2, 6, 14], "total": [20] * 3})
    with pytest.raises(ValueError, match="only 3 calibration anchors"):
        fit_binomial_logit(pd.Series({"a": -1.0, "b": 0.0, "c": 1.0}), outcomes)


@pytest.mark.parametrize(
    ("ability", "successes"),
    [
        (pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}), [2, 6, 14, 18]),
        (pd.Series({"a": -1.5, "b": -0.5, "c": 0.5, "d": 1.5}), [0, 0, 20, 20]),
    ],
)
def test_binomial_logit_rejects_unidentifiable_or_separated_fits(ability, successes):
    outcomes = pd.DataFrame({"model": ability.index, "successes": successes, "total": [20] * 4})
    with pytest.raises(RuntimeError, match="unidentifiable|separated"):
        fit_binomial_logit(ability, outcomes)


def test_target_hard_outcomes_never_change_prediction():
    ability = pd.Series({"a": -1.5, "b": -0.5, "c": 0.5, "d": 1.5,
                         "frontier": 2.0})
    outcomes = pd.DataFrame({"model": ["a", "b", "c", "d", "frontier"],
                             "successes": [2, 6, 14, 18, 0], "total": [20] * 5})
    first = predict_from_ability(ability, outcomes, "frontier", ["a", "b", "c", "d"])
    outcomes.loc[outcomes.model.eq("frontier"), "successes"] = 20
    second = predict_from_ability(ability, outcomes, "frontier", ["a", "b", "c", "d"])
    assert first.predicted == second.predicted
    assert first.intercept == second.intercept
    assert first.slope == second.slope
    assert first.actual == 0.0
    assert second.actual == 1.0
    assert first.residual == first.predicted
    assert second.residual == second.predicted - 1.0


@pytest.mark.parametrize("pair_limit", [
    pytest.param(8, id="headline-and-paired"),
    pytest.param(None, id="uncapped"),
])
def test_target_truth_and_item_availability_never_change_bootstrap(pair_limit):
    successes = {"a": 2, "b": 6, "c": 14, "d": 18, "frontier": 0}
    patterns = {model: [int(item < successes[model]) for item in range(20)]
                for model in successes}
    models, verdicts, hard = bootstrap_inputs(patterns)
    first, _ = predict_block(verdicts, hard, "frontier", models[:-1],
                             resamples=30, seed=7, pair_limit=pair_limit)
    changed = pd.concat([
        hard[hard.model.ne("frontier")],
        pd.DataFrame({"bench": ["gpqa"] * 7, "model": ["frontier"] * 7,
                      "item": [f"target-only-{item}" for item in range(7)],
                      "correct": [1] * 7}),
    ], ignore_index=True)
    second, _ = predict_block(verdicts, changed, "frontier", models[:-1],
                              resamples=30, seed=7, pair_limit=pair_limit)
    fit_fields = [
        "predicted", "predicted_low", "predicted_median", "predicted_high",
        "intercept", "slope", "n_bootstrap", "bootstrap_unstable",
    ]
    assert first["actual"] == 0.0 and second["actual"] == 1.0
    assert first["residual"] != second["residual"]
    assert {key: first[key] for key in fit_fields} == {key: second[key] for key in fit_fields}


def test_selected_anchor_missing_from_hard_outcomes_fails_instead_of_fitting_subset():
    anchors = ["a", "b", "c", "d", "e"]
    ability = pd.Series({**dict(zip(anchors, [-2, -1, 0, 1, 2])), "frontier": 2.5})
    outcomes = pd.DataFrame({"model": ["a", "b", "c", "d", "frontier"],
                             "successes": [2, 6, 14, 18, 0], "total": [20] * 5})
    with pytest.raises(ValueError, match="selected anchors unavailable.*outcomes=e"):
        predict_from_ability(ability, outcomes, "frontier", anchors)


def test_selected_anchor_missing_from_ability_fails_instead_of_fitting_subset():
    anchors = ["a", "b", "c", "d", "e"]
    ability = pd.Series({"a": -2.0, "b": -1.0, "c": 0.0, "d": 1.0,
                         "frontier": 2.5})
    outcomes = pd.DataFrame({"model": anchors,
                             "successes": [2, 6, 10, 14, 18], "total": [20] * 5})
    with pytest.raises(ValueError, match="selected anchors unavailable.*ability=e"):
        predict_from_ability(ability, outcomes, "frontier", anchors)


def test_fitted_anchor_rows_exactly_match_requested_ids_and_order():
    anchors = ["e", "c", "a", "d", "b"]
    ability = pd.Series({"a": -2.0, "b": -1.0, "c": 0.0, "d": 1.0,
                         "e": 2.0, "frontier": 2.5})
    outcomes = pd.DataFrame({"model": ["a", "b", "c", "d", "e"],
                             "successes": [2, 6, 10, 14, 18], "total": [20] * 5})
    fit = predict_from_ability(ability, outcomes, "frontier", anchors)
    assert fit.anchor_table.model.tolist() == anchors


def bootstrap_inputs(patterns, q1_items=10):
    models = ["a", "b", "c", "d", "frontier"]
    verdicts = []
    for item in range(q1_items):
        for weak, strong in zip(models, models[1:]):
            verdicts.extend([
                {"bench": "gpqa", "judge": "j", "item": str(item),
                 "model_left": strong, "model_right": weak, "pick_left": 1},
                {"bench": "gpqa", "judge": "j", "item": str(item),
                 "model_left": weak, "model_right": strong, "pick_left": 0},
            ])
    hard = pd.DataFrame([
        {"bench": "gpqa", "model": model, "item": str(item),
         "correct": correct}
        for model, outcomes in patterns.items() for item, correct in enumerate(outcomes)
    ])
    return models, pd.DataFrame(verdicts), hard


def test_two_source_bootstrap_is_seeded_and_exports_success_count():
    successes = {"a": 2, "b": 6, "c": 14, "d": 18, "frontier": 20}
    patterns = {model: [int(item < successes[model]) for item in range(20)]
                for model in successes}
    models, verdicts, hard = bootstrap_inputs(patterns)

    first, anchors = predict_block(verdicts, hard, "frontier",
                                   models[:-1], resamples=20, seed=7)
    second, _ = predict_block(verdicts, hard, "frontier",
                              models[:-1], resamples=20, seed=7)

    assert first == second
    assert first["n_bootstrap"] == 20
    assert not first["bootstrap_unstable"]
    assert first["actual"] == 1.0
    assert first["residual"] == first["predicted"] - 1.0
    assert not first["inside_anchor_range"]
    assert first["predicted_low"] < first["predicted_median"] < first["predicted_high"]
    assert set(anchors.model) == set(models[:-1])


def test_uncapped_sensitivity_keeps_all_compatible_pair_items():
    successes = {"a": 2, "b": 6, "c": 14, "d": 18, "frontier": 20}
    patterns = {model: [int(item < successes[model]) for item in range(20)]
                for model in successes}
    models, verdicts, hard = bootstrap_inputs(patterns, q1_items=10)
    capped, _ = predict_block(verdicts, hard, "frontier", models[:-1],
                              resamples=2, seed=7)
    uncapped, _ = predict_block(verdicts, hard, "frontier", models[:-1],
                                resamples=2, seed=7, pair_limit=None)
    assert capped["n_pair_items_max"] == 8
    assert uncapped["n_pair_items_max"] == 10


def test_metrics_report_overall_inside_and_outside_anchor_range():
    predictions = pd.DataFrame({
        "bench": ["gpqa"] * 4, "judge": ["j"] * 4,
        "target_quartile": [2] * 4, "model": ["a", "b", "c", "d"],
        "actual": [0.1, 0.3, 0.6, 0.9], "predicted": [0.2, 0.4, 0.5, 0.7],
        "inside_anchor_range": [True, True, False, False],
    })
    metrics = summarize_metrics(predictions)
    assert set(metrics.anchor_range) == {"overall", "inside", "outside"}
    assert metrics.set_index("anchor_range").n_models.to_dict() == {
        "overall": 4, "inside": 2, "outside": 2,
    }


def test_metrics_return_nan_without_warning_for_constant_inputs(recwarn):
    predictions = pd.DataFrame({
        "bench": ["gpqa"] * 2, "judge": ["j"] * 2,
        "target_quartile": [2] * 2, "actual": [1.0, 1.0],
        "predicted": [0.5, 0.5], "inside_anchor_range": [True, True],
    })
    metrics = summarize_metrics(predictions)
    assert pd.isna(metrics.loc[metrics.anchor_range.eq("overall"), "spearman"]).all()
    assert not recwarn.list


def test_block_analysis_exports_exact_anchors_and_frozen_missing_reason():
    successes = {"a": 2, "b": 6, "c": 14, "d": 18, "frontier": 20}
    patterns = {model: [int(item < successes[model]) for item in range(20)]
                for model in successes}
    _, verdicts, hard = bootstrap_inputs(patterns)
    hard["target_quartile"] = 2
    coverage = pd.DataFrame({
        "bench": ["gpqa", "gpqa"], "judge": ["j", "j"],
        "target": ["frontier", "missing"],
        "status": ["identifiable", "needs_historical_backfill"],
        "historical_backfill_anchors": ["[]", "[]"],
        "exclusion_reasons": ["[]", '["needs_historical_backfill"]'],
    })
    predictions, fits, residuals, exported = analyze_blocks(
        coverage, verdicts, hard, {"gpqa": ["a", "b", "c", "d"]},
        {"frontier", "missing"},
        {"a": "2024-01-01", "b": "2024-02-01", "c": "2024-03-01",
         "d": "2024-04-01", "frontier": "2025-01-01",
         "missing": "2025-01-01", "j": "2025-01-01"},
        resamples=2, seed=3, target_quartiles=(2,))
    assert exported.iloc[0].anchor_ids == "a|b|c|d"
    assert set(residuals.model) == {"a", "b", "c", "d"}
    assert fits.iloc[0].fit_status == "ok"
    assert fits.iloc[1].failure_reason == "coverage:needs_historical_backfill"
    missing = predictions[predictions.model.eq("missing")].iloc[0]
    assert missing.fit_status == "failed"
    assert missing.failure_reason == "coverage:needs_historical_backfill"
    paired, _, _, _ = analyze_blocks(
        coverage.iloc[:1], verdicts, hard, {"gpqa": ["a", "b", "c", "d"]},
        {"frontier"},
        {"a": "2024-01-01", "b": "2024-02-01", "c": "2024-03-01",
         "d": "2024-04-01", "frontier": "2025-01-01", "j": "2025-01-01"},
        resamples=2, seed=3, target_quartiles=(2,),
        anchor_overrides={"gpqa|j|frontier": ["a", "b", "c"]})
    assert predictions.iloc[0].anchor_ids == "a|b|c|d"
    assert paired.iloc[0].anchor_ids == "a|b|c"


def test_parallel_block_analysis_matches_serial_results():
    successes = {"a": 2, "b": 6, "c": 14, "d": 18, "frontier": 20}
    patterns = {model: [int(item < successes[model]) for item in range(20)]
                for model in successes}
    _, verdicts, hard = bootstrap_inputs(patterns)
    hard["target_quartile"] = 2
    coverage = pd.DataFrame({
        "bench": ["gpqa"], "judge": ["j"], "target": ["frontier"],
        "status": ["identifiable"], "historical_backfill_anchors": ["[]"],
        "exclusion_reasons": ["[]"],
    })
    args = (coverage, verdicts, hard, {"gpqa": ["a", "b", "c", "d"]},
            {"frontier"},
            {"a": "2024-01-01", "b": "2024-02-01", "c": "2024-03-01",
             "d": "2024-04-01", "frontier": "2025-01-01", "j": "2025-01-01"})
    serial = analyze_blocks(*args, resamples=2, seed=3, target_quartiles=(2,))
    parallel = analyze_blocks_parallel(*args, resamples=2, seed=3,
                                       target_quartiles=(2,), workers=2)
    for serial_table, parallel_table in zip(serial, parallel):
        pd.testing.assert_frame_equal(serial_table, parallel_table)


def test_analysis_tasks_are_single_block_quartile_in_stable_order():
    coverage = pd.DataFrame({"target": ["first", "second"]})
    tasks = analysis_tasks(coverage, (2, 3))
    assert [(task.iloc[0].target, quartile) for task, quartile in tasks] == [
        ("first", 2), ("first", 3), ("second", 2), ("second", 3),
    ]
    assert all(len(task) == 1 for task, _ in tasks)


def test_phase_checkpoint_replaces_csvs_before_writing_complete_marker(tmp_path):
    checkpoint_tables(tmp_path, "headline", {"predictions.csv": pd.DataFrame({"x": [1]})})
    checkpoint_tables(tmp_path, "headline", {"predictions.csv": pd.DataFrame({"x": [2]})})
    assert pd.read_csv(tmp_path / "predictions.csv").x.tolist() == [2]
    marker = pd.read_json(tmp_path / "checkpoint-headline.json", typ="series")
    assert marker["rows"] == {"predictions.csv": 1}
    assert not list(tmp_path.glob("*.tmp"))
    clear_checkpoints(tmp_path)
    assert (tmp_path / "predictions.csv").exists()
    assert not list(tmp_path.glob("checkpoint-*.json"))


def test_bootstrap_marks_low_success_and_raises_on_zero_success():
    models, verdicts, hard = bootstrap_inputs({
        "a": [0, 0, 0, 1], "b": [0, 0, 1, 1], "c": [0, 1, 1, 1],
        "d": [1, 1, 1, 0], "frontier": [1, 1, 1, 1],
    }, q1_items=8)
    summary, _ = predict_block(verdicts, hard, "frontier", models[:-1], resamples=10, seed=10)
    assert summary["n_bootstrap"] == 7
    assert summary["bootstrap_unstable"]
    with pytest.raises(RuntimeError, match="every two-source item-bootstrap calibration failed"):
        predict_block(verdicts, hard, "frontier", models[:-1], resamples=1, seed=3)
