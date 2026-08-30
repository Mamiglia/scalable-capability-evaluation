from pathlib import Path
from runpy import run_path

import pandas as pd
import pytest


module = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/canonical_analysis.py")
cap_pair_items = module["cap_pair_items"]
fit_binomial_logit = module["fit_binomial_logit"]
predict_from_ability = module["predict_from_ability"]
predict_block = module["predict_block"]


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
    assert first["predicted_low"] < first["predicted_median"] < first["predicted_high"]
    assert set(anchors.model) == set(models[:-1])


def test_bootstrap_marks_low_success_and_raises_on_zero_success():
    models, verdicts, hard = bootstrap_inputs({
        "a": [0, 0, 0, 1], "b": [0, 0, 1, 1], "c": [0, 1, 1, 1],
        "d": [1, 1, 1, 0], "frontier": [1, 1, 1, 1],
    }, q1_items=8)
    summary, _ = predict_block(verdicts, hard, "frontier", models[:-1], resamples=10, seed=6)
    assert summary["n_bootstrap"] == 7
    assert summary["bootstrap_unstable"]
    with pytest.raises(RuntimeError, match="every two-source item-bootstrap calibration failed"):
        predict_block(verdicts, hard, "frontier", models[:-1], resamples=1, seed=0)
