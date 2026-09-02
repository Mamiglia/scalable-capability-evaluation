import numpy as np
import pandas as pd
import pytest


def test_exact_model_folds_do_not_group_families():
    from bias_crossfit import assign_exact_model_folds

    got = assign_exact_model_folds(
        ["qwen-low", "qwen-high", "other-a", "other-b", "other-c"],
        n_splits=5, seed=0,
    )

    assert got["model"].nunique() == 5
    assert got["fold"].nunique() == 5


def test_fold_frames_exclude_targets_globally_and_use_easy_target_rows_only():
    from bias_crossfit import build_fold_frames

    votes = pd.DataFrame([
        {"source": source, "item_id": item, "left_model": left,
         "right_model": right, "judge": "j", "chosen_left": 1.0,
         "benchmark": "gpqa"}
        for source in ("pilot", "hf")
        for item in ("easy", "hard")
        for left, right in (("target", "anchor-a"), ("anchor-a", "anchor-b"))
    ])
    folds = pd.DataFrame({"model": ["target", "anchor-a", "anchor-b"], "fold": [0, 1, 2]})
    splits = pd.DataFrame({"benchmark": ["gpqa", "gpqa"], "item": ["easy", "hard"],
                           "half": ["easy", "hard"]})
    anchors = pd.DataFrame({"model": ["anchor-a", "anchor-b"], "theta": [0.0, 1.0]})

    calibration, target = build_fold_frames(
        votes, anchors, splits, folds, fold=0, irt_model="fixed_3pl",
    )

    assert "target" not in set(calibration["left_model"]) | set(calibration["right_model"])
    assert set(target["item_id"]) == {"easy"}
    assert set(target["source"]) == {"pilot", "hf"}


def test_anchor_coverage_pools_different_judge_and_source_target_edges():
    from bias_crossfit import anchor_coverage

    rows = pd.DataFrame([
        {"source": "pilot", "judge": "j1", "left_model": "target", "right_model": "anchor-a"},
        {"source": "hf", "judge": "j2", "left_model": "target", "right_model": "anchor-b"},
    ])

    eligible = anchor_coverage(rows, {"anchor-a", "anchor-b"}).iloc[0]
    insufficient = anchor_coverage(rows.iloc[:1], {"anchor-a", "anchor-b"}).iloc[0]

    assert bool(eligible["eligible"])
    assert eligible["n_anchors"] == 2
    assert not bool(insufficient["eligible"])
    assert insufficient["reason"] == "fewer than 2 anchors"


def test_fold_frames_add_fixed_deltas_and_pooled_components():
    from bias_crossfit import build_fold_frames

    votes = pd.DataFrame([
        {"source": source, "item_id": "easy", "left_model": left,
         "right_model": right, "judge": judge, "chosen_left": 1.0,
         "benchmark": "gpqa"}
        for source in ("z-source", "a-source")
        for judge, left, right in (
            ("j1", "anchor-a", "anchor-b"),
            ("j1", "target", "anchor-a"),
            ("j2", "target", "anchor-b"),
        )
    ])
    folds = pd.DataFrame({"model": ["target", "anchor-a", "anchor-b"], "fold": [0, 1, 2]})
    splits = pd.DataFrame({"benchmark": ["gpqa"], "item": ["easy"], "half": ["easy"]})
    anchors = pd.DataFrame({"model": ["anchor-a", "anchor-b"], "theta": [0.0, 1.5]})

    calibration, target = build_fold_frames(
        votes, anchors, splits, folds, fold=0, irt_model="fixed_3pl",
    )

    assert set(calibration["delta_theta"]) == {-1.5}
    assert set(calibration["component"]) == {"pooled:0"}
    assert set(target["component"]) == {"pooled:0"}


def test_fold_frames_exclude_hard_calibration_rows():
    from bias_crossfit import build_fold_frames

    votes = pd.DataFrame([
        {"source": "pilot", "benchmark": "gpqa", "item_id": item,
         "left_model": "anchor-a", "right_model": "anchor-b", "judge": "j",
         "chosen_left": 1.0}
        for item in ("easy", "hard")
    ])
    folds = pd.DataFrame({"model": ["target", "anchor-a", "anchor-b"], "fold": [0, 1, 2]})
    splits = pd.DataFrame({"benchmark": ["gpqa", "gpqa"], "item": ["easy", "hard"],
                           "half": ["easy", "hard"]})
    anchors = pd.DataFrame({"model": ["anchor-a", "anchor-b"], "theta": [0.0, 1.0]})

    calibration, _ = build_fold_frames(
        votes, anchors, splits, folds, fold=0, irt_model="fixed_3pl",
    )

    assert set(calibration["item_id"]) == {"easy"}


def test_fold_frames_require_benchmark_and_discard_non_gpqa_rows():
    from bias_crossfit import build_fold_frames

    folds = pd.DataFrame({"model": ["target", "anchor-a", "anchor-b"], "fold": [0, 1, 2]})
    splits = pd.DataFrame({"benchmark": ["gpqa"], "item": ["shared"], "half": ["easy"]})
    anchors = pd.DataFrame({"model": ["anchor-a", "anchor-b"], "theta": [0.0, 1.0]})
    benchmarkless = pd.DataFrame([{
        "source": "pilot", "item_id": "shared", "left_model": "anchor-a",
        "right_model": "anchor-b", "judge": "j", "chosen_left": 1.0,
    }])
    votes = pd.concat([
        benchmarkless.assign(benchmark="other"),
        benchmarkless.assign(benchmark="gpqa"),
    ], ignore_index=True)

    with pytest.raises(ValueError, match="benchmark"):
        build_fold_frames(
            benchmarkless, anchors, splits, folds, fold=0,
            irt_model="fixed_3pl",
        )
    calibration, _ = build_fold_frames(
        votes, anchors, splits, folds, fold=0, irt_model="fixed_3pl",
    )

    assert set(calibration["benchmark"]) == {"gpqa"}


def test_fold_frames_filter_mixed_artifacts_to_requested_irt_form():
    from bias_crossfit import build_fold_frames

    votes = pd.DataFrame([
        {"source": "s", "benchmark": "gpqa", "item_id": "shared",
         "left_model": left, "right_model": right, "judge": "j",
         "chosen_left": 1.0}
        for left, right in (("target", "a"), ("target", "b"), ("a", "b"))
    ])
    folds = pd.DataFrame({"model": ["target", "a", "b"], "fold": [0, 1, 2]})
    splits = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
         "item": "shared", "half": "easy"},
        {"irt_model": "2pl", "fit": "gpqa", "benchmark": "gpqa",
         "item": "shared", "half": "hard"},
    ])
    anchors = pd.DataFrame([
        {"irt_model": form, "fit": "gpqa", "model": model, "theta": theta}
        for form, values in (("fixed_3pl", (0.0, 1.0)), ("2pl", (10.0, 20.0)))
        for model, theta in zip(("a", "b"), values)
    ])

    calibration, target = build_fold_frames(
        votes, anchors, splits, folds, fold=0, irt_model="fixed_3pl",
    )

    assert set(calibration["delta_theta"]) == {-1.0}
    assert set(calibration["half"]) == {"easy"}
    assert set(target["half"]) == {"easy"}


def test_run_fold_hides_target_outcome_until_after_inference():
    from bias_crossfit import run_fold

    seen = {}
    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": 1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
        "benchmark": "gpqa", "half": "easy",
    }, {
        "left_model": "other-a", "right_model": "other-b", "delta_theta": 9.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "benchmark": "other", "half": "easy",
    }])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "easy",
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
        "benchmark": "gpqa", "half": "easy",
    }, {
        "left_model": "other-held", "right_model": "other-a", "item_id": "easy",
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
        "benchmark": "other", "half": "easy",
    }])
    hard = pd.DataFrame({
        "model": ["a", "b", "held"], "hard_accuracy": [0.8, 0.4, 0.9],
    })

    def fake_fit(frame, **_kwargs):
        seen["fit_models"] = set(frame["left_model"]) | set(frame["right_model"])
        return {
            "judges": ["j"], "sources": ["s"], "conditions": ["c"],
            "features": [], "feature_mean": np.array([]), "feature_scale": np.array([]),
            "alpha": np.zeros((1, 1)), "source": np.zeros((1, 1, 1)),
            "condition": np.zeros((1, 1, 1)), "discrimination": np.ones((1, 1)),
            "beta": np.zeros((1, 1, 0)),
        }

    def fake_infer(frame, anchors, fit, **_kwargs):
        assert "hard_accuracy" not in frame
        seen["target_items"] = set(frame["item_id"])
        return pd.DataFrame({
            "model": ["held"], "theta": [1.2],
            "theta_p5": [0.8], "theta_p95": [1.6],
        })

    result = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=pd.DataFrame({"model": ["a", "b"], "theta": [1.0, 0.0]}),
        hard_accuracy=hard, fit_calibration_fn=fake_fit,
        infer_targets_fn=fake_infer,
    )
    got = result["predictions"]

    assert seen == {"fit_models": {"a", "b"}, "target_items": {"easy"}}
    assert set(got["method"]) == {
        "corrected", "uncorrected", "raw_elo", "easy_accuracy", "mean",
    }
    assert set(got["model"]) == {"held"}
    assert np.isclose(
        got.set_index("method").loc["corrected", "prediction"], 0.88,
    )
    assert np.isclose(got.set_index("method").loc["mean", "prediction"], 0.6)
    assert set(result["calibration"]["method"]) == {"corrected", "uncorrected"}


def _ablation_fixtures():
    calibration = pd.DataFrame([
        {
            "left_model": left, "right_model": right, "delta_theta": -1.0,
            "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
            "delta_style": style, "delta_length": length, "half": "easy",
        }
        for left, right, style, length in (
            ("a", "b", -1.0, 0.5), ("b", "c", 1.0, -0.5),
        )
    ])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "easy",
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
        "delta_style": 2.0, "delta_length": 1.0, "half": "easy",
    }])
    anchors = pd.DataFrame({"model": ["a", "b", "c"], "theta": [0.0, 1.0, 2.0]})
    hard = pd.DataFrame({
        "model": ["a", "b", "c", "held"],
        "hard_accuracy": [0.2, 0.5, 0.8, 0.7],
    })

    def fake_fit(_frame, feature_columns, **_kwargs):
        features = list(feature_columns)
        return {"features": features, "beta": np.ones((2, 1, len(features)))}

    def fake_infer(_frame, _anchors, fit, **_kwargs):
        theta = float(fit["beta"].sum())
        return pd.DataFrame({
            "model": ["held"], "theta": [theta],
            "theta_p5": [theta - 0.1], "theta_p95": [theta + 0.1],
        })

    return calibration, target, anchors, hard, fake_fit, fake_infer


def test_run_fold_ablates_feature_groups_from_the_corrected_fit():
    from bias_crossfit import run_fold

    calibration, target, anchors, hard, fake_fit, fake_infer = (
        _ablation_fixtures()
    )
    got = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=anchors, hard_accuracy=hard,
        feature_columns=("delta_style", "delta_length"), arm="corrected",
        feature_ablations={"presentation": ["delta_style", "delta_length"]},
        fit_calibration_fn=fake_fit, infer_targets_fn=fake_infer,
    )["predictions"].set_index("method")

    assert got.loc["corrected", "predictor"] == 4.0
    assert got.loc["corrected_without_presentation", "predictor"] == 0.0


def test_run_fold_rejects_ablation_features_missing_from_the_fit():
    from bias_crossfit import run_fold

    calibration, target, anchors, hard, fake_fit, fake_infer = (
        _ablation_fixtures()
    )
    with pytest.raises(ValueError, match="not in fit.*delta_markdown"):
        run_fold(
            fold=0, calibration_frame=calibration, target_frame=target,
            anchors=anchors, hard_accuracy=hard,
            feature_columns=("delta_style",), arm="corrected",
            feature_ablations={"presentation": ["delta_markdown"]},
            fit_calibration_fn=fake_fit, infer_targets_fn=fake_infer,
        )


def test_run_fold_executes_only_the_requested_arm():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c",
        "chosen_left": 0.0, "delta_style": 1.0, "half": "easy",
    }])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "easy",
        "judge": "j", "source": "s", "condition": "c",
        "chosen_left": 1.0, "delta_style": 1.0, "half": "easy",
    }])
    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]})
    hard = pd.DataFrame({
        "model": ["a", "b", "held"], "hard_accuracy": [0.2, 0.6, 0.4],
    })
    fitted = []

    def fake_fit(_frame, feature_columns, **_kwargs):
        features = list(feature_columns)
        fitted.append(features)
        return {"features": features, "beta": np.zeros((1, 1, len(features)))}

    def fake_infer(*_args, **_kwargs):
        return pd.DataFrame({
            "model": ["held"], "theta": [0.5],
            "theta_p5": [0.4], "theta_p95": [0.6],
        })

    got = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=anchors, hard_accuracy=hard, feature_columns=("delta_style",),
        arm="uncorrected", fit_calibration_fn=fake_fit,
        infer_targets_fn=fake_infer,
    )

    assert fitted == [[]]
    assert set(got["predictions"]["method"]) == {"uncorrected"}
    assert got["coefficients"].empty


def test_run_fold_labels_and_forwards_each_production_inference():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c",
        "chosen_left": 0.0, "delta_style": 1.0, "half": "easy",
    }])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "easy",
        "judge": "j", "source": "s", "condition": "c",
        "chosen_left": 1.0, "delta_style": 1.0, "half": "easy",
    }])
    seen = []

    def fake_fit(_frame, feature_columns, **_kwargs):
        features = list(feature_columns)
        return {"features": features, "beta": np.ones((1, 1, len(features)))}

    def fake_infer(*_args, inference_name, chains, collect_diagnostics, **_kwargs):
        seen.append((inference_name, chains, collect_diagnostics))
        return pd.DataFrame({
            "model": ["held"], "theta": [0.5],
            "theta_p5": [0.4], "theta_p95": [0.6],
        })

    run_fold(
        fold=2, calibration_frame=calibration, target_frame=target,
        anchors=pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]}),
        hard_accuracy=pd.DataFrame({
            "model": ["a", "b", "held"], "hard_accuracy": [0.2, 0.6, 0.4],
        }),
        feature_columns=("delta_style",), arm="corrected",
        inference_options={"chains": 4, "collect_diagnostics": True},
        fit_calibration_fn=fake_fit, infer_targets_fn=fake_infer,
    )

    assert seen == [
        ("corrected", 4, True),
    ]


def test_run_fold_keeps_one_anchor_components_in_coverage_only():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "half": "easy",
    }])
    target = pd.DataFrame([
        {
            "left_model": target_model, "right_model": anchor, "item_id": "easy",
            "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
            "half": "easy",
        }
        for target_model, anchor in (("eligible", "a"), ("eligible", "b"), ("excluded", "c"))
    ])
    anchors = pd.DataFrame({"model": ["a", "b", "c"], "theta": [0.0, 1.0, 2.0]})
    hard = pd.DataFrame({
        "model": ["a", "b", "c", "eligible", "excluded"],
        "hard_accuracy": [0.2, 0.5, 0.8, 0.6, 0.7],
    })
    easy = pd.DataFrame({
        "model": ["a", "b", "c", "eligible", "excluded"],
        "easy_accuracy": [0.3, 0.5, 0.7, 0.6, 0.6],
    })

    def fake_fit(_frame, feature_columns, **_kwargs):
        return {"features": list(feature_columns), "beta": np.zeros((1, 1, 0))}

    def fake_infer(*_args, **_kwargs):
        return pd.DataFrame({
            "model": ["eligible", "excluded"], "theta": [1.2, 1.4],
            "theta_p5": [1.0, 1.2], "theta_p95": [1.4, 1.6],
        })

    result = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=anchors, hard_accuracy=hard, easy_accuracy=easy,
        fit_calibration_fn=fake_fit, infer_targets_fn=fake_infer,
    )
    coverage = result["coverage"].set_index("model")

    assert set(result["predictions"]["model"]) == {"eligible"}
    assert coverage.loc["eligible", "n_anchors"] == 2
    assert bool(coverage.loc["eligible", "eligible"])
    assert coverage.loc["excluded", "n_anchors"] == 1
    assert not bool(coverage.loc["excluded", "eligible"])
    assert coverage.loc["excluded", "reason"] == "fewer than 2 anchors"


def test_raw_elo_counterbalances_expanded_presentation_orders():
    from bias_crossfit import _fold_coverage, _raw_elo_predictors

    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]})
    shared = {"benchmark": "gpqa", "source": "s", "condition": "c", "judge": "j"}
    fixed = pd.DataFrame([
        {**shared, "item_id": "anchor", "left_model": "a", "right_model": "b",
         "chosen_left": 0.0},
        {**shared, "item_id": "other", "left_model": "target", "right_model": "b",
         "chosen_left": 0.0},
    ])
    expanded = pd.concat([fixed, pd.DataFrame([
        {**shared, "item_id": "orders", "left_model": "target", "right_model": "a",
         "chosen_left": 1.0},
        {**shared, "item_id": "orders", "left_model": "a", "right_model": "target",
         "chosen_left": 1.0},
    ])], ignore_index=True)
    collapsed = pd.concat([fixed, pd.DataFrame([{
        **shared, "item_id": "orders", "left_model": "a", "right_model": "target",
        "chosen_left": 0.5,
    }])], ignore_index=True)

    def predictor(rows):
        coverage = _fold_coverage(rows, anchors, {"target"}, 0, "fixed_3pl")
        return _raw_elo_predictors(rows, anchors, coverage).set_index("model").loc[
            "target", "raw_elo"
        ]

    assert np.isclose(predictor(expanded), predictor(collapsed))


def test_run_fold_canonicalizes_source_priority_and_epoch_for_easy_accuracy():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "half": "easy",
    }])
    target = pd.DataFrame([
        {"left_model": "held", "right_model": anchor, "item_id": item,
         "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
         "half": "easy"}
        for anchor, item in (("a", "e1"), ("b", "e2"))
    ])
    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]})
    hard = pd.DataFrame({
        "model": ["a", "b", "held"], "hard_accuracy": [0.2, 0.6, 0.5],
    })
    responses = pd.DataFrame([
        {"benchmark": "gpqa", "model": model, "item": item, "epoch": 1,
         "answer": "A", "correct": correct, "source_priority": 2}
        for model, values in (("a", (0, 1)), ("b", (1, 1)))
        for item, correct in zip(("e1", "e2"), values)
    ] + [
        {"benchmark": "gpqa", "model": "held", "item": "e1", "epoch": 1,
         "answer": "A", "correct": False, "source_priority": 0},
        {"benchmark": "gpqa", "model": "held", "item": "e1", "epoch": 2,
         "answer": "A", "correct": True, "source_priority": 2},
        {"benchmark": "gpqa", "model": "held", "item": "e1", "epoch": 3,
         "answer": "A", "correct": False, "source_priority": 2},
        {"benchmark": "gpqa", "model": "held", "item": "e2", "epoch": 1,
         "answer": "A", "correct": False, "source_priority": 2},
        {"benchmark": "gpqa", "model": "held", "item": "hard", "epoch": 1,
         "answer": "A", "correct": True, "source_priority": 2},
    ])

    def fake_fit(_frame, feature_columns, **_kwargs):
        return {"features": list(feature_columns), "beta": np.zeros((1, 1, 0))}

    def fake_infer(*_args, **_kwargs):
        return pd.DataFrame({
            "model": ["held"], "theta": [0.5],
            "theta_p5": [0.4], "theta_p95": [0.6],
        })

    got = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=anchors, hard_accuracy=hard, responses=responses,
        easy_items=["e1", "e2"], fit_calibration_fn=fake_fit,
        infer_targets_fn=fake_infer,
    )["predictions"].query("method == 'easy_accuracy'").iloc[0]

    assert got["predictor"] == 0.5


def test_run_fold_uses_only_the_requested_irt_form():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -10.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "half": "easy",
    }])
    target = pd.DataFrame([
        {"left_model": "held", "right_model": anchor, "item_id": item,
         "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
         "half": "easy"}
        for anchor, item in (("a", "e1"), ("b", "e2"))
    ])
    anchors = pd.DataFrame([
        {"irt_model": form, "model": model, "theta": theta}
        for form, values in (("fixed_3pl", (0.0, 1.0)), ("2pl", (10.0, 20.0)))
        for model, theta in zip(("a", "b"), values)
    ])
    hard = pd.DataFrame([
        {"irt_model": form, "model": model, "hard_accuracy": accuracy}
        for form in ("fixed_3pl", "2pl")
        for model, accuracy in (("a", 0.2), ("b", 0.6), ("held", 0.4))
    ])
    splits = pd.DataFrame([
        {"irt_model": "fixed_3pl", "benchmark": "gpqa", "item": "e1", "half": "easy"},
        {"irt_model": "2pl", "benchmark": "gpqa", "item": "e2", "half": "easy"},
    ])
    responses = pd.DataFrame([
        {"benchmark": "gpqa", "model": model, "item": item, "answer": "A",
         "correct": correct, "epoch": 1, "source_priority": 2}
        for model, results in (("a", (1, 0)), ("b", (1, 1)), ("held", (1, 0)))
        for item, correct in zip(("e1", "e2"), results)
    ])

    def fake_fit(_frame, feature_columns, **_kwargs):
        return {"features": list(feature_columns), "beta": np.zeros((1, 1, 0))}

    def fake_infer(_frame, seen_anchors, _fit, **_kwargs):
        assert set(seen_anchors["irt_model"]) == {"2pl"}
        return pd.DataFrame({
            "model": ["held"], "theta": [15.0],
            "theta_p5": [14.0], "theta_p95": [16.0],
        })

    got = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=anchors, hard_accuracy=hard, responses=responses,
        easy_items=splits, irt_model="2pl", fit_calibration_fn=fake_fit,
        infer_targets_fn=fake_infer,
    )["predictions"].set_index("method")

    assert np.isclose(got.loc["corrected", "prediction"], 0.4)
    assert got.loc["easy_accuracy", "predictor"] == 0.0


def test_run_fold_requires_easy_items_with_canonical_responses():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "half": "easy",
    }])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "e1",
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
        "half": "easy",
    }])
    responses = pd.DataFrame([{
        "benchmark": "gpqa", "model": "held", "item": "e1", "epoch": 1,
        "answer": "A", "correct": True, "source_priority": 1,
    }])

    def forbid_fit(*_args, **_kwargs):
        raise AssertionError("fit must not run without an explicit easy bank")

    with pytest.raises(ValueError, match="easy_items"):
        run_fold(
            fold=0, calibration_frame=calibration, target_frame=target,
            anchors=pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]}),
            hard_accuracy=pd.DataFrame({
                "model": ["a", "b", "held"], "hard_accuracy": [0.2, 0.6, 0.4],
            }),
            responses=responses, fit_calibration_fn=forbid_fit,
        )


def test_run_fold_rejects_hard_judgment_rows_before_fit():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "half": "hard",
    }])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "e1",
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
        "half": "easy",
    }])

    def forbid_fit(*_args, **_kwargs):
        raise AssertionError("fit must not run on hard judging evidence")

    with pytest.raises(ValueError, match="easy"):
        run_fold(
            fold=0, calibration_frame=calibration, target_frame=target,
            anchors=pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]}),
            hard_accuracy=pd.DataFrame({
                "model": ["a", "b", "held"], "hard_accuracy": [0.2, 0.6, 0.4],
            }),
            fit_calibration_fn=forbid_fit,
        )


def test_posterior_predictive_summary_is_grouped_by_judge_and_source():
    from bias_crossfit import posterior_predictive_summary

    frame = pd.DataFrame([
        {"judge": "j1", "source": "s1", "condition": "c", "delta_theta": 0.0,
         "chosen_left": observed}
        for observed in (0.0, 1.0)
    ] + [{
        "judge": "j2", "source": "s2", "condition": "c", "delta_theta": 0.0,
        "chosen_left": 0.0,
    }])
    fit = {
        "judges": ["j1", "j2"], "sources": ["s1", "s2"], "conditions": ["c"],
        "features": [], "feature_mean": np.array([]), "feature_scale": np.array([]),
        "alpha": np.array([[0.0, np.log(99.0)], [0.0, np.log(99.0)]]),
        "source": np.zeros((2, 2, 2)), "condition": np.zeros((2, 2, 1)),
        "discrimination": np.zeros((2, 2)), "beta": np.zeros((2, 2, 0)),
    }

    got = posterior_predictive_summary(frame, fit).set_index(["judge", "source"])

    assert got.loc[("j1", "s1"), "observed_rate"] == 0.5
    assert got.loc[("j1", "s1"), "predicted_rate"] == 0.5
    assert got.loc[("j1", "s1"), "brier"] == 0.25
    assert got.loc[("j1", "s1"), "interval_coverage_90"] == 1.0
    assert got.loc[("j2", "s2"), "interval_coverage_90"] == 0.0


def test_reduce_folds_concatenates_each_artifact_with_stable_empty_tables():
    from bias_crossfit import PREDICTION_COLUMNS, reduce_folds

    row = {column: np.nan for column in PREDICTION_COLUMNS}
    row.update({"benchmark": "gpqa", "fold": 0, "method": "mean", "model": "a"})
    got = reduce_folds([
        {"predictions": pd.DataFrame([row])},
        {"predictions": pd.DataFrame([{**row, "fold": 1, "model": "b"}])},
    ])

    assert got["predictions"]["model"].tolist() == ["a", "b"]
    assert list(got["predictions"].columns) == PREDICTION_COLUMNS
    assert got["coverage"].empty
    assert got["calibration"].empty


def test_run_fold_returns_exact_empty_prediction_schema_without_targets():
    from bias_crossfit import PREDICTION_COLUMNS, run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 0.0,
        "half": "easy",
    }])
    target = pd.DataFrame(columns=[
        "left_model", "right_model", "item_id", "judge", "source",
        "condition", "chosen_left", "half",
    ])

    def fake_fit(_frame, feature_columns, **_kwargs):
        return {"features": list(feature_columns), "beta": np.zeros((1, 1, 0))}

    def fake_infer(*_args, **_kwargs):
        return pd.DataFrame(columns=["model", "theta", "theta_p5", "theta_p95"])

    got = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]}),
        hard_accuracy=pd.DataFrame({
            "model": ["a", "b"], "hard_accuracy": [0.2, 0.6],
        }),
        fit_calibration_fn=fake_fit, infer_targets_fn=fake_infer,
    )["predictions"]

    assert got.empty
    assert list(got.columns) == PREDICTION_COLUMNS


def test_judge_ordering_uses_raw_answer_and_within_gpqa_scores_not_inspect_value(
    tmp_path,
):
    from types import SimpleNamespace as NS
    from bias_crossfit import judge_ordering_rows
    from inputs import load_hf_votes

    manifest = tmp_path / "raw" / "pilot-j-gpqa-a-vs-b"
    manifest.mkdir(parents=True)
    (manifest / "logs.json").write_text(
        '{"run.eval": {"status": "success"}}'
    )
    eval_path = tmp_path / "evals" / "run.eval"
    eval_path.parent.mkdir()
    eval_path.touch()
    log = NS(
        status="success",
        eval=NS(
            metadata={
                "major_path": "data/gpqa/hash/a/0/a.eval",
                "minor_path": "data/gpqa/hash/b/0/b.eval",
            },
            model="judge",
        ),
        samples=[NS(id="item_1", scores={
            "judge": NS(answer="1", value="I", correctness=False),
        })],
    )
    votes = load_hf_votes(
        tmp_path / "raw", tmp_path / "evals", reader=lambda _: log,
    ).assign(inspect_correctness=False, major="b", minor="a")

    got = judge_ordering_rows(
        votes,
        pd.DataFrame({
            "model": ["a", "b"], "hard_accuracy": [0.9, 0.2],
        }),
    ).iloc[0]

    assert got["chosen_left"] == 1.0
    assert bool(got["left_is_stronger"])
    assert got["judge_ordering_correct"] == 1


def test_run_fold_filters_unseen_target_evidence_and_audits_excluded_model():
    from bias_crossfit import run_fold

    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": -1.0,
        "judge": "j", "source": "s", "condition": "c",
        "chosen_left": 0.0, "benchmark": "gpqa", "half": "easy",
        "item_id": "cal", "round": "r1",
    }])
    target = pd.DataFrame([
        {"left_model": "kept", "right_model": anchor, "item_id": item,
         "judge": "j", "source": "s", "condition": "c",
         "chosen_left": 1.0, "benchmark": "gpqa", "half": "easy",
         "round": "r1"}
        for anchor, item in (("a", "e1"), ("b", "e2"))
    ] + [{
        "left_model": "excluded", "right_model": "a", "item_id": "e3",
        "judge": "j", "source": "unseen", "condition": "c",
        "chosen_left": 1.0, "benchmark": "gpqa", "half": "easy",
        "round": "r2",
    }])

    def fake_fit(_frame, feature_columns, **_kwargs):
        return {
            "judges": ["j"], "sources": ["s"], "conditions": ["c"],
            "features": list(feature_columns),
            "beta": np.zeros((1, 1, len(feature_columns))),
        }

    def fake_infer(frame, _anchors, _fit, **_kwargs):
        assert set(frame["source"]) == {"s"}
        assert set(frame["left_model"]) == {"kept"}
        return pd.DataFrame({
            "model": ["kept"], "theta": [0.5],
            "theta_p5": [0.4], "theta_p95": [0.6],
        })

    result = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]}),
        hard_accuracy=pd.DataFrame({
            "model": ["a", "b", "kept", "excluded"],
            "hard_accuracy": [0.2, 0.6, 0.5, 0.4],
        }),
        arm="corrected", fit_calibration_fn=fake_fit,
        infer_targets_fn=fake_infer,
        inference_options={"target_models": {"kept", "excluded"}},
    )
    coverage = result["coverage"].set_index("model")

    assert set(result["predictions"]["model"]) == {"kept"}
    assert result["target_exclusions"].to_dict("records") == [{
        "model": "excluded", "reason": "unseen source: unseen", "count": 1,
    }]
    assert coverage.loc["kept", [
        "n_evidence_remaining", "n_evidence_excluded",
    ]].tolist() == [2, 0]
    assert coverage.loc["excluded", [
        "n_evidence_remaining", "n_evidence_excluded",
    ]].tolist() == [0, 1]


def test_raw_elo_counterbalances_repeated_items_within_round_and_counts_rounds():
    from bias_crossfit import _fold_coverage, _raw_elo_predictors

    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]})
    shared = {
        "benchmark": "gpqa", "source": "s", "condition": "c", "judge": "j",
    }
    repeated = pd.DataFrame([
        {**shared, "round": "r1", "item_id": "anchor", "left_model": "a",
         "right_model": "b", "chosen_left": 0.0},
        {**shared, "round": "r1", "item_id": "bridge", "left_model": "target",
         "right_model": "b", "chosen_left": 0.0},
        {**shared, "round": "r1", "item_id": "same", "left_model": "target",
         "right_model": "a", "chosen_left": 1.0},
        {**shared, "round": "r2", "item_id": "same", "left_model": "target",
         "right_model": "a", "chosen_left": 0.0},
        {**shared, "round": "r2", "item_id": "same", "left_model": "target",
         "right_model": "a", "chosen_left": 0.0},
    ])
    separated = repeated.copy()
    separated.loc[separated["round"].eq("r1") & separated["item_id"].eq("same"), "item_id"] = "same-r1"
    separated.loc[separated["round"].eq("r2") & separated["item_id"].eq("same"), "item_id"] = "same-r2"

    def predictor(rows):
        coverage = _fold_coverage(rows, anchors, {"target"}, 0, "fixed_3pl")
        value = _raw_elo_predictors(rows, anchors, coverage).set_index("model").loc[
            "target", "raw_elo"
        ]
        return value, coverage.set_index("model").loc["target", "n_rounds"]

    repeated_value, n_rounds = predictor(repeated)
    separated_value, _ = predictor(separated)

    assert np.isclose(repeated_value, separated_value)
    assert n_rounds == 2


def test_build_fold_frames_respects_predictor_bin():
    from bias_crossfit import build_fold_frames

    votes = pd.DataFrame([
        {
            "benchmark": "gpqa", "item_id": item, "judge": "j", "source": "s",
            "condition": "c", "chosen_left": 1.0, "left_model": left,
            "right_model": right, "half": "unused",
        }
        for item, left, right in (
            ("i1", "a", "b"), ("i2", "a", "b"), ("i1", "a", "held"),
            ("i2", "b", "held"),
        )
    ])
    splits = pd.DataFrame({
        "benchmark": "gpqa", "item": ["i1", "i2"], "tertile": ["T1", "T2"],
    })
    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]})
    folds = pd.DataFrame({"model": ["held"], "fold": [0]})

    calibration, target = build_fold_frames(
        votes.drop(columns="half"), anchors, splits, folds, 0, "fixed_3pl",
        bin_column="tertile", bin_value="T1",
    )

    assert set(calibration["item_id"]) == {"i1"}
    assert set(target["item_id"]) == {"i1"}
    assert (calibration["tertile"] == "T1").all()
