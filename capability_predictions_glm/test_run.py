from dataclasses import replace

import pandas as pd

from capability_predictions_glm.config import CONFIG
from capability_predictions_glm.data import InputTables, feature_variants
from capability_predictions_glm.run import load_report, run_analysis, write_results


def _inputs() -> InputTables:
    models = [f"m{i}" for i in range(6)]
    items = [f"i{i:02d}" for i in range(12)]
    # Solve rates decrease by quartile while retaining within-quartile model variation.
    thresholds = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    responses = pd.DataFrame([
        {"bench": "b", "model": model, "item": item,
         "correct": int(model_index >= thresholds[item_index]), "is_canonical": True}
        for item_index, item in enumerate(items)
        for model_index, model in enumerate(models)
    ])
    transcripts = pd.DataFrame([
        {"bench": "b", "model": model, "item": item, "epoch": 0, "log": "l",
         "signal": float(model_index), "quality": float(model_index >= 3)}
        for item_index, item in enumerate(items[:3])
        for model_index, model in enumerate(models)
        if model_index >= thresholds[item_index]
    ])
    pairs = []
    for item_index, item in enumerate(items[:3]):
        eligible = [i for i in range(6) if i >= thresholds[item_index]]
        pair_index = 0
        for low_position, low in enumerate(eligible):
            for high in eligible[low_position + 1:]:
                ordered = (low, high) if pair_index % 2 == 0 else (high, low)
                pair_index += 1
                for left, right in (ordered, ordered[::-1]):
                    pairs.append({
                        "bench": "b", "judge": "judge-1", "item": item,
                        "model_left": f"m{left}", "model_right": f"m{right}",
                        "epoch_left": 0, "epoch_right": 0, "log_left": "l", "log_right": "l",
                        "pick_left": int(left > right),
                        "d_signal": float(left - right),
                        "d_quality": float((left >= 3) - (right >= 3)),
                    })
    dictionary = pd.DataFrame({
        "feature": ["signal", "quality"], "block": ["surface", "rubric"]})
    return InputTables({}, dictionary, feature_variants(dictionary), responses,
                       transcripts, pd.DataFrame(pairs))


def test_run_analysis_compares_both_glms_and_judge_on_q2_to_q4() -> None:
    config = replace(
        CONFIG,
        model_folds=(("fold", ("m5",)),),
        direct_elastic_alphas=(0.001,), pair_elastic_cs=(1.0,), elastic_l1_ratios=(0.5,),
        ridge_alphas=(1.0,), logistic_cs=(1.0,), inner_folds=2,
    )

    result = run_analysis(_inputs(), config)

    assert set(result.predictions.target_quartile) == {"Q2", "Q3", "Q4"}
    assert set(result.predictions.variant) == {"no_llm_annotations", "with_llm_annotations"}
    assert set(result.predictions.method) == {
        "direct_glm", "pairwise_glm", "llm_judge", "q1_accuracy", "training_mean"}
    assert set(result.predictions.model) == {"m5"}
    assert set(result.comparisons.comparator) == {"judge-1", "pooled_judges"}
    assert not result.item_quartiles.empty


def test_dry_run_reports_load_and_result_writer_is_atomic(tmp_path) -> None:
    config = replace(
        CONFIG,
        output_root=tmp_path / "results",
        model_folds=(("fold", ("m5",)),),
        direct_elastic_alphas=(0.001,), pair_elastic_cs=(1.0,), elastic_l1_ratios=(0.5,),
        ridge_alphas=(1.0,), logistic_cs=(1.0,), inner_folds=2,
    )
    inputs = _inputs()

    report = load_report(inputs, config)

    assert report["feature_counts"] == {"no_llm_annotations": 1, "with_llm_annotations": 2}
    assert report["target_quartiles"] == ["Q2", "Q3", "Q4"]
    assert not config.output_root.exists()

    write_results(run_analysis(inputs, config), config, inputs.manifest)
    assert sorted(path.name for path in config.output_root.iterdir()) == [
        "calibrations.csv", "coefficients.csv", "comparisons.csv", "item-quartiles.csv",
        "manifest.json", "pair-metrics.csv", "predictions.csv", "tuning.csv",
    ]
