import importlib
import json
import xml.etree.ElementTree as ET
import warnings

import matplotlib.image as mpimg
import pandas as pd
import pytest

from capability_predictions_glm.config import MODEL_FOLDS, SOURCE_MANIFEST_SHA256, SOURCE_NAME


def _module():
    return importlib.import_module("capability_predictions_glm.ranking_baselines")


def test_target_accuracies_pool_q2_to_q4_by_item_count() -> None:
    accuracies = pd.DataFrame([
        {"bench": "b", "model": "m", "quartile": target, "accuracy": accuracy}
        for target, accuracy in (("Q1", 0.9), ("Q2", 0.0), ("Q3", 0.5), ("Q4", 1.0))
    ])
    quartiles = pd.DataFrame([
        {"bench": "b", "quartile": target, "item": item}
        for target, items in (("Q2", ("i1",)), ("Q3", ("i2", "i3")),
                              ("Q4", ("i4",)))
        for item in items
    ])

    targets = _module().target_accuracies(accuracies, quartiles).set_index("target_quartile")

    assert set(targets.index) == {"Q2", "Q3", "Q4", "Q2-Q4"}
    assert targets.loc["Q2-Q4", "observed_accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(("schema", "value"), (
    ("task", "pairwise"), ("model", "pairwise_glm"),
))
def test_crossfit_pair_predictions_produce_one_global_bt_rank_per_model(
    schema: str, value: str,
) -> None:
    pairs = pd.DataFrame([
        {"bench": "b", "item": item, "judge": "j", "model_left": left,
         "model_right": right, "d_signal": difference}
        for item, left, right, difference in (
            ("i1", "c", "b", 1.0), ("i2", "b", "a", 1.0),
            ("i3", "c", "a", 2.0),
        )
    ])
    coefficients = pd.DataFrame([
        {schema: value, "fold": fold, "target_quartile": target,
         "benchmark": "b", "feature": "signal", "effective": 2.0}
        for fold in ("f0", "f1") for target in ("Q2", "Q3", "Q4")
    ])

    rankings = _module().crossfit_glm_rankings(
        pairs, coefficients, (("f0", ("a", "b")), ("f1", ("c",))),
        "Length only",
    )

    assert "fold" not in rankings
    assert set(rankings.target_quartile) == {"Q2", "Q3", "Q4", "Q2-Q4"}
    assert rankings.groupby(["target_quartile", "model"]).size().eq(1).all()
    q2 = rankings[rankings.target_quartile == "Q2"].set_index("model").score
    assert q2["c"] > q2["b"] > q2["a"]


def test_judge_bt_rankings_have_one_score_per_model_without_targets_or_folds() -> None:
    pairs = pd.DataFrame([
        {"bench": "b", "item": item, "judge": judge, "model_left": left,
         "model_right": right, "pick_left": pick}
        for judge, picks in (
            ("j1", (("i1", "c", "b", 1), ("i2", "b", "a", 1))),
            ("j2", (("i1", "c", "b", 0), ("i2", "b", "a", 0))),
        )
        for item, left, right, pick in picks
    ])

    rankings = _module().judge_rankings(pairs)

    assert "fold" not in rankings and "target_quartile" not in rankings
    assert set(rankings.condition) == {"Judge: j1", "Judge: j2", "Pooled judges"}
    assert rankings.groupby(["condition", "model"]).size().eq(1).all()
    j1 = rankings[rankings.condition == "Judge: j1"].set_index("model").score
    assert j1["c"] > j1["b"] > j1["a"]


def test_saved_pair_predictions_collapse_duplicates_into_one_bt_ranking() -> None:
    predictions = pd.DataFrame([
        {"bench": "b", "item": item, "model_left": left, "model_right": right,
         "probability": probability, "fold": fold}
        for item, left, right, values in (
            ("i1", "c", "b", ((0.8, "f0"), (0.6, "f1"))),
            ("i2", "b", "a", ((0.9, "f0"),)),
            ("i3", "c", "a", ((0.95, "f1"),)),
        )
        for probability, fold in values
    ])

    rankings = _module().saved_pair_rankings(predictions, "Deterministic GLM")

    assert "fold" not in rankings
    assert rankings.groupby("model").size().eq(1).all()
    scores = rankings.set_index("model").score
    assert scores["c"] > scores["b"] > scores["a"]


def test_linear_ensemble_predicts_every_pair_before_bt_ranking() -> None:
    pairs = pd.DataFrame([
        {"bench": "b", "item": item, "model_left": left, "model_right": right,
         "d_signal": difference}
        for item, left, right, difference in (
            ("i1", "c", "b", 1.0), ("i2", "b", "a", 1.0),
            ("i3", "c", "a", 2.0),
        )
    ])
    coefficients = [pd.DataFrame([{
        "feature": "signal", "benchmark": "b", "effective": weight,
    }]) for weight in (1.0, 2.0)]

    rankings = _module().linear_ensemble_rankings(
        pairs, coefficients, "Deterministic GLM",
    )

    assert rankings.groupby("model").size().eq(1).all()
    scores = rankings.set_index("model").score
    assert scores["c"] > scores["b"] > scores["a"]


def test_global_rank_points_have_one_matched_point_per_model_and_no_folds() -> None:
    rankings = pd.DataFrame([
        {"bench": "b", "target_quartile": "Q2", "condition": condition,
         "kind": "baseline", "model": model, "score": score}
        for condition, values in (
            ("broad", (("a", 1.0), ("b", 2.0), ("c", 3.0))),
            ("narrow", (("b", 2.0), ("c", 3.0))),
            ("Judge GLM L1", (("b", 2.0),)),
        )
        for model, score in values
    ])
    targets = pd.DataFrame([
        {"bench": "b", "target_quartile": "Q2", "model": model,
         "observed_accuracy": observed}
        for model, observed in (("a", 0.1), ("b", 0.5), ("c", 0.9))
    ])

    points, taus = _module().global_rank_points(rankings, targets)

    assert "fold" not in points
    assert points.groupby("condition").model.apply(set).to_dict() == {
        "broad": {"b", "c"}, "narrow": {"b", "c"}, "Judge GLM L1": {"b"},
    }
    assert points.groupby(["condition", "model"]).size().eq(1).all()
    values = taus.set_index("condition").tau
    assert values[["broad", "narrow"]].to_dict() == pytest.approx({
        "broad": 1.0, "narrow": 1.0,
    })
    assert pd.isna(values["Judge GLM L1"])


def test_single_model_rankings_are_unscored_without_warning() -> None:
    rankings = pd.DataFrame([{
        "bench": "b", "target_quartile": "Q2", "condition": "only",
        "kind": "baseline", "model": "m", "score": 1.0,
    }])
    targets = pd.DataFrame([{
        "bench": "b", "target_quartile": "Q2", "model": "m",
        "observed_accuracy": 1.0,
    }])

    with warnings.catch_warnings(record=True) as caught:
        _, taus = _module().global_rank_points(rankings, targets)

    assert pd.isna(taus.loc[0, "tau"])
    assert not caught


def test_tau_a_keeps_predictor_ties_in_the_denominator() -> None:
    rankings = pd.DataFrame([
        {"bench": "b", "target_quartile": "Q2", "condition": "tied",
         "kind": "baseline", "model": model, "score": score}
        for model, score in (("a", 0.0), ("b", 0.0), ("c", 1.0))
    ])
    targets = pd.DataFrame([
        {"bench": "b", "target_quartile": "Q2", "model": model,
         "observed_accuracy": observed}
        for model, observed in (("a", 0.0), ("b", 0.5), ("c", 1.0))
    ])

    _, taus = _module().global_rank_points(rankings, targets)

    assert taus.loc[0, "tau"] == pytest.approx(2 / 3)


def test_global_best_and_worst_judges_are_fixed_from_pooled_target() -> None:
    taus = pd.DataFrame([
        {"bench": "gpqa", "target_quartile": target, "condition": judge,
         "kind": "judge", "tau": tau, "n_models": 4}
        for target, values in (
            ("Q2-Q4", (("Judge one", 0.8), ("Judge two", 0.2))),
            ("Q2", (("Judge one", -0.5), ("Judge two", 0.9))),
        )
        for judge, tau in values
    ])

    summary, selections, wide = _module().summarize_global_taus(taus)

    assert selections.to_dict("records") == [{
        "bench": "gpqa", "best_judge": "Judge one", "worst_judge": "Judge two",
    }]
    q2 = summary[summary.target_quartile == "Q2"].set_index("condition").tau
    assert q2["Best judge"] == pytest.approx(-0.5)
    assert q2["Worst judge"] == pytest.approx(0.9)
    assert q2["Mean judge"] == pytest.approx(0.2)
    assert wide.set_index(["target_quartile", "condition"]).loc[
        ("Q2", "Mean judge"), "gpqa"
    ] == pytest.approx(0.2)


def test_target_item_bootstrap_penalizes_resampled_truth_ties() -> None:
    points = pd.DataFrame([
        {"bench": "b", "target_quartile": "Q2", "condition": "perfect",
         "kind": "baseline", "model": model, "score": score,
         "predicted_rank": score / 2, "true_rank": score / 2}
        for model, score in (("a", 0.0), ("b", 1.0), ("c", 2.0))
    ])
    points = pd.concat([points, pd.DataFrame([
        {"bench": "b", "target_quartile": "Q2", "condition": "Judge GLM L1",
         "kind": "baseline", "model": model, "score": score,
         "predicted_rank": score, "true_rank": score}
        for model, score in (("a", 0.0), ("c", 1.0))
    ])], ignore_index=True)
    outcomes = {
        "a": (0, 0, 0, 0), "b": (0, 0, 1, 1), "c": (1, 1, 1, 1),
    }
    responses = pd.DataFrame([
        {"bench": "b", "model": model, "item": f"i{item}", "correct": correct,
         "is_canonical": True}
        for model, values in outcomes.items() for item, correct in enumerate(values)
    ])
    quartiles = pd.DataFrame([
        {"bench": "b", "item": f"i{item}", "quartile": "Q2"}
        for item in range(4)
    ])

    intervals = _module().bootstrap_target_taus(
        points, responses, quartiles, n_resamples=200, seed=0,
    ).set_index("condition")

    assert intervals.loc["perfect", "lower"] == pytest.approx(2 / 3)
    assert intervals.loc["perfect", "upper"] == pytest.approx(1.0)
    assert intervals.n_resamples.eq(200).all()
    assert intervals.loc["Judge GLM L1", ["lower", "upper"]].notna().all()


def test_bar_summary_orders_tau_best_to_worst() -> None:
    frame = pd.DataFrame([
        {"condition": condition, "predicted_rank": predicted, "true_rank": true}
        for condition, values in (
            ("middle", ((0, 0), (0, 0.5), (1, 1))),
            ("worst", ((1, 0), (0.5, 0.5), (0, 1))),
            ("best", ((0, 0), (0.5, 0.5), (1, 1))),
        )
        for predicted, true in values
    ])

    summary = _module().bar_summary(frame)

    assert summary.condition.tolist() == ["best", "middle", "worst"]
    assert summary.tau.tolist() == pytest.approx([1.0, 2 / 3, -1.0])


def test_rank_scatter_has_one_unclipped_point_per_model_and_no_fold_legend(tmp_path) -> None:
    conditions = [
        "Judge: Llama-3.1-8B-Instruct", "Judge: deepseek-r1-0528",
        "Judge: deepseek-v4-flash-0731", "Judge: gemma-4-31b-it",
        "Judge: gpt-oss-120b", "Judge: ling-2.6-flash", "Judge: o4-mini",
        "Pooled judges", "Judge GLM L1", "Judge GLM L2", "Q1 accuracy",
        "Length only", "Length + surface",
        "Deterministic GLM", "Full pairwise GLM",
    ]
    points = pd.DataFrame([
        {"bench": "gpqa", "target_quartile": "Q2", "condition": condition,
         "kind": "judge" if condition.startswith("Judge:") else "baseline",
         "model": model, "predicted_rank": rank, "true_rank": rank}
        for condition in conditions
        for model, rank in (("m0", 0.0), ("m1", 0.5), ("m2", 1.0))
    ])
    intervals = pd.DataFrame([
        {"condition": condition, "lower": 0.5, "upper": 1.0}
        for condition in conditions
    ])

    with warnings.catch_warnings(record=True) as caught:
        paths = _module().plot_rank_scatter(points, "gpqa", "Q2", tmp_path, intervals)

    assert {path.suffix for path in paths} == {".png", ".svg", ".csv"}
    assert not [warning for warning in caught if "constrained_layout" in str(warning.message)]
    image = mpimg.imread(tmp_path / "gpqa-q2-ranks.png")
    assert image.shape[1] / 300 <= 4.4 and image.shape[0] / 300 <= 7.6
    svg_path = tmp_path / "gpqa-q2-ranks.svg"
    svg = svg_path.read_text()
    assert "<text" in svg and "DeepSeek V4" in svg
    assert "deepseek-v4-flash-0731" not in svg and "fold-0" not in svg
    assert "o4 + V4" in svg
    assert "Kendall τ-a" in svg
    assert all(color in svg for color in ("#4292c6", "#082f5a", "#eb6834",
                                           "#1baf7a", "#20a4b8", "#c05a9d",
                                           "#e87ba4", "#4a3aa7", "#eda100"))
    root = ET.parse(svg_path).getroot()
    text_elements = list(root.iter("{http://www.w3.org/2000/svg}text"))
    texts = [element.text for element in text_elements]
    assert texts.count("0") >= 24 and texts.count("1") >= 24
    r1 = next(element for element in text_elements if element.text == "R1")
    assert "rotate(-90" not in r1.attrib.get("transform", "")
    assert "Full GLM" in texts
    assert texts.count("1.00") >= len(conditions)
    collections = [group for group in root.iter("{http://www.w3.org/2000/svg}g")
                   if group.attrib.get("id", "").startswith("PathCollection")]
    assert collections
    assert all("stroke" not in use.attrib.get("style", "")
               for group in collections
               for use in group.iter("{http://www.w3.org/2000/svg}use"))
    assert any(group.attrib.get("id", "").startswith("LineCollection")
               for group in root.iter("{http://www.w3.org/2000/svg}g"))
    rects = list(root.iter("{http://www.w3.org/2000/svg}rect"))
    widths = sorted({round(float(rect.attrib["width"]), 2) for rect in rects if "width" in rect.attrib})
    heights = sorted({round(float(rect.attrib["height"]), 2) for rect in rects if "height" in rect.attrib})
    assert 1.7 < widths[-1] / widths[0] < 2.3
    assert heights[-1] > 1.2 * heights[0]
    scatter = min(
        ((float(rect.attrib["width"]), float(rect.attrib["height"])) for rect in rects
         if "width" in rect.attrib and "height" in rect.attrib),
        key=lambda size: size[0],
    )
    assert 0.9 < scatter[0] / scatter[1] < 1.1
    source = pd.read_csv(tmp_path / "gpqa-q2-ranks.csv")
    assert source.groupby(["condition", "model"]).size().eq(1).all()
    assert source.condition.drop_duplicates().tolist()[-7:] == [
        "Judge GLM L1", "Judge GLM L2",
        "Length only", "Length + surface", "Deterministic GLM",
        "Full pairwise GLM", "Q1 accuracy",
    ]


def test_updated_loader_requires_every_pinned_fold_target_cell(tmp_path) -> None:
    root = tmp_path / "updated"
    for fold, _ in MODEL_FOLDS:
        for target in ("Q2", "Q3", "Q4"):
            cell = root / f"{fold}-{target}"
            cell.mkdir(parents=True)
            pd.DataFrame([{
                "fold": fold, "bench": "gpqa", "target_quartile": target,
                "method": "pairwise_glm", "variant": "with_llm_annotations",
                "comparator": "pooled_judges", "model": "m", "score": 0.5,
                "observed_accuracy": 0.5,
            }]).to_csv(cell / "predictions.csv", index=False)
            (cell / "manifest.json").write_text(json.dumps({
                "source_manifest": {"source": {
                    "name": SOURCE_NAME,
                    "design_matrix_manifest_sha256": SOURCE_MANIFEST_SHA256,
                }}
            }))

    loaded = _module().load_updated_predictions(root)
    assert len(loaded) == 12
    (root / "model-fold-3-Q4" / "predictions.csv").unlink()
    with pytest.raises(ValueError, match="12 complete fold-target cells"):
        _module().load_updated_predictions(root)


def test_report_writes_wide_benchmark_table_and_four_target_figures(tmp_path) -> None:
    conditions = (
        ("Judge: judge-good", "judge", 1), ("Judge: judge-bad", "judge", -1),
        ("Pooled judges", "pooled_judge", 1), ("Length only", "baseline", -1),
        ("Judge GLM L1", "baseline", 1), ("Judge GLM L2", "baseline", 1),
        ("Length + surface", "baseline", 1), ("Deterministic GLM", "baseline", 1),
        ("Full pairwise GLM", "baseline", 1),
        ("Q1 accuracy", "easy", 1),
    )
    rankings = pd.DataFrame([
        {"bench": "gpqa", "target_quartile": target, "condition": condition,
         "kind": kind, "model": model, "score": direction * observed}
        for target in ("Q2", "Q3", "Q4", "Q2-Q4")
        for condition, kind, direction in conditions
        for model, observed in (("weak", 0.0), ("mid", 0.5), ("strong", 1.0))
    ])
    targets = pd.DataFrame([
        {"bench": "gpqa", "target_quartile": target, "model": model,
         "observed_accuracy": observed}
        for target in ("Q2", "Q3", "Q4", "Q2-Q4")
        for model, observed in (("weak", 0.0), ("mid", 0.5), ("strong", 1.0))
    ])
    intervals = pd.DataFrame([
        {"bench": "gpqa", "target_quartile": target, "condition": condition,
         "lower": -0.1, "upper": 1.0, "n_resamples": 20}
        for target in ("Q2", "Q3", "Q4", "Q2-Q4")
        for condition, _, _ in conditions
    ])

    _module().write_global_report(rankings, targets, tmp_path, intervals)

    wide = pd.read_csv(tmp_path / "tables/tau-wide.csv")
    assert list(wide.columns) == ["target_quartile", "condition", "GPQA"]
    assert {"Best judge", "Pooled judges", "Mean judge", "Worst judge",
            "Length only", "Length + surface", "Full pairwise GLM", "Q1 accuracy",
            "Deterministic GLM", "Judge GLM L1", "Judge GLM L2",
            "Judge: judge-good", "Judge: judge-bad"} <= set(wide.condition)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    assert len(list((tmp_path / "figures").glob("*.svg"))) == 4
    assert len(list((tmp_path / "figures").glob("*.csv"))) == 4
    assert (tmp_path / "tables/bootstrap-tau-intervals.csv").exists()
