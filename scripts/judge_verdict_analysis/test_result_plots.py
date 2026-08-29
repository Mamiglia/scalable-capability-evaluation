from pathlib import Path
from xml.etree import ElementTree

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from .result_plots import (
    accuracy_comparison,
    coefficient_selection_stability,
    feature_effects,
    group_effects,
    load_lesion_summary,
    load_prediction_artifacts,
    load_rejudge_summary,
    plot_accuracy_comparison,
    plot_annotation_removal,
    plot_feature_scatter,
    plot_group_effects,
    plot_ranked_features,
    plot_stability,
    write_result_plots,
)


VARIANTS = (
    "readable", "pca-20", "readable-surprisal", "readable-sae",
    "readable-surprisal-sae",
)


def _prediction_files(path: Path, variants=VARIANTS) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = {}
    for variant_index, variant in enumerate(variants):
        output[variant] = []
        for fold_index, probabilities in enumerate(((0.9, 0.1, 0.8, 0.2), (0.9, 0.9, 0.8, 0.2))):
            fold = path / variant / f"model-{fold_index}__item-0"
            fold.mkdir(parents=True)
            artifact = fold / "predictions.parquet"
            pd.DataFrame({
                "bench": ["a", "a", "b", "b"],
                "judge": ["j1", "j1", "j2", "j2"],
                "item": ["i0", "i1", "i2", "i3"],
                "model_left": ["left"] * 4, "model_right": ["right"] * 4,
                "observed": [1, 0, 1, 0],
                "probability": np.clip(np.asarray(probabilities) - variant_index * 0.01, 0, 1),
            }).to_parquet(artifact, index=False)
            output[variant].append(artifact)
    return output


def _rejudge(path: Path) -> Path:
    output = path / "rejudge.csv"
    pd.DataFrame({
        "judge": ["j1", "j2"], "benchmark": ["a", "b"],
        "wilson_95_low": [0.7, 0.9], "wilson_95_high": [0.9, 1.0],
    }).to_csv(output, index=False)
    return output


def _lesions(path: Path) -> tuple[Path, Path]:
    lesion_path = path / "lesions.csv"
    pd.DataFrame([
        {"kind": "singular-feature", "variable": "signal", "reference": "intercept-only",
         "estimate": 0.20, "lower": 0.10, "upper": 0.30},
        {"kind": "lesion-feature", "variable": "signal", "reference": "full",
         "estimate": 0.08, "lower": 0.03, "upper": 0.13},
        {"kind": "singular-feature", "variable": "noise", "reference": "intercept-only",
         "estimate": -0.05, "lower": -0.09, "upper": -0.01},
        {"kind": "lesion-feature", "variable": "noise", "reference": "full",
         "estimate": 0.02, "lower": -0.01, "upper": 0.05},
        {"kind": "singular-block", "variable": "rubric", "reference": "intercept-only",
         "estimate": 0.12, "lower": 0.06, "upper": 0.18},
        {"kind": "lesion-block", "variable": "rubric", "reference": "full",
         "estimate": 0.04, "lower": 0.01, "upper": 0.07},
        {"kind": "singular-family", "variable": "annotations", "reference": "intercept-only",
         "estimate": 0.16, "lower": 0.08, "upper": 0.24},
        {"kind": "lesion-family", "variable": "annotations", "reference": "full",
         "specification": "no-annotations", "estimate": 0.06, "lower": 0.02, "upper": 0.10},
        {"kind": "singular-family", "variable": "planning", "reference": "intercept-only",
         "estimate": 0.05, "lower": 0.01, "upper": 0.09},
        {"kind": "lesion-family", "variable": "planning", "reference": "full",
         "estimate": 0.01, "lower": -0.01, "upper": 0.03},
    ]).to_csv(lesion_path, index=False)
    dictionary_path = path / "feature-dictionary.parquet"
    pd.DataFrame({
        "feature": ["signal", "noise"], "block": ["rubric", "clause-plan"],
    }).to_parquet(dictionary_path, index=False)
    return lesion_path, dictionary_path


def _stability_files(path: Path) -> tuple[list[Path], list[Path]]:
    selections, coefficients = [], []
    for fold, selected, values in (
        ("fold-0", ["signal"], {"signal": (1.0, 1.0)}),
        ("fold-1", ["signal", "noise"], {"signal": (2.0, 4.0), "noise": (-2.0, -2.0)}),
    ):
        fold_dir = path / "fits" / "frequentist" / fold
        fold_dir.mkdir(parents=True)
        selection_path, coefficient_path = fold_dir / "selection.csv", fold_dir / "coefficients.csv"
        pd.DataFrame({"feature": selected}).to_csv(selection_path, index=False)
        pd.DataFrame([
            {"feature": feature, "judge": judge, "benchmark": "a", "effective": effective}
            for feature, pair in values.items()
            for judge, effective in zip(("j1", "j2"), pair)
        ]).to_csv(coefficient_path, index=False)
        selections.append(selection_path)
        coefficients.append(coefficient_path)
    return selections, coefficients


def test_saved_prediction_comparison_uses_fold_clusters_and_rejudge_floor(tmp_path: Path) -> None:
    predictions = load_prediction_artifacts(_prediction_files(tmp_path, ("readable",)))
    rejudge = load_rejudge_summary(_rejudge(tmp_path))

    summary, folds = accuracy_comparison(predictions, rejudge)

    assert predictions.columns.tolist() == [
        "variant", "fold", "model_fold", "item_fold", "judge", "benchmark",
        "item", "model_left", "model_right", "observed", "probability",
    ]
    assert set(folds.scope) == {"overall", "judge×benchmark"}
    overall = summary[(summary.variant == "readable") & (summary.scope == "overall")].iloc[0]
    assert overall.estimate == pytest.approx(0.875)
    assert overall.exact_floor == pytest.approx(0.8)
    assert overall.model_fold_count == 2 and overall.item_fold_count == 1
    assert np.isnan(overall.lower) and np.isnan(overall.upper)
    cell = summary[(summary.judge == "j1") & (summary.benchmark == "a")].iloc[0]
    assert cell.estimate == pytest.approx(0.75)
    assert cell.exact_floor == pytest.approx(0.7)
    model_points = folds[
        (folds.scope == "overall") & (folds.variant == "readable")
        & (folds.replication_dimension == "model_fold")
    ]
    item_points = folds[
        (folds.scope == "overall") & (folds.variant == "readable")
        & (folds.replication_dimension == "item_fold")
    ]
    assert model_points.accuracy.tolist() == [1.0, 0.75]
    assert item_points.accuracy.tolist() == [0.875]


def test_overall_rejudge_floor_uses_the_same_imbalanced_prediction_rows() -> None:
    predictions = pd.DataFrame({
        "variant": ["readable"] * 10, "fold": ["model-0__item-0"] * 10,
        "model_fold": ["model-0"] * 10, "item_fold": ["0"] * 10,
        "judge": ["j1"] * 9 + ["j2"], "benchmark": ["a"] * 9 + ["b"],
        "observed": [1] * 10, "probability": [0.9] * 10,
    })
    rejudge = pd.DataFrame({
        "judge": ["j1", "j2"], "benchmark": ["a", "b"],
        "wilson_95_low": [0.1, 0.9],
    })

    summary, _ = accuracy_comparison(predictions, rejudge)

    overall = summary[summary.scope == "overall"].iloc[0]
    assert overall.estimate == 1.0
    assert overall.exact_floor == pytest.approx(0.18)


def test_accuracy_interval_uses_folds_not_prediction_rows() -> None:
    predictions = pd.DataFrame([
        {
            "variant": "readable", "fold": f"model-{model}__item-{item}",
            "model_fold": f"model-{model}", "item_fold": str(item), "judge": "j1",
            "benchmark": "a", "observed": 1, "probability": 0.9 if row < correct else 0.1,
        }
        for model, correct in enumerate((7, 8, 9))
        for item in range(3)
        for row in range(10)
    ])
    rejudge = pd.DataFrame({
        "judge": ["j1"], "benchmark": ["a"], "wilson_95_low": [0.65],
    })

    summary, _ = accuracy_comparison(predictions, rejudge)

    overall = summary[summary.scope == "overall"].iloc[0]
    assert overall.estimate == pytest.approx(0.8)
    assert overall.model_fold_count == 3 and overall.item_fold_count == 3
    assert overall.row_count == 90
    assert overall.lower == pytest.approx(0.551586, abs=1e-6)
    assert overall.upper == 1.0
    assert overall.uncertainty_method == "conservative union of model-fold/item-fold t95"


def test_crossed_two_by_two_folds_show_dimension_points_without_a_ci() -> None:
    predictions = pd.DataFrame([
        {
            "variant": "readable", "fold": f"model-{model}__item-{item}",
            "model_fold": f"model-{model}", "item_fold": str(item), "judge": "j1",
            "benchmark": "a", "observed": 1, "probability": probability,
        }
        for model, item, probability in (
            (0, 0, 0.9), (0, 1, 0.9), (1, 0, 0.1), (1, 1, 0.1),
        )
    ])
    rejudge = pd.DataFrame({
        "judge": ["j1"], "benchmark": ["a"], "wilson_95_low": [0.7],
    })

    summary, points = accuracy_comparison(predictions, rejudge)

    overall = summary[summary.scope == "overall"].iloc[0]
    assert overall.model_fold_count == overall.item_fold_count == 2
    assert np.isnan(overall.lower) and np.isnan(overall.upper)
    assert overall.uncertainty_method == "model-fold/item-fold points"
    overall_points = points[points.scope == "overall"]
    assert overall_points.groupby("replication_dimension").size().to_dict() == {
        "item_fold": 2, "model_fold": 2,
    }


def test_lesion_and_stability_tables_preserve_requested_effect_directions(tmp_path: Path) -> None:
    lesion_path, dictionary_path = _lesions(tmp_path)
    lesions = load_lesion_summary(lesion_path, dictionary_path)
    features = feature_effects(lesions).set_index("feature")
    groups = group_effects(lesions).set_index(["level", "group"])
    selections, coefficients = _stability_files(tmp_path)
    stability, fold_coefficients = coefficient_selection_stability(selections, coefficients)
    stability = stability.set_index("feature")

    assert features.loc["signal", ["singular", "unique", "block"]].tolist() == [
        0.20, 0.08, "rubric",
    ]
    assert features.loc["noise", "singular"] == -0.05
    assert groups.loc[("family", "annotations"), ["singular", "unique"]].tolist() == [0.16, 0.06]
    assert ("family", "planning") in groups.index
    assert stability.loc["signal", "selection_rate"] == 1.0
    assert stability.loc["noise", "selection_rate"] == 0.5
    assert stability.loc["signal", "coefficient_mean"] == pytest.approx(2.0)
    assert stability.loc["signal", "coefficient_fold_count"] == 2
    assert np.isnan(stability.loc["signal", "coefficient_lower"])
    assert stability.loc["noise", "coefficient_mean"] == pytest.approx(-1.0)
    assert stability.loc["noise", "coefficient_fold_count"] == 2
    assert fold_coefficients.set_index(["fold", "feature"]).loc[("fold-1", "signal"), "effective"] == 3.0
    assert fold_coefficients.set_index(["fold", "feature"]).loc[("fold-0", "noise"), "effective"] == 0.0


def test_stability_requires_identical_selection_and_coefficient_fold_sets(tmp_path: Path) -> None:
    selections, coefficients = _stability_files(tmp_path)

    with pytest.raises(ValueError, match="fold sets differ"):
        coefficient_selection_stability(selections, coefficients[:1])


def test_effect_tables_reject_incomplete_singular_unique_pairs(tmp_path: Path) -> None:
    lesion_path, dictionary_path = _lesions(tmp_path)
    lesions = load_lesion_summary(lesion_path, dictionary_path)

    with pytest.raises(ValueError, match="incomplete paired effects"):
        feature_effects(lesions[~((lesions.kind == "lesion-feature") & (lesions.variable == "noise"))])
    with pytest.raises(ValueError, match="incomplete paired effects"):
        group_effects(lesions[~((lesions.kind == "lesion-family") & (lesions.variable == "planning"))])


@pytest.mark.parametrize("variants", [
    {"readable": []},
    {**{variant: [] for variant in VARIANTS}, "unknown": []},
])
def test_top_level_workflow_requires_exactly_the_five_variants(
    tmp_path: Path, variants: dict[str, list[Path]],
) -> None:
    with pytest.raises(ValueError, match="exactly the five requested variants"):
        write_result_plots(
            variants, tmp_path / "rejudge.csv", tmp_path / "lesions.csv",
            tmp_path / "dictionary.parquet", [], [], tmp_path / "figures",
        )


def test_result_plots_are_compact_editable_shared_scale_and_unclipped(tmp_path: Path) -> None:
    predictions = load_prediction_artifacts(_prediction_files(tmp_path))
    summary, fold_scores = accuracy_comparison(predictions, load_rejudge_summary(_rejudge(tmp_path)))
    lesion_path, dictionary_path = _lesions(tmp_path)
    lesions = load_lesion_summary(lesion_path, dictionary_path)
    features, groups = feature_effects(lesions), group_effects(lesions)
    selection_paths, coefficient_paths = _stability_files(tmp_path)
    stability, fold_coefficients = coefficient_selection_stability(selection_paths, coefficient_paths)
    output_dir = tmp_path / "figures"

    outputs = {
        **plot_accuracy_comparison(summary, fold_scores, output_dir),
        "feature-scatter": plot_feature_scatter(features, output_dir, top_labels=1),
        **plot_ranked_features(features, output_dir),
        "group-effects": plot_group_effects(groups, output_dir),
        "annotation-removal": plot_annotation_removal(groups, output_dir),
        "stability": plot_stability(stability, fold_coefficients, output_dir),
    }

    assert set(outputs) == {
        "oof-accuracy-overall", "oof-accuracy-by-cell", "feature-scatter",
        "feature-singular-ranked", "feature-unique-ranked", "group-effects",
        "annotation-removal", "stability",
    }
    svgs = []
    for png, svg in outputs.values():
        assert png.exists() and svg.exists()
        pixels = mpimg.imread(png).shape
        assert pixels[1] <= 1320 and pixels[0] <= 960
        if png.name == "oof-accuracy-overall.png":
            assert pixels[1] <= 540
        source = svg.read_text()
        assert "<text" in source
        assert "clip-path" not in "".join(
            ElementTree.tostring(element, encoding="unicode")
            for element in ElementTree.fromstring(source).iter()
            if element.attrib.get("id", "").startswith("PathCollection")
        )
        svgs.append(source)
    overall_source = outputs["oof-accuracy-overall"][1].read_text()
    assert 'id="legend_1"' in overall_source
    assert "Model fold" in overall_source and "Item fold" in overall_source
    readable_tick = next(
        element for element in ElementTree.fromstring(overall_source).iter()
        if element.text == "Readable"
    )
    assert "rotate(-90" in readable_tick.attrib.get("transform", "")
    assert 'id="legend_1"' in outputs["feature-scatter"][1].read_text()
    singular_svg = outputs["feature-singular-ranked"][1].read_text()
    unique_svg = outputs["feature-unique-ranked"][1].read_text()
    assert ">−0.3</text>" in singular_svg and ">0.3</text>" in singular_svg
    assert ">−0.3</text>" in unique_svg and ">0.3</text>" in unique_svg
    assert not any("suptitle" in source.lower() for source in svgs)


def test_loaders_reject_incomplete_or_invalid_saved_artifacts(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.parquet"
    pd.DataFrame({"judge": ["j1"], "bench": ["a"], "observed": [1]}).to_parquet(invalid)
    with pytest.raises(ValueError, match="probability"):
        load_prediction_artifacts({"readable": [invalid]})

    invalid_rejudge = tmp_path / "invalid.csv"
    pd.DataFrame({"judge": ["j1"], "benchmark": ["a"], "wilson_95_low": [1.2]}).to_csv(
        invalid_rejudge, index=False)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_rejudge_summary(invalid_rejudge)
