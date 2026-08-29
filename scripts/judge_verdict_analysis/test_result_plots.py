from pathlib import Path
from xml.etree import ElementTree

import matplotlib
import matplotlib.image as mpimg
from matplotlib.text import Annotation, Text
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import result_plots
from .result_plots import (
    accuracy_comparison,
    coefficient_selection_stability,
    feature_effects,
    group_effects,
    load_benchmark_feature_effects,
    load_lesion_summary,
    load_prediction_artifacts,
    load_rejudge_summary,
    plot_accuracy_comparison,
    plot_annotation_comparison,
    plot_feature_scatter,
    plot_feature_overview,
    plot_feature_blocks,
    plot_benchmark_feature_grid,
    plot_group_effects,
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
    artifact_rows = []
    for fold_index in range(2):
        coefficient_path = path / "lesion-folds" / f"fold-{fold_index}" / "full.parquet"
        coefficient_path.parent.mkdir(parents=True)
        pd.DataFrame([
            {
                "feature": feature, "judge": judge, "benchmark": benchmark,
                "effective": base + fold_index + judge_index / 10 + benchmark_index / 100,
            }
            for feature, base in (("signal", 1.0), ("noise", -1.0))
            for judge_index, judge in enumerate(("j1", "j2"))
            for benchmark_index, benchmark in enumerate(
                ("aime", "gpqa", "intercode", "swe_bench_mini")
            )
        ]).to_parquet(coefficient_path, index=False)
        artifact_rows.append({
            "fold": f"fold-{fold_index}", "specification": "full",
            "coefficients": str(coefficient_path),
        })
    pd.DataFrame(artifact_rows).to_csv(path / "artifact-index.csv", index=False)
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
            {"feature": feature, "judge": judge, "benchmark": benchmark,
             "effective": effective}
            for feature, pair in values.items()
            for benchmark in ("aime", "gpqa", "intercode", "swe_bench_mini")
            for judge, effective in zip(("j1", "j2"), pair)
        ]).to_csv(coefficient_path, index=False)
        selections.append(selection_path)
        coefficients.append(coefficient_path)
    return selections, coefficients


def _item_deltas(path: Path) -> Path:
    output = path / "item-deltas.parquet"
    pd.DataFrame([
        {
            "bench": bench, "item": f"{bench}-item-{item_fold}",
            "fold": f"model-{model_fold}__item-{item_fold}", "kind": kind,
            "variable": feature, "row_count": 1,
            "delta_log_loss": base + model_fold / 1000,
        }
        for bench in ("aime", "gpqa", "intercode", "swe_bench_mini")
        for model_fold in range(3)
        for item_fold in range(3)
        for feature, kind, base in (
            ("signal", "singular-feature", 0.04),
            ("signal", "lesion-feature", 0.008),
            ("noise", "singular-feature", -0.01),
            ("noise", "lesion-feature", 0.002),
        )
    ]).to_parquet(output, index=False)
    return output


def _judge_item_deltas(path: Path) -> Path:
    output = path / "judge-item-deltas.parquet"
    rows = pd.read_parquet(_item_deltas(path)).merge(
        pd.DataFrame({"judge": ["j1", "j2"]}), how="cross",
    )
    rows["delta_log_loss"] += rows.judge.map({"j1": 0.001, "j2": -0.001})
    rows.to_parquet(output, index=False)
    return output


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


def test_result_plots_are_compact_editable_and_unclipped(tmp_path: Path) -> None:
    predictions = load_prediction_artifacts(_prediction_files(tmp_path))
    summary, fold_scores = accuracy_comparison(predictions, load_rejudge_summary(_rejudge(tmp_path)))
    lesion_path, dictionary_path = _lesions(tmp_path)
    lesions = load_lesion_summary(lesion_path, dictionary_path)
    features, groups = feature_effects(lesions), group_effects(lesions)
    coefficients = result_plots.summarize_effective_coefficients(
        result_plots.load_lesion_coefficients(lesion_path.with_name("artifact-index.csv")),
    )
    selection_paths, coefficient_paths = _stability_files(tmp_path)
    stability, fold_coefficients = coefficient_selection_stability(selection_paths, coefficient_paths)
    output_dir = tmp_path / "figures"

    outputs = {
        **plot_accuracy_comparison(summary, fold_scores, output_dir),
        "feature-effects-overview": plot_feature_overview(features, coefficients, output_dir),
        **plot_feature_blocks(features, coefficients, output_dir),
        "feature-scatter": plot_feature_scatter(features, output_dir, top_labels=1),
        "group-effects": plot_group_effects(groups, output_dir),
        "annotation-comparison": plot_annotation_comparison(groups, output_dir),
        "stability": plot_stability(stability, fold_coefficients, output_dir),
    }

    assert set(outputs) == {
        "oof-accuracy-overall", "oof-accuracy-by-cell", "feature-scatter",
        "feature-effects-overview", "feature-effects-rubric", "feature-effects-clause-plan",
        "group-effects", "annotation-comparison", "stability",
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
    overview_svg = outputs["feature-effects-overview"][1].read_text()
    assert overview_svg.count('id="axes_') == 3
    assert all(label in overview_svg for label in ("Singular", "Unique", "Coefficient"))
    assert ">−.05</text>" in overview_svg and ">.2</text>" in overview_svg
    assert ">.08</text>" in overview_svg and ">−1.6</text>" in overview_svg
    for name, feature in (("feature-effects-rubric", "signal"),
                          ("feature-effects-clause-plan", "noise")):
        block_svg = outputs[name][1].read_text()
        assert block_svg.count('id="axes_') == 3 and feature in block_svg
    assert "planning (clause-plan)" in outputs["group-effects"][1].read_text()
    assert "Singular" in outputs["annotation-comparison"][1].read_text()
    assert "Unique removal" in outputs["annotation-comparison"][1].read_text()
    assert not any("suptitle" in source.lower() for source in svgs)


def test_dense_legends_and_scatter_labels_stay_inside_canvas_without_overlap(
    tmp_path: Path, monkeypatch,
) -> None:
    blocks = list(result_plots.BLOCK_COLORS)
    features = pd.DataFrame({
        "feature": [f"dense_feature_label_{index}" for index in range(len(blocks))],
        "block": blocks,
        "singular": np.linspace(0.010, 0.018, len(blocks)),
        "singular_lower": np.linspace(0.008, 0.016, len(blocks)),
        "singular_upper": np.linspace(0.012, 0.020, len(blocks)),
        "unique": np.linspace(-0.0004, 0.0004, len(blocks)),
        "unique_lower": np.linspace(-0.0014, -0.0006, len(blocks)),
        "unique_upper": np.linspace(0.0006, 0.0014, len(blocks)),
    })
    groups = pd.DataFrame([
        {
            "level": level, "group": name, "singular": 0.04, "singular_lower": 0.03,
            "singular_upper": 0.05, "unique": 0.01, "unique_lower": 0.0,
            "unique_upper": 0.02,
        }
        for level, names in (
            ("block", blocks),
            ("family", ["annotations", "clause-structure", "hand-engineered-text", "planning"]),
        )
        for name in names
    ])
    coefficients = pd.DataFrame({
        "feature": features.feature, "estimate": np.linspace(-0.5, 0.5, len(features)),
        "lower": np.linspace(-0.6, 0.4, len(features)),
        "upper": np.linspace(-0.4, 0.6, len(features)), "fold_count": 2,
    })

    def checked_save(fig, output_dir, name):
        fig.canvas.draw()
        renderer, canvas = fig.canvas.get_renderer(), fig.bbox
        text = [artist for artist in fig.findobj(Text) if artist.get_visible() and artist.get_text()]
        for artist in [*fig.legends, *text]:
            bounds = artist.get_window_extent(renderer)
            assert bounds.x0 >= canvas.x0 - 1 and bounds.x1 <= canvas.x1 + 1, (name, artist)
            assert bounds.y0 >= canvas.y0 - 1 and bounds.y1 <= canvas.y1 + 1, (name, artist)
        annotations = [Text.get_window_extent(artist, renderer) for artist in text
                       if isinstance(artist, Annotation)]
        assert not any(left.overlaps(right) for index, left in enumerate(annotations)
                       for right in annotations[index + 1:]), name
        plt.close(fig)
        return Path(output_dir) / f"{name}.png", Path(output_dir) / f"{name}.svg"

    monkeypatch.setattr(result_plots, "_save", checked_save)

    result_plots.plot_feature_scatter(features, tmp_path, top_labels=3)
    result_plots.plot_feature_overview(features, coefficients, tmp_path, top=len(features))
    result_plots.plot_feature_blocks(features, coefficients, tmp_path)
    result_plots.plot_group_effects(groups, tmp_path)


def test_feature_scatter_and_overview_panels_use_independent_rounded_scales(
    tmp_path: Path, monkeypatch,
) -> None:
    features = pd.DataFrame({
        "feature": ["low", "high"], "block": ["surface", "rubric"],
        "singular": [-0.008, 0.054], "singular_lower": [-0.014, 0.044],
        "singular_upper": [-0.006, 0.064], "unique": [-0.0019, 0.0045],
        "unique_lower": [-0.0050, 0.0024], "unique_upper": [-0.0009, 0.0070],
    })
    coefficients = pd.DataFrame({
        "feature": ["low", "high"], "estimate": [-0.4, 0.6],
        "lower": [-0.5, 0.5], "upper": [-0.3, 0.7], "fold_count": [2, 2],
    })
    scales = {}

    def capture_scale(fig, output_dir, name):
        for suffix, axis in zip(("singular", "unique", "coefficient"), fig.axes):
            key = f"{name}-{suffix}" if len(fig.axes) > 1 else name
            scales[key] = (axis.get_xlim(), axis.get_ylim(), axis.get_xticks(), axis.get_yticks())
        plt.close(fig)
        return Path(output_dir) / f"{name}.png", Path(output_dir) / f"{name}.svg"

    monkeypatch.setattr(result_plots, "_save", capture_scale)

    result_plots.plot_feature_scatter(features, tmp_path, top_labels=0)
    result_plots.plot_feature_overview(features, coefficients, tmp_path, top=2)

    assert scales["feature-scatter"][:2] == ((-0.01, 0.06), (-0.002, 0.005))
    assert scales["feature-scatter"][2].tolist() == [-0.01, 0.0, 0.06]
    assert scales["feature-scatter"][3].tolist() == [-0.002, 0.0, 0.005]
    assert scales["feature-effects-overview-singular"][0] == (-0.01, 0.06)
    assert scales["feature-effects-overview-unique"][0] == (-0.002, 0.005)
    assert scales["feature-effects-overview-coefficient"][0] == (-0.6, 0.6)


def test_scatter_labels_requested_features_outward_without_overlap(
    tmp_path: Path, monkeypatch,
) -> None:
    values = (
        ("hedging_count", "surface", 0.003868, 0.003361),
        ("think_aloud_per_100_words", "exposition", 0.009888, 0.003379),
        ("bare_answer", "structure", 0.033129, -0.000135),
        ("tool_errors", "structure", 0.015287, 0.001644),
        ("rigour", "rubric", 0.053963, 0.001151),
        ("depth", "rubric", 0.041681, 0.002910),
        ("self_corrections", "rubric", 0.020767, 0.004508),
        ("efficiency", "rubric", 0.043386, 0.001868),
        ("clarity", "rubric", 0.045217, -0.000037),
    )
    features = pd.DataFrame([
        {
            "feature": feature, "block": block, "singular": singular,
            "singular_lower": singular - 0.02, "singular_upper": singular + 0.02,
            "unique": unique, "unique_lower": unique - 0.004,
            "unique_upper": unique + 0.004,
        }
        for feature, block, singular, unique in values
    ])
    expected = {feature.replace("_", " ") for feature, *_ in values[:-1]}
    placements = []
    real_save = result_plots._save

    def inspect_and_save(fig, output_dir, name):
        fig.canvas.draw()
        renderer, ax = fig.canvas.get_renderer(), fig.axes[0]
        annotations = [artist for artist in fig.findobj(Annotation)
                       if artist.get_visible() and artist.get_text()]
        assert {artist.get_text().replace("\n", " ") for artist in annotations} == expected
        boxes = [artist.get_window_extent(renderer) for artist in annotations]
        assert not any(left.overlaps(right) for index, left in enumerate(boxes)
                       for right in boxes[index + 1:])
        assert all(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
                   for box in boxes)
        inverse = ax.transAxes.inverted()
        origin = inverse.transform(ax.transData.transform((0, 0)))
        positions = []
        for artist in annotations:
            point = inverse.transform(ax.transData.transform(artist.xy))
            label = np.asarray(artist.get_position(), dtype=float)
            assert np.dot(label - point, point - origin) > 0
            positions.append((artist.get_text(), *np.round(label, 6)))
        placements.append(positions)
        return real_save(fig, output_dir, name)

    monkeypatch.setattr(result_plots, "_save", inspect_and_save)

    _, svg = plot_feature_scatter(features, tmp_path)
    plot_feature_scatter(features, tmp_path)

    assert placements[0] == placements[1]
    assert "<text" in svg.read_text()


def test_dense_source_block_labels_do_not_wrap_or_overlap(tmp_path: Path, monkeypatch) -> None:
    features = pd.DataFrame({
        "feature": [f"long_clause_motif_feature_{index:02d}" for index in range(24)],
        "block": ["clause-motifs"] * 24,
        "singular": np.linspace(-0.01, 0.04, 24),
        "singular_lower": np.linspace(-0.012, 0.038, 24),
        "singular_upper": np.linspace(-0.008, 0.042, 24),
        "unique": np.linspace(-0.002, 0.002, 24),
        "unique_lower": np.linspace(-0.0025, 0.0015, 24),
        "unique_upper": np.linspace(-0.0015, 0.0025, 24),
    })
    coefficients = pd.DataFrame({
        "feature": features.feature, "estimate": np.linspace(-0.5, 0.5, len(features)),
        "lower": np.linspace(-0.6, 0.4, len(features)),
        "upper": np.linspace(-0.4, 0.6, len(features)), "fold_count": 2,
    })

    def inspect_labels(fig, output_dir, name):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [label for label in fig.axes[0].get_yticklabels() if label.get_visible()]
        assert all("\n" not in label.get_text() for label in labels)
        boxes = [label.get_window_extent(renderer) for label in labels]
        assert not any(left.overlaps(right) for index, left in enumerate(boxes)
                       for right in boxes[index + 1:])
        plt.close(fig)
        return Path(output_dir) / f"{name}.png", Path(output_dir) / f"{name}.svg"

    monkeypatch.setattr(result_plots, "_save", inspect_labels)

    plot_feature_blocks(features, coefficients, tmp_path)


def test_top_level_map_prioritizes_capability_grammar_and_marks_scatter_secondary(tmp_path: Path) -> None:
    prediction_paths = _prediction_files(tmp_path / "predictions")
    rejudge_path = _rejudge(tmp_path)
    lesion_path, dictionary_path = _lesions(tmp_path)
    selection_paths, coefficient_paths = _stability_files(tmp_path)

    outputs = write_result_plots(
        prediction_paths, rejudge_path, lesion_path, dictionary_path,
        selection_paths, coefficient_paths, tmp_path / "outputs",
        item_deltas_path=_item_deltas(tmp_path),
        judge_item_deltas_path=_judge_item_deltas(tmp_path),
    )["figures"]

    keys = list(outputs)
    assert keys.index("feature-effects-overview") < keys.index("group-effects")
    assert keys.index("group-effects") < keys.index("feature-scatter-secondary")
    assert "feature-scatter" not in outputs
    assert {"feature-effects-rubric", "feature-effects-clause-plan"} <= set(outputs)
    assert {"feature-effects-by-benchmark", "annotation-comparison"} <= set(outputs)
    assert {"feature-effects-j1-by-benchmark", "feature-effects-j2-by-benchmark"} <= set(outputs)
    assert outputs["feature-effects-overview"][0].parent.name == "overview"
    assert outputs["feature-effects-rubric"][0].parent.name == "by-feature-group"
    assert outputs["feature-effects-by-benchmark"][0].parent.name == "by-benchmark"
    assert outputs["feature-effects-j1-by-benchmark"][0].parent.name == "by-judge"
    assert outputs["feature-scatter-secondary"][0].parent.name == "secondary"
    assert {path.name for path in (tmp_path / "outputs").iterdir() if path.is_dir()} == {
        "overview", "by-feature-group", "by-benchmark", "by-judge", "secondary",
    }


def test_saved_item_deltas_write_editable_four_by_three_benchmark_grid(
    tmp_path: Path,
) -> None:
    lesion_path, dictionary_path = _lesions(tmp_path)
    effects = load_benchmark_feature_effects(_item_deltas(tmp_path), dictionary_path)
    raw_coefficients = result_plots.load_lesion_coefficients(
        lesion_path.with_name("artifact-index.csv"),
    )
    coefficients = result_plots.summarize_effective_coefficients(
        raw_coefficients, by=("benchmark",),
    )

    assert set(effects.effect) == {"singular", "unique"}
    assert set(effects.benchmark) == {"aime", "gpqa", "intercode", "swe_bench_mini"}
    assert effects.model_fold_count.eq(3).all() and effects.item_fold_count.eq(3).all()
    assert coefficients.loc[
        (coefficients.benchmark == "aime") & (coefficients.feature == "signal"),
        ["estimate", "lower", "upper", "fold_count"],
    ].iloc[0].tolist() == pytest.approx([1.55, 1.05, 2.05, 2])

    png, svg = plot_benchmark_feature_grid(effects, coefficients, tmp_path)

    assert png.exists() and svg.exists()
    source = svg.read_text()
    assert source.count('id="axes_') == 12
    assert all(label in source for label in (
        "AIME", "GPQA", "InterCode", "SWE", "Singular", "Unique", "Coefficient",
    ))
    assert "<text" in source and "β (1 SD)" in source


def test_judge_grids_use_exact_judge_by_benchmark_coefficients(tmp_path: Path) -> None:
    lesion_path, dictionary_path = _lesions(tmp_path)
    effects = load_benchmark_feature_effects(
        _judge_item_deltas(tmp_path), dictionary_path, by_judge=True,
    )
    coefficients = result_plots.summarize_effective_coefficients(
        result_plots.load_lesion_coefficients(
            lesion_path.with_name("artifact-index.csv"),
        ),
        by=("judge", "benchmark"),
    )

    signal = coefficients[
        (coefficients.judge == "j1") & (coefficients.benchmark == "aime")
        & (coefficients.feature == "signal")
    ].iloc[0]
    assert signal[["estimate", "lower", "upper", "fold_count"]].tolist() == pytest.approx(
        [1.5, 1.0, 2.0, 2],
    )

    outputs = result_plots.plot_judge_benchmark_grids(effects, coefficients, tmp_path)

    assert set(outputs) == {
        "feature-effects-j1-by-benchmark", "feature-effects-j2-by-benchmark",
    }
    for png, svg in outputs.values():
        assert png.exists() and svg.exists()
        source = svg.read_text()
        assert source.count('id="axes_') == 12 and "<text" in source
        assert all(label in source for label in ("AIME", "GPQA", "InterCode", "SWE"))


def test_benchmark_grid_uses_shared_support_rows_and_keeps_labels_apart(
    tmp_path: Path, monkeypatch,
) -> None:
    rows = []
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    features = [f"long_feature_label_{index:02d}" for index in range(12)]
    for benchmark in benchmarks:
        for index, feature in enumerate(features):
            for effect in ("singular", "unique"):
                if effect == "singular":
                    estimate = 0.07 - index * 0.01
                    spread = 0.006
                else:
                    estimate = 0.008 - index * 0.001
                    spread = 0.0007
                rows.append({
                    "benchmark": benchmark, "feature": feature,
                    "block": "surface" if index % 2 else "rubric", "effect": effect,
                    "estimate": estimate, "lower": estimate - spread,
                    "upper": estimate + spread,
                })
    effects = pd.DataFrame(rows)
    coefficients = pd.DataFrame([
        {
            "benchmark": benchmark, "feature": feature,
            "estimate": index / 10 - 0.5, "lower": index / 10 - 0.6,
            "upper": index / 10 - 0.4, "fold_count": 2,
        }
        for benchmark in benchmarks
        for index, feature in enumerate(features)
    ])

    def inspect(fig, output_dir, name):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert fig.get_size_inches().tolist() == pytest.approx([4.0, 5.6])
        assert len(fig.axes) == 12
        for axis_index, ax in enumerate(fig.axes):
            labels = [label for label in ax.get_yticklabels()
                      if label.get_visible() and label.get_text()]
            assert len(labels) == (10 if axis_index % 3 == 0 else 0)
            boxes = [label.get_window_extent(renderer) for label in labels]
            assert not any(left.overlaps(right) for index, left in enumerate(boxes)
                           for right in boxes[index + 1:])
            assert all(fig.bbox.contains(box.x0, box.y0)
                       and fig.bbox.contains(box.x1, box.y1) for box in boxes)
            x_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()
                       if label.get_visible() and label.get_text()]
            assert not any(left.overlaps(right) for index, left in enumerate(x_boxes)
                           for right in x_boxes[index + 1:])
        for column in range(3):
            assert len({fig.axes[row * 3 + column].get_xlim() for row in range(4)}) == 1
        assert [ax.get_title(loc="left") for ax in fig.axes[:3]] == [
            "Singular", "Unique", "Coefficient",
        ]
        plt.close(fig)
        return Path(output_dir) / f"{name}.png", Path(output_dir) / f"{name}.svg"

    monkeypatch.setattr(result_plots, "_save", inspect)

    output = plot_benchmark_feature_grid(effects, coefficients, tmp_path, top=10)

    assert output == (
        tmp_path / "feature-effects-by-benchmark.png",
        tmp_path / "feature-effects-by-benchmark.svg",
    )


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
