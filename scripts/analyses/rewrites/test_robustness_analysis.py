import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml


PATH = Path(__file__).with_name("robustness_analysis.py")
SPEC = importlib.util.spec_from_file_location("robustness_analysis", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def cell_frame(judges=analysis.JUDGE_ORDER):
    rows = []
    for judge_index, judge in enumerate(judges):
        for condition_index, condition in enumerate(analysis.CONDITIONS):
            for gap_index, gap in enumerate((5.0, 15.0, 30.0)):
                rows.append({
                    "judge": judge,
                    "condition": condition,
                    "model_pair": f"pair-{gap_index}",
                    "item": f"item-{gap_index}",
                    "capability_gap": gap,
                    "observed": 0.55 + 0.05 * judge_index + 0.04 * condition_index,
                    "predicted": 0.57 + 0.04 * judge_index + 0.02 * condition_index,
                })
    return pd.DataFrame(rows)


def test_analysis_is_original_vs_style_standardised_gpt_only():
    assert analysis.CONDITIONS == ("a_baseline", "style_standardised_gpt")
    assert analysis.CONDITION_LABELS == ("Original", "Style-standardised")
    assert set(analysis.STYLE_RATES) == set(analysis.CONDITIONS)


def test_load_verdicts_requires_baseline_and_combines_explicit_inputs(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    rewrite_path = tmp_path / "rewrite.csv"
    rewrite = pd.DataFrame({
        "condition": ["style_standardised_gpt"] * 3000,
        "row": range(3000),
    })
    rewrite.to_csv(baseline_path, index=False)
    rewrite.to_csv(rewrite_path, index=False)

    with pytest.raises(ValueError, match="baseline.*a_baseline"):
        analysis.load_verdicts(baseline_path, rewrite_path)

    pd.DataFrame({
        "condition": ["a_baseline", "old_rewriter"], "row": [-1, -2],
    }).to_csv(baseline_path, index=False)
    pd.DataFrame({"condition": ["a_baseline"], "row": [-1]}).to_csv(
        rewrite_path, index=False
    )
    with pytest.raises(ValueError, match="rewrite.*style_standardised_gpt"):
        analysis.load_verdicts(baseline_path, rewrite_path)

    rewrite.to_csv(rewrite_path, index=False)
    loaded = analysis.load_verdicts(baseline_path, rewrite_path)

    assert len(loaded) == 3001
    assert set(loaded.condition) == set(analysis.CONDITIONS)
    assert analysis.BASELINE_VERDICTS == (
        analysis.ROOT / "judge_results/judging_variations/verdicts.csv"
    )
    assert analysis.REWRITE_VERDICTS == (
        analysis.ROOT / "judge_results/robustness/style_standardised_gpt/verdicts_rewrite.csv"
    )


def test_predictor_order_matches_the_frozen_correctness_config():
    config = yaml.safe_load(
        (analysis.ROOT / "scripts/hibayes/config_correctness.yaml").read_text()
    )
    frozen_order = config["model"]["models"][0]["config"]["predictors"]

    assert analysis.PREDICTORS == tuple(frozen_order)


def test_default_yaml_campaign_loads_and_requires_exactly_50_by_30_cells():
    assert analysis.CAMPAIGN == (
        analysis.ROOT / "scripts/pairwise-newjudges/configs/manipulations/"
        "style_standardised_gpt.yaml"
    )
    loaded = analysis.load_campaign(analysis.CAMPAIGN)
    assert len(loaded) == 50
    assert {len(items) for items in loaded.values()} == {30}

    valid = {f"major-{index}-vs-minor-{index}": [f"item-{item}" for item in range(30)]
             for index in range(50)}
    analysis.validate_campaign(valid)

    invalid = dict(valid)
    invalid[next(iter(invalid))] = invalid[next(iter(invalid))][:-1]
    with pytest.raises(ValueError, match="50 pairs × 30 items"):
        analysis.validate_campaign(invalid)


def test_style_rates_schema_is_exactly_the_frozen_models_six_categories():
    columns = ["model", "item", "n_chars", "reasoning_chars", *analysis.STYLE_CATEGORIES]
    analysis.validate_style_rates(pd.DataFrame(columns=columns), Path("rates.csv"))

    with pytest.raises(ValueError, match="canonical style categories"):
        analysis.validate_style_rates(
            pd.DataFrame(columns=[*columns, "invented_seventh_rate"]), Path("bad.csv")
        )


def test_summarise_requires_both_orders_and_complete_conditions():
    rows = []
    for judge in ("judge-a", "judge-b"):
        for condition in analysis.CONDITIONS:
            for item in ("i1", "i2"):
                for order in (1, 2):
                    rows.append({
                        "condition": condition, "judge": judge, "major": "strong",
                        "minor": "weak", "item_id": item, "order": order,
                        "picked_major": item == "i1",
                        "parse_ok": (judge, condition, item, order) != (
                            "judge-b", "style_standardised_gpt", "i2", 2
                        ),
                    })

    estimates, coverage = analysis.summarise(
        pd.DataFrame(rows), {"strong-vs-weak": ["i1", "i2"]}
    )

    assert set(estimates.n_cells) == {1}
    assert set(estimates.condition) == set(analysis.CONDITIONS)
    cov = coverage.set_index(["judge", "condition"])
    assert cov.loc[("judge-b", "style_standardised_gpt"), "complete_cells"] == 1


def test_paired_cells_uses_one_intersection_for_all_judges_and_conditions():
    rows = []
    for judge in analysis.JUDGE_ORDER:
        for condition in analysis.CONDITIONS:
            for item in ("i1", "i2"):
                for order in (1, 2):
                    if (judge, condition, item, order) == (
                        analysis.JUDGE_ORDER[-1], analysis.CONDITIONS[-1], "i2", 2
                    ):
                        continue
                    rows.append({
                        "judge": judge, "transcript_condition": condition,
                        "model_pair": "strong|weak", "item": item, "order": order,
                        "is_correct": item == "i1", "capability_gap": 10.0,
                        "predicted": 0.7,
                    })

    cells = analysis.paired_cells(
        pd.DataFrame(rows), {"strong-vs-weak": ["i1", "i2"]}
    )

    assert len(cells) == len(analysis.JUDGE_ORDER) * len(analysis.CONDITIONS)
    assert set(cells.item) == {"i1"}


def test_accuracy_tables_use_judges_as_paired_replication_units():
    by_judge, summary = analysis.accuracy_tables(cell_frame())

    assert len(by_judge) == len(analysis.JUDGE_ORDER) * len(analysis.CONDITIONS)
    assert set(by_judge[by_judge.condition == "a_baseline"].delta_vs_baseline) == {0.0}
    rewritten = summary[summary.condition == "style_standardised_gpt"].iloc[0]
    assert rewritten.n_judges == 3
    assert rewritten.delta == pytest.approx(0.04)
    assert rewritten.delta_ci_lo <= rewritten.delta <= rewritten.delta_ci_hi


def test_accuracy_tables_show_points_but_no_fake_ci_for_two_judges():
    by_judge, summary = analysis.accuracy_tables(cell_frame(analysis.JUDGE_ORDER[:2]))

    assert len(by_judge.judge.unique()) == 2
    assert summary[["ci_lo", "ci_hi", "delta_ci_lo", "delta_ci_hi"]].isna().all().all()


def test_build_designs_recomputes_every_transcript_feature_per_condition(
    tmp_path, monkeypatch
):
    style_paths = {}
    for condition, marker in zip(analysis.CONDITIONS, (0.25, 1.75), strict=True):
        path = tmp_path / f"{condition}.csv"
        pd.DataFrame({
            "model": ["a"], "item": ["item-a"], "n_chars": [marker],
            "reasoning_chars": [1],
            **{category: [0.0] for category in analysis.STYLE_CATEGORIES},
        }).to_csv(path, index=False)
        style_paths[condition] = path

    reference = pd.DataFrame({name: [0.0, 2.0] for name in analysis.SCALED_FEATURES})
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    def fake_build(selected, capabilities, bench, conditions, style_rates):
        marker = float(style_rates.n_chars.iloc[0])
        frame = pd.DataFrame({
            "judge": selected.judge.to_numpy(), "item": selected.item_id.to_numpy(),
            "order": selected.order.to_numpy(), "model_pair": ["a|b"] * len(selected),
            "same_model_family": [1] * len(selected), "truth_side": [1.0] * len(selected),
            "capability_gap": [0.5] * len(selected),
        })
        for feature in analysis.TRANSCRIPT_FEATURES:
            frame[feature] = marker
        return frame, {}

    fake_correctness = types.SimpleNamespace(
        difference_columns=lambda rates: [], to_correctness=lambda frame, differences: (frame, [])
    )
    fake_from_verdicts = types.SimpleNamespace(
        build=fake_build, load_capabilities=lambda path, field: {}, CAPABILITY_CONFIG=Path("caps")
    )
    monkeypatch.setitem(sys.modules, "prepare_correctness", fake_correctness)
    monkeypatch.setitem(sys.modules, "prepare_from_verdicts", fake_from_verdicts)

    verdicts = pd.DataFrame([
        {"condition": condition, "judge": "judge-a", "item_id": "item-a", "order": 1}
        for condition in analysis.CONDITIONS
    ])
    designs = analysis.build_designs(verdicts, style_paths, reference_path)
    original = designs[designs.transcript_condition == "a_baseline"].iloc[0]
    rewritten = designs[designs.transcript_condition == "style_standardised_gpt"].iloc[0]

    for feature in analysis.TRANSCRIPT_FEATURES:
        assert original[f"{feature}_sd"] != rewritten[f"{feature}_sd"]
    assert original.capability_gap_sd == rewritten.capability_gap_sd
    assert original.same_model_family == rewritten.same_model_family == 1
    assert rewritten.truth_side_x_total_length == rewritten.total_length_sd
    assert np.isfinite(designs[list(analysis.PREDICTORS)].to_numpy()).all()


def test_predict_glm_uses_each_named_judges_fit_and_item_effect(tmp_path):
    coords = {"judge": ["judge-a", "judge-b"], "item": ["item-a"]}
    coords_path = tmp_path / "coords.json"
    coords_path.write_text(json.dumps(coords))
    terms = [
        {"term": "judge_intercepts[0]", "mean": 0.0},
        {"term": "judge_intercepts[1]", "mean": math.log(2)},
        {"term": "item_effects[item-a]", "mean": 0.1},
    ]
    for judge_index, weight in enumerate((0.2, -0.3)):
        terms.extend({
            "term": f"judge_predictor_effects[{judge_index}, {predictor_index}]",
            "mean": weight if predictor == "hedging_difference_stronger_sd" else 0.0,
        } for predictor_index, predictor in enumerate(analysis.PREDICTORS))
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(terms).to_csv(summary_path, index=False)
    frame = pd.DataFrame([
        {**{name: 0.0 for name in analysis.PREDICTORS}, "judge": judge,
         "item": "item-a", "hedging_difference_stronger_sd": 2.0}
        for judge in coords["judge"]
    ])

    prediction = analysis.predict_glm(frame, summary_path, coords_path)

    assert prediction.iloc[0] == pytest.approx(1 / (1 + math.exp(-0.5)))
    assert prediction.iloc[1] == pytest.approx(1 / (1 + math.exp(-(math.log(2) - 0.5))))


def test_gap_summary_uses_judges_for_ci_and_suppresses_it_below_three():
    full = analysis.gap_summary(cell_frame())
    sparse = analysis.gap_summary(cell_frame(analysis.JUDGE_ORDER[:2]))

    assert full.obs_ci_lo.notna().all()
    assert full.pred_ci_lo.notna().all()
    assert sparse[["obs_ci_lo", "obs_ci_hi", "pred_ci_lo", "pred_ci_hi"]].isna().all().all()


def test_style_delta_tables_use_major_minus_minor_and_pair_means(tmp_path):
    columns = ["model", "item", "n_chars", "reasoning_chars", *analysis.STYLE_CATEGORIES]
    original = pd.DataFrame([
        ["major", "i1", 10, 9, 5, 4, 3, 2, 1, 0],
        ["minor", "i1", 10, 9, 1, 2, 3, 4, 5, 6],
        ["major", "i2", 10, 9, 7, 6, 5, 4, 3, 2],
        ["minor", "i2", 10, 9, 2, 3, 4, 5, 6, 7],
    ], columns=columns)
    rewritten = original.copy()
    rewritten.loc[rewritten.model == "major", list(analysis.STYLE_CATEGORIES)] = 0
    rewritten.loc[rewritten.model == "minor", list(analysis.STYLE_CATEGORIES)] = 1
    paths = {}
    for condition, frame in zip(analysis.CONDITIONS, (original, rewritten), strict=True):
        paths[condition] = tmp_path / f"{condition}.csv"
        frame.to_csv(paths[condition], index=False)

    cells, pairs = analysis.style_delta_tables(
        {"major-vs-minor": ["i1", "i2"]}, paths
    )

    assert len(cells) == 2 * 2 * len(analysis.STYLE_CATEGORIES)
    assert len(pairs) == 2 * len(analysis.STYLE_CATEGORIES)
    pronouns = cells[
        (cells.condition == "a_baseline") &
        (cells.feature == "personal_pronouns")
    ]
    assert pronouns.delta_major_minus_minor.tolist() == [4.0, 5.0]
    pair = pairs[
        (pairs.condition == "a_baseline") &
        (pairs.feature == "personal_pronouns")
    ].iloc[0]
    assert pair.delta_major_minus_minor == pytest.approx(4.5)
    assert pair.n_items == 2


def test_figures_are_small_nature_style_and_show_judges_and_dashed_glm():
    cells = cell_frame()
    accuracy = analysis.make_accuracy_figure(cells)
    by_judge = analysis.gap_by_judge(cells)
    gap = analysis.make_gap_figure(analysis.gap_summary(cells), by_judge)

    assert tuple(accuracy.get_size_inches()) == pytest.approx((3.6, 1.9))
    assert tuple(gap.get_size_inches()) == pytest.approx((2.9, 1.9))
    assert len(accuracy.axes) == len(gap.axes) == 2
    assert all(not ax.spines["top"].get_visible() for ax in [*accuracy.axes, *gap.axes])
    assert all(ax.spines["left"].get_position() == ("outward", 5)
               for ax in [*accuracy.axes, *gap.axes])
    assert any(line.get_linestyle() == "--" for line in gap.axes[0].lines)
    assert accuracy.axes[0].get_ylabel() == "Judge accuracy (%)"
    assert accuracy.axes[1].get_ylabel() == "Rewrite − original (pp)"
    assert not {"a", "b"} & {text.get_text() for ax in accuracy.axes for text in ax.texts}
    assert max(line.get_markersize() for ax in accuracy.axes for line in ax.lines) >= 4
    assert accuracy.legends and gap.legends
    assert not accuracy.legends[0].get_frame_on() and not gap.legends[0].get_frame_on()


def test_gap_shared_xlabel_does_not_overlap_ticks_or_legend():
    cells = cell_frame()
    gap = analysis.make_gap_figure(analysis.gap_summary(cells))
    gap.canvas.draw()
    renderer = gap.canvas.get_renderer()
    xlabel = gap._supxlabel.get_window_extent(renderer)
    tick_labels = [
        tick.get_window_extent(renderer)
        for ax in gap.axes for tick in ax.get_xticklabels()
    ]
    legend = gap.legends[0].get_window_extent(renderer)

    assert not any(xlabel.overlaps(tick) for tick in tick_labels)
    assert not xlabel.overlaps(legend)


def test_style_delta_figures_are_compact_and_cover_all_six_regressors():
    rows = []
    for condition_index, condition in enumerate(analysis.CONDITIONS):
        for pair_index in range(3):
            for feature_index, feature in enumerate(analysis.STYLE_CATEGORIES):
                rows.append({
                    "condition": condition, "pair": f"pair-{pair_index}",
                    "major": "major", "minor": "minor", "feature": feature,
                    "delta_major_minus_minor": feature_index - condition_index,
                    "n_items": 30,
                })
    pairs = pd.DataFrame(rows)
    figures = [
        analysis.make_style_delta_figure(pairs, features)
        for features in (analysis.TARGETED_STYLE_CATEGORIES,
                         analysis.OTHER_STYLE_CATEGORIES)
    ]

    assert set(analysis.TARGETED_STYLE_CATEGORIES + analysis.OTHER_STYLE_CATEGORIES) == set(
        analysis.STYLE_CATEGORIES
    )
    for figure, features in zip(
        figures,
        (analysis.TARGETED_STYLE_CATEGORIES, analysis.OTHER_STYLE_CATEGORIES),
        strict=True,
    ):
        assert tuple(figure.get_size_inches()) == pytest.approx((4.3, 1.9))
        assert len(figure.axes) == 3
        assert {ax.get_title(loc="left") for ax in figure.axes} == {
            analysis.STYLE_LABELS[feature] for feature in features
        }
        assert all(ax.get_xticklabels()[0].get_text() == "Original" for ax in figure.axes)
        assert all(not ax.spines["top"].get_visible() for ax in figure.axes)
        assert figure.legends and not figure.legends[0].get_frame_on()


def test_axes_end_at_endpoint_ticks_and_panel_labels_are_above_data():
    cells = cell_frame()
    accuracy = analysis.make_accuracy_figure(cells)
    gap = analysis.make_gap_figure(analysis.gap_summary(cells))
    pairs = pd.DataFrame([{
        "condition": condition, "pair": "pair", "major": "major",
        "minor": "minor", "feature": feature,
        "delta_major_minus_minor": 0.0, "n_items": 30,
    } for condition in analysis.CONDITIONS for feature in analysis.TARGETED_STYLE_CATEGORIES])
    styles = analysis.make_style_delta_figure(
        pairs, analysis.TARGETED_STYLE_CATEGORIES
    )

    endpoint_axes = [accuracy.axes[0], *gap.axes, *styles.axes]
    for ax in endpoint_axes:
        assert ax.get_xlim() == pytest.approx((ax.get_xticks()[0], ax.get_xticks()[-1]))
        assert ax.get_ylim() == pytest.approx((ax.get_yticks()[0], ax.get_yticks()[-1]))
    assert not accuracy.axes[1].spines["bottom"].get_visible()

    titled = [
        (gap, analysis.CONDITION_LABELS),
        (styles, tuple(analysis.STYLE_LABELS[name]
                       for name in analysis.TARGETED_STYLE_CATEGORIES)),
    ]
    for figure, expected in titled:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for ax, label in zip(figure.axes, expected, strict=True):
            assert ax.get_title(loc="left") == label
            title_box = ax._left_title.get_window_extent(renderer)
            axes_box = ax.get_window_extent(renderer)
            assert title_box.y0 >= axes_box.y1
            assert figure.bbox.contains(*title_box.get_points()[0])
            assert figure.bbox.contains(*title_box.get_points()[1])


def test_style_delta_condition_labels_do_not_overlap_or_clip():
    pairs = pd.DataFrame([{
        "condition": condition, "pair": "pair", "major": "major",
        "minor": "minor", "feature": feature,
        "delta_major_minus_minor": 0.0, "n_items": 30,
    } for condition in analysis.CONDITIONS for feature in analysis.TARGETED_STYLE_CATEGORIES])
    figure = analysis.make_style_delta_figure(
        pairs, analysis.TARGETED_STYLE_CATEGORIES
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = [
        label.get_window_extent(renderer)
        for ax in figure.axes for label in ax.get_xticklabels()
    ]
    ylabel = figure.axes[0].yaxis.label.get_window_extent(renderer)
    legend = figure.legends[0].get_window_extent(renderer)

    assert not any(
        left.overlaps(right)
        for index, left in enumerate(labels) for right in labels[index + 1:]
    )
    assert all(figure.bbox.contains(*box.get_points()[0]) and
               figure.bbox.contains(*box.get_points()[1]) for box in labels)
    assert figure.bbox.contains(*ylabel.get_points()[0])
    assert figure.bbox.contains(*ylabel.get_points()[1])
    assert not any(label.overlaps(legend) for label in labels)


def test_write_outputs_uses_shared_cells_and_keeps_svg_text_editable(tmp_path):
    cells = cell_frame()
    coverage = pd.DataFrame([{"judge": "j", "condition": "c", "observed_rows": 1}])
    style_cells = pd.DataFrame([{
        "condition": condition, "pair": "major-vs-minor", "major": "major",
        "minor": "minor", "item": "item", "feature": feature,
        "delta_major_minus_minor": 0.0,
    } for condition in analysis.CONDITIONS for feature in analysis.STYLE_CATEGORIES])
    style_pairs = style_cells.assign(n_items=1)
    manuscript = tmp_path / "manuscript" / "04-ablations"

    analysis.write_outputs(
        cells, coverage, tmp_path, style_cells, style_pairs, manuscript,
    )

    expected = {
        "accuracy_by_judge.csv", "accuracy_summary.csv", "coverage.csv",
        "cell_predictions.csv", "gap_by_judge.csv", "gap_estimates.csv",
        "overall_observed_vs_glm.csv", "accuracy_original_vs_rewrite.png",
        "accuracy_original_vs_rewrite.svg", "gap_curves.png", "gap_curves.svg",
        "style_deltas_by_cell.csv", "style_deltas_by_pair.csv",
        "style_deltas_targeted.png", "style_deltas_targeted.svg",
        "style_deltas_other.png", "style_deltas_other.svg",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert "<text" in (tmp_path / "accuracy_original_vs_rewrite.svg").read_text()
    assert "<text" in (tmp_path / "gap_curves.svg").read_text()
    assert "<text" in (tmp_path / "style_deltas_targeted.svg").read_text()
    assert "<text" in (tmp_path / "style_deltas_other.svg").read_text()
    manuscript_names = {
        "style_standardisation_accuracy.png",
        "style_standardisation_accuracy.svg",
        "style_standardisation_gap_curves.png",
        "style_standardisation_gap_curves.svg",
        "style_standardisation_targeted_features.png",
        "style_standardisation_targeted_features.svg",
        "style_standardisation_other_features.png",
        "style_standardisation_other_features.svg",
    }
    assert manuscript_names == {path.name for path in manuscript.iterdir()}
    mapping = {
        "accuracy_original_vs_rewrite": "style_standardisation_accuracy",
        "gap_curves": "style_standardisation_gap_curves",
        "style_deltas_targeted": "style_standardisation_targeted_features",
        "style_deltas_other": "style_standardisation_other_features",
    }
    for source, destination in mapping.items():
        for suffix in ("png", "svg"):
            assert (tmp_path / f"{source}.{suffix}").read_bytes() == (
                manuscript / f"{destination}.{suffix}"
            ).read_bytes()
