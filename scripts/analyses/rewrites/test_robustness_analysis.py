import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


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


def test_figures_are_small_nature_style_and_show_judges_and_dashed_glm():
    cells = cell_frame()
    accuracy = analysis.make_accuracy_figure(cells)
    by_judge = analysis.gap_by_judge(cells)
    gap = analysis.make_gap_figure(analysis.gap_summary(cells), by_judge)

    assert tuple(accuracy.get_size_inches()) == pytest.approx((3.8, 2.4))
    assert tuple(gap.get_size_inches()) == pytest.approx((4.2, 2.4))
    assert len(accuracy.axes) == len(gap.axes) == 2
    assert all(not ax.spines["top"].get_visible() for ax in [*accuracy.axes, *gap.axes])
    assert all(ax.spines["left"].get_position() == ("outward", 5)
               for ax in [*accuracy.axes, *gap.axes])
    assert any(line.get_linestyle() == "--" for line in gap.axes[0].lines)
    assert accuracy.legends and gap.legends
    assert not accuracy.legends[0].get_frame_on() and not gap.legends[0].get_frame_on()


def test_write_outputs_uses_shared_cells_and_keeps_svg_text_editable(tmp_path):
    cells = cell_frame()
    coverage = pd.DataFrame([{"judge": "j", "condition": "c", "observed_rows": 1}])

    analysis.write_outputs(cells, coverage, tmp_path)

    expected = {
        "accuracy_by_judge.csv", "accuracy_summary.csv", "coverage.csv",
        "cell_predictions.csv", "gap_by_judge.csv", "gap_estimates.csv",
        "overall_observed_vs_glm.csv", "accuracy_original_vs_rewrite.png",
        "accuracy_original_vs_rewrite.svg", "gap_curves.png", "gap_curves.svg",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert "<text" in (tmp_path / "accuracy_original_vs_rewrite.svg").read_text()
    assert "<text" in (tmp_path / "gap_curves.svg").read_text()
