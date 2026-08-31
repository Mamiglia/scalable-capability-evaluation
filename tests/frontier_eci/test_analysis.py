from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest


module = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/analyze.py")
calibrate = module["calibrate"]
merge_rollouts = module["merge_rollouts"]
prefix_verdicts = module["prefix_verdicts"]
consensus_verdicts = module["consensus_verdicts"]
predict_block = module["predict_block"]
plot_frontier_style = module["plot_frontier_style"]


def test_target_actual_does_not_enter_calibration():
    ability = pd.Series({"a": -2.0, "b": -0.5, "c": 0.5, "d": 2.0,
                         "frontier-a": 1.0, "frontier-b": 1.5})
    actual = pd.Series({"a": 0.1, "b": 0.3, "c": 0.6, "d": 0.9,
                        "frontier-a": 0.0, "frontier-b": 1.0})
    first, anchors = calibrate(ability, actual, ["a", "b", "c", "d"], ["frontier-a"])
    changed = actual.copy()
    changed[["frontier-a", "frontier-b"]] = [1.0, 0.0]
    second, _ = calibrate(ability, changed, ["a", "b", "c", "d"], ["frontier-a"])
    pd.testing.assert_series_equal(first, second)
    assert anchors == 4


def test_future_edges_are_absent_from_an_earlier_prefix():
    verdicts = pd.DataFrame({
        "model_left": ["past", "target"], "model_right": ["target", "future"],
        "pick_left": [0, 1], "bench": ["gpqa", "gpqa"],
        "judge": ["j", "j"], "item": ["x", "x"],
    })
    releases = {"past": "2024-01-01", "target": "2025-01-01", "future": "2026-01-01"}
    prefix = prefix_verdicts(verdicts, "target", releases)
    assert len(prefix) == 1
    assert set(prefix.iloc[0][["model_left", "model_right"]]) == {"past", "target"}


def test_later_judge_allows_models_available_by_judge_release():
    verdicts = pd.DataFrame({
        "model_left": ["target"], "model_right": ["intermediate"],
        "pick_left": [1], "bench": ["gpqa"], "judge": ["later-judge"], "item": ["x"],
    })
    releases = {"target": "2024-01-01", "intermediate": "2025-01-01",
                "later-judge": "2026-01-01"}
    assert len(prefix_verdicts(verdicts, "target", releases, "later-judge")) == 1


def test_consensus_weights_judges_not_uploaded_rounds():
    rows = pd.DataFrame({
        "bench": ["gpqa"] * 3, "item": ["x"] * 3,
        "model_left": ["a"] * 3, "model_right": ["b"] * 3,
        "epoch_left": [0] * 3, "epoch_right": [0] * 3,
        "judge": ["many", "many", "one"], "judge_epoch": [1, 2, 1],
        "pick_left": [0, 0, 1],
    })
    assert consensus_verdicts(rows).iloc[0].pick_left == 0.5


def test_empty_earliest_prefix_is_actual_only():
    row = predict_block(pd.DataFrame(), pd.Series({"first": 0.7}), "first",
                        {"first": "2024-01-01"}, 10, np.random.default_rng(0))
    assert row["actual"] == 0.7
    assert pd.isna(row["predicted"])
    assert row["n_prefix_verdicts"] == 0


def test_canonical_rollouts_win_when_model_is_already_present():
    canonical = pd.DataFrame({"bench": ["gpqa"], "model": ["a"], "item": ["x"],
                              "epoch": [0], "correct": [True], "is_canonical": [True]})
    frontier = pd.DataFrame({"bench": ["gpqa", "gpqa"], "model": ["a", "b"],
                             "item": ["x", "x"], "epoch": [0, 0],
                             "correct": [False, False], "is_canonical": [True, True]})
    merged = merge_rollouts(canonical, frontier)
    assert merged.set_index("model").correct.to_dict() == {"a": True, "b": False}


def test_frontier_plot_is_narrow_and_keeps_svg_text_editable(tmp_path):
    table = pd.DataFrame({
        "target_release": ["2024-11-20", "2026-07-16"],
        "judge_release": ["2025-03-25"] * 2,
        "actual": [0.4, 0.8], "predicted": [0.5, 0.7],
        "predicted_low": [0.3, 0.6], "predicted_high": [0.7, 0.8],
    })
    plot_frontier_style(table, "gpt-4o-2024-11-20", "gpqa", 2,
                        tmp_path, "band")
    stem = tmp_path / "gpt-4o-2024-11-20_gpqa_q2_frontier_band"
    assert stem.with_suffix(".png").exists()
    assert stem.with_suffix(".pdf").exists()
    svg = stem.with_suffix(".svg")
    assert "<text" in svg.read_text()
    from PIL import Image
    with Image.open(stem.with_suffix(".png")) as image:
        assert image.width <= 650


def test_frontier_plot_breaks_band_at_missing_prediction_and_does_not_clip_actuals(tmp_path):
    table = pd.DataFrame({
        "target_release": ["2024-11-20", "2025-06-01", "2026-07-16"],
        "judge_release": ["2025-03-25"] * 3,
        "actual": [0.0, 0.5, 1.0], "predicted": [0.2, np.nan, 0.8],
        "predicted_low": [0.1, np.nan, 0.7],
        "predicted_high": [0.3, np.nan, 0.9],
    })
    ax = plot_frontier_style(table, "gpt-4o-2024-11-20", "gpqa", 2,
                             tmp_path, "band")
    prediction = next(line for line in ax.lines
                      if line.get_color() == "#2a78d6" and len(line.get_ydata()) == 3)
    actual = next(line for line in ax.lines
                  if line.get_color() == "#222222" and line.get_marker() == "o")
    assert np.isnan(prediction.get_ydata()[1])
    assert len(ax.collections[0].get_paths()) == 2
    assert actual.get_clip_on() is False
    assert ax.get_yticks().tolist() == [0.0, 0.5, 1.0]


def test_rank_regression_plot_uses_exact_in_sample_label_and_clips_display_only(tmp_path):
    table = pd.DataFrame({
        "target_release": ["2024-11-20", "2025-06-01"],
        "judge_release": ["2025-03-25"] * 2,
        "actual": [0.0, 1.0], "fitted": [-0.2, 1.2],
        "fitted_low": [-0.4, 1.0], "fitted_high": [0.0, 1.4],
    })
    original = table.copy()
    ax = plot_frontier_style(
        table, "gpt-4o-2024-11-20", "gpqa", 2, tmp_path, "line",
        prediction_label="Rank–accuracy fit (in-sample)",
        value_column="fitted", low_column="fitted_low", high_column="fitted_high",
        clip_prediction_display=True,
    )
    blue = next(line for line in ax.lines if line.get_color() == "#2a78d6")
    assert blue.get_ydata().tolist() == [0.0, 1.0]
    assert ax.get_legend().get_texts()[1].get_text() == "Rank–accuracy fit (in-sample)"
    pd.testing.assert_frame_equal(table, original)


def test_combined_rank_plot_has_item_errorbars_and_band(tmp_path):
    table = pd.DataFrame({
        "target_release": ["2024-11-20", "2025-06-01"],
        "judge_release": ["2025-03-25"] * 2,
        "actual": [0.3, 0.7], "fitted": [0.4, 0.6],
        "fitted_low": [0.2, 0.45], "fitted_high": [0.6, 0.8],
    })
    points = plot_frontier_style(
        table, "gpt-4o-2024-11-20", "gpqa", "2-4", tmp_path, "points",
        prediction_label="Rank–accuracy fit (in-sample)",
        value_column="fitted", low_column="fitted_low", high_column="fitted_high",
        panel_label="GPQA · Q2–Q4", y_label="Combined hard accuracy",
        file_quartile="2-q4",
    )
    assert any(len(collection.get_segments()) == 2 for collection in points.collections
               if hasattr(collection, "get_segments"))
    assert points.get_ylabel() == "Combined hard accuracy"
    band = plot_frontier_style(
        table, "gpt-4o-2024-11-20", "gpqa", "2-4", tmp_path, "band",
        prediction_label="Rank–accuracy fit (in-sample)",
        value_column="fitted", low_column="fitted_low", high_column="fitted_high",
        panel_label="GPQA · Q2–Q4", y_label="Combined hard accuracy",
        file_quartile="2-q4",
    )
    assert any(collection.get_alpha() == pytest.approx(0.18)
               for collection in band.collections)


def test_pooled_plot_omits_single_judge_release_line(tmp_path):
    table = pd.DataFrame({
        "target_release": ["2025-01-01", "2025-04-01"],
        "actual": [0.3, 0.7], "predicted": [0.4, 0.6],
        "predicted_low": [0.2, 0.45], "predicted_high": [0.6, 0.8],
    })
    ax = plot_frontier_style(
        table, "pooled", "gpqa", "2-4", tmp_path, "band",
        prediction_label="Pooled pre-release judges",
        panel_label="GPQA · Q2–Q4", y_label="Combined hard accuracy",
        file_quartile="2-q4", show_judge_release=False,
    )
    assert "Judge release" not in [text.get_text() for text in ax.get_legend().get_texts()]
    assert not any(line.get_linestyle() == "--" for line in ax.lines)
