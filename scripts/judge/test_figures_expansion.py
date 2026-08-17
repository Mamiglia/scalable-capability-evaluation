"""The per-transcript filter re-tallies run stats from the item-level verdicts.

With the floor at zero it must reproduce the exported stats exactly, or the
filtered numbers would be measuring the re-tally rather than the filter.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).parent))
import figures_expansion as expansion

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["data-expansion-gpqa", "data-expansion-aime2025", "data-expansion-aime2026"]
STATS = ("score", "n_items", "slot_a", "noanswer_rate")


@pytest.mark.parametrize("dataset", DATASETS)
def test_refit_reproduces_the_exported_run_stats(dataset, monkeypatch):
    d = ROOT / "interactive_dashboards" / "transcript_viewer" / "data" / dataset
    if not (d / "index.json").exists():
        pytest.skip(f"{dataset} not exported here")
    monkeypatch.setattr(expansion, "MIN_REASONING_CHARS", 0)
    expansion.transcript_chars.cache_clear()
    expansion.refit_runs.cache_clear()

    stats = expansion.refit_runs(d)
    checked = 0
    for run in json.load(open(d / "index.json"))["runs"]:
        # self-pairs have no major to pick, so the refit leaves them alone
        if run.get("frac", "full") != "full" or run["is_self"]:
            continue
        refit = stats[(run["judge"], run["major"], run["minor"])]
        for name in STATS:
            if run[name] is not None:
                assert refit[name] == pytest.approx(run[name]), f"{run['run_id']} {name}"
        checked += 1
    assert checked > 0


def test_gap_bins_follow_unique_design_points_not_repeated_judges():
    gaps = [i / 20 for i in range(20)]
    edges = expansion._gap_bin_edges(gaps)

    assert len(edges) - 1 > 4
    assert edges == pytest.approx(expansion._gap_bin_edges(gaps * 6))


def test_gap_bins_reuse_full_data_width_for_a_narrow_subset():
    full = [i / 100 for i in range(101)]
    subset = [x for x in full if 0.05 <= x <= 0.20]

    full_edges = expansion._gap_bin_edges(full)
    subset_edges = expansion._gap_bin_edges(subset, reference=full)

    assert len(subset_edges) < len(full_edges)
    assert subset_edges[1] - subset_edges[0] == pytest.approx(
        full_edges[1] - full_edges[0])


def test_gap_mean_is_drawn_at_an_observed_x_not_the_bin_midpoint():
    rows = [
        {"judge": "a", "is_self": False, "gap": 0.19, "score": 0.4, "n_items": 10},
        {"judge": "b", "is_self": False, "gap": 0.19, "score": 0.6, "n_items": 10},
    ]
    fig, ax = plt.subplots()

    expansion._judge_mean_band(ax, rows, "gap", reference_gaps=[i / 100 for i in range(101)])

    assert ax.lines[-1].get_xdata() == pytest.approx([0.19])
    plt.close(fig)


def test_gap_mean_can_include_self_pairs_at_zero():
    rows = [
        {"judge": "a", "is_self": True, "gap": 0, "score": 0.4, "n_items": 10},
        {"judge": "b", "is_self": True, "gap": 0, "score": 0.6, "n_items": 10},
        {"judge": "a", "is_self": False, "gap": 0.1, "score": 0.5, "n_items": 10},
        {"judge": "b", "is_self": False, "gap": 0.1, "score": 0.7, "n_items": 10},
    ]
    fig, ax = plt.subplots()

    expansion._judge_mean_band(ax, rows, "gap", include_self=True)

    assert ax.lines[-1].get_xdata()[0] == 0
    assert ax.lines[-1].get_ydata()[0] == pytest.approx(0.5)
    plt.close(fig)


def test_gap_mean_can_overlay_binned_per_judge_lines(monkeypatch):
    rows = [
        {"judge": judge, "is_self": gap == 0, "gap": gap,
         "score": 0.5 + gap, "n_items": 10}
        for judge in ("a", "b") for gap in (0, 0.05, 0.15, 0.25)
    ]
    monkeypatch.setattr(expansion, "_gap_bin_edges",
                        lambda *args, **kwargs: [0.01, 0.1, 0.2, 0.3])
    monkeypatch.setattr(expansion, "fit_sigmoid", lambda *args: (10, 0))
    fig, ax = plt.subplots()

    expansion._judge_mean_band(
        ax, rows, "gap", include_self=True,
        per_judge_colors={"a": "#ff0000", "b": "#0000ff"}, fit_mean=True)

    red = next(line for line in ax.lines if line.get_color() == "#ff0000")
    assert red.get_xdata() == pytest.approx([0, 0.05, 0.15, 0.25])
    assert red.get_clip_on() is False
    assert len(ax.lines[-1].get_xdata()) == 100
    plt.close(fig)


def test_main_gap_curve_overlays_judges_and_mean_in_one_narrow_panel(
        tmp_path, monkeypatch):
    judges = ("qwen3-8b", "qwen3-32b")
    rows = [
        {"judge": judge, "eval": "gpqa", "is_self": gap == 0,
         "gap": gap, "eci_gap": gap * 100,
         "rel_gap_months": gap * (100 if judge == "qwen3-8b" else -100),
         "score": 0.5 + gap, "score_eci": 0.5 + gap, "n_items": 20}
        for judge in judges for gap in (0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21)
    ]
    models = {"gpqa": {
        "qwen3-8b": {"label": "Qwen3 8B", "eci": 128},
        "qwen3-32b": {"label": "Qwen3 32B", "eci": 130},
    }}
    figures = {}
    monkeypatch.setattr(expansion, "GAP_AXIS", "accuracy")
    monkeypatch.setattr(expansion, "GAP_LABEL", "capability gap (accuracy diff, per bench)")
    monkeypatch.setattr(expansion, "BENCHES", ["gpqa"])
    monkeypatch.setattr(expansion, "save",
                        lambda fig, out, name, **kwargs: figures.update({name: fig}))

    expansion.fig_gap_curves(rows, models, tmp_path)

    fig = figures["2a_gap_curve_by_capability"]
    ax = fig.axes[0]
    assert len(fig.axes) == 1
    assert fig.get_size_inches()[0] <= 2.2
    colors = {line.get_color() for line in ax.lines}
    assert expansion.JUDGE_COLORS["qwen3-8b"] in colors
    assert expansion.JUDGE_COLORS["qwen3-32b"] in colors
    assert expansion.MEAN_COLOUR in colors
    assert max(len(line.get_xdata()) for line in ax.lines
               if line.get_color() == expansion.MEAN_COLOUR) == 100
    assert ax.get_xlim()[0] == 0
    assert ax.get_xticks()[[0, -1]] == pytest.approx(ax.get_xlim())
    assert ax.get_ylim() == pytest.approx((0, 1))
    assert ax.get_yticks()[[0, -1]] == pytest.approx((0, 1))
    assert not ax.get_title()
    fig.canvas.draw()
    xlabel_box = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
    assert 0 <= xlabel_box.x0 < xlabel_box.x1 <= 1
    assert len(figures["2a_gap_curve_by_capability_by_eval"].axes) == 2
    release_ax = figures["2b_gap_curve_by_release"].axes[0]
    assert release_ax.get_xlim()[0] <= -21
    assert release_ax.get_xlim()[1] >= 21
    assert release_ax.get_xticks()[[0, -1]] == pytest.approx(release_ax.get_xlim())
    for made in figures.values():
        plt.close(made)


def test_zero_gap_self_comparison_matches_judges_and_weights_self_runs(monkeypatch):
    rows = [
        {"eval": "gpqa", "judge": judge, "is_self": False, "gap": gap,
         "score": 0.6, "n_items": 10}
        for judge in ("a", "b") for gap in (0.05, 0.1, 0.15, 0.2)
    ] + [
        {"eval": "gpqa", "judge": "a", "is_self": True, "gap": 0,
         "score": 0.4, "n_items": 10},
        {"eval": "gpqa", "judge": "a", "is_self": True, "gap": 0,
         "score": 0.6, "n_items": 30},
    ]
    monkeypatch.setattr(expansion, "fit_sigmoid", lambda *args: (10, 0))

    result = expansion._zero_gap_self_comparison(rows)

    assert result == [{"judge": "a", "fitted_gap0": pytest.approx(0.5),
                       "self_accuracy": pytest.approx(0.55),
                       "delta": pytest.approx(-0.05), "n_self_runs": 2}]


def test_family_gap_cells_match_within_and_across_inside_judge_bin(monkeypatch):
    rows = [
        {"eval": "gpqa", "judge": judge, "is_self": False, "gap": gap,
         "score": score, "n_items": 10, "major": major, "minor": minor}
        for judge, gap, score, major, minor in [
            ("a", 0.1, 0.7, "qwen3-a", "qwen3-b"),
            ("a", 0.1, 0.5, "qwen3-a", "gpt-b"),
            ("b", 0.1, 0.8, "qwen3-a", "qwen3-b"),
            ("b", 0.1, 0.6, "qwen3-a", "gpt-b"),
            ("c", 0.1, 0.9, "qwen3-a", "qwen3-b"),
        ]
    ]
    monkeypatch.setattr(expansion, "_gap_bin_edges", lambda *args, **kwargs: [0, 0.2])

    cells = expansion._family_gap_cells(rows, reference_gaps=[0.1])

    assert [(c["judge"], c["within"], c["across"], c["delta"]) for c in cells] == [
        ("a", pytest.approx(0.7), pytest.approx(0.5), pytest.approx(0.2)),
        ("b", pytest.approx(0.8), pytest.approx(0.6), pytest.approx(0.2)),
    ]


def test_family_delta_slopes_are_computed_within_judge():
    cells = [
        {"judge": "a", "gap": 0.1, "delta": 0.1},
        {"judge": "a", "gap": 0.2, "delta": 0.3},
        {"judge": "b", "gap": 0.1, "delta": -0.1},
    ]

    slopes = expansion._family_delta_slopes(cells)
    means = expansion._family_judge_means(cells)

    assert slopes == [{"judge": "a", "slope": pytest.approx(2.0), "n_bins": 2}]
    assert means == [
        {"judge": "a", "delta": pytest.approx(0.2), "n_bins": 2},
        {"judge": "b", "delta": pytest.approx(-0.1), "n_bins": 1},
    ]


def test_new_analysis_figures_are_narrow_and_wired_to_their_helpers(tmp_path, monkeypatch):
    comparisons = [
        {"judge": j, "fitted_gap0": 0.58 + i * 0.01, "self_accuracy": 0.5,
         "delta": 0.08 + i * 0.01, "n_self_runs": 1}
        for i, j in enumerate(("a", "b", "c"))]
    cells = [
        {"bin": i, "gap": gap, "judge": j, "within": 0.63 + k * 0.01 + i * 0.02,
         "across": 0.55, "delta": 0.08 + k * 0.01 + i * 0.02}
        for i, gap in enumerate((0.05, 0.15, 0.25))
        for k, j in enumerate(("a", "b", "c") if i != 1 else ("a", "b"))]
    saved = []
    figures = {}
    monkeypatch.setattr(expansion, "_zero_gap_self_comparison", lambda rows: comparisons)
    monkeypatch.setattr(expansion, "_family_gap_cells",
                        lambda rows, reference_gaps: cells)
    def capture(fig, out, name, **kwargs):
        saved.append((name, len(fig.axes), fig.get_size_inches()[0]))
        figures[name] = fig

    monkeypatch.setattr(expansion, "save", capture)

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expansion.fig_zero_gap_self([], tmp_path)
        expansion.fig_family_gap([], tmp_path)

    intervals = [segment for collection in figures["2h_family_gap"].axes[0].collections
                 if isinstance(collection, LineCollection)
                 for segment in collection.get_segments()]
    assert any(np.allclose(segment[:, 0], 0.15) for segment in intervals)
    delta_intervals = [segment
                       for collection in figures["2h_family_gap"].axes[1].collections
                       if isinstance(collection, LineCollection)
                       for segment in collection.get_segments()]
    assert any(np.allclose(segment[:, 0], 0.15) for segment in delta_intervals)
    assert figures["2h_family_gap"].axes[1].get_xlim() == pytest.approx((0, 0.3))
    assert figures["2h_family_gap"].axes[1].get_xticks() == pytest.approx(
        (0, 0.1, 0.2, 0.3))
    assert figures["2h_family_gap"].axes[1].get_ylim() == pytest.approx((-0.4, 0.4))
    assert figures["2h_family_gap"].axes[1].get_yticks() == pytest.approx((-0.4, 0, 0.4))
    figures["2h_family_gap"].canvas.draw()
    assert all(ax.yaxis.label.get_window_extent()
               .transformed(figures["2h_family_gap"].transFigure.inverted()).x0 >= 0
               for ax in figures["2h_family_gap"].axes)
    assert saved == [
        ("2g_zero_gap_vs_self_pair", 1, pytest.approx(1.6)),
        ("2h_family_gap", 2, pytest.approx(2.8)),
    ]
    assert not any("tight_layout" in str(w.message) for w in caught)


def test_forecasting_estimates_held_out_target_rank_from_bracketing_opponents():
    judge = {"accuracy": 0.45, "release": "2024-01"}
    target = {"accuracy": 0.60, "release": "2025-01"}
    opponents = {f"o{i}": {"accuracy": accuracy, "release": "2024-01"}
                 for i, accuracy in enumerate((0.40, 0.50, 0.70, 0.80))}
    rows = []
    for opponent, meta in opponents.items():
        p_target = 1 / (1 + np.exp(-20 * (target["accuracy"] - meta["accuracy"])))
        target_major = target["accuracy"] > meta["accuracy"]
        rows.append({
            "eval": "gpqa", "judge": "j", "is_self": False,
            "major": "target" if target_major else opponent,
            "minor": opponent if target_major else "target",
            "score": p_target if target_major else 1 - p_target,
            "n_items": 100, "noanswer_rate": 0,
            "_major_meta": target if target_major else meta,
            "_minor_meta": meta if target_major else target,
            "_judge_meta": judge,
        })

    cells = expansion._forecast_rank_cells(rows)

    assert len(cells) == 1
    cell = cells[0]
    assert cell["judge"] == "j" and cell["target"] == "target"
    assert cell["release_gap_months"] == 12
    assert cell["n_opponents"] == 4
    assert cell["n_stronger"] == cell["n_weaker"] == 2
    assert cell["estimated_accuracy"] == pytest.approx(0.60, abs=0.01)
    assert cell["absolute_rank_error"] < 0.1
    assert cell["loo_lo"] <= cell["absolute_rank_error"] <= cell["loo_hi"]


def test_forecasting_threshold_rejects_nonmonotone_false_crossing():
    points = [
        (0.730, 0.331, 526), (0.818, 0.173, 602),
        (0.884, 0.683, 634), (0.927, 0.046, 656),
    ]

    threshold = expansion._fit_forecast_threshold(points)

    assert threshold is None


def test_forecasting_plot_is_narrow_and_uses_endpoint_ticks(tmp_path, monkeypatch):
    cells = [
        {"judge": judge, "target": f"m{i}", "release_gap_months": gap,
         "n_opponents": 4, "n_stronger": 2, "n_weaker": 2,
         "estimated_accuracy": 0.6, "actual_accuracy": 0.6,
         "estimated_rank": error + 1, "actual_rank": 1,
         "rank_error": error, "absolute_rank_error": error,
         "loo_lo": max(0, error - 0.5), "loo_hi": error + 0.5}
        for i, (judge, gap, error) in enumerate((
            ("qwen3-8b", 3, 1), ("qwen3-8b", 12, 5),
            ("qwen3-32b", 6, 3)))
    ]
    captured = {}
    monkeypatch.setattr(expansion, "_forecast_rank_cells", lambda rows: cells)
    monkeypatch.setattr(expansion, "save",
                        lambda fig, out, name, **kwargs: captured.update(
                            fig=fig, name=name))

    expansion.fig_forecasting([], tmp_path)

    ax = captured["fig"].axes[0]
    assert captured["name"] == "2i_forecasting_rank_error"
    assert captured["fig"].get_size_inches() == pytest.approx((1.8, 4.0))
    assert ax.get_xlim() == pytest.approx((0, 12))
    assert ax.get_xticks() == pytest.approx((0, 3, 6, 9, 12))
    assert ax.get_ylim() == pytest.approx((0, 10))
    assert ax.get_yticks()[[0, -1]] == pytest.approx((0, 10))
    mean = [line for line in ax.lines
            if line.get_color() == expansion.MEAN_COLOUR
            and line.get_linewidth() >= 1.5]
    assert len(mean) == 1


def test_style_offsets_visible_spines():
    fig, ax = plt.subplots()

    expansion.style(ax)

    assert ax.spines["left"].get_position() == ("outward", 5)
    assert ax.spines["bottom"].get_position() == ("outward", 5)
    plt.close(fig)


def test_save_keeps_svg_labels_as_editable_text(tmp_path):
    fig, ax = plt.subplots()
    ax.set_xlabel("Whole editable label")

    expansion.save(fig, tmp_path, "editable")

    root = ET.parse(tmp_path / "editable.svg").getroot()
    labels = ["".join(node.itertext()) for node in root.iter()
              if node.tag.endswith("}text")]
    assert "Whole editable label" in labels


def test_zero_gap_plot_colors_judges_and_draws_a_black_mean_line(tmp_path, monkeypatch):
    comparisons = [
        {"judge": j, "fitted_gap0": 0.55 + i * 0.02, "self_accuracy": 0.5,
         "delta": 0.05 + i * 0.02, "n_self_runs": 1}
        for i, j in enumerate(("qwen3-8b", "qwen3-32b", "ling-2.6-flash"))]
    captured = {}
    monkeypatch.setattr(expansion, "_zero_gap_self_comparison", lambda rows: comparisons)
    monkeypatch.setattr(expansion, "save",
                        lambda fig, *args, **kwargs: captured.update(fig=fig))

    expansion.fig_zero_gap_self([], tmp_path)

    lines = captured["fig"].axes[0].lines
    judge_colors = {line.get_color() for line in lines
                    if line.get_color() in expansion.JUDGE_COLORS.values()}
    mean = [line for line in lines
            if line.get_color() == expansion.MEAN_COLOUR and len(line.get_xdata()) == 2]
    assert len(judge_colors) == 3
    assert len(mean) == 1 and mean[0].get_linewidth() >= 1.5
    assert captured["fig"].axes[0].get_xlim() == pytest.approx((0, 1))
    assert captured["fig"].axes[0].get_ylim() == pytest.approx((0, 1))
    point_collections = [c for c in captured["fig"].axes[0].collections
                         if hasattr(c, "get_offsets") and len(c.get_offsets())]
    assert all(c.get_clip_on() is False for c in point_collections)


def test_oversight_rows_use_gpqa_accuracy_and_focal_release_date():
    models = {"gpqa": {
        "judge": {"accuracy": 0.60, "release": "2024-06"},
        "focal": {"accuracy": 0.80, "release": "2025-06"},
        # Newer and stronger than the judge is allowed: only focal-relative age matters.
        "other": {"accuracy": 0.90, "release": "2025-05"},
        "same_age": {"accuracy": 0.50, "release": "2025-06"},
        "not_stronger": {"accuracy": 0.60, "release": "2025-05"},
    }}
    base = {"eval": "gpqa", "judge": "judge", "minor": "other", "score": 0.75,
            "is_self": False, "n_items": 10}
    rows = [
        dict(base, major="focal"),
        dict(base, major="focal", minor="same_age"),
        dict(base, major="not_stronger"),
        dict(base, eval="aime", major="focal"),
    ]

    selected = expansion._oversight_rows(rows, models)

    assert len(selected) == 1
    assert selected[0]["gap"] == pytest.approx(0.10)
    # Input score is P(pick focal); "other" is more accurate, so it must flip.
    assert selected[0]["score"] == pytest.approx(0.25)
    assert selected[0]["_series"] == "judge"


def test_oversight_rows_use_the_run_dataset_model_metadata():
    models = {"gpqa": {
        "judge": {"accuracy": 0.95, "release": "2025-12"},
        "focal": {"accuracy": 0.50, "release": "2024-01"},
        "other": {"accuracy": 0.40, "release": "2024-01"},
    }}
    row = {"eval": "gpqa", "judge": "judge", "major": "focal", "minor": "other",
           "score": 0.75, "is_self": False, "n_items": 10,
           "_major_meta": {"accuracy": 0.90, "release": "2026-02"},
           "_minor_meta": {"accuracy": 0.80, "release": "2026-01"},
           "_judge_meta": {"accuracy": 0.60, "release": "2025-01"}}

    selected = expansion._oversight_rows([row], models)

    assert len(selected) == 1


def test_load_fills_missing_model_metadata_from_later_dataset(tmp_path, monkeypatch):
    dirs = []
    for i, release in enumerate(("", "2025-04")):
        d = tmp_path / str(i)
        d.mkdir()
        (d / "index.json").write_text(json.dumps({
            "models": {"model": {"label": "Model", "accuracy": 0.7,
                                    "release": release, "arm": "A"}},
            "runs": [],
        }))
        dirs.append(d)
    monkeypatch.setattr(expansion, "BENCHES", ["gpqa"])
    monkeypatch.setattr(expansion, "DIRS", {"gpqa": dirs})

    _, models = expansion.load()

    assert models["gpqa"]["model"]["release"] == "2025-04"


def test_gap_curve_draws_empirical_line_when_fit_is_disabled():
    rows = [
        {"_series": "judge", "is_self": False, "gap": 0.1,
         "score": 0.6, "n_items": 10},
        {"_series": "judge", "is_self": False, "gap": 0.2,
         "score": 0.7, "n_items": 10},
    ]
    fig, ax = plt.subplots()

    expansion._gap_curve(ax, rows, "gap", ["judge"], {"judge": "#123456"}, fit=False)

    assert len(ax.lines) == 2  # chance + empirical series
    plt.close(fig)


def test_oversight_figure_includes_judges_outside_default_order(tmp_path, monkeypatch):
    judge_labels = {
        "qwen3-8b": "Qwen3 8B", "qwen3-32b": "Qwen3 32B",
        "qwen3-vl-30b-a3b-instruct": "Qwen3 VL 30B A3B Instruct",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
        "gpt-4.1-nano": "GPT-4.1 Nano", "gpt-4o": "GPT-4o",
        "gpt-oss-120b": "GPT-OSS 120B",
        "qwen-2.5-72b-instruct": "Qwen2.5 72B Instruct",
    }
    models = {"gpqa": {
        **{j: {"label": label, "accuracy": 0.5, "release": "2024-01"}
           for j, label in judge_labels.items()},
        "focal": {"label": "Focal", "accuracy": 0.9, "release": "2025-02"},
        "other": {"label": "Other", "accuracy": 0.8, "release": "2025-01"},
    }}
    rows = [row for judge in judge_labels for row in (
        {"eval": "gpqa", "judge": judge, "major": "focal", "minor": "other",
         "score": 0.75, "is_self": False, "n_items": 10, "gap": 0.1},
        {"eval": "gpqa", "judge": judge, "major": "focal", "minor": "focal",
         "score": 0.5, "is_self": True, "n_items": 10, "gap": 0},
    )]
    captured = {}

    def capture(ax, plot_rows, key, **kwargs):
        expansion.style(ax)
        captured["key"] = key
        captured["judges"] = set(kwargs["per_judge_colors"])
        captured["include_self"] = kwargs["include_self"]
        captured["fit_mean"] = kwargs["fit_mean"]
        captured["self_controls"] = sum(r["is_self"] for r in plot_rows)
        return 1

    def capture_save(fig, *args, **kwargs):
        fig.canvas.draw()
        legend_box = fig.legends[0].get_window_extent().transformed(
            fig.transFigure.inverted())
        captured.update(
            axes=len(fig.axes), width=fig.get_size_inches()[0],
            height=fig.get_size_inches()[1], xmin=fig.axes[0].get_xlim()[0],
            xmax=fig.axes[0].get_xlim()[1],
            xticks=tuple(fig.axes[0].get_xticks()),
            ylabel=fig.axes[0].get_ylabel(),
            labels_inside=all(
                artist.get_window_extent().transformed(fig.transFigure.inverted()).x0 >= 0
                for artist in (fig.axes[0].xaxis.label, fig.axes[0].yaxis.label)),
            legend_inside=all(0 <= v <= 1 for v in legend_box.extents))

    monkeypatch.setattr(expansion, "_judge_mean_band", capture)
    monkeypatch.setattr(expansion, "save", capture_save)

    expansion.fig_oversight(rows, models, tmp_path)

    assert captured == {"key": "gap", "judges": set(judge_labels),
                        "include_self": True, "fit_mean": True,
                        "self_controls": len(judge_labels), "axes": 1,
                        "width": pytest.approx(1.8), "height": pytest.approx(4.2),
                        "xmin": pytest.approx(0), "xmax": pytest.approx(0.2),
                        "xticks": pytest.approx((0, 0.1, 0.2)),
                        "ylabel": "Judge accuracy",
                        "labels_inside": True, "legend_inside": True}


def test_control_grid_ignores_models_outside_the_design():
    models = {"gpqa": {
        "designed": {"arm": "A", "eci": 1, "release": "2025-01", "label": "Qwen3 8B"},
        "judge-only": {"arm": "A", "eci": 2, "release": "2025-01", "label": "Qwen3 32B"},
    }}

    assert set(expansion._grid_meta(models)) == {"designed"}


def test_excluded_answer_model_can_still_be_the_judge(tmp_path, monkeypatch):
    d = tmp_path / "gpqa"
    d.mkdir()
    meta = lambda accuracy: {"label": "model", "accuracy": accuracy,
                             "release": "2025-01", "eci": accuracy, "arm": "A"}
    run = {"run_id": "run", "judge": "deepseek-judge", "major": "a", "minor": "b",
           "frac": "full", "score": 0.7, "score_lo": 0.6, "score_hi": 0.8,
           "n_items": 10, "slot_a": 0.5, "noanswer_rate": 0, "is_self": False,
           "weak_judge": False}
    (d / "index.json").write_text(json.dumps({
        "models": {"deepseek-judge": meta(0.5), "a": meta(0.8), "b": meta(0.6)},
        "runs": [run, dict(run, run_id="excluded-answer", major="deepseek-judge")],
    }))
    monkeypatch.setattr(expansion, "BENCHES", ["gpqa"])
    monkeypatch.setattr(expansion, "DIRS", {"gpqa": [d]})
    monkeypatch.setattr(expansion, "GAP_AXIS", "accuracy")
    monkeypatch.setattr(expansion, "EXCLUDE", {"deepseek"})
    monkeypatch.setattr(expansion, "EXCLUDE_EXACT", set())

    rows, _ = expansion.load()

    assert [(r["judge"], r["major"], r["minor"]) for r in rows] == [
        ("deepseek-judge", "a", "b")]


def test_judge_bootstrap_interval_stays_inside_observed_rates():
    lo, hi = expansion._bootstrap_mean_ci([0.35, 0.65])

    assert 0.35 <= lo <= hi <= 0.65


def test_model_inventory_distinguishes_judges_from_evaluated_models(tmp_path):
    rows = [
        {"eval": "gpqa", "judge": "judge-only", "major": "both", "minor": "evaluated-only"},
        {"eval": "gpqa", "judge": "both", "major": "both", "minor": "evaluated-only"},
        {"eval": "aime", "judge": "aime-only", "major": "x", "minor": "y"},
    ]
    models = {"gpqa": {
        "both": {"accuracy": 0.73106},
        "evaluated-only": {"accuracy": 0.625},
        "unused-metadata": {"accuracy": 0.99},
    }}

    expansion.write_model_inventory(rows, models, tmp_path)

    assert (tmp_path / "0_model_inventory.csv").read_text().splitlines() == [
        "model,used_as_judge,evaluated,gpqa_accuracy",
        "both,Y,Y,0.731",
        "evaluated-only,N,Y,0.625",
        "judge-only,Y,N,—",
    ]
    assert "| judge-only | Y | N | — |" in (
        tmp_path / "0_model_inventory.md").read_text()
