import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "pairwise-newjudges" / "analyze_truncation.py"


def load_script():
    spec = importlib.util.spec_from_file_location("analyze_truncation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_estimates_counterbalance_orders_and_drop_incomplete_cells():
    rows = []
    scores = {
        0.1: {"p1": (1, 1), "p2": (0, 0), "p3": (1, 1)},
        0.5: {"p1": (1, 0), "p2": (0, 0), "p3": (1,)},
        0.9: {"p1": (0, 0), "p2": (0, 0), "p3": (1, 1)},
        1.0: {"p1": (1, 1), "p2": (0, 0), "p3": (1, 1)},
    }
    for fraction, pairs in scores.items():
        for pair, picks in pairs.items():
            for order, picked_major in enumerate(picks, 1):
                rows.append({
                    "judge": "complete", "direction": "first", "fraction": fraction,
                    "pair": pair, "item": "item", "order": order,
                    "picked_major": picked_major, "parse_ok": True,
                })
    for fraction in (0.1, 1.0):
        for order in (1, 2):
            rows.append({
                "judge": "partial", "direction": "end", "fraction": fraction,
                "pair": "p1", "item": "item", "order": order,
                "picked_major": 1, "parse_ok": True,
            })

    estimates = load_script().matched_estimates(pd.DataFrame(rows))

    complete = estimates.set_index("fraction")
    assert complete.index.tolist() == [0.1, 0.5, 0.9, 1.0]
    assert complete["accuracy"].tolist() == [0.5, 0.25, 0.0, 0.5]
    assert complete["delta_vs_100"].tolist() == [0.0, -0.25, -0.5, 0.0]
    assert complete["n_cells"].tolist() == [2, 2, 2, 2]
    assert complete["n_pairs"].tolist() == [2, 2, 2, 2]
    assert complete["judge"].unique().tolist() == ["complete"]


def test_collect_logs_reads_successful_pair_runs_and_preserves_failed_runs(tmp_path):
    good = tmp_path / "trunc_first_10" / "trunc-10" / "judge" / "a-vs-b" / "good.eval"
    bad = tmp_path / "trunc_end_10" / "trunc-end-10" / "judge" / "a-vs-b" / "bad.eval"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.touch()
    bad.touch()

    samples = [
        SimpleNamespace(id="item_1", scores={"judge": SimpleNamespace(answer="1")}),
        SimpleNamespace(id="item_2", scores={"judge": SimpleNamespace(answer="1")}),
    ]
    success = SimpleNamespace(
        status="success",
        eval=SimpleNamespace(metadata={"truncate_frac": 0.1, "truncate_keep": "first"}),
        samples=samples,
    )
    failed = SimpleNamespace(status="error")

    def read_log(path, header_only=False):
        return success if Path(path).name == "good.eval" else failed

    rows, runs = load_script().collect_logs(tmp_path, read_log=read_log)

    assert rows[["condition", "judge", "pair", "item", "order"]].to_dict("records") == [
        {"condition": "trunc_first_10", "judge": "judge", "pair": "a-vs-b",
         "item": "item", "order": 1},
        {"condition": "trunc_first_10", "judge": "judge", "pair": "a-vs-b",
         "item": "item", "order": 2},
    ]
    assert rows["picked_major"].tolist() == [True, False]
    assert runs[["condition", "success"]].to_dict("records") == [
        {"condition": "trunc_end_10", "success": False},
        {"condition": "trunc_first_10", "success": True},
    ]


def test_direction_contrasts_are_matched_within_pair_item():
    rows = []
    for direction in ("first", "end"):
        for fraction in (0.1, 0.5, 0.9, 1.0):
            for pair, picked_major in (("p1", 1), ("p2", direction == "end")):
                for order in (1, 2):
                    rows.append({
                        "judge": "judge", "direction": direction, "fraction": fraction,
                        "pair": pair, "item": "item", "order": order,
                        "picked_major": picked_major, "parse_ok": True,
                    })

    contrasts = load_script().matched_direction_contrasts(pd.DataFrame(rows))

    assert contrasts["fraction"].tolist() == [0.1, 0.5, 0.9, 1.0]
    assert contrasts["end_minus_first"].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert contrasts["n_cells"].tolist() == [2, 2, 2, 2]
    assert contrasts["n_pairs"].tolist() == [2, 2, 2, 2]


def test_original_baseline_supplies_the_shared_100_percent_condition():
    verdicts = pd.DataFrame([
        {"condition": "a_baseline", "judge": "judge", "major": "a", "minor": "b",
         "item_id": "kept", "order": 1, "picked": 1, "picked_major": True,
         "parse_ok": True},
        {"condition": "a_baseline", "judge": "judge", "major": "a", "minor": "b",
         "item_id": "dropped", "order": 1, "picked": 1, "picked_major": True,
         "parse_ok": True},
        {"condition": "other", "judge": "judge", "major": "a", "minor": "b",
         "item_id": "kept", "order": 1, "picked": 1, "picked_major": True,
         "parse_ok": True},
    ])

    rows = load_script().baseline_rows(verdicts, {"a-vs-b": ["kept"]})

    assert rows[["condition", "direction", "fraction", "pair", "item"]].to_dict("records") == [
        {"condition": "a_baseline", "direction": "first", "fraction": 1.0,
         "pair": "a-vs-b", "item": "kept"},
        {"condition": "a_baseline", "direction": "end", "fraction": 1.0,
         "pair": "a-vs-b", "item": "kept"},
    ]


def experiment_rows(judges=("j1", "j2", "j3"), pairs=("a-vs-b",)):
    rows = []
    for judge in judges:
        for direction in ("first", "end"):
            for fraction in (0.1, 0.5, 0.9, 1.0):
                for pair in pairs:
                    for item in ("shared", "incomplete"):
                        if (judge, direction, fraction, pair, item) == (
                            "j3", "end", 0.5, "a-vs-b", "incomplete"
                        ):
                            continue
                        for order in (1, 2):
                            rows.append({
                                "judge": judge, "direction": direction,
                                "fraction": fraction, "pair": pair, "item": item,
                                "order": order, "picked_major": True, "parse_ok": True,
                            })
    return pd.DataFrame(rows)


def test_experiment_cells_use_one_intersection_and_one_shared_original_condition():
    cells = load_script().experiment_cells(experiment_rows())

    assert set(cells.condition) == {
        "original", "first_10", "first_50", "first_90",
        "end_10", "end_50", "end_90",
    }
    assert set(cells.item) == {"shared"}
    assert len(cells) == 3 * 7
    assert len(cells[cells.condition == "original"]) == 3


def test_overall_summary_uses_judges_as_the_replication_unit():
    cells = experiment_rows(pairs=("a-vs-b", "c-vs-d"))
    cells = load_script().experiment_cells(cells)
    cells["observed"] = cells.judge.map({"j1": 0.5, "j2": 0.7, "j3": 0.9})

    by_judge, summary = load_script().overall_tables(cells)

    original = summary[summary.condition == "original"].iloc[0]
    assert original.accuracy == pytest.approx(0.7)
    assert original.n_judges == 3
    assert np.isfinite([original.ci_low, original.ci_high]).all()
    assert len(by_judge[by_judge.condition == "original"]) == 3


def test_overall_summary_suppresses_ci_below_three_judges():
    cells = load_script().experiment_cells(experiment_rows(judges=("j1", "j2")))

    _, summary = load_script().overall_tables(cells)

    assert summary[["ci_low", "ci_high"]].isna().all().all()


def test_plot_tables_use_pair_clustered_intervals():
    rows = []
    gaps = (3.0, 5.0, 12.0, 15.0, 25.0, 30.0)
    for condition in (
        "original", "first_10", "first_50", "first_90",
        "end_10", "end_50", "end_90",
    ):
        for pair_index, gap in enumerate(gaps):
            observed, predicted = ((0.0, 0.2) if pair_index % 2 == 0 else (1.0, 0.8))
            for item_index in range(3 if pair_index % 2 == 0 else 1):
                rows.append({
                    "judge": "j1", "condition": condition,
                    "pair": f"pair-{pair_index}", "item": f"item-{item_index}",
                    "observed": observed, "predicted": predicted,
                    "capability_gap": gap,
                })
    cells = pd.DataFrame(rows)

    overall, _ = load_script().overall_tables(cells)
    gaps, _ = load_script().gap_tables(cells)

    original = overall[overall.condition == "original"].iloc[0]
    assert original.accuracy == pytest.approx(0.5)
    assert original.n_pairs == 6
    assert np.isfinite([original.ci_low, original.ci_high]).all()
    assert gaps.observed.tolist() == pytest.approx([0.5] * len(gaps))
    assert gaps.predicted.tolist() == pytest.approx([0.5] * len(gaps))
    assert gaps.n_pairs.tolist() == [2] * len(gaps)
    assert np.isfinite(gaps[["observed_ci_low", "observed_ci_high",
                             "predicted_ci_low", "predicted_ci_high"]]).all().all()


def test_average_gap_table_averages_judges_before_weighting_pairs():
    rows = []
    for condition in ("original", "first_90", "end_90"):
        for pair_index, gap in enumerate((3.0, 5.0, 12.0, 15.0, 25.0, 30.0)):
            scores = {"j1": 1.0, "j2": 0.0} if pair_index % 2 == 0 else {
                "j1": 0.0, "j2": 1.0,
            }
            for judge, score in scores.items():
                repetitions = 3 if pair_index == 0 and judge == "j1" else 1
                for item_index in range(repetitions):
                    rows.append({
                        "judge": judge, "condition": condition,
                        "pair": f"pair-{pair_index}", "item": f"item-{item_index}",
                        "observed": score, "predicted": 0.2 + 0.6 * score,
                        "capability_gap": gap,
                    })

    averages = load_script().average_gap_table(pd.DataFrame(rows))

    first_bin = averages[(averages.condition == "original") &
                         (averages.gap_bin == 0)].iloc[0]
    assert len(averages) == 9
    assert first_bin.observed == pytest.approx(0.5)
    assert first_bin.observed_ci_low == pytest.approx(0.5)
    assert first_bin.observed_ci_high == pytest.approx(0.5)
    assert first_bin.predicted == pytest.approx(0.5)
    assert first_bin.predicted_ci_low == pytest.approx(0.5)
    assert first_bin.predicted_ci_high == pytest.approx(0.5)
    assert first_bin.n_judges == 2
    assert first_bin.n_pairs == 2


def test_glm_frame_uses_exact_original_and_pair_dependent_truncated_lengths():
    module = load_script()
    cells = pd.DataFrame([
        {"judge": "j1", "condition": condition, "pair": "a-vs-b", "item": "i"}
        for condition in ("original", "first_90", "end_90")
    ])
    transcripts = {
        ("a", "i"): "abcdefghij\nANSWER: A",
        ("b", "i"): "123456\nANSWER: B",
    }
    reference = pd.DataFrame({feature: [0.0, 1.0] for feature in module.SCALED_FEATURES})
    lexicon = {category: [] for category in module.STYLE_CATEGORIES}

    frame = module.build_glm_frame(
        cells, transcripts, {"a-vs-b": ("a", "b")}, {"a": 2.0, "b": 1.0},
        reference, lexicon,
    )

    original = frame[frame.condition == "original"]
    truncated = frame[frame.condition != "original"]
    assert sorted(original.left_chars) == [16, 20]
    assert set(truncated.left_chars) == {5}
    assert set(truncated.right_chars) == {5}
    assert original.total_length.unique().tolist() == pytest.approx([np.log1p(36)])
    assert set(frame.truth_side) == {-1.0, 1.0}
    assert len(frame) == 6


def test_load_transcripts_replays_first_correct_epoch_and_length_limit(tmp_path):
    def sample(item, epoch, text, correct=True):
        return SimpleNamespace(
            id=item, epoch=epoch, score=SimpleNamespace(value="C" if correct else "I"),
            messages=None,
            output=SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=text)
            )]),
        )

    logs = {
        "a.eval": SimpleNamespace(samples=[
            sample("i", 2, "later-long"), sample("i", 1, "first-long"),
        ]),
        "b.eval": SimpleNamespace(samples=[sample("i", 1, "bravo-long")]),
    }
    config = {
        "transcripts": {"a": "a.eval", "b": "b.eval"},
        "comparisons": [["a", "b"]], "sample_ids": {"a-vs-b": ["i"]},
        "max_length": 5,
    }

    transcripts = load_script().load_transcripts(
        config, root=tmp_path, read_log=lambda path: logs[Path(path).name]
    )

    assert transcripts == {("a", "i"): "first", ("b", "i"): "bravo"}


def test_collapse_predictions_averages_orders_onto_observed_cells():
    cells = pd.DataFrame([{
        "judge": "j", "condition": "first_90", "pair": "a-vs-b",
        "item": "i", "observed": 0.5,
    }])
    frame = pd.DataFrame([
        {"judge": "j", "condition": "first_90", "pair": "a-vs-b",
         "item": "i", "order": 1, "predicted": 0.2, "capability_gap": 10.0},
        {"judge": "j", "condition": "first_90", "pair": "a-vs-b",
         "item": "i", "order": 2, "predicted": 0.6, "capability_gap": 10.0},
    ])

    merged = load_script().collapse_predictions(cells, frame)

    assert merged[["observed", "predicted", "capability_gap"]].iloc[0].tolist() == [
        0.5, 0.4, 10.0,
    ]


def figure_cells():
    rows = []
    for judge_index, judge in enumerate(("j1", "j2", "j3")):
        for condition in (
            "original", "first_10", "first_50", "first_90",
            "end_10", "end_50", "end_90",
        ):
            for pair_index, gap in enumerate((3.0, 5.0, 12.0, 15.0, 25.0, 30.0)):
                rows.append({
                    "judge": judge, "condition": condition,
                    "pair": f"pair-{pair_index}", "item": f"item-{pair_index}",
                    "observed": 0.55 + 0.05 * judge_index + 0.02 * (pair_index % 2),
                    "predicted": 0.57 + 0.04 * judge_index + 0.01 * (pair_index % 2),
                    "capability_gap": gap,
                })
    return pd.DataFrame(rows)


def test_overall_figure_is_compact_nature_style():
    fig = load_script().make_overall_figure(figure_cells())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    xlabel = fig.axes[0].xaxis.label.get_window_extent(renderer)
    legend = fig.legends[0].get_window_extent(renderer)

    assert tuple(fig.get_size_inches()) == pytest.approx((1.8, 1.9))
    assert len(fig.axes) == 1
    assert not ax.spines["top"].get_visible()
    assert ax.spines["left"].get_position() == ("outward", 5)
    assert ax.get_ylim() == pytest.approx((0.5, 1.0))
    assert ax.get_yticks().tolist() == pytest.approx([0.5, 0.75, 1.0])
    result_lines = [line for line in ax.lines
                    if line.get_color() != "#888888" and line.get_marker() == "None"]
    assert len(result_lines) == 6
    assert {line.get_linestyle() for line in result_lines} == {"-", "--"}
    assert all({line.get_linestyle() for line in result_lines
                if line.get_color() == color} == {"-", "--"}
               for color in {line.get_color() for line in result_lines})
    assert len(ax.containers) == 6
    assert all(container.has_yerr and not container.lines[1]
               for container in ax.containers)
    marker_x = []
    for container in ax.containers:
        line = container.lines[0]
        marker_x.append(line.get_transform().transform(
            [[line.get_xdata()[0], line.get_ydata()[0]]]
        )[0, 0])
    assert len(set(np.round(marker_x, 3))) == 6
    assert all(not container.lines[0].get_clip_on() for container in ax.containers)
    assert fig.legends and not fig.legends[0].get_frame_on()
    legend_items = dict(zip(
        (text.get_text() for text in fig.legends[0].get_texts()),
        fig.legends[0].legend_handles,
    ))
    assert set(legend_items) == {
        "j1", "j2", "j3", "First retained", "End retained",
    }
    markers = [container.lines[0] for container in ax.containers]
    assert all(line.get_markerfacecolor() != "none" for line in markers[::2])
    assert all(line.get_markerfacecolor() == "none" for line in markers[1::2])
    assert legend_items["First retained"].get_markerfacecolor() != "none"
    assert legend_items["End retained"].get_markerfacecolor() == "none"
    assert not xlabel.overlaps(legend)
    tick_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
    assert all(not left.overlaps(right)
               for left, right in zip(tick_boxes, tick_boxes[1:]))


def test_gap_figure_colours_models_and_styles_empirical_vs_glm(tmp_path):
    fig = load_script().make_gap_figure(figure_cells())
    path = tmp_path / "gap.svg"
    fig.savefig(path)

    assert tuple(fig.get_size_inches()) == pytest.approx((4.3, 1.9))
    assert len(fig.axes) == 3
    assert all(not ax.get_title() for ax in fig.axes)
    assert all(not ax.spines["right"].get_visible() for ax in fig.axes)
    assert all(ax.get_ylim() == pytest.approx((0.5, 1.0)) for ax in fig.axes)
    assert all(ax.get_yticks().tolist() == pytest.approx([0.5, 0.75, 1.0])
               for ax in fig.axes)
    for ax in fig.axes:
        result_lines = [line for line in ax.lines if line.get_marker() == "None"
                        and line.get_color() != "#888888"]
        assert len(result_lines) == 6
        assert len({line.get_color() for line in result_lines}) == 3
        assert all({line.get_linestyle() for line in result_lines
                    if line.get_color() == color} == {"-", "--"}
                   for color in {line.get_color() for line in result_lines})
        assert len(ax.containers) == 6
        assert all(container.has_yerr and not container.lines[1]
                   for container in ax.containers)
        marker_x = []
        for container in ax.containers:
            line = container.lines[0]
            marker_x.append(line.get_transform().transform(
                [[line.get_xdata()[0], line.get_ydata()[0]]]
            )[0, 0])
        assert len(set(np.round(marker_x, 3))) == 6
    assert all(container.lines[0].get_markerfacecolor() != "none"
               for ax in fig.axes[:2] for container in ax.containers)
    assert all(container.lines[0].get_markerfacecolor() == "none"
               for container in fig.axes[2].containers)
    assert fig.legends and not fig.legends[0].get_frame_on()
    assert {text.get_text() for text in fig.legends[0].get_texts()} == {
        "j1", "j2", "j3", "Empirical accuracy", "Frozen GLM prediction",
    }
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    assert all(not ax.xaxis.label.get_window_extent(renderer).overlaps(legend)
               for ax in fig.axes)
    assert fig.axes[0].yaxis.label.get_window_extent(renderer).x0 >= 0
    assert "<text" in path.read_text()


def test_figures_use_standard_qwen_and_gpt_oss_colours():
    cells = figure_cells()
    cells = cells[cells.judge.isin(("j1", "j2"))].copy()
    cells["judge"] = cells.judge.map({"j1": "qwen3-8b", "j2": "gpt-oss-20b"})
    expected = {"qwen3-8b": "#2a78d6", "gpt-oss-20b": "#eb6834"}

    for make_figure in (load_script().make_overall_figure, load_script().make_gap_figure):
        fig = make_figure(cells)
        handles = dict(zip(
            (text.get_text() for text in fig.legends[0].get_texts()),
            fig.legends[0].legend_handles,
        ))
        assert {judge: handles[judge].get_color() for judge in expected} == expected


def test_average_gap_figure_is_small_and_shows_empirical_and_glm_means(tmp_path):
    fig = load_script().make_average_gap_figure(figure_cells())
    path = tmp_path / "gap-average.svg"
    fig.savefig(path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]

    assert tuple(fig.get_size_inches()) == pytest.approx((1.8, 1.9))
    assert len(fig.axes) == 1
    assert not ax.get_title()
    assert not ax.spines["top"].get_visible()
    assert ax.get_ylim() == pytest.approx((0.5, 1.0))
    assert ax.get_yticks().tolist() == pytest.approx([0.5, 0.75, 1.0])
    result_lines = [line for line in ax.lines
                    if line.get_color() != "#888888" and line.get_marker() == "None"]
    assert [(line.get_color(), line.get_linestyle()) for line in result_lines] == [
        ("#111111", "-"),
        ("#1baf7a", "-"), ("#1baf7a", ":"),
        ("#eda100", "-"), ("#eda100", ":"),
    ]
    assert "#2a78d6" not in {line.get_color() for line in result_lines}
    assert len(ax.containers) == 3
    assert all(container.has_yerr and not container.lines[1]
               for container in ax.containers)
    markers = [container.lines[0] for container in ax.containers]
    assert [marker.get_markerfacecolor() == "none" for marker in markers] == [
        False, False, True,
    ]
    marker_x = [marker.get_transform().transform(
        [[marker.get_xdata()[0], marker.get_ydata()[0]]]
    )[0, 0] for marker in markers]
    assert len(set(np.round(marker_x, 3))) == 3
    assert max(marker_x) - min(marker_x) > 5
    assert all(not marker.get_clip_on() for marker in markers)
    legend = fig.legends[0]
    assert not legend.get_frame_on()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Original", "First 90%", "End 90%",
        "Empirical", "GLM prediction",
    ]
    assert legend.legend_handles[-1].get_linestyle() == ":"
    legend_box = legend.get_window_extent(renderer)
    assert 0 <= legend_box.x0 and legend_box.x1 <= fig.bbox.x1
    assert not ax.xaxis.label.get_window_extent(renderer).overlaps(
        legend_box
    )
    assert ax.yaxis.label.get_window_extent(renderer).x0 >= 0
    assert "<text" in path.read_text()


def test_write_plot_outputs_writes_figures_and_source_tables(tmp_path):
    results = tmp_path / "results"
    manuscript = tmp_path / "manuscript" / "04-ablations"
    load_script().write_plot_outputs(figure_cells(), results, manuscript)

    assert {
        "overall_by_judge.csv", "overall_summary.csv", "gap_by_judge.csv",
        "gap_estimates.csv", "gap_average.csv",
        "overall_accuracy.png", "overall_accuracy.svg",
        "gap_curves_90.png", "gap_curves_90.svg",
        "gap_curves_90_average.png", "gap_curves_90_average.svg",
    } == {path.name for path in results.iterdir()}
    assert {"overall_accuracy.png", "overall_accuracy.svg",
            "gap_curves_90.png", "gap_curves_90.svg",
            "gap_curves_90_average.png", "gap_curves_90_average.svg"} == {
                path.name for path in manuscript.iterdir()
            }
    for name in ("overall_accuracy.png", "overall_accuracy.svg",
                 "gap_curves_90.png", "gap_curves_90.svg",
                 "gap_curves_90_average.png", "gap_curves_90_average.svg"):
        assert (results / name).read_bytes() == (manuscript / name).read_bytes()
    assert "<text" in (results / "overall_accuracy.svg").read_text()
    assert "<text" in (manuscript / "gap_curves_90.svg").read_text()
