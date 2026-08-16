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


def figure_cells():
    rows = []
    for judge_index, judge in enumerate(("j1", "j2", "j3")):
        for condition in (
            "original", "first_10", "first_50", "first_90",
            "end_10", "end_50", "end_90",
        ):
            for pair_index, gap in enumerate((5.0, 15.0, 30.0)):
                rows.append({
                    "judge": judge, "condition": condition,
                    "pair": f"pair-{pair_index}", "item": f"item-{pair_index}",
                    "observed": 0.55 + 0.05 * judge_index,
                    "predicted": 0.57 + 0.04 * judge_index,
                    "capability_gap": gap,
                })
    return pd.DataFrame(rows)


def test_overall_figure_is_compact_nature_style():
    fig = load_script().make_overall_figure(figure_cells())

    assert tuple(fig.get_size_inches()) == pytest.approx((2.2, 2.4))
    assert len(fig.axes) == 1
    assert not fig.axes[0].spines["top"].get_visible()
    assert fig.axes[0].spines["left"].get_position() == ("outward", 5)
    assert fig.legends and not fig.legends[0].get_frame_on()


def test_gap_figure_has_three_conditions_frozen_glm_and_editable_svg(tmp_path):
    fig = load_script().make_gap_figure(figure_cells())
    path = tmp_path / "gap.svg"
    fig.savefig(path)

    assert tuple(fig.get_size_inches()) == pytest.approx((5.4, 2.4))
    assert len(fig.axes) == 3
    assert all(not ax.get_title() for ax in fig.axes)
    assert all(not ax.spines["right"].get_visible() for ax in fig.axes)
    assert any(line.get_linestyle() == "--" for line in fig.axes[0].lines)
    assert fig.legends and not fig.legends[0].get_frame_on()
    assert "<text" in path.read_text()
