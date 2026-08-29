import math

import pandas as pd

from transcript_length_overview import (
    _paired_curve,
    bounded_ci,
    extract_pair_cells,
    filter_judges,
    judge_color_map,
    load_benchmarks,
    length_accuracy_ci,
    length_only_pair_stats,
    length_residual_stats,
    log_axis_limits,
    mean_ci,
    median_ci,
    paired_significance,
    render_overview,
    render_length_guidance,
    symmetric_limit,
)


def test_extract_pair_cells_counterbalances_and_splits_on_length():
    items = {
        "stronger-longer": {
            "judge": {"orders": [{"picked_major": 1}, {"picked_major": 0}]}
        },
        "stronger-shorter": {
            "judge": {"orders": [{"picked_major": 1}, {"picked_major": 1}]}
        },
        "tie": {"judge": {"orders": [{"picked_major": 1}]}},
        "unparsed": {"judge": {"orders": [{"picked_major": None}]}},
    }
    lengths = {
        "strong": {"stronger-longer": 10, "stronger-shorter": 4, "tie": 3, "unparsed": 8},
        "weak": {"stronger-longer": 5, "stronger-shorter": 9, "tie": 3, "unparsed": 2},
    }

    rows = extract_pair_cells(items, lengths, "strong", "weak", gap=0.2, pair="pair")

    assert rows == [
        {"judge": "judge", "pair": "pair", "item": "stronger-longer", "gap": 0.2,
         "condition": "Stronger longer", "judge_accuracy": 0.5, "length_accuracy": 1.0,
         "chose_longer": 0.5},
        {"judge": "judge", "pair": "pair", "item": "stronger-shorter", "gap": 0.2,
         "condition": "Stronger shorter", "judge_accuracy": 1.0, "length_accuracy": 0.0,
         "chose_longer": 0.0},
        {"judge": "judge", "pair": "pair", "item": "tie", "gap": 0.2,
         "condition": "Tie", "judge_accuracy": 1.0, "length_accuracy": 0.5,
         "chose_longer": None},
    ]


def test_mean_ci_omits_interval_below_three_replication_units():
    mean, low, high = mean_ci([0.4, 0.6])

    assert mean == 0.5
    assert math.isnan(low)
    assert math.isnan(high)


def test_bounded_ci_keeps_missing_interval_missing():
    low, high = bounded_ci(math.nan, math.nan)

    assert math.isnan(low)
    assert math.isnan(high)


def test_length_accuracy_ci_does_not_floor_half_credit():
    mean, low, high = length_accuracy_ci([0, 0.5])

    assert mean == 0.25
    assert low > 0
    assert high > mean


def test_median_ci_contains_the_observed_median():
    median, low, high = median_ci([1, 2, 3, 100], ["a", "b", "c", "d"], n_boot=100)

    assert median == 2.5
    assert low <= median <= high


def test_per_benchmark_axis_limits_expand_to_readable_round_bounds():
    assert log_axis_limits(0.15, 5.4) == (0.1, 10, (0.1, 1, 10))
    assert log_axis_limits(5.8, 104) == (1, 200, (1, 10, 100, 200))
    assert symmetric_limit([-0.12, 0.08]) == 0.25
    assert symmetric_limit([-0.60, 0.10]) == 0.75


def test_paired_significance_uses_within_judge_differences():
    summary = pd.DataFrame({
        "Stronger longer": [0.80, 0.72, 0.90, 0.85, 0.77, 0.88],
        "Stronger shorter": [0.50, 0.44, 0.61, 0.57, 0.48, 0.60],
    })

    assert paired_significance(summary) == "***"


def test_gap_curve_drops_bins_with_fewer_than_three_paired_judges():
    rows = []
    for judge in ("a", "b"):
        for condition in ("Stronger longer", "Stronger shorter"):
            rows.append({"judge": judge, "gap": 0.15, "condition": condition,
                         "pair": "thin", "judge_accuracy": 0.6})
    for judge in ("a", "b", "c"):
        for condition in ("Stronger longer", "Stronger shorter"):
            rows.append({"judge": judge, "gap": 0.25, "condition": condition,
                         "pair": "kept", "judge_accuracy": 0.7})

    curve = _paired_curve(pd.DataFrame(rows))

    assert curve.bin.astype(float).tolist() == [25.0, 25.0, 25.0]


def test_length_only_panel_uses_only_gaps_above_five_points():
    cells = pd.DataFrame([
        {"pair": "low", "item": "1", "judge": "j", "gap": 0.05,
         "length_accuracy": 1.0, "judge_accuracy": 0.5},
        {"pair": "high", "item": "1", "judge": "j", "gap": 0.06,
         "length_accuracy": 0.0, "judge_accuracy": 1.0},
        {"pair": "high", "item": "1", "judge": "k", "gap": 0.06,
         "length_accuracy": 0.0, "judge_accuracy": 0.5},
    ])

    points = length_only_pair_stats(cells)

    assert points.pair.tolist() == ["high"]
    assert points.judge_accuracy.tolist() == [0.75]


def test_length_residual_is_real_minus_length_choice_for_each_judge_pair():
    cells = pd.DataFrame([
        {"pair": "low", "item": "1", "judge": "j", "gap": 0.05,
         "length_accuracy": 1.0, "judge_accuracy": 0.0},
        {"pair": "high", "item": "1", "judge": "j", "gap": 0.20,
         "length_accuracy": 0.0, "judge_accuracy": 1.0},
        {"pair": "high", "item": "2", "judge": "j", "gap": 0.20,
         "length_accuracy": 1.0, "judge_accuracy": 1.0},
    ])

    points = length_residual_stats(cells)

    assert points[["pair", "judge"]].values.tolist() == [["high", "j"]]
    assert points.length_accuracy.tolist() == [0.5]
    assert points.judge_accuracy.tolist() == [1.0]
    assert points.residual.tolist() == [0.5]


def test_strong_judge_filter_drops_llama_and_ling_only():
    benchmark = {
        "cells": pd.DataFrame({"judge": ["Llama-3.1-8B", "ling-2.6-flash", "o4-mini"]}),
        "lengths": pd.DataFrame({"model": ["Llama-3.1-8B", "ling-2.6-flash"]}),
        "models": pd.DataFrame({"model": ["Llama-3.1-8B", "ling-2.6-flash"]}),
    }

    filtered = filter_judges({"bench": benchmark})["bench"]

    assert filtered["cells"].judge.tolist() == ["o4-mini"]
    assert filtered["lengths"].equals(benchmark["lengths"])
    assert filtered["models"].equals(benchmark["models"])


def test_requested_benchmarks_render_to_editable_svg(tmp_path):
    benchmarks = load_benchmarks()

    assert list(benchmarks) == [
        "GPQA (all)", "GPQA (canonical)", "SWE-bench Mini", "Intercode", "AIME (canonical)"
    ]
    assert all({"Stronger longer", "Stronger shorter"}.issubset(set(bench["cells"].condition))
               for bench in benchmarks.values())
    gpqa_gemini = benchmarks["GPQA (all)"]["models"].set_index("model").loc["gemini-3.1-flash-lite"]
    assert gpqa_gemini.n_trials == 792
    assert gpqa_gemini.accuracy == 579 / 792
    colors = judge_color_map(benchmarks)
    assert len(set(colors.values())) == len(colors)

    out = tmp_path / "overview"
    render_overview(benchmarks, out, n_boot=10)
    guidance = tmp_path / "length_guidance"
    render_length_guidance(benchmarks, guidance)

    assert out.with_suffix(".png").is_file()
    overview_svg = out.with_suffix(".svg").read_text()
    assert "<text" in overview_svg
    assert "P1" not in overview_svg
    assert "stronger model" in overview_svg
    assert "Capability gap" not in overview_svg
    assert guidance.with_suffix(".png").is_file()
    assert "<text" in guidance.with_suffix(".svg").read_text()
