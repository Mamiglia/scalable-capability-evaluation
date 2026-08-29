from pathlib import Path
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import capability_predictions_glm.contributions as contributions
from capability_predictions_glm.contributions import bootstrap_delta


def test_bootstrap_delta_averages_within_replication_unit() -> None:
    rows = pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "delta": [0.1, 0.3, -0.2, 0.0],
    })

    summary = bootstrap_delta(rows, "model", iterations=100, seed=3)

    # model a contributes +0.2 and model b -0.1; models receive equal weight.
    assert summary["estimate"] == pytest.approx(0.05)
    assert summary["n_units"] == 2
    assert summary["lower"] <= summary["estimate"] <= summary["upper"]


def test_summaries_include_overall_and_per_benchmark_rows() -> None:
    rows = pd.DataFrame({
        "bench": ["a", "b", "a", "b"],
        "model": ["m1", "m1", "m2", "m2"],
        "target_quartile": ["Q2"] * 4,
        "delta": [0.1, 0.3, -0.1, 0.1],
    })

    summary = pd.DataFrame(contributions._summaries(rows, "model", {}, 100))

    assert "benchmark" in summary
    assert set(summary.benchmark) == {"all", "a", "b"}
    overall = summary[(summary.benchmark == "all") & (summary.target_quartile == "all")].iloc[0]
    assert overall.estimate == pytest.approx(0.1)
    assert overall.n_units == 2


def test_cross_benchmark_importance_counts_helpful_and_harmful_support() -> None:
    rows = pd.DataFrame([
        {"variable": "shared", "block": "surface", "benchmark": bench,
         "estimate": 0.2, "lower": 0.1, "upper": 0.3}
        for bench in ("a", "b")
    ] + [
        {"variable": "specific", "block": "rubric", "benchmark": "a",
         "estimate": 0.2, "lower": 0.1, "upper": 0.3},
        {"variable": "specific", "block": "rubric", "benchmark": "b",
         "estimate": 0.0, "lower": -0.1, "upper": 0.1},
        {"variable": "harmful", "block": "structure", "benchmark": "a",
         "estimate": -0.2, "lower": -0.3, "upper": -0.1},
        {"variable": "harmful", "block": "structure", "benchmark": "b",
         "estimate": 0.0, "lower": -0.1, "upper": 0.1},
    ])

    summary = contributions.cross_benchmark_importance(rows).set_index("variable")

    assert summary.loc["shared", "positive_support"] == 2
    assert summary.loc["specific", "positive_support"] == 1
    assert summary.loc["harmful", "negative_support"] == 1
    assert summary.loc["shared", "positive_benchmarks"] == "a, b"


def test_shared_feature_selection_prefers_cross_benchmark_support() -> None:
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    support = {"bare_answer": 4, "all": 4, "two": 2} | {
        f"one_{index}": 1 for index in range(8)} | {"none": 0}
    rows = pd.DataFrame([
        {"task": "direct", "target_quartile": "all", "contribution": "unique",
         "variable": feature, "block": "surface", "benchmark": benchmark,
         "estimate": 0.2 - feature_index / 100,
         "lower": 0.1 if benchmark_index < count else -0.1, "upper": 0.3}
        for feature_index, (feature, count) in enumerate(support.items())
        for benchmark_index, benchmark in enumerate(benchmarks)
    ])

    selected = contributions.shared_feature_selection(rows, "direct", count=10)

    assert selected == ["all", "two", *[f"one_{index}" for index in range(8)]]


def test_cross_benchmark_profile_plots_are_editable_and_use_shared_features(tmp_path: Path) -> None:
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    rows = pd.DataFrame([
        {"task": "direct", "target_quartile": "all", "contribution": contribution,
         "variable": feature, "block": block, "benchmark": benchmark,
         "estimate": estimate * multiplier,
         "lower": estimate * multiplier - 0.02, "upper": estimate * multiplier + 0.02}
        for contribution, multiplier in (("singular", 2), ("unique", 1))
        for feature, block, values in (
            ("shared", "surface", (0.002, 0.001, 0.002, 0.001)),
            ("agentic", "structure", (0.0, 0.0, 0.004, 0.003)),
        )
        for benchmark, estimate in zip(benchmarks, values)
    ])

    for contribution in ("singular", "unique"):
        contributions.plot_cross_benchmark_profiles(
            rows, ["shared", "agentic"], tmp_path, task="direct", contribution=contribution)

    expected = {f"{contribution}-effect-{kind}.{extension}"
                for contribution in ("singular", "unique")
                for kind in ("heatmap", "diamonds") for extension in ("png", "svg")}
    assert {path.name for path in tmp_path.iterdir()} == expected
    for svg_path in tmp_path.glob("*.svg"):
        svg = svg_path.read_text()
        assert "<text" in svg and "shared" in svg and "agentic" in svg
        assert all(label in svg for label in ("AIME", "GPQA", "InterCode", "SWE"))
    for contribution, outer in (("singular", "1"), ("unique", "0.5")):
        diamond = (tmp_path / f"{contribution}-effect-diamonds.svg").read_text()
        assert f"Outer ring = {outer} pp" in diamond
        assert all(diamond.count(f">{label}</text>") == 2
                   for label in ("AIME", "GPQA", "InterCode", "SWE"))


def test_pairwise_cross_benchmark_profiles_use_accuracy_percentage_points(tmp_path: Path) -> None:
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    rows = pd.DataFrame([
        {"task": "pairwise", "target_quartile": "all", "contribution": "unique",
         "variable": "feature", "block": "surface", "benchmark": benchmark,
         "estimate": estimate, "lower": estimate - 0.01, "upper": estimate + 0.01}
        for benchmark, estimate in zip(benchmarks, (0.01, 0.02, -0.01, 0.03))
    ])

    contributions.plot_cross_benchmark_profiles(
        rows, ["feature"], tmp_path, task="pairwise", contribution="unique")

    heatmap = (tmp_path / "unique-effect-heatmap.svg").read_text()
    assert "Unique Δ accuracy (pp)" in heatmap
    assert "MAE" not in heatmap and "log loss" not in heatmap


def test_pairwise_coefficient_heatmap_shows_effective_weights_and_fold_stability(tmp_path: Path) -> None:
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    rows = pd.DataFrame([
        {"task": "pairwise", "term": "benchmark_deviation", "benchmark": benchmark,
         "feature": feature, "target_quartile": target, "fold": fold,
         "effective": value, "coefficient": 99.0}
        for benchmark_index, benchmark in enumerate(benchmarks)
        for feature, base in (("shared", 0.2), ("agentic", -0.3))
        for fold, fold_shift in (("f1", -0.05), ("f2", 0.05))
        for target in ("Q2", "Q3")
        for value in (base + benchmark_index / 10 + fold_shift,)
    ])

    contributions.plot_coefficient_heatmap(
        rows, ["shared", "agentic"], tmp_path, task="pairwise")

    svg = (tmp_path / "coefficient-heatmap.svg").read_text()
    assert (tmp_path / "coefficient-heatmap.png").exists()
    assert "Effective β (1 SD)" in svg and "Fold range excludes 0" in svg
    assert "shared" in svg and "agentic" in svg
    assert all(label in svg for label in ("AIME", "GPQA", "InterCode", "SWE"))
    assert "MAE" not in svg and "log loss" not in svg


def test_pairwise_cross_profiles_expand_to_twenty_readable_rows() -> None:
    assert contributions._cross_feature_count("direct") == 10
    assert contributions._cross_feature_count("pairwise") == 20
    assert contributions._cross_heatmap_height(20) >= 4.7
    assert contributions._cross_diamond_height(20) >= 8.5


def test_by_feature_figures_use_independent_singular_and_unique_axes(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        {"task": "direct", "target_quartile": "all", "benchmark": "all",
         "variable": variable, "block": "surface", "contribution": contribution,
         "estimate": estimate, "lower": estimate - 0.01, "upper": estimate + 0.01}
        for variable in ("one", "two")
        for contribution, estimate in (("singular", 0.5), ("unique", 0.02))
    ])

    contributions.plot_each_block(rows, tmp_path, "direct")

    svg = (tmp_path / "direct-surface-contributions.svg").read_text()
    assert svg.count('id="axes_') == 2
    assert ">Singular</text>" in svg and ">Unique</text>" in svg
    assert ">one</text>" in svg and ">two</text>" in svg


def test_coefficient_summary_averages_targets_within_fold_then_shows_fold_range() -> None:
    rows = pd.DataFrame([
        {"task": "direct", "term": "benchmark_slope", "benchmark": "a",
         "feature": "f", "target_quartile": target, "fold": fold, "coefficient": value}
        for fold, target, value in (("f1", "Q2", 1.0), ("f1", "Q3", 3.0),
                                    ("f2", "Q2", 5.0), ("f2", "Q3", 7.0))
    ])

    summary = contributions.coefficient_fold_summary(rows, "direct", "a", ["f"]).iloc[0]

    assert summary["mean"] == pytest.approx(4.0)
    assert summary["lower"] == pytest.approx(2.0)
    assert summary["upper"] == pytest.approx(6.0)
    assert summary["n_folds"] == 2


def test_display_limits_follow_points_not_wider_intervals() -> None:
    assert contributions._display_limits(pd.Series([0.1, 1.19]).to_numpy(), nonnegative=True) == (
        (0, 1.25), [0, 1.25])
    assert contributions._display_limits(pd.Series([-0.13, 1.19]).to_numpy()) == (
        (-1.25, 1.25), [-1.25, 0, 1.25])
    assert contributions._display_limits(pd.Series([-0.13, 1.19]).to_numpy(), symmetric=False) == (
        (-0.25, 1.25), [-0.25, 1.25])
    assert contributions._display_limits(pd.Series([1.1, 12.57]).to_numpy(), nonnegative=True) == (
        (0, 15), [0, 15])


def test_grid_tick_labels_are_compact() -> None:
    assert [contributions._format_tick(value) for value in (-0.25, 0, 1.25, 15)] == [
        "−.25", "0", "1.25", "15"]


def test_heatmap_limits_do_not_waste_singular_colour_range() -> None:
    assert contributions._heatmap_limits(pd.Series([1.1, 12.57]).to_numpy(), "singular") == (
        (0, 15), [0, 15])
    assert contributions._heatmap_limits(pd.Series([-0.13, 1.19]).to_numpy(), "unique") == (
        (-0.25, 1.25), [-0.25, 1.25])


def test_endpoint_markers_are_unclipped_but_error_bars_stay_clipped() -> None:
    fig, ax = plt.subplots()
    marker, interval = contributions._plot_error_point(ax, 0.2, 0, 0.1, 0.3, "#222222")

    assert marker.get_clip_on() is False
    assert all(collection.get_clip_on() for collection in interval.lines[2])
    plt.close(fig)


def test_pairwise_coefficient_summary_uses_effective_benchmark_slopes() -> None:
    rows = pd.DataFrame([
        {"task": "pairwise", "term": "benchmark_deviation", "benchmark": "a",
         "feature": "f", "target_quartile": "Q2", "fold": fold,
         "coefficient": 99.0, "effective": value}
        for fold, value in (("f1", -0.2), ("f2", 0.4))
    ])

    summary = contributions.coefficient_fold_summary(rows, "pairwise", "a", ["f"]).iloc[0]

    assert summary["mean"] == pytest.approx(0.1)
    assert summary["lower"] == pytest.approx(-0.2)
    assert summary["upper"] == pytest.approx(0.4)


def test_pairwise_contributions_are_accuracy_gains() -> None:
    keys = {
        "fold": ["f", "f"], "bench": ["a", "a"], "target_quartile": ["Q2", "Q2"],
        "item": ["i1", "i2"],
    }
    candidate = pd.DataFrame(keys | {"accuracy": [0.7, 0.4], "log_loss": [0.6, 0.8]})
    full = pd.DataFrame(keys | {"accuracy": [0.8, 0.6], "log_loss": [0.4, 0.5]})

    singular = contributions._pair_contribution(candidate, None, "singular")
    unique = contributions._pair_contribution(candidate, full, "unique")

    assert singular.delta.tolist() == pytest.approx([0.2, -0.1])
    assert unique.delta.tolist() == pytest.approx([0.1, 0.2])
    scaled = contributions._scale(
        pd.DataFrame({"estimate": [0.1], "lower": [0.05], "upper": [0.15]}), "pairwise")
    assert scaled.estimate.iloc[0] == pytest.approx(10.0)


def test_benchmark_grid_is_four_by_three_without_target_shapes(tmp_path: Path) -> None:
    plot_benchmarks = getattr(contributions, "plot_benchmark_grid", None)
    assert callable(plot_benchmarks)

    effect_rows = []
    benchmarks = ("aime", "gpqa", "intercode", "swe_bench_mini")
    blocks = ("surface", "rubric", "clause-descriptors")
    variables = tuple((f"feature_{index:02d}", blocks[index % len(blocks)]) for index in range(12))
    for bench in benchmarks:
        for index, (variable, block) in enumerate(variables):
            for contribution, estimate in (("singular", 0.12 - index / 100),
                                           ("unique", (index - 6) / 100)):
                effect_rows.append({
                    "task": "direct", "benchmark": bench, "target_quartile": "all",
                    "variable": variable, "block": block, "contribution": contribution,
                    "estimate": estimate, "lower": estimate - 0.01, "upper": estimate + 0.01,
                })
    effects = pd.DataFrame(effect_rows)
    coefficients = pd.DataFrame([
        {
            "task": "direct", "term": "benchmark_slope", "benchmark": bench,
            "feature": feature, "target_quartile": target, "fold": fold,
            "coefficient": value,
        }
        for bench in benchmarks
        for feature, _ in variables
        for target, value in (("Q2", -0.1), ("Q3", 0.1), ("Q4", 0.2))
        for fold in ("f1", "f2")
    ])

    contributions.plot_benchmark_grid(effects, coefficients, tmp_path, "direct")

    assert {path.name for path in tmp_path.iterdir()} == {
        "feature-summary-by-benchmark.png", "feature-summary-by-benchmark.svg"}
    image = mpimg.imread(tmp_path / "feature-summary-by-benchmark.png")
    assert image.shape[0] > image.shape[1]
    svg = (tmp_path / "feature-summary-by-benchmark.svg").read_text()
    assert "<text" in svg
    assert all(label in svg for label in ("AIME", "GPQA", "InterCode", "SWE", "Singular", "Unique", "Coefficient"))
    assert ">AIME</text>" in svg and "AIME · Singular" not in svg
    assert len(re.findall(r">feature \d{2}</text>", svg)) == 40
    assert "· surface" not in svg and ">surface</text>" in svg
    assert ">rubric</text>" in svg and ">clause descriptors</text>" in svg
    assert svg.count("β (1 SD)") == 1
    assert all(target not in svg for target in ("Q2", "Q3", "Q4"))
