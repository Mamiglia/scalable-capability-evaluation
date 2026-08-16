import pandas as pd
import pytest
import struct


def test_parse_args_names_the_provisional_graph_bypass_plainly():
    from run import parse_args

    assert parse_args(["--allow-incomplete-graph"]).allow_incomplete_graph


def test_cutoff_runs_write_beside_instead_of_over_primary_results(tmp_path):
    from run import output_directories, parse_args

    assert parse_args(["--predictor-cutoff", "0.95"]).predictor_cutoff == 0.95
    assert output_directories(tmp_path, None) == (
        tmp_path / "tables",
        tmp_path / "figures",
    )
    assert output_directories(tmp_path, 0.95) == (
        tmp_path / "tables" / "cut95",
        tmp_path / "figures" / "cut95",
    )
    assert output_directories(tmp_path, 0.80) == (
        tmp_path / "tables" / "cut80",
        tmp_path / "figures" / "cut80",
    )


def test_write_outputs_enforces_the_analysis_table_contract(tmp_path):
    from run import write_outputs

    expected = {
        "fit_difficulties.csv",
        "fit_capabilities.csv",
        "item_splits.csv",
        "judge_diagnostics.csv",
        "coverage.csv",
        "elo_scores.csv",
        "rank_metrics.csv",
        "lomo_predictions.csv",
        "value_metrics.csv",
        "quartile_metrics.csv",
        "partition_metrics.csv",
        "partition_predictions.csv",
        "null_metrics.csv",
        "cross_benchmark_metrics.csv",
        "newer_stronger_metrics.csv",
        "monte_carlo_raw.csv",
        "monte_carlo_summary.csv",
    }
    tables = {
        name: pd.DataFrame({"value": [1]})
        if name in {
            "fit_difficulties.csv",
            "fit_capabilities.csv",
            "item_splits.csv",
            "judge_diagnostics.csv",
        }
        else pd.DataFrame({"component": [0], "value": [1]})
        for name in expected
    }

    write_outputs(tables, tmp_path)

    assert {path.name for path in tmp_path.glob("*.csv")} == expected


def test_write_irt_posterior_round_trips_without_pickle(tmp_path):
    import numpy as np
    from run import write_irt_posterior

    path = tmp_path / "irt_posterior.npz"
    write_irt_posterior(path, {
        "fixed_3pl__gpqa": {
            "theta": np.array([[1.0, -1.0]]),
            "a": np.array([[1.2]]),
            "b": np.array([[0.3]]),
            "floor": np.array([0.25]),
            "models": ["m1", "m2"],
            "items": [("gpqa", "i1")],
        }
    })

    stored = np.load(path, allow_pickle=False)
    assert stored["fixed_3pl__gpqa__models"].tolist() == ["m1", "m2"]
    assert stored["fixed_3pl__gpqa__theta"].shape == (1, 2)


def test_load_prefit_tables_reuses_saved_irt_outputs(tmp_path):
    from run import load_prefit_tables

    expected = tuple(
        pd.DataFrame({"name": [name]})
        for name in ("difficulty", "capability", "split")
    )
    for filename, table in zip(
        ("fit_difficulties.csv", "fit_capabilities.csv", "item_splits.csv"),
        expected,
    ):
        table.to_csv(tmp_path / filename, index=False)

    actual = load_prefit_tables(tmp_path)

    for got, wanted in zip(actual, expected):
        pd.testing.assert_frame_equal(got, wanted)


def test_partition_item_banks_add_each_pooled_harder_slice():
    from run import partition_item_banks

    splits = pd.DataFrame([
        {"item": "i1", "tertile": "T1", "quartile": "Q1", "quintile": "P1"},
        {"item": "i2", "tertile": "T1", "quartile": "Q1", "quintile": "P2"},
        {"item": "i3", "tertile": "T2", "quartile": "Q2", "quintile": "P3"},
        {"item": "i4", "tertile": "T2", "quartile": "Q3", "quintile": "P4"},
        {"item": "i5", "tertile": "T3", "quartile": "Q4", "quintile": "P5"},
    ])

    banks = partition_item_banks(splits)
    pooled = {
        subset: set(rows["item"])
        for subset, rows in banks.groupby("subset")
        if "–" in subset
    }

    assert pooled == {
        "T2–T3": {"i3", "i4", "i5"},
        "Q2–Q4": {"i3", "i4", "i5"},
        "P2–P5": {"i2", "i3", "i4", "i5"},
    }


def test_render_figures_writes_each_png_and_svg(tmp_path):
    from figures import render_figures

    rank = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "judge": "j", "component": 0, "method": method, "tau_a": value}
        for method, value in (("elo", 0.6), ("easy_accuracy", 0.3))
    ])
    lomo = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "judge": "consensus", "component": 0, "model": "m", "method": method, "prediction": value, "observed": 0.5}
        for method, value in (("elo", 0.6), ("easy_accuracy", 0.4))
    ])
    sensitivity = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "judge": "j", "component": 0, "method": method, "metric": "tau_a", "value": value, "target_quartile": "Q2", "direction": "gpqa_to_aime"}
        for method, value in (("elo", 0.6), ("easy_accuracy", 0.3))
    ])
    monte_carlo = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "component": "0.0", "axis": axis, "budget": budget, "tau_median": value, "tau_p5": value - 0.1, "tau_p95": value + 0.1}
        for axis in ("models", "items", "rounds")
        for budget, value in ((4, 0.3), (8, 0.5))
    ])
    partition_bins = {
        "halves": ("hard",),
        "tertile": ("T2", "T3", "T2–T3"),
        "quartile": ("Q2", "Q3", "Q4", "Q2–Q4"),
        "quintile": ("P2", "P3", "P4", "P5", "P2–P5"),
    }
    partition_predictions = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": fit, "partition": partition, "target_bin": target_bin,
         "predictor_bin": {"halves": "easy", "tertile": "T1", "quartile": "Q1", "quintile": "P1"}[partition],
        "judge": judge, "component": "0:0", "model": model, "method": method,
         "predictor": predictor, "prediction": predictor, "observed": observed,
         "predictor_accuracy_threshold": 0.9}
        for partition, bins in partition_bins.items()
        for fit in ("aime", "gpqa", "joint")
        for target_bin in bins
        for judge in ("consensus", "judge-a", "judge-b", "judge-c")
        for method in ("elo", "easy_accuracy")
        for model, predictor, observed in (("m1", 0.6, 0.5), ("m2", 0.3, 0.4))
    ])
    partition_metrics = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": fit, "partition": partition, "target_bin": target_bin,
         "judge": "consensus", "component": "0:0", "method": method, "metric": metric,
         "value": value, "comparison_p_value": 0.009}
        for partition, bins in partition_bins.items()
        for fit in ("aime", "gpqa", "joint")
        for target_bin in bins
        for method in ("elo", "easy_accuracy")
        for metric, value in (("tau_a", 0.5), ("mae", 0.1))
    ])
    null_metrics = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
            "partition": partition, "predictor_bin": predictor_bin,
            "target_bin": target_bin, "judge": judge, "component": "0:0",
            "method": method, "metric": metric, "null_median": median,
            "ci_low": low, "ci_high": high, "p_value": 0.05,
            "n_randomizations": 1000,
            "null_type": null_type,
        }
        for partition, bins in partition_bins.items()
        for predictor_bin in ({"halves": "easy", "tertile": "T1", "quartile": "Q1", "quintile": "P1"}[partition],)
        for target_bin in bins
        for judge in ("consensus", "judge-a", "judge-b", "judge-c")
        for method, null_type in (
            ("elo", "coin-flip judgments"),
            ("easy_accuracy", "permuted easy accuracy"),
        )
        for metric, median, low, high in (
            ("tau_a", 0.0, -0.2, 0.2),
            ("mae", 0.2, 0.1, 0.3),
        )
    ])

    render_figures({
        "rank_metrics.csv": rank,
        "lomo_predictions.csv": lomo,
        "quartile_metrics.csv": sensitivity,
        "cross_benchmark_metrics.csv": sensitivity,
        "newer_stronger_metrics.csv": sensitivity,
        "monte_carlo_summary.csv": monte_carlo,
        "partition_predictions.csv": partition_predictions,
        "partition_metrics.csv": partition_metrics,
        "null_metrics.csv": null_metrics,
    }, tmp_path)

    folder_names = {
        "halves": "halves",
        "tertile": "tertiles",
        "quartile": "quartiles",
        "quintile": "quintiles",
    }
    split_figure_names = {
        f"{folder}/{fit}_{kind}_scatter"
        for folder in folder_names.values()
        for fit in ("gpqa",)
        for kind in ("rank", "score_prediction")
    }
    split_figure_names |= {
        f"{folder}/{fit}_{kind}_prediction_by_slice"
        for folder in folder_names.values()
        for fit in ("gpqa",)
        for kind in ("rank", "score")
    }
    split_figure_names |= {
        f"{folder}/gpqa_{kind}_prediction_easy_vs_rest"
        for folder in ("tertiles", "quartiles", "quintiles")
        for kind in ("rank", "score")
    }
    split_figure_names.add("quartiles/gpqa_rank_scatter_easy_vs_rest")
    split_figure_names |= {
        f"{folder}/per_judge/{fit}_{kind}_prediction_by_slice"
        for folder in folder_names.values()
        for fit in ("gpqa",)
        for kind in ("rank", "score")
    }
    expected = {
        f"{name}.{extension}"
        for name in split_figure_names
        for extension in ("png", "svg")
    }
    assert {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == expected

    # A model keeps its colour across the side-by-side predictor panels.
    rank_svg = (tmp_path / "quartiles" / "gpqa_rank_scatter.svg").read_text()
    from partition_figures import MODEL_ACCURACY_COLOR, SUMMARY_X_OFFSETS, _model_colors

    colors = _model_colors(partition_predictions["model"])
    assert len(set(colors.values())) == partition_predictions["model"].nunique()
    assert all(color in rank_svg for color in colors.values())
    assert "Observed accuracy" in rank_svg
    assert "Predictor rank" in rank_svg
    assert "Q2–Q4" not in rank_svg

    pooled_scatter = tmp_path / "quartiles" / "gpqa_rank_scatter_easy_vs_rest"
    pooled_scatter_svg = pooled_scatter.with_suffix(".svg").read_text()
    assert "Q2–Q4" in pooled_scatter_svg
    assert "Elo predictor" in pooled_scatter_svg
    assert "accuracy predictor" in pooled_scatter_svg
    pooled_width, pooled_height = struct.unpack(
        ">II", pooled_scatter.with_suffix(".png").read_bytes()[16:24]
    )
    full_width, full_height = struct.unpack(
        ">II", (tmp_path / "quartiles" / "gpqa_rank_scatter.png").read_bytes()[16:24]
    )
    assert pooled_width <= full_width
    assert pooled_height < full_height

    # Pooled harder slices get a separate, narrower summary while retaining
    # the same height as the corresponding multi-slice figure.
    for folder, pooled_bin in (
        ("tertiles", "T2–T3"),
        ("quartiles", "Q2–Q4"),
        ("quintiles", "P2–P5"),
    ):
        main = tmp_path / folder / "gpqa_rank_prediction_by_slice"
        pooled = tmp_path / folder / "gpqa_rank_prediction_easy_vs_rest"
        assert pooled_bin not in main.with_suffix(".svg").read_text()
        assert pooled_bin in pooled.with_suffix(".svg").read_text()
        main_width, main_height = struct.unpack(
            ">II", main.with_suffix(".png").read_bytes()[16:24]
        )
        pooled_width, pooled_height = struct.unpack(
            ">II", pooled.with_suffix(".png").read_bytes()[16:24]
        )
        assert pooled_width < main_width
        assert abs(pooled_height - main_height) <= 2

    # Even the four-row quintile scatter stays comfortably below a large
    # screen-oriented canvas at 300 dpi.
    quintile_png = tmp_path / "quintiles" / "gpqa_rank_scatter.png"
    width, height = struct.unpack(">II", quintile_png.read_bytes()[16:24])
    assert width <= 1200
    assert height <= 1500

    # The three by-slice marks are dodged rather than printing their error bars
    # on top of one another, and the secondary-axis series stays visually pale.
    assert len(set(SUMMARY_X_OFFSETS.values())) == 3
    summary_svg = (
        tmp_path / "quartiles" / "gpqa_rank_prediction_by_slice.svg"
    ).read_text()
    assert MODEL_ACCURACY_COLOR in summary_svg
    assert "Q1 &gt;90%" in summary_svg
    assert "blue *: Elo vs null" in summary_svg
    assert "red *: acc. vs null" in summary_svg
    assert "bar: Elo vs acc." in summary_svg
    assert "Item slice" in summary_svg
    assert "GPQA accuracy" in summary_svg

    per_judge_svg = (
        tmp_path / "quartiles" / "per_judge" / "gpqa_rank_prediction_by_slice.svg"
    ).read_text()
    assert "model-bootstrap CI" in per_judge_svg
    assert "judge mean" not in per_judge_svg
    assert "Target item slice" in per_judge_svg


def test_per_judge_tau_pools_only_within_component_pairs():
    from judge_figures import _per_judge_metrics

    predictions = pd.DataFrame([
        {
            "partition": "halves", "target_bin": "hard", "judge": "j",
            "component": component, "model": model, "method": "elo",
            "predictor": predictor, "prediction": predictor,
            "observed": observed,
        }
        for component, rows in (
            ("0:0", (("a", 0.1, 0.1), ("b", 0.9, 0.9))),
            ("1:0", (("x", 0.1, 0.9), ("y", 0.9, 0.1))),
        )
        for model, predictor, observed in rows
    ])

    got = _per_judge_metrics(predictions, "tau_a").iloc[0]

    # One concordant and one discordant within-component pair cancel. The four
    # meaningless cross-component pairs are never constructed.
    assert got["value"] == 0.0


def test_build_model_info_rejects_conflicting_release_dates():
    from run import build_model_info

    metadata = pd.DataFrame([
        {"model": "m", "release": "2025-01"},
        {"model": "m", "release": "2025-02"},
    ])
    full_accuracy = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": "all", "model": "m", "accuracy": 0.5}
    ])

    with pytest.raises(ValueError, match="conflicting release"):
        build_model_info(metadata, full_accuracy)
