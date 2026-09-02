import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import warnings

import numpy as np
import pandas as pd
import pytest


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
    split_figure_names.add("quartiles/gpqa_rank_scatter_legend")
    split_figure_names.add("quartiles/gpqa_rank_scatter_labelled")
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


def test_bias_cli_parses_only_declared_fold_arms():
    from correct_biases import parse_args

    args = parse_args([
        "fit-fold", "--irt-model", "fixed_3pl", "--fold", "3",
        "--model", "corrected",
    ])
    assert (args.command, args.irt_model, args.fold, args.model) == (
        "fit-fold", "fixed_3pl", 3, "corrected",
    )
    with pytest.raises(SystemExit):
        parse_args([
            "fit-fold", "--irt-model", "fixed_3pl", "--fold", "5",
            "--model", "corrected",
        ])
    with pytest.raises(SystemExit):
        parse_args([
            "fit-fold", "--irt-model", "fixed_3pl", "--fold", "1",
            "--model", "other",
        ])
    with pytest.raises(SystemExit):
        parse_args([
            "fit-fold", "--irt-model", "3pl", "--fold", "1",
            "--model", "corrected",
        ])


def test_bias_table_writer_enforces_contract(tmp_path):
    from correct_biases import write_tables

    tables = {name: pd.DataFrame({"x": [1]}) for name in (
        "predictions.csv", "coefficients.csv", "coverage.csv", "metrics.csv",
        "gap_pairs.csv",
    )}
    write_tables(tables, tmp_path)
    assert {path.name for path in tmp_path.glob("*.csv")} == set(tables)
    with pytest.raises(ValueError, match="coverage.csv"):
        write_tables({
            key: value for key, value in tables.items()
            if key != "coverage.csv"
        }, tmp_path)


def test_bias_fold_artifact_round_trips_and_refuses_overwrite(tmp_path):
    from correct_biases import read_fold_artifact, write_fold_artifact

    path = tmp_path / "corrected.npz"
    expected = {
        "predictions": pd.DataFrame({"model": ["a"], "prediction": [0.75]}),
        "coverage": pd.DataFrame({"model": ["a"], "eligible": [True]}),
    }
    write_fold_artifact(path, expected)

    actual = read_fold_artifact(path)

    assert set(actual) == set(expected)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
    with pytest.raises(FileExistsError, match="corrected.npz"):
        write_fold_artifact(path, expected)


def test_bias_reduce_requires_all_fold_arm_artifacts(tmp_path):
    from correct_biases import IRT_MODELS, require_fold_artifacts

    with pytest.raises(FileNotFoundError, match="fold artifacts"):
        require_fold_artifacts(tmp_path)

    for irt_model in IRT_MODELS:
        for fold in range(5):
            directory = tmp_path / "folds" / irt_model / f"fold-{fold}"
            directory.mkdir(parents=True)
            for model in ("corrected", "uncorrected"):
                (directory / f"{model}.npz").touch()

    assert len(require_fold_artifacts(tmp_path)) == len(IRT_MODELS) * 10


def test_bias_primary_responses_keep_gpqa_only():
    from correct_biases import primary_responses

    responses = pd.DataFrame({
        "benchmark": ["gpqa", "aime2025"],
        "model": ["g", "a"],
    })

    assert primary_responses(responses)["model"].tolist() == ["g"]


def test_bias_prepare_requires_exactly_the_three_declared_sources():
    from correct_biases import _validate_prepared

    votes = pd.DataFrame([{
        "source": "gpqa-pilot-rd2", "benchmark": "gpqa", "judge": "j",
        "item_id": "i", "left_model": "a", "right_model": "b",
        "chosen_left": 1.0, "delta_log_chars": 0.0,
    }])
    responses = pd.DataFrame([{
        "benchmark": "gpqa", "model": "a", "item": "i", "answer": "A",
        "correct": True, "source_priority": 1,
    }])

    with pytest.raises(ValueError, match="three declared sources"):
        _validate_prepared(votes, responses)


def test_bias_prepare_rejects_missing_verdicts():
    from correct_biases import _validate_prepared

    votes = pd.DataFrame([
        {
            "source": source, "benchmark": "gpqa", "judge": "j",
            "item_id": "i", "left_model": "a", "right_model": "b",
            "chosen_left": choice, "delta_log_chars": 0.0,
        }
        for source, choice in (
            ("gpqa-pilot-rd2", 1.0),
            ("data-expansion-gpqa", 0.0),
            ("hf-judging-2026-08-17", float("nan")),
        )
    ])
    responses = pd.DataFrame([{
        "benchmark": "gpqa", "model": "a", "item": "i", "answer": "A",
        "correct": True, "source_priority": 1,
    }])

    with pytest.raises(ValueError, match="missing or non-binary verdicts"):
        _validate_prepared(votes, responses)


def test_bias_input_preflight_requires_each_viewer_dataset(tmp_path):
    from correct_biases import validate_input_paths

    viewer = tmp_path / "interactive_dashboards" / "transcript_viewer" / "data"
    (viewer / "gpqa-pilot-rd2").mkdir(parents=True)
    config = {
        "paths": {"viewer_data": "interactive_dashboards/transcript_viewer/data"},
        "correction": {
            "viewer_datasets": {
                "gpqa-pilot-rd2": "gpqa",
                "data-expansion-gpqa": "gpqa",
            },
        },
    }

    with pytest.raises(FileNotFoundError, match="data-expansion-gpqa"):
        validate_input_paths(tmp_path, config)

    (viewer / "data-expansion-gpqa").mkdir()
    validate_input_paths(tmp_path, config)


def test_bias_diagnostic_rows_keep_exact_fold_arm_and_inference():
    from correct_biases import _diagnostic_rows

    got = _diagnostic_rows([{
        "phase": "target", "inference": "corrected_without_style",
        "divergences": 0,
        "diagnostics": {
            "theta": {"r_hat": np.array([1.0]), "n_eff": np.array([40.0])},
        },
    }], irt_model="2pl", fold=3, arm="corrected")

    assert got.loc[0, [
        "irt_model", "fold", "arm", "phase", "inference", "parameter",
    ]].tolist() == [
        "2pl", 3, "corrected", "target", "corrected_without_style", "theta",
    ]


def test_bias_reference_artifacts_save_both_irt_diagnostics(tmp_path):
    from correct_biases import write_reference_artifacts

    fits = {
        irt_model: {
            "a": np.ones((2, 1)), "b": np.zeros((2, 1)),
            "floor": np.array([floor]), "items": [("gpqa", "i")],
            "divergences": 0,
            "diagnostics": {
                "b": {"r_hat": np.array([1.0]), "n_eff": np.array([20.0])},
            },
        }
        for irt_model, floor in (("fixed_3pl", 0.25), ("2pl", 0.0))
    }
    write_reference_artifacts(
        tmp_path,
        pd.DataFrame({"irt_model": ["fixed_3pl", "2pl"]}),
        pd.DataFrame({"irt_model": ["fixed_3pl", "2pl"]}),
        fits,
    )

    diagnostics = pd.read_csv(tmp_path / "diagnostics.csv")
    assert set(diagnostics["irt_model"]) == {"fixed_3pl", "2pl"}
    assert set(diagnostics["divergences"]) == {0}
    assert set(diagnostics["r_hat_max"]) == {1.0}
    assert set(diagnostics["n_eff_min"]) == {20.0}


def test_bias_diagnostics_keep_non_estimable_deterministics_without_warning():
    from correct_biases import _diagnostic_rows

    artifact = {"diagnostics": {
        "constant": {"r_hat": np.array([np.nan]), "n_eff": np.array([np.nan])},
    }}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _diagnostic_rows(
            [artifact], irt_model="fixed_3pl", fold=0, arm="corrected",
        ).iloc[0]

    assert np.isnan(got["r_hat_max"])
    assert np.isnan(got["n_eff_min"])


def _valid_fold_tables(irt_model="fixed_3pl", fold=0, arm="corrected"):
    from bias_crossfit import (
        CALIBRATION_COLUMNS, COEFFICIENT_COLUMNS, COVERAGE_COLUMNS,
        PREDICTION_COLUMNS, TARGET_EXCLUSION_COLUMNS,
    )
    from correct_biases import DIAGNOSTIC_COLUMNS, IDENTITY_COLUMNS

    return {
        "predictions": pd.DataFrame(columns=PREDICTION_COLUMNS),
        "coverage": pd.DataFrame(columns=COVERAGE_COLUMNS),
        "calibration": pd.DataFrame(columns=CALIBRATION_COLUMNS),
        "coefficients": pd.DataFrame(columns=COEFFICIENT_COLUMNS),
        "target_exclusions": pd.DataFrame(columns=TARGET_EXCLUSION_COLUMNS),
        "diagnostics": pd.DataFrame(columns=DIAGNOSTIC_COLUMNS),
        "identity": pd.DataFrame([[irt_model, fold, arm]], columns=IDENTITY_COLUMNS),
    }


@pytest.mark.parametrize("contents", (b"", b"not an npz"))
def test_bias_fold_validator_rejects_empty_or_unreadable_npz(tmp_path, contents):
    from correct_biases import validate_fold_artifact

    path = tmp_path / "corrected.npz"
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="nonempty readable NPZ"):
        validate_fold_artifact(path, "fixed_3pl", 0, "corrected")


def test_bias_fold_validator_requires_every_table_and_exact_schema(tmp_path):
    from correct_biases import validate_fold_artifact, write_fold_artifact

    missing_path = tmp_path / "missing.npz"
    missing = _valid_fold_tables()
    missing.pop("coverage")
    write_fold_artifact(missing_path, missing)
    with pytest.raises(ValueError, match="missing.*coverage"):
        validate_fold_artifact(missing_path, "fixed_3pl", 0, "corrected")

    schema_path = tmp_path / "schema.npz"
    wrong_schema = _valid_fold_tables()
    wrong_schema["predictions"] = wrong_schema["predictions"].drop(columns="model")
    write_fold_artifact(schema_path, wrong_schema)
    with pytest.raises(ValueError, match="predictions schema"):
        validate_fold_artifact(schema_path, "fixed_3pl", 0, "corrected")


def test_bias_fold_loader_checks_all_embedded_identities_before_reduce(tmp_path):
    from correct_biases import IRT_MODELS, load_fold_artifacts, write_fold_artifact

    for irt_model in IRT_MODELS:
        for fold in range(5):
            for arm in ("corrected", "uncorrected"):
                path = tmp_path / "folds" / irt_model / f"fold-{fold}" / f"{arm}.npz"
                embedded = "2pl" if (irt_model, fold, arm) == (
                    "fixed_3pl", 4, "uncorrected",
                ) else irt_model
                write_fold_artifact(
                    path, _valid_fold_tables(embedded, fold, arm),
                )

    with pytest.raises(ValueError, match="identity.*fixed_3pl.*uncorrected"):
        load_fold_artifacts(tmp_path)


def test_bias_reference_preflight_uses_lowercase_pyyaml_wheel_name():
    script = (Path(__file__).parent / "bias_reference_fit.sbatch").read_text()

    assert "'pyyaml-6.0.3-*.whl'" in script
    assert "'PyYAML-6.0.3-*.whl'" not in script


def test_bias_reference_fit_does_not_require_matplotlib_on_compute_nodes():
    script = (Path(__file__).parent / "bias_reference_fit.sbatch").read_text()

    assert "matplotlib" not in script.lower()


def test_bias_candidate_closure_requires_easy_hard_cells_and_one_anchor():
    from correct_biases import candidate_closure

    candidates = {"a", "b"}
    responses = pd.DataFrame([
        {"benchmark": "gpqa", "model": model, "item": item, "epoch": 1,
         "answer": "A", "correct": True, "source_priority": 1}
        for model in candidates
        for item in ("e1", "e2", "h1", "h2")
    ])
    splits = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
         "item": item, "half": half}
        for half, items in (("easy", ("e1", "e2")), ("hard", ("h1", "h2")))
        for item in items
    ])
    anchors = pd.DataFrame([
        {"irt_model": "fixed_3pl", "model": model, "theta": index}
        for index, model in enumerate(sorted(candidates))
    ])

    got = candidate_closure(
        responses, candidates, splits=splits, anchors=anchors,
        irt_model="fixed_3pl",
    ).set_index("model")

    assert got[["n_easy", "n_hard", "n_anchors"]].to_dict("index") == {
        "a": {"n_easy": 2, "n_hard": 2, "n_anchors": 1},
        "b": {"n_easy": 2, "n_hard": 2, "n_anchors": 1},
    }
    with pytest.raises(ValueError, match="b.*hard"):
        candidate_closure(
            responses[~(
                responses["model"].eq("b") & responses["item"].str.startswith("h")
            )],
            candidates, splits=splits, anchors=anchors,
            irt_model="fixed_3pl",
        )
    with pytest.raises(ValueError, match="exactly one anchor.*a"):
        candidate_closure(
            responses, candidates, splits=splits,
            anchors=pd.concat([anchors, anchors.iloc[[0]]], ignore_index=True),
            irt_model="fixed_3pl",
        )


def test_bias_prepare_writes_audited_exclusions_and_scored_pool(
    tmp_path, monkeypatch,
):
    import correct_biases
    import inputs

    sources = (
        "gpqa-pilot-rd2", "data-expansion-gpqa", "hf-judging-2026-08-17",
    )
    votes = pd.DataFrame([
        {"source": source, "benchmark": "gpqa", "judge": "j",
         "artifact": f"{source}/artifact", "round": f"{source}:round",
         "item_id": "i1", "left_model": f"m{index}",
         "right_model": f"m{(index + 1) % 5}", "chosen_left": 1.0,
         "delta_log_chars": 0.0}
        for index, source in enumerate(sources)
    ] + [
        {"source": "gpqa-pilot-rd2", "benchmark": "gpqa", "judge": "j",
         "artifact": "gpqa-pilot-rd2/extra", "round": "pilot:extra",
         "item_id": "i2", "left_model": "m3", "right_model": "m4",
         "chosen_left": 0.0, "delta_log_chars": 0.0},
    ])
    responses = pd.DataFrame([
        {"benchmark": "gpqa", "model": model, "item": item, "epoch": 1,
         "answer": "A", "correct": True, "source_priority": 1,
         "source": "epoch", "source_path": f"{model}.eval"}
        for model in (f"m{index}" for index in range(5))
        for item in ("i1", "i2")
    ])
    exclusions = pd.DataFrame([{
        "source": "gpqa-pilot-rd2", "judge": "j", "artifact": "bad.json",
        "reason": "invalid viewer verdict", "count": 2,
    }])
    monkeypatch.setattr(correct_biases, "ROOT", tmp_path)
    monkeypatch.setattr(correct_biases, "validate_input_paths", lambda *_: None)
    monkeypatch.setattr(
        inputs, "load_correction_data",
        lambda *_args, **_kwargs: (votes, responses, exclusions),
    )
    config = {
        "paths": {},
        "correction": {"output": "analysis/bias_correction"},
        "irt": {"seed": 0},
    }

    correct_biases.prepare(config)

    prepared = tmp_path / "analysis" / "bias_correction" / "input"
    pd.testing.assert_frame_equal(pd.read_parquet(prepared / "responses.parquet"), responses)
    pd.testing.assert_frame_equal(pd.read_parquet(prepared / "exclusions.parquet"), exclusions)
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["excluded_verdict_rows"] == 2


def test_bias_prepare_import_does_not_require_matplotlib():
    root = Path(__file__).resolve().parents[2]
    code = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise RuntimeError('matplotlib import blocked')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import inputs
import correct_biases
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "src"), str(root / "src" / "judge"),
        str(root / "scripts" / "irt_proxy"),
    ))

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env,
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr


def test_bias_prepare_import_does_not_require_draccus():
    root = Path(__file__).resolve().parents[2]
    code = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'draccus' or name.startswith('draccus.'):
        raise RuntimeError('draccus import blocked')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import inputs
import correct_biases
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "src"), str(root / "src" / "judge"),
        str(root / "scripts" / "irt_proxy"),
    ))

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env,
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
