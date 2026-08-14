import pandas as pd
import pytest


def test_write_outputs_enforces_the_analysis_table_contract(tmp_path):
    from run import write_outputs

    expected = {
        "fit_difficulties.csv",
        "fit_capabilities.csv",
        "item_splits.csv",
        "coverage.csv",
        "elo_scores.csv",
        "rank_metrics.csv",
        "lomo_predictions.csv",
        "value_metrics.csv",
        "quartile_metrics.csv",
        "cross_benchmark_metrics.csv",
        "newer_stronger_metrics.csv",
        "monte_carlo_raw.csv",
        "monte_carlo_summary.csv",
    }
    tables = {
        name: pd.DataFrame({"value": [1]})
        if name in {"fit_difficulties.csv", "fit_capabilities.csv", "item_splits.csv"}
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


def test_render_figures_writes_each_png_and_svg(tmp_path):
    from figures import render_figures

    rank = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "judge": "j", "component": 0, "method": method, "tau_b": value}
        for method, value in (("elo", 0.6), ("easy_accuracy", 0.3))
    ])
    lomo = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "judge": "consensus", "component": 0, "model": "m", "method": method, "prediction": value, "observed": 0.5}
        for method, value in (("elo", 0.6), ("easy_accuracy", 0.4))
    ])
    sensitivity = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "judge": "j", "component": 0, "method": method, "metric": "tau_b", "value": value, "target_quartile": "Q2", "direction": "gpqa_to_aime"}
        for method, value in (("elo", 0.6), ("easy_accuracy", 0.3))
    ])
    monte_carlo = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "component": "0.0", "axis": axis, "budget": budget, "tau_median": value, "tau_p5": value - 0.1, "tau_p95": value + 0.1}
        for axis in ("models", "items", "rounds")
        for budget, value in ((4, 0.3), (8, 0.5))
    ])

    render_figures({
        "rank_metrics.csv": rank,
        "lomo_predictions.csv": lomo,
        "quartile_metrics.csv": sensitivity,
        "cross_benchmark_metrics.csv": sensitivity,
        "newer_stronger_metrics.csv": sensitivity,
        "monte_carlo_summary.csv": monte_carlo,
    }, tmp_path)

    expected = {
        f"{name}.{extension}"
        for name in (
            "rank_tau",
            "lomo_predictions",
            "quartile_prediction",
            "cross_benchmark",
            "newer_stronger",
            "monte_carlo_budget",
        )
        for extension in ("png", "svg")
    }
    assert {path.name for path in tmp_path.iterdir()} == expected


def test_cross_benchmark_plot_omits_components_without_an_estimate(tmp_path):
    from figures import plot_cross_benchmark

    rows = []
    for component, value in ((0, 0.5), (1, float("nan"))):
        for method in ("elo", "easy_accuracy"):
            rows.append({
                "irt_model": "fixed_3pl",
                "fit": "gpqa",
                "judge": "j",
                "component": component,
                "method": method,
                "metric": "tau_b",
                "value": value,
                "direction": "gpqa_to_aime",
            })

    plot_cross_benchmark(pd.DataFrame(rows), tmp_path)

    # A component with no finite judge estimate should not leave a blank tick.
    svg = (tmp_path / "cross_benchmark.svg").read_text()
    assert "C0" in svg
    assert "C1" not in svg


def test_quartile_plot_facets_connected_components(tmp_path):
    from figures import plot_quartile_prediction

    table = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "judge": "j",
            "component": component,
            "method": method,
            "metric": "tau_b",
            "value": 0.5,
            "target_quartile": quartile,
        }
        for component in ("0:0", "1:0")
        for method in ("elo", "easy_accuracy")
        for quartile in ("Q2", "Q3", "Q4")
    ])

    plot_quartile_prediction(table, tmp_path)

    svg = (tmp_path / "quartile_prediction.svg").read_text()
    assert svg.count('id="axes_') == 2


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
