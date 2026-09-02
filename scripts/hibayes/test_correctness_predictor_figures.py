"""Checks for predictor-importance arithmetic and generated fit configs."""

import pandas as pd

import correctness_predictor_figures as figures
from correctness_predictor_figures import GROUP_PREDICTORS, importance_rows, mean_ci, regressor_rows


def test_importance_is_group_level_and_capability_first():
    accuracies = {
        "full": pd.Series({"judge-a": 0.75, "judge-b": 0.80}),
        "minus_capability": pd.Series({"judge-a": 0.72, "judge-b": 0.74}),
        "only_capability": pd.Series({"judge-a": 0.66, "judge-b": 0.70}),
    }

    rows = importance_rows(accuracies, groups=["Capability"], judges=("judge-a", "judge-b"))

    assert list(GROUP_PREDICTORS) == ["Capability", "Length", "Style", "Position"]
    assert rows[0]["group"] == "Capability"
    assert rows[0]["judge"] == "judge-a"
    assert abs(rows[0]["ablation_delta_accuracy"] + 0.03) < 1e-12
    assert rows[0]["only_accuracy"] == 0.66
    assert abs(rows[1]["ablation_delta_accuracy"] + 0.06) < 1e-12


def test_mean_ci_uses_replication_units():
    mean, half_width = mean_ci([1.0, 2.0, 3.0])

    assert mean == 2.0
    assert 2.48 < half_width < 2.49


def test_regressor_table_averages_judge_specific_full_fit_effects():
    effects = pd.DataFrame(
        [
            {"arm": "full", "predictor": "capability_gap_sd", "level": "qwen3-0.6b", "mean": 0.7},
            {"arm": "full", "predictor": "capability_gap_sd", "level": "qwen3-8b", "mean": 0.9},
            {"arm": "full", "predictor": "capability_gap_sd", "level": "population", "mean": 99.0},
        ]
    )

    rows = regressor_rows(effects, judges=("qwen3-0.6b", "qwen3-8b"), predictors=("capability_gap_sd",))

    assert rows[0]["mean_beta"] == 0.8
    assert rows[0]["regressor"] == "Capability gap"
    assert rows[0]["calculation"] == "zscore(abs(ECI left − ECI right))"


def test_regressor_table_keeps_names_clear_of_beta_column(monkeypatch):
    effects = pd.DataFrame(
        [
            {"arm": "full", "predictor": predictor, "level": judge, "mean": 0.1}
            for predictor in figures.PREDICTORS
            for judge in figures.JUDGES
        ]
    )
    captured = []
    monkeypatch.setattr(figures, "save", lambda figure, stem: captured.append(figure))

    figures.plot_regressor_table(effects)

    figure = captured[0]
    assert figure.get_size_inches()[0] <= 4.0
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    names = [text for text in figure.axes[0].texts if text.get_text() in figures.LABELS.values()]
    betas = [text for text in figure.axes[0].texts if text.get_text() == "+0.100"]
    for name, beta in zip(names, betas, strict=True):
        assert name.get_window_extent(renderer).x1 + 4 < beta.get_window_extent(renderer).x0


def test_weight_plot_uses_a_narrow_manuscript_canvas(monkeypatch):
    effects = pd.DataFrame(
        [
            {"arm": "full", "predictor": predictor, "level": judge, "mean": 0.1}
            for predictor in figures.PREDICTORS
            for judge in figures.JUDGES
        ]
    )
    captured = []
    monkeypatch.setattr(figures, "save", lambda figure, stem: captured.append(figure))

    figures.plot_weights(effects)

    figure = captured[0]
    assert figure.get_size_inches()[0] == 3.62
    assert "Mean ± 95% CI" in [container.get_label() for container in figure.axes[0].containers]
    figure.canvas.draw()
    assert figure.legends[0].get_window_extent().width <= figure.bbox.width


def test_importance_plot_is_side_by_side_at_weight_plot_width(monkeypatch):
    frame = pd.DataFrame(
        [
            {"group": group, "judge": judge, "only_accuracy": 0.7, "ablation_delta_accuracy": -0.08}
            for group in figures.GROUP_PREDICTORS
            for judge in figures.JUDGES
        ]
    )
    captured = []
    monkeypatch.setattr(figures, "save", lambda figure, stem: captured.append(figure))

    figures.plot_importance(frame)

    assert captured[0].get_size_inches()[0] == 3.84
    assert [axis.get_ylabel() for axis in captured[0].axes] == [
        "Single variable model accuracy",
        "Regressor ablation Δaccuracy",
    ]
    assert captured[0].axes[0].get_position().y0 == captured[0].axes[1].get_position().y0
    assert captured[0].axes[0].get_position().x1 < captured[0].axes[1].get_position().x0
    assert captured[0].axes[1].get_ylim()[0] <= -0.08
    assert captured[0].axes[1].get_ylim()[1] == 0.01
    assert all(
        label.get_rotation() == 90
        for axis in captured[0].axes
        for label in axis.get_xticklabels()
    )
    judge_lines = [
        line for line in captured[0].axes[0].lines if line.get_label() in figures.JUDGE_LABELS.values()
    ]
    assert len(judge_lines) == len(figures.JUDGES)
    assert len({line.get_color() for line in judge_lines}) == len(figures.JUDGES)
    assert [text.get_text() for text in captured[0].legends[0].get_texts()] == [
        *(figures.JUDGE_LABELS[judge] for judge in figures.JUDGES),
        "Mean ± 95% CI",
    ]


def test_save_writes_editable_svg_alongside_raster_and_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    figures.style()
    figure, axis = figures.plt.subplots()
    axis.text(0.5, 0.5, "Editable")

    figures.save(figure, "test")

    assert {path.suffix for path in tmp_path.iterdir()} == {".png", ".pdf", ".svg"}
    assert "<text" in (tmp_path / "test.svg").read_text()
