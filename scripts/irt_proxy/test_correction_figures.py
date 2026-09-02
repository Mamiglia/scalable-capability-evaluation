import struct

import numpy as np
import pandas as pd

from test_prediction import prediction_fixture


def test_correction_figures_are_small_and_keep_svg_text(tmp_path):
    from correction_figures import _gap_curve_table, render_correction_figures
    from metrics import capability_gap_rows

    predictions = prediction_fixture()
    gap_pairs = capability_gap_rows(predictions)
    bins = _gap_curve_table(gap_pairs, n_resamples=20)
    assert len(bins) == 60
    assert bins["n_pairs"].eq(0).any()
    render_correction_figures(
        predictions,
        gap_pairs,
        tmp_path,
    )

    for stem in ("capability_predictions", "gap_curve"):
        png, svg = tmp_path / f"{stem}.png", tmp_path / f"{stem}.svg"
        width, _ = struct.unpack(">II", png.read_bytes()[16:24])
        text = svg.read_text()
        assert width <= 900
        assert "<text" in text
        assert "Bias-corrected capability" not in text


def test_gap_curve_artist_keeps_nan_at_empty_middle_bin(tmp_path):
    from correction_figures import _gap_curve
    from metrics import capability_gap_rows

    predictions = pd.DataFrame([
        {
            "benchmark": "gpqa", "fold": index, "component": "c0",
            "method": "corrected", "model": f"m{index}",
            "predictor": value, "prediction": value,
            "prediction_p5": value, "prediction_p95": value,
            "observed": value,
        }
        for index, value in enumerate((0.0, 0.02, 0.12))
    ])

    figure = _gap_curve(capability_gap_rows(predictions), tmp_path)

    assert figure is not None
    corrected = next(
        line for line in figure.axes[0].lines if line.get_label() == "Corrected"
    )
    assert np.allclose(corrected.get_xdata()[:3], [0.025, 0.075, 0.125])
    assert corrected.get_ydata()[0] == 1
    assert np.isnan(corrected.get_ydata()[1])
    assert corrected.get_ydata()[2] == 1
