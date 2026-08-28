from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import pandas as pd

matplotlib.use("Agg")

from .plots import plot_coefficients, plot_lesions


def _coefficients(path: Path) -> Path:
    rows = []
    for backend in ("frequentist", "hibayes"):
        for judge in ("judge-a", "judge-b"):
            for benchmark in ("a", "b"):
                for feature, effective in (("signal", -0.8), ("noise", 0.8)):
                    row = {
                        "backend": backend, "fold": "fold-0", "judge": judge,
                        "benchmark": benchmark, "feature": feature, "effective": effective,
                    }
                    if backend == "hibayes":
                        row.update(lower=effective - 0.2, upper=effective + 0.2)
                    rows.append(row)
    output = path / "coefficients.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _lesions(path: Path) -> Path:
    output = path / "lesions.csv"
    pd.DataFrame([
        {"backend": "frequentist", "feature": "signal", "delta_log_loss": 0.12,
         "lower": 0.08, "upper": 0.16},
        {"backend": "hibayes", "feature": "signal", "delta_log_loss": 0.10,
         "lower": 0.06, "upper": 0.14},
        {"backend": "frequentist", "feature": "noise", "delta_log_loss": 0.01,
         "lower": -0.01, "upper": 0.03},
        {"backend": "hibayes", "feature": "noise", "delta_log_loss": 0.02,
         "lower": 0.00, "upper": 0.04},
    ]).to_csv(output, index=False)
    return output


def test_model_plot_artifacts_are_compact_editable_and_complete(tmp_path: Path) -> None:
    coefficient_outputs = plot_coefficients(_coefficients(tmp_path), tmp_path / "figures")
    lesion_outputs = plot_lesions(_lesions(tmp_path), tmp_path / "figures")

    assert set(coefficient_outputs) == {"frequentist", "hibayes"}
    assert all(paths[0].exists() and paths[1].exists() for paths in coefficient_outputs.values())
    assert lesion_outputs[0].exists() and lesion_outputs[1].exists()
    assert "<text" in coefficient_outputs["hibayes"][1].read_text()
    assert "<text" in lesion_outputs[1].read_text()
    assert mpimg.imread(coefficient_outputs["hibayes"][0]).shape[1] <= 1320
    assert "−1" in coefficient_outputs["hibayes"][1].read_text()
    assert "1" in coefficient_outputs["hibayes"][1].read_text()
    assert "Frequentist" in lesion_outputs[1].read_text()
    assert "HiBayes" in lesion_outputs[1].read_text()
