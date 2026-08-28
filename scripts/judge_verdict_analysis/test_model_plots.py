from pathlib import Path
from xml.etree import ElementTree

import matplotlib
import matplotlib.image as mpimg
import pandas as pd

matplotlib.use("Agg")

from .config import AnalysisConfig
from .plots import plot_coefficients, plot_lesions, plot_saved_model_results


def _coefficients(path: Path) -> Path:
    rows = []
    for backend in ("frequentist", "hibayes"):
        for judge in ("judge-a", "judge-b"):
            for benchmark in ("a", "b"):
                folds = ("fold-0", "fold-1") if backend == "frequentist" else ("fold-0",)
                for fold in folds:
                    for feature, effective in (("signal", -0.8), ("noise", 0.8)):
                        effective += -0.7 if fold == "fold-1" and feature == "signal" else 0.7 if fold == "fold-1" else 0
                        row = {
                            "backend": backend, "fold": fold, "judge": judge,
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


def _saved_results(path: Path) -> AnalysisConfig:
    config = AnalysisConfig(results_dir=path / "results", regressors=("noise", "signal")).validate()
    coefficients = pd.read_csv(_coefficients(path))
    for (backend, fold), table in coefficients.groupby(["backend", "fold"]):
        output = config.fit_dir / backend / fold
        output.mkdir(parents=True)
        table.drop(columns=["backend", "fold"]).to_csv(output / "coefficients.csv", index=False)
    config.lesion_dir.mkdir(parents=True)
    pd.read_csv(_lesions(path)).to_csv(config.lesion_dir / "lesions.csv", index=False)
    return config


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


def test_saved_results_share_scales_and_preserve_fold_points(tmp_path: Path) -> None:
    outputs = plot_saved_model_results(_saved_results(tmp_path))
    frequentist = outputs["frequentist"][1].read_text()
    hibayes = outputs["hibayes"][1].read_text()
    lesions = outputs["lesions"][1].read_text()

    assert "−1.5" in frequentist and "−1.5" in hibayes
    assert ">1.5</text>" in frequentist and ">1.5</text>" in hibayes
    assert frequentist.count("PathCollection") >= 2
    assert "#555555" not in frequentist
    assert "#555555" in hibayes
    assert "Frequentist" not in frequentist and "HiBayes" not in hibayes
    for svg in (frequentist, hibayes):
        root = ElementTree.fromstring(svg)
        marker_groups = [element for element in root.iter()
                         if element.attrib.get("id", "").startswith("PathCollection")]
        assert marker_groups
        assert all("clip-path" not in element.attrib
                   for group in marker_groups for element in group.iter())
    assert "#999999" in lesions
    assert lesions.index("noise") < lesions.index("signal")
    assert "Frequentist" in lesions and "HiBayes" in lesions
    assert 'id="legend_1"' in lesions
    assert 'id="legend_1"' not in frequentist and 'id="legend_1"' not in hibayes
