from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

from capability_predictions_glm.plots import plot_results


def test_plots_are_small_and_keep_svg_text_editable(tmp_path) -> None:
    predictions = pd.DataFrame([
        {
            "fold": "f", "bench": "b", "target_quartile": quartile,
            "variant": "no_llm_annotations", "comparator": "judge-1",
            "method": method, "model": model,
            "observed_accuracy": observed, "predicted_accuracy": predicted,
        }
        for quartile, shift in (("Q2", 0.0), ("Q3", -0.1), ("Q4", -0.2))
        for method, offset in (("direct_glm", 0.0), ("pairwise_glm", 0.05), ("llm_judge", -0.05))
        for model, observed, predicted in (("a", 0.25, 0.3 + shift + offset),
                                           ("b", 0.75, 0.7 + shift + offset))
    ])
    comparisons = pd.DataFrame([
        {
            "fold": "f", "bench": "b", "target_quartile": quartile,
            "variant": "no_llm_annotations", "comparator": "judge-1",
            "method": method, "metric": "mae", "value": value,
        }
        for quartile, addition in (("Q2", 0.0), ("Q3", 0.03), ("Q4", 0.06))
        for method, value in (("direct_glm", 0.10 + addition),
                              ("pairwise_glm", 0.12 + addition),
                              ("llm_judge", 0.14 + addition))
    ])
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    comparisons.to_csv(tmp_path / "comparisons.csv", index=False)

    outputs = plot_results(tmp_path, tmp_path / "figures")

    assert {path.name for path in outputs} == {
        "hard-accuracy-calibration.png", "hard-accuracy-calibration.svg",
        "matched-mae.png", "matched-mae.svg",
    }
    image = mpimg.imread(tmp_path / "figures/hard-accuracy-calibration.png")
    assert image.shape[1] <= 1050  # 3.5 inches at 300 dpi
    for svg in Path(tmp_path / "figures").glob("*.svg"):
        assert "<text" in svg.read_text()
