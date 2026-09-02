import importlib.util
from itertools import combinations

from matplotlib import image as mpimg
import pandas as pd
import pytest


def test_forecast_lomo_keeps_a_four_model_judge():
    import forecast_figures
    assert hasattr(forecast_figures, "forecast_lomo_predictions")

    models = list("abcd")
    primary = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
            "judge": "consensus", "model": model, "method": "easy_accuracy",
            "predictor": 0.91 + index / 100, "observed": 0.4 + index / 10,
        }
        for index, model in enumerate(models)
    ])
    rounds = pd.DataFrame([
        {
            "source": "s", "benchmark": "gpqa", "item_id": "i",
            "vote_id": "s:i", "judge": "judge-four", "major": major,
            "minor": minor, "picked_major": 0.5, "n_calls": 2,
        }
        for major, minor in combinations(models, 2)
    ])
    splits = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "half": "easy",
    }])

    got = forecast_figures.forecast_lomo_predictions(primary, rounds, splits)

    valid = got[
        got["judge"].eq("judge-four") & got["method"].eq("elo")
    ].dropna(subset=["prediction"])
    assert valid["model"].tolist() == models


@pytest.mark.filterwarnings(
    "error:This figure includes Axes that are not compatible with tight_layout",
)
def test_temporal_predictions_keep_future_elo_errors_in_release_bins(tmp_path):
    spec = importlib.util.find_spec("forecast_figures")
    assert spec is not None, "forecast figure module is missing"
    from forecast_figures import prepare_temporal_predictions

    predictions = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
         "judge": "judge-a", "component": "0:0", "model": "target-now",
         "method": "elo", "prediction": 0.70, "observed": 0.60},
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
         "judge": "judge-a", "component": "0:0", "model": "target-six",
         "method": "elo", "prediction": 0.40, "observed": 0.65},
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
         "judge": "judge-a", "component": "0:0", "model": "target-old",
         "method": "elo", "prediction": 0.80, "observed": 0.50},
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
         "judge": "judge-a", "component": "0:0", "model": "target-six",
         "method": "easy_accuracy", "prediction": 0.10, "observed": 0.65},
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
         "judge": "consensus", "component": "0:0", "model": "target-six",
         "method": "elo", "prediction": 0.20, "observed": 0.65},
    ])
    metadata = pd.DataFrame([
        {"model": "judge-a", "eci": 130.0, "release": "2025-01"},
        {"model": "target-now", "eci": 135.0, "release": "2025-02"},
        {"model": "target-six", "eci": None, "release": "2025-07"},
        {"model": "target-old", "eci": 125.0, "release": "2024-12"},
    ])
    pool = [
        {"name": "Target Now", "eci": None, "release": None},
        {"name": "Target Six", "eci": 145.0, "release": "2025-07-15"},
    ]

    got = prepare_temporal_predictions(predictions, metadata, pool)

    assert got[[
        "model", "release_gap_months", "horizon", "judge_eci", "target_eci",
        "absolute_error",
    ]].to_dict("records") == [
        {"model": "target-now", "release_gap_months": 1, "horizon": "~0 mo",
         "judge_eci": 130.0, "target_eci": 135.0, "absolute_error": 0.1},
        {"model": "target-six", "release_gap_months": 6, "horizon": "~6 mo",
         "judge_eci": 130.0, "target_eci": 145.0, "absolute_error": 0.25},
    ]

    import forecast_figures
    assert hasattr(forecast_figures, "render_temporal_heatmaps")
    forecast_figures.render_temporal_heatmaps(got, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "gpqa_forecast_eci_by_horizon.png",
        "gpqa_forecast_eci_by_horizon.svg",
        "gpqa_forecast_release_dates.png",
        "gpqa_forecast_release_dates.svg",
        "gpqa_forecast_eci_by_release_gap.png",
        "gpqa_forecast_eci_by_release_gap.svg",
    }
    pixels = mpimg.imread(tmp_path / "gpqa_forecast_eci_by_horizon.png")[..., :3]
    coloured = pixels[pixels.max(axis=2) - pixels.min(axis=2) > 0.15]
    assert (coloured[:, 2] > coloured[:, 0]).sum() > (
        coloured[:, 0] > coloured[:, 2]
    ).sum()
    eci_svg = (tmp_path / "gpqa_forecast_eci_by_horizon.svg").read_text()
    release_svg = (tmp_path / "gpqa_forecast_release_dates.svg").read_text()
    gap_svg = (tmp_path / "gpqa_forecast_eci_by_release_gap.svg").read_text()
    assert all(label in eci_svg for label in (
        "Judge ECI", "Target ECI", "~0 mo", "~6 mo", "n=1", "<text",
    ))
    assert "~12 mo" not in eci_svg and "~18 mo" not in eci_svg
    assert all(label in release_svg for label in ("Judge release", "Target release"))
    assert all(label in gap_svg for label in ("Judge ECI", "Release gap (months)"))
