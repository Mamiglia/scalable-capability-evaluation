import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DASHBOARDS = ROOT / "interactive_dashboards"


def test_transcript_viewer_is_self_contained_and_cost_free():
    viewer = DASHBOARDS / "transcript_viewer"
    for name in ("index.html", "app.js", "style.css", "plotly.min.js", "data/datasets.json"):
        assert (viewer / name).is_file()
    assert not list(viewer.rglob("mc_cost*"))

    menu = json.loads((viewer / "data/datasets.json").read_text())
    assert not any(entry["id"].startswith("jv") for entry in menu)
    assert not (viewer / "data/jv").exists()


def test_jv_dashboard_is_self_contained():
    viewer = DASHBOARDS / "jv"
    for name in ("index.html", "app.js", "style.css", "plotly.min.js", "data/datasets.json"):
        assert (viewer / name).is_file()
    menu = json.loads((viewer / "data/datasets.json").read_text())
    assert menu and all(entry["id"].startswith("jv-") for entry in menu)
    assert all((viewer / "data" / entry["id"] / "index.json").is_file() for entry in menu)


def test_cost_estimator_is_self_contained():
    cost = DASHBOARDS / "cost_estimator"
    for name in ("index.html", "mc_cost.css", "mc_cost.js", "mc_cost_calculator.js",
                 "plotly.min.js", "data/mc_cost.json", "data/mc_cost_calculator.json"):
        assert (cost / name).is_file()
