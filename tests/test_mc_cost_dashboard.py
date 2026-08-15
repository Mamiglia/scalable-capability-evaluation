import json
from pathlib import Path

import pytest

import pandas as pd

from mc_cost.simulation import (
    SimulationResult,
    dashboard_payload,
    dashboard_ranking_recovery,
    validate_dashboard_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def minimal_payload(**updates):
    payload = {
        "schema_version": 1,
        "generated_at": "2026-08-15T00:00:00Z",
        "source_hashes": {},
        "quantiles": [0.05, 0.5, 0.95],
        "catalogue": [],
        "automatic": [],
        "manual": [],
        "scaling": [],
        "ranking_recovery": [],
        "warnings": [],
    }
    payload.update(updates)
    return payload


def test_dashboard_payload_requires_exact_versioned_schema():
    validate_dashboard_payload(minimal_payload())

    with pytest.raises(ValueError, match="quantiles"):
        validate_dashboard_payload(minimal_payload(quantiles=[0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match="schema fields"):
        validate_dashboard_payload({**minimal_payload(), "formula": "browser-side"})


def test_dashboard_payload_requires_six_field_manual_identity_and_finite_quantiles():
    row = {
        "mode": "manual", "metric": "p_choose_m1",
        "judge": "j", "m1": "a", "m2": "b",
        "judge_reasoning_class": "non_reasoning",
        "m1_reasoning_class": "reasoning",
        "m2_reasoning_class": "non_reasoning",
        "p5": 0.2, "p50": 0.5, "p95": 0.8,
    }
    validate_dashboard_payload(minimal_payload(manual=[row]))

    with pytest.raises(ValueError, match="six-field"):
        validate_dashboard_payload(minimal_payload(manual=[{
            key: value for key, value in row.items() if key != "m2_reasoning_class"
        }]))
    with pytest.raises(ValueError, match="finite"):
        validate_dashboard_payload(minimal_payload(manual=[{**row, "p95": float("inf")}]))


def test_dashboard_builder_sorts_every_row_array_and_derives_scaling_rows():
    automatic = [
        {
            "mode": "automatic", "configuration": "b", "metric": "cost",
            "model": "__total__", "role": "all", "p5": 2.0, "p50": 3.0, "p95": 4.0,
        },
        {
            "mode": "automatic", "configuration": "a", "metric": "choice_accuracy",
            "p5": 0.4, "p50": 0.5, "p95": 0.6,
        },
    ]
    payload = dashboard_payload(
        generated_at="2026-08-15T00:00:00Z",
        source_hashes={"decision": "abc"},
        catalogue=[{"model_id": "z"}, {"model_id": "a"}],
        automatic=reversed(automatic),
        manual=[],
        ranking_recovery=[],
        warnings=[{"code": "z", "message": "last"}, {"code": "a", "message": "first"}],
    )

    assert list(payload) == [
        "schema_version", "generated_at", "source_hashes", "quantiles", "catalogue",
        "automatic", "manual", "scaling", "ranking_recovery", "warnings",
    ]
    for key in ("catalogue", "automatic", "manual", "scaling", "ranking_recovery", "warnings"):
        assert payload[key] == sorted(
            payload[key], key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        )
    assert {row["metric"] for row in payload["scaling"]} == {"choice_accuracy", "cost"}


def test_dashboard_ranking_recovery_keeps_components_separate_and_summarizes_trials():
    rows = []
    for trial in (0, 1):
        for comparison_id, (m1, m2) in enumerate((
            ("a", "b"), ("c", "d"), ("a", "a"),
        )):
            context = {
                "trial": trial, "configuration": "grid", "mode": "automatic",
                "comparison_id": comparison_id, "judge": "j", "m1": m1, "m2": m2,
            }
            rows.extend([
                {**context, "metric": "judgment_calls", "value": 10},
                {**context, "metric": "m1_wins", "value": 8},
                # Deliberately contradictory: ranking must use exact M1 wins,
                # not reverse-engineer votes from a point-theta accuracy label.
                {**context, "metric": "choice_accuracy", "value": 0.2},
            ])
    result = SimulationResult(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
    catalogue = pd.DataFrame({
        "model_id": ["a", "b", "c", "d"], "theta": [4.0, 3.0, 2.0, 1.0],
    })

    recovery, warnings = dashboard_ranking_recovery(result, catalogue, {"J": 1})

    assert len(recovery) == 2
    assert {row["component_index"] for row in recovery} == {0, 1}
    assert all(row["metric"] == "kendall_tau" and row["p50"] == 1.0 for row in recovery)
    assert {row["code"] for row in warnings} == {"disconnected_graph", "ranking_recovery_partial"}
    assert "self comparison" in next(
        row["message"] for row in warnings if row["code"] == "ranking_recovery_partial"
    )


def test_dashboard_markup_has_accessible_native_controls_and_local_plotly():
    html = (ROOT / "interactive_viewer/mc_cost.html").read_text()
    for control in (
        "mode", "automatic-controls", "manual-controls", "judge-pool", "anchor-pool",
        "comparator-pool", "judge-count", "anchor-count", "comparator-count",
        "attempted-items", "presentations", "strength-filter", "manual-rows",
        "add-manual-row", "run-summary", "accuracy-plot", "scaling-axis", "scaling-plot",
        "token-table", "manual-results", "ranking-recovery", "warning-list",
    ):
        assert f'id="{control}"' in html
    assert html.count("<label") >= 10
    assert 'src="plotly.min.js"' in html
    assert 'src="mc_cost.js"' in html
    assert '<input id="judge-count" type="number"' in html
    assert '<table' in html


def test_dashboard_assets_keep_existing_viewer_separate_and_support_dark_mode():
    index = (ROOT / "interactive_viewer/index.html").read_text()
    script = (ROOT / "interactive_viewer/mc_cost.js").read_text()
    styles = (ROOT / "interactive_viewer/mc_cost.css").read_text()

    assert '<a href="mc_cost.html">Monte Carlo cost simulator</a>' in index
    assert 'fetch("data/mc_cost.json")' in script
    assert "Plotly.react" in script
    assert "@media (prefers-color-scheme: dark)" in styles
    assert "@media (max-width:" in styles
