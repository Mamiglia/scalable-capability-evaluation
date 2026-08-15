import json
from pathlib import Path

import numpy as np
import pytest

import pandas as pd

from mc_cost.simulation import (
    SimulationConfig,
    SimulationResult,
    _trial_rows,
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


def test_dashboard_payload_requires_six_field_manual_identity_and_ordered_finite_quantiles():
    row = {
        "mode": "manual", "metric": "p_choose_m1",
        "judge": "j", "m1": "a", "m2": "b",
        "judge_reasoning_class": "non_reasoning",
        "m1_reasoning_class": "reasoning",
        "m2_reasoning_class": "non_reasoning",
        "p5": 0.2, "p50": 0.5, "p95": 0.8,
    }
    validate_dashboard_payload(minimal_payload(manual=[row]))

    with pytest.raises(ValueError, match="warning_state"):
        validate_dashboard_payload(minimal_payload(manual=[{
            **row, "warning_state": "made_up",
        }]))

    with pytest.raises(ValueError, match="six-field"):
        validate_dashboard_payload(minimal_payload(manual=[{
            key: value for key, value in row.items() if key != "m2_reasoning_class"
        }]))
    with pytest.raises(ValueError, match="finite"):
        validate_dashboard_payload(minimal_payload(manual=[{**row, "p95": float("inf")}]))

    invalid_rows = [
        ({key: value for key, value in row.items() if key != "p5"}, "p5, p50, and p95"),
        ({**row, "p50": "0.5"}, "finite numeric"),
        ({**row, "p5": float("nan")}, "finite numeric"),
        ({**row, "p5": 0.7, "p50": 0.5}, "ordered"),
    ]
    for invalid, message in invalid_rows:
        with pytest.raises(ValueError, match=message):
            validate_dashboard_payload(minimal_payload(manual=[invalid]))


def test_dashboard_payload_requires_numeric_exact_values_and_only_all_null_unavailable_ranking():
    exact = {"metric": "judgment_call_upper_bound", "exact": True, "value": 12}
    unavailable = {
        "metric": "kendall_tau", "valid": False,
        "reason": "no usable ranking edges",
        "p5": None, "p50": None, "p95": None,
    }
    validate_dashboard_payload(minimal_payload(automatic=[exact], ranking_recovery=[unavailable]))

    for invalid in (
        {"metric": "judgment_call_upper_bound", "exact": True},
        {**exact, "value": "12"},
        {**exact, "value": np.inf},
    ):
        with pytest.raises(ValueError, match="exact numeric value"):
            validate_dashboard_payload(minimal_payload(automatic=[invalid]))
    for invalid in (
        {**unavailable, "p50": 0.0},
        {**unavailable, "valid": True},
        {**unavailable, "reason": ""},
    ):
        with pytest.raises(ValueError, match="quantile"):
            validate_dashboard_payload(minimal_payload(ranking_recovery=[invalid]))


def test_dashboard_payload_allows_only_explicitly_unavailable_null_quantiles():
    identity = {
        "mode": "manual", "metric": "cost", "warning_state": "missing_price",
        "reason": "missing input or output price for a",
        "judge": "j", "m1": "a", "m2": "b",
        "judge_reasoning_class": "non_reasoning",
        "m1_reasoning_class": "reasoning", "m2_reasoning_class": "non_reasoning",
        "p5": None, "p50": None, "p95": None,
    }

    validate_dashboard_payload(minimal_payload(manual=[identity]))

    for invalid in (
        {**identity, "warning_state": "ok"},
        {**identity, "warning_state": "made_up"},
        {**identity, "reason": ""},
        {key: value for key, value in identity.items() if key != "reason"},
        {**identity, "warning_state": None, "valid": False},
    ):
        with pytest.raises(ValueError, match="all-null|unavailable|reason"):
            validate_dashboard_payload(minimal_payload(manual=[invalid]))


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


def test_trial_rows_export_generating_theta_and_per_role_cost_totals():
    comparisons = pd.DataFrame([{
        "comparison_id": 0, "judge": "j", "m1": "a", "m2": "b",
    }])
    presented = pd.DataFrame([{
        "comparison_id": 0, "chose_m1": 1, "m1_theta": 1.5, "m2_theta": -0.5,
        "p_choose_m1": 0.8,
    }])
    usage = pd.DataFrame([
        {"role": "candidate", "model": "a", "reasoning_class": "reasoning", "calls": 2,
         "input_tokens": 3, "output_tokens": 4, "ordinary_output_tokens": 1, "reasoning_tokens": 3},
        {"role": "judge", "model": "j", "reasoning_class": "non_reasoning", "calls": 1,
         "input_tokens": 5, "output_tokens": 6, "ordinary_output_tokens": 6, "reasoning_tokens": 0},
    ])
    priced = usage.assign(cost=[0.25, 0.75])

    rows = _trial_rows(
        SimulationConfig(trials=1, attempted_items=1, mode="manual"),
        0, comparisons, presented, usage, priced, set(),
    )

    metrics = {(row["metric"], row["model"], row["role"]): row["value"] for row in rows}
    comparison_metrics = {
        row["metric"]: row["value"] for row in rows
        if not pd.isna(row["comparison_id"]) and row["comparison_id"] == 0
    }
    assert comparison_metrics["m1_theta"] == 1.5
    assert comparison_metrics["m2_theta"] == -0.5
    assert metrics[("cost", "__total__", "candidate")] == 0.25
    assert metrics[("cost", "__total__", "judge")] == 0.75


def ranking_rows(trial, comparison_id, m1, m2, wins, m1_theta, m2_theta, calls=10):
    context = {
        "trial": trial, "configuration": "grid", "mode": "automatic",
        "comparison_id": comparison_id, "judge": "j", "m1": m1, "m2": m2,
    }
    return [
        {**context, "metric": "judgment_calls", "value": calls},
        {**context, "metric": "m1_wins", "value": wins},
        {**context, "metric": "m1_theta", "value": m1_theta},
        {**context, "metric": "m2_theta", "value": m2_theta},
    ]


def test_dashboard_ranking_recovery_uses_trial_theta_and_stable_component_membership():
    rows = []
    # The generating order reverses between trials. A static catalogue theta
    # would report +1 for both; the exact trial theta yields median zero.
    rows.extend(ranking_rows(0, 0, "a", "b", 8, 2.0, 1.0))
    rows.extend(ranking_rows(1, 0, "a", "b", 8, 0.0, 1.0))
    # Membership changes by trial and must never be merged by ordinal position.
    rows.extend(ranking_rows(0, 1, "c", "d", 8, 2.0, 1.0))
    rows.extend(ranking_rows(1, 1, "c", "e", 8, 2.0, 1.0))
    # Both partial reasons must survive export, even beside valid components.
    rows.extend(ranking_rows(0, 2, "a", "a", 8, 2.0, 2.0))
    rows.extend(ranking_rows(1, 2, "b", "c", 0, 1.0, 0.0, calls=0))
    result = SimulationResult(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
    catalogue = pd.DataFrame({
        "model_id": ["a", "b", "c", "d", "e"], "theta": [4.0, 3.0, 2.0, 1.0, 0.0],
    })

    recovery, warnings = dashboard_ranking_recovery(
        result, catalogue, {"J": 1, "presentations": 1},
    )

    assert len(recovery) == 1
    row = recovery[0]
    assert row["component_models"] == "trial-local components"
    assert row["p50"] == pytest.approx(0.5)
    assert row["n_trials"] == 2
    assert {
        reason for row in recovery for reason in row["partial_reasons"]
    } >= {"self comparison has no ranking edge", "no usable judgment calls"}
    assert {row["code"] for row in warnings} == {"disconnected_graph", "ranking_recovery_partial"}
    partial_messages = {
        row["message"] for row in warnings if row["code"] == "ranking_recovery_partial"
    }
    assert partial_messages == {"self comparison has no ranking edge", "no usable judgment calls"}


def test_dashboard_ranking_recovery_normalizes_repeated_presentations():
    base = []
    base.extend(ranking_rows(0, 0, "a", "b", 8, 3.0, 2.0, calls=10))
    base.extend(ranking_rows(0, 1, "b", "c", 7, 2.0, 1.0, calls=10))
    doubled = pd.DataFrame(base).copy()
    doubled.loc[doubled.metric == "judgment_calls", "value"] *= 2
    doubled.loc[doubled.metric == "m1_wins", "value"] *= 2
    catalogue = pd.DataFrame({"model_id": ["a", "b", "c"]})

    once, _ = dashboard_ranking_recovery(
        SimulationResult(pd.DataFrame(base), pd.DataFrame(), pd.DataFrame()),
        catalogue, {"presentations": 1},
    )
    twice, _ = dashboard_ranking_recovery(
        SimulationResult(doubled, pd.DataFrame(), pd.DataFrame()),
        catalogue, {"presentations": 2},
    )

    for key in ("p5", "p50", "p95", "valid", "recovery_type"):
        assert once[0][key] == twice[0][key]


def test_dashboard_ranking_recovery_keeps_fractional_normalized_evidence():
    rows = []
    rows.extend(ranking_rows(0, 0, "a", "b", 1, 2.0, 1.0, calls=1))
    rows.extend(ranking_rows(0, 1, "b", "c", 0, 1.0, 0.0, calls=1))

    recovery, _ = dashboard_ranking_recovery(
        SimulationResult(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame()),
        pd.DataFrame({"model_id": ["a", "b", "c"]}),
        {"presentations": 2},
    )

    assert recovery[0]["valid"] is True
    assert recovery[0]["n_finite_trials"] == 1


@pytest.mark.parametrize("presentations", [None, 0, -1, True])
def test_dashboard_ranking_recovery_requires_positive_presentations(presentations):
    result = SimulationResult(pd.DataFrame(columns=[
        "trial", "comparison_id", "m1", "m2", "metric", "value",
    ]), pd.DataFrame(), pd.DataFrame())

    with pytest.raises(ValueError, match="presentations"):
        dashboard_ranking_recovery(
            result, pd.DataFrame({"model_id": []}), {"presentations": presentations},
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
    assert '<select id="judge-count"' in html
    assert 'id="automatic-guidance"' in html
    assert '<table' in html


def test_dashboard_assets_keep_existing_viewer_separate_and_support_dark_mode():
    index = (ROOT / "interactive_viewer/index.html").read_text()
    script = (ROOT / "interactive_viewer/mc_cost.js").read_text()
    styles = (ROOT / "interactive_viewer/mc_cost.css").read_text()

    assert '<a href="mc_cost.html">Monte Carlo cost simulator</a>' in index
    assert 'fetch("data/mc_cost.json")' in script
    assert "Plotly.react" in script
    assert "mc-cost export-dashboard --config configs/mc_cost/default.json" in script
    assert "dashboard.joint_scenarios" in script
    assert "max_judgment_calls" in script
    assert "partial_reasons" in script
    assert "<th>Candidate calls</th>" in (ROOT / "interactive_viewer/mc_cost.html").read_text()
    assert "<th>Judge calls</th>" in (ROOT / "interactive_viewer/mc_cost.html").read_text()
    assert "@media (prefers-color-scheme: dark)" in styles
    assert "@media (max-width:" in styles


def test_dashboard_badges_never_invent_support_for_a_missing_scenario():
    script = (ROOT / "interactive_viewer/mc_cost.js").read_text()

    assert 'for (const status of statuses)' in script
    assert 'statuses.length ? statuses : ["predicted"]' not in script


def test_dashboard_headline_total_cost_requires_the_all_role():
    rows = [
        {"metric": "cost", "model": "__total__", "role": "candidate", "p50": 1.0},
        {"metric": "cost", "model": "__total__", "role": "judge", "p50": 2.0},
        {"metric": "cost", "model": "__total__", "role": "all", "p50": 3.0},
    ]
    script = (ROOT / "interactive_viewer/mc_cost.js").read_text()

    expected = next(
        row for row in rows
        if row["metric"] == "cost" and row["model"] == "__total__" and row["role"] == "all"
    )
    assert expected["p50"] == 3.0
    assert (
        'summaryItem("Total cost", metric(rows, "cost", '
        '(row) => row.model === "__total__" && row.role === "all"), false, true)'
    ) in script
