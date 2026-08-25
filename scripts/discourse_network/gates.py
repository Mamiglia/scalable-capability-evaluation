"""Frozen outcome-blind gate artifacts."""

from __future__ import annotations

from typing import Any
import math


ATTACHMENT_BASELINES = {"recency", "tfidf", "shared_symbol", "shared_entity", "no_text"}


def validate_attachment_gate(gate: dict[str, Any], freeze_manifest_sha256: str,
                             config_hash: str, attachment_manifest_sha256: str) -> dict[str, Any]:
    bindings = {"purpose": "attachment_gate", "freeze_manifest_sha256": freeze_manifest_sha256,
                "config_hash": config_hash,
                "attachment_manifest_sha256": attachment_manifest_sha256}
    for field, expected in bindings.items():
        if gate.get(field) != expected:
            raise ValueError(f"attachment gate {field} mismatch")
    if set(gate.get("baselines", ())) != ATTACHMENT_BASELINES or gate.get("bootstrap_unit") != "opaque_item":
        raise ValueError("attachment gate must include every full-list baseline and item bootstrap")
    if set(gate.get("calibration_unit_ids", ())) & set(gate.get("gate_unit_ids", ())):
        raise ValueError("attachment calibration and gate rows must be disjoint")
    floors = {"candidate_recall": 0.95, "bootstrap_improvement_ci_low": 0.0,
              "precision": 0.65, "recall": 0.50, "over_parent_cap": 0.05,
              "synthetic_macro_f1": 0.80, "synthetic_family_min_f1": 0.70,
              "paraphrase_f1": 0.75}
    for field, floor in floors.items():
        value = float(gate.get(field, -1))
        if field == "over_parent_cap":
            if value > floor:
                raise ValueError("attachment gate parent-cap audit failed")
        elif field == "bootstrap_improvement_ci_low":
            if value <= floor:
                raise ValueError("attachment grouped-bootstrap baseline improvement failed")
        elif value < floor:
            raise ValueError(f"attachment gate {field} failed")
    for subset, metrics in gate.get("subset_metrics", {}).items():
        if metrics.get("precision", 0) < 0.55 or metrics.get("recall", 0) < 0.40:
            raise ValueError(f"attachment subset transfer failed for {subset}")
    if not gate.get("subset_metrics"):
        raise ValueError("attachment gate needs per-subset transfer metrics")
    statistics = gate.get("nonconstant_headline_statistics")
    correlations = gate.get("topology_spearman")
    if (gate.get("topology_grid_cells") != 27 or not isinstance(statistics, list) or
            not isinstance(correlations, dict) or set(statistics) != set(correlations) or
            any(not math.isfinite(float(correlations[name])) or float(correlations[name]) < 0.80
                for name in statistics)):
        raise ValueError("attachment topology robustness gate is absent or failed")
    populations = gate.get("candidate_budget_populations")
    if populations is not None:
        try:
            values = [int(populations[str(budget)]) for budget in (16, 32, 64)]
            realized = int(gate["max_realized_candidate_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("attachment candidate-budget populations are invalid") from error
        if (values != sorted(values) or (realized > 16 and values[1] <= values[0]) or
                (realized > 32 and values[2] <= values[1])):
            raise ValueError("attachment candidate-budget populations are not genuinely distinct")
    return gate | {"status": "pass"}


def validate_dense_retrieval_gate(gate: dict[str, Any], *, validation_only: bool = True) -> bool:
    return (gate.get("purpose") == "dense_retrieval_gate" and
            gate.get("split") == "representation_validation" and
            gate.get("bootstrap_ci_low", 0) > 0 and validation_only)


def validate_circuit_encoder_gate(gate: dict[str, Any], *,
                                  expected_input_mask: dict[str, bool] | None = None) -> bool:
    mask = gate.get("input_mask", {})
    if expected_input_mask is not None and mask != expected_input_mask:
        return False
    return (gate.get("purpose") == "circuit_encoder_gate" and
            gate.get("split") == "representation_validation" and mask.get("edge_dense") is True and
            gate.get("synthetic_macro_f1", 0) >= 0.80 and gate.get("retrieval_ci_low", 0) > 0 and
            gate.get("cv") == "leave_base_trace_group_out" and
            gate.get("explicit_motifs_removed") is True and
            set(gate.get("baselines", ())) == {"length", "recency", "benchmark"} and
            gate.get("baseline_improvement_ci_low", 0) > 0)


SHORTCUT_PROBES = {"distance", "adjacency", "lexical", "benchmark", "provider",
                   "node_kind_pair", "edge_source", "transcript_length", "a1_marker_density"}


def apply_shortcut_recurrence_gate(statuses: dict[str, str], gate: dict[str, Any], *,
                                   retained_units: dict[str, set[int]] | None = None
                                   ) -> dict[str, Any]:
    if gate.get("purpose") != "shortcut_recurrence_gate":
        raise ValueError("shortcut/recurrence gate purpose mismatch")
    if gate.get("selection_changed") is not False:
        raise ValueError("post-freeze shortcut audit may fail or label a view, never tune selection")
    probes = gate.get("probes", {})
    if set(probes) != SHORTCUT_PROBES or any(value not in {"pass", "surface", "failed"}
                                             for value in probes.values()):
        raise ValueError("shortcut audit is incomplete")
    updated = dict(statuses)
    global_failure = any(value == "failed" for value in probes.values())
    surface_labels = sorted(name for name, value in probes.items() if value == "surface")
    if surface_labels:
        updated = {view: "surface" if status == "pass" else status
                   for view, status in updated.items()}
    recurrence = gate.get("recurrence", {})
    retained_units = retained_units or {
        view: {int(unit) for unit in recurrence.get(view, {})} for view in ("cluster", "sae")}
    unit_statuses = {}
    for view in ("cluster", "sae"):
        if statuses.get(view) != "pass":
            unit_statuses[view] = {}
            continue
        expected = {str(unit) for unit in retained_units.get(view, set())}
        support = recurrence.get(view)
        if not expected or not isinstance(support, dict) or set(support) != expected:
            raise ValueError(f"shortcut recurrence must cover every retained {view} unit")
        unit_statuses[view] = {}
        for unit in sorted(expected, key=int):
            record = support[unit]
            scope = record.get("scope")
            if scope != "global" and not (isinstance(scope, str) and scope.startswith("benchmark:")):
                raise ValueError(f"shortcut recurrence {view} unit {unit} lacks a declared scope")
            if global_failure or record.get("items", 0) < 10 or record.get("provider_groups", 0) < 3:
                unit_statuses[view][unit] = "failed"
            else:
                unit_statuses[view][unit] = "surface" if surface_labels else "pass"
        if not any(status in {"pass", "surface"} for status in unit_statuses[view].values()):
            updated[view] = "failed"
    if global_failure:
        updated = {view: "failed" if status in {"pass", "surface"} else status
                   for view, status in updated.items()}
    return {"view_statuses": updated, "unit_statuses": unit_statuses,
            "surface_labels": surface_labels}
