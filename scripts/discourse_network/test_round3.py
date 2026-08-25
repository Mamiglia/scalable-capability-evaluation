import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.adapters import (_atomic_adapter, FrozenSegmenterAdapter,
                                         verify_embedding_input, write_bytes_once)
from discourse_network.artifacts import load_c_manifest_chain
from discourse_network.circuits import extract_circuits
from discourse_network.gates import (apply_shortcut_recurrence_gate,
                                      validate_attachment_gate,
                                      validate_circuit_encoder_gate)
from discourse_network.network_outputs import (aggregate_circuit_transcripts,
                                                build_topology_outputs)
from discourse_network.nodes import stable_id


def graph_nodes():
    return [
        {"transcript_key": "t", "item_key": "i", "node_id": 0, "kind": "question",
         "rendered_text": "q", "target_eligible": False},
        {"transcript_key": "t", "item_key": "i", "node_id": 1, "kind": "assistant_edu",
         "rendered_text": "a", "target_eligible": True},
        {"transcript_key": "t", "item_key": "i", "node_id": 2, "kind": "assistant_edu",
         "rendered_text": "b", "target_eligible": True, "is_final_sink": True},
    ]


def test_topology_outputs_include_primary_and_full_grid():
    nodes = pd.DataFrame(graph_nodes())
    candidates = pd.DataFrame([
        {"transcript_key": "t", "source_node_id": 0, "target_node_id": 1,
         "probability": .9, "candidate_rank": 1},
        {"transcript_key": "t", "source_node_id": 1, "target_node_id": 2,
         "probability": .8, "candidate_rank": 1},
        {"transcript_key": "t", "source_node_id": 0, "target_node_id": 2,
         "probability": .7, "candidate_rank": 2},
    ])
    primary, grid = build_topology_outputs(nodes, candidates, [], threshold=.75,
                                           primary_parents=3, primary_budget=32,
                                           parent_grid=(2, 3, 4), threshold_offsets=(-.1, 0, .1),
                                           budget_grid=(16, 32, 64), config_hash="f" * 64)
    assert len(primary) == 1 and len(grid) == 27
    assert {"longest_path_edges", "branch_nodes", "convergence_nodes", "edge_count"} <= set(primary)


def test_attachment_gate_requires_complete_topology_robustness():
    gate = {"purpose": "attachment_gate", "freeze_manifest_sha256": "a" * 64,
            "config_hash": "b" * 64, "attachment_manifest_sha256": "c" * 64,
            "baselines": ["recency", "tfidf", "shared_symbol", "shared_entity", "no_text"],
            "bootstrap_unit": "opaque_item", "candidate_recall": .96,
            "bootstrap_improvement_ci_low": .01, "precision": .7, "recall": .55,
            "subset_metrics": {"gpqa": {"precision": .6, "recall": .45}},
            "over_parent_cap": .04, "synthetic_macro_f1": .81,
            "synthetic_family_min_f1": .71, "paraphrase_f1": .76,
            "calibration_unit_ids": ["c"], "gate_unit_ids": ["g"],
            "topology_grid_cells": 27,
            "nonconstant_headline_statistics": ["depth", "branching"],
            "topology_spearman": {"depth": .81, "branching": .85}}
    validate_attachment_gate(gate, "a" * 64, "b" * 64, "c" * 64)
    with pytest.raises(ValueError, match="topology"):
        validate_attachment_gate(gate | {"topology_spearman": {"depth": .79, "branching": .85}},
                                 "a" * 64, "b" * 64, "c" * 64)


def _manifest(directory, stage, upstream_path=None):
    directory.mkdir(parents=True)
    payload = directory / "data.json"
    payload.write_text(json.dumps({"stage": stage}))
    upstream_sha = hashlib.sha256(upstream_path.read_bytes()).hexdigest() if upstream_path else None
    manifest = {"schema_version": "workstream-c-v1", "stage": stage,
                "purpose": "representation_discovery", "reconciliation_sha256": "d" * 64,
                "freeze_manifest_sha256": "a" * 64, "config_hash": "b" * 64,
                "code_hash": "c" * 64, "upstream_c_manifest_sha256": upstream_sha,
                "upstream_c_manifest_path": str(upstream_path) if upstream_path else None,
                "row_counts": {"data.json": 1},
                "schema_hashes": {"data.json": stable_id({"stage": stage})},
                "files": {"data.json": {"path": "data.json",
                                           "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                                           "size": payload.stat().st_size}}}
    path = directory / "artifact_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_parent_chain_uses_actual_parent_not_child_self_attestation(tmp_path):
    nodes = _manifest(tmp_path / "nodes", "nodes")
    attachments = _manifest(tmp_path / "attachments", "attachments", nodes)
    attachment_sha = hashlib.sha256(attachments.read_bytes()).hexdigest()
    validate_attachment_gate({"purpose": "attachment_gate", "freeze_manifest_sha256": "a" * 64,
        "config_hash": "b" * 64, "attachment_manifest_sha256": attachment_sha,
        "baselines": ["recency", "tfidf", "shared_symbol", "shared_entity", "no_text"],
        "bootstrap_unit": "opaque_item", "candidate_recall": .96,
        "bootstrap_improvement_ci_low": .01, "precision": .7, "recall": .55,
        "subset_metrics": {"gpqa": {"precision": .6, "recall": .45}},
        "over_parent_cap": .04, "synthetic_macro_f1": .81, "synthetic_family_min_f1": .71,
        "paraphrase_f1": .76, "calibration_unit_ids": ["c"], "gate_unit_ids": ["g"],
        "topology_grid_cells": 27, "nonconstant_headline_statistics": ["depth"],
        "topology_spearman": {"depth": .81}}, "a" * 64, "b" * 64, attachment_sha)
    edge_views = _manifest(tmp_path / "edge-views", "edge-views", attachments)
    circuits = _manifest(tmp_path / "circuits", "circuits", edge_views)
    loaded = load_c_manifest_chain(circuits, expected_sha256=hashlib.sha256(
        circuits.read_bytes()).hexdigest(), expected_stage="circuits",
        freeze_manifest_sha256="a" * 64, config_hash="b" * 64, code_hash="c" * 64)
    assert loaded.manifest["upstream_c_manifest_sha256"] == hashlib.sha256(
        edge_views.read_bytes()).hexdigest()
    nodes.write_text(nodes.read_text() + " ")
    with pytest.raises(ValueError, match="parent"):
        load_c_manifest_chain(circuits, expected_sha256=hashlib.sha256(
            circuits.read_bytes()).hexdigest(), expected_stage="circuits",
            freeze_manifest_sha256="a" * 64, config_hash="b" * 64, code_hash="c" * 64)


def test_segmenter_recomputes_source_hash_and_embedding_input_tamper(tmp_path):
    class Segmenter:
        def segment(self, text):
            return [(0, len(text))]
    with pytest.raises(ValueError, match="source text/hash"):
        FrozenSegmenterAdapter(tmp_path, lambda _: Segmenter()).segment_rows([{
            "transcript_key": "t", "part_id": 0, "subpart_id": 0, "start": 0,
            "text": "actual", "source_text": "whole actual", "source_text_hash": "0" * 64,
        }], revision="r", config_hash="a" * 64)
    rows = FrozenSegmenterAdapter(tmp_path, lambda _: Segmenter()).segment_rows([{
        "transcript_key": "t", "part_id": 0, "subpart_id": 0, "start": 0,
        "text": "actual", "source_text": "whole actual",
        "source_text_hash": hashlib.sha256(b"whole actual").hexdigest(),
    }], revision="r", config_hash="a" * 64)
    assert rows[0]["source_text_hash"] == hashlib.sha256(b"whole actual").hexdigest()

    nodes_dir = tmp_path / "nodes"
    nodes_manifest = _manifest(nodes_dir, "nodes")
    nodes_path = nodes_dir / "nodes.parquet"
    pd.DataFrame({"transcript_key": ["t"], "node_id": [0], "semantic_text": ["x"]}).to_parquet(
        nodes_path, index=False)
    # Replace the generic payload record with the actual nodes table.
    manifest = json.loads(nodes_manifest.read_text())
    frame = pd.read_parquet(nodes_path)
    manifest["files"] = {"nodes.parquet": {"path": "nodes.parquet",
        "sha256": hashlib.sha256(nodes_path.read_bytes()).hexdigest(), "size": nodes_path.stat().st_size}}
    manifest["row_counts"] = {"nodes.parquet": 1}
    manifest["schema_hashes"] = {"nodes.parquet": stable_id([
        {"name": col, "dtype": str(frame[col].dtype)} for col in frame.columns])}
    nodes_manifest.write_text(json.dumps(manifest))
    verify_embedding_input(nodes_path, nodes_manifest, hashlib.sha256(nodes_manifest.read_bytes()).hexdigest(),
                           "a" * 64, "b" * 64, "c" * 64)
    nodes_path.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        verify_embedding_input(nodes_path, nodes_manifest,
                               hashlib.sha256(nodes_manifest.read_bytes()).hexdigest(),
                               "a" * 64, "b" * 64, "c" * 64)


def test_edge_dense_false_makes_learned_circuits_null():
    gate = {"purpose": "circuit_encoder_gate", "split": "representation_validation",
            "input_mask": {"edge_dense": False}, "synthetic_macro_f1": .9,
            "retrieval_ci_low": .1, "cv": "leave_base_trace_group_out",
            "explicit_motifs_removed": True,
            "baselines": ["length", "recency", "benchmark"], "baseline_improvement_ci_low": .1}
    assert not validate_circuit_encoder_gate(gate, expected_input_mask={"edge_dense": False})


def test_shortcut_and_recurrence_audit_can_only_fail_or_label_views():
    statuses = {"dense": "pass", "cluster": "pass", "sae": "pass"}
    gate = {"purpose": "shortcut_recurrence_gate", "selection_changed": False,
        "probes": {name: "pass" for name in ("distance", "adjacency", "lexical",
            "benchmark", "provider", "node_kind_pair", "edge_source", "transcript_length",
            "a1_marker_density")},
        "recurrence": {"cluster": {"0": {"items": 11, "provider_groups": 3,
                                             "scope": "global"}},
                       "sae": {"4": {"items": 4, "provider_groups": 2,
                                       "scope": "benchmark:gpqa"}}}}
    updated = apply_shortcut_recurrence_gate(
        statuses, gate, retained_units={"cluster": {0}, "sae": {4}})
    assert updated["view_statuses"] == {"dense": "pass", "cluster": "pass", "sae": "failed"}
    with pytest.raises(ValueError, match="tune"):
        apply_shortcut_recurrence_gate(statuses, gate | {"selection_changed": True})


def test_circuit_gate_requires_group_cv_motif_removal_and_baseline_gain():
    gate = {"purpose": "circuit_encoder_gate", "split": "representation_validation",
            "input_mask": {"edge_dense": True}, "synthetic_macro_f1": .81,
            "retrieval_ci_low": .01, "cv": "leave_base_trace_group_out",
            "explicit_motifs_removed": True, "baselines": ["length", "recency", "benchmark"],
            "baseline_improvement_ci_low": .01}
    assert validate_circuit_encoder_gate(gate, expected_input_mask={"edge_dense": True})
    assert not validate_circuit_encoder_gate(gate | {"explicit_motifs_removed": False},
                                             expected_input_mask={"edge_dense": True})


def test_tool_circuit_requires_both_assistant_paths():
    nodes = [{"transcript_key": "t", "node_id": 0, "kind": "question"},
             {"transcript_key": "t", "node_id": 1, "kind": "tool_call", "call_id": "c"},
             {"transcript_key": "t", "node_id": 2, "kind": "tool_output", "call_id": "c"},
             {"transcript_key": "t", "node_id": 3, "kind": "assistant_edu", "is_final_sink": True}]
    edges = [{"edge_id": stable_id("e", a, b), "source_node_id": a, "target_node_id": b}
             for a, b in ((1, 2), (2, 3))]
    circuits = extract_circuits(nodes, edges, config_hash="f" * 64, max_learned=64)
    assert not any(row["extractor_kind"] == "tool" for row in circuits)


def test_adapter_partial_resume_never_overwrites_checkpoint(tmp_path):
    output = tmp_path / "adapter"
    calls = []
    def interrupted(directory):
        write_bytes_once(directory / "payload.bin", b"same")
        calls.append("first")
        raise RuntimeError("interrupt")
    with pytest.raises(RuntimeError):
        _atomic_adapter(output, interrupted)
    def resumed(directory):
        write_bytes_once(directory / "payload.bin", b"same")
        calls.append("second")
        return {"complete": True}
    assert _atomic_adapter(output, resumed) == {"complete": True}
    assert calls == ["first", "second"] and (output / "payload.bin").read_bytes() == b"same"


def test_circuit_aggregates_are_zero_aware_and_exposure_adjusted():
    circuits = pd.DataFrame([
        {"transcript_key": "t", "circuit_id": "a", "extractor_kind": "path",
         "precap_count": 3, "true_kind_proportion": .75, "final_ancestry": True, "cluster_id": 0},
        {"transcript_key": "t", "circuit_id": "b", "extractor_kind": "tool",
         "precap_count": 1, "true_kind_proportion": .25, "final_ancestry": False, "cluster_id": 1},
    ])
    dense = np.array([[1., 1.], [3., 3.]])
    activations = pd.DataFrame([{"circuit_id": "a", "feature_id": 4, "activation": 2.,
                                 "sae_fit_id": "f", "dictionary_id": "d"}])
    summary, features = aggregate_circuit_transcripts(circuits, dense, activations)
    combined = summary[summary["summary_scope"].eq("combined_weighted")].iloc[0]
    assert combined["precap_count"] == 4
    combined_feature = features[features["summary_scope"].eq("combined_weighted")].iloc[0]
    assert combined_feature["mean"] == 1.5  # inactive kind contributes zero at its true 0.25 weight
