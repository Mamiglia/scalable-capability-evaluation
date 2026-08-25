import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.attachments import (candidate_recall, full_earlier_baselines,
                                            parent_cap_audit, transfer_metrics)
from discourse_network.artifacts import c_code_hash, load_c_manifest
from discourse_network.circuits import extract_circuits
from discourse_network.edge_views import (TopKSAE, aggregate_activations,
                                           fit_sae_refits, fit_scaled_pca, fit_spherical_clusters,
                                           hungarian_match, retrieval_precision_at_k)
from discourse_network.nodes import FrozenBundle, build_nodes, sha256_file, stable_id, validate_artifact_rows
from discourse_network.stages import STAGES, StageContext, attachments_stage, dispatch_stage


def plain_nodes():
    return [
        {"transcript_key": "t", "node_id": 0, "kind": "question", "rendered_text": "find alpha"},
        {"transcript_key": "t", "node_id": 1, "kind": "assistant_edu", "rendered_text": "alpha is one",
         "target_eligible": True},
        {"transcript_key": "t", "node_id": 2, "kind": "assistant_edu", "rendered_text": "unrelated beta",
         "target_eligible": True},
        {"transcript_key": "t", "node_id": 3, "kind": "assistant_edu", "rendered_text": "therefore alpha",
         "target_eligible": True, "is_final_sink": True},
    ]


def edge(source, target):
    return {"edge_id": stable_id("e", source, target), "source_node_id": source,
            "target_node_id": target}


def test_primary_boundaries_and_provenance_are_consumed_not_resegmented():
    transcript = "[Assistant]: First clause and second clause."
    content = "First clause and second clause."
    start = len("[Assistant]: ")
    parts = [{"role": "assistant", "content": content, "start": start,
              "end": start + len(content), "bare": False, "subparts": [{
                  "kind": "content", "content": content, "start": start,
                  "end": start + len(content), "call_id": None, "function": None,
                  "raw_payload": None,
              }]}]
    boundaries = [{"part_id": 0, "subpart_id": 0, "start": start, "end": start + 12},
                  {"part_id": 0, "subpart_id": 0, "start": start + 13,
                   "end": start + len(content), "verification_marker": True}]
    provenance = {"system": "isanlp_rst", "revision": "a" * 40,
                  "config_hash": "b" * 64, "blind_gate_selected": True}
    source_hash = __import__("hashlib").sha256(content.encode()).hexdigest()
    boundaries = [row | {"source_text_hash": source_hash,
                         "segmenter_revision": provenance["revision"],
                         "segmenter_config_hash": provenance["config_hash"]} for row in boundaries]
    nodes = build_nodes("t", "Q", transcript, parts, prose_boundaries=boundaries,
                        segmenter_provenance=provenance)
    assert [node["rendered_text"] for node in nodes[1:]] == ["First clause", "and second clause."]
    assert all(node["segmenter_name"] == "isanlp_rst" for node in nodes[1:])
    assert all(node["source_span_id"] == "0:0" for node in nodes[1:])
    assert nodes[2]["verification_marker"] is True
    with pytest.raises(ValueError, match="blind gate"):
        build_nodes("t", "Q", transcript, parts, prose_boundaries=boundaries,
                    segmenter_provenance=provenance | {"blind_gate_selected": False})


def test_tfidf_full_list_baselines_and_attachment_gates():
    nodes = plain_nodes()
    rows = full_earlier_baselines(nodes, target_node_id=3)
    assert [row["source_node_id"] for row in rows] == [0, 1, 2]
    assert rows[1]["tfidf"] > rows[2]["tfidf"]
    assert candidate_recall({0, 1}, {0, 1, 2}) == 1.0
    assert parent_cap_audit([{0, 1, 2, 3}, {0}], cap=3) == pytest.approx(0.5)
    metrics = transfer_metrics([1, 0, 1, 0], [1, 0, 0, 0], ["a", "a", "b", "b"])
    assert metrics["overall"]["precision"] == 1.0
    assert metrics["b"]["recall"] == 0.0


def test_scaled_pca_is_exact_and_views_fit_apply_and_match(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    rng = np.random.default_rng(0)
    values = np.r_[rng.normal(-2, .1, (12, 4)), rng.normal(2, .1, (12, 4))]
    with pytest.raises(ValueError, match="exactly 256"):
        fit_scaled_pca(values, 256, canonical=True)
    geometry = fit_scaled_pca(values, 3)
    dense = geometry.transform(values)
    clusters = fit_spherical_clusters(dense, 2, seed=0)
    assert set(clusters.predict(dense)) == {0, 1}
    sae = TopKSAE.fit(dense, width=6, top_k=2, seed=0, iterations=6)
    activations = sae.transform(dense)
    assert np.all((activations > 0).sum(1) <= 2)
    permutation = np.array([2, 0, 1, 5, 3, 4])
    match = hungarian_match(sae.decoder, sae.decoder[permutation])
    assert np.allclose(match["cosines"], 1.0)
    canonical, refits, matches = fit_sae_refits(dense, width=6, top_k=2,
                                                 groups=np.repeat(np.arange(8), 3),
                                                 seeds=(1,), iterations=3, batch_size=8)
    assert canonical.decoder.shape == refits[0].decoder.shape and len(matches) == 1
    labels = np.array([0] * 12 + [1] * 12)
    assert retrieval_precision_at_k(dense, labels, k=3) > 0.9


def test_sparse_aggregation_uses_inventory_and_rejects_duplicate_keys():
    inventory = [{"unit_id": "a", "in_final_ancestry": True},
                 {"unit_id": "b", "in_final_ancestry": False},
                 {"unit_id": "c", "in_final_ancestry": False}]
    rows = [{"unit_id": "a", "feature_id": 1, "activation": 3.0}]
    summary = aggregate_activations(rows, inventory=inventory)
    assert summary[1]["mean"] == 1.0
    assert summary[1]["proportion_active"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="duplicate activation key"):
        aggregate_activations(rows + rows, inventory=inventory)


def test_paths_keep_only_path_edges_and_radius_two_ancestry_is_explicit():
    nodes = plain_nodes()
    edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(0, 2)]
    circuits = extract_circuits(nodes, edges, config_hash="f" * 64, max_learned=64)
    path = next(row for row in circuits if row["extractor_kind"] == "path" and
                row["node_ids"] == [0, 1, 2])
    assert path["edge_count"] == 2
    radius = next(row for row in circuits if row["extractor_kind"] == "conclusion_radius_two")
    assert radius["node_ids"] == [0, 1, 2, 3]


def test_final_ancestry_truncates_by_graph_distance_and_reports_precap_proportions():
    nodes = [{"transcript_key": "t", "node_id": i, "kind": "assistant_edu",
              "rendered_text": str(i), "target_eligible": True,
              "is_final_sink": i == 7, "verification_marker": i == 5}
             for i in range(8)]
    nodes[0]["kind"] = "question"
    edges = [edge(0, 1), edge(1, 6), edge(6, 7), edge(2, 3), edge(3, 4), edge(4, 5), edge(5, 7)]
    circuits = extract_circuits(nodes, edges, config_hash="f" * 64, max_final_nodes=4,
                                max_learned=64)
    final = next(row for row in circuits if row["extractor_kind"] == "final_ancestry")
    assert 6 in final["node_ids"] and 2 not in final["node_ids"]
    verification = next(row for row in circuits if row["extractor_kind"] == "verification")
    assert verification["marker_gate"] == "verification_marker"
    assert all(row["precap_count"] >= 1 and 0 < row["true_kind_proportion"] <= 1
               for row in circuits)


def test_artifact_validator_rejects_forbidden_fields_and_dispatches_real_stages(tmp_path):
    text_hash = __import__("hashlib").sha256(b"x").hexdigest()
    row = {"transcript_key": "t", "node_id": 0, "part_id": 0, "role": "assistant",
           "kind": "assistant_edu", "rendered_text": "x", "semantic_text": "x",
           "target_eligible": True, "is_final_response": True, "is_final_sink": True,
           "rendered_start": 0, "rendered_end": 1, "part_start": 0, "part_end": 1,
           "rendered_text_hash": text_hash,
           "semantic_text_hash": text_hash, "source_span_id": "0:0",
           "segmenter_name": "test", "segmenter_revision": "r", "segmenter_config_hash": "b" * 64,
           "protected_span_kind": None, "chunk_count": 1, "token_count": 1, "coverage": 1.0,
           "embedding_row": 0, "input_row_hash": "a" * 64, "input_file_sha256": "c" * 64}
    validate_artifact_rows("nodes", [row])
    with pytest.raises(ValueError, match="forbidden"):
        validate_artifact_rows("nodes", [row | {"model": "leak"}])
    assert set(STAGES) == {"nodes", "attachments", "edge-views", "circuits"}
    called = []
    dispatch_stage("nodes", handlers={"nodes": lambda: called.append("nodes")})
    assert called == ["nodes"]


def test_nodes_stage_dispatch_writes_atomic_manifest_and_confirmation_is_apply_only(tmp_path):
    transcript = "[Assistant]: One sentence."
    start = len("[Assistant]: ")
    parts = [{"role": "assistant", "content": "One sentence.", "start": start,
              "end": len(transcript), "bare": False, "subparts": [{
                  "kind": "content", "content": "One sentence.", "start": start,
                  "end": len(transcript), "call_id": None, "function": None, "raw_payload": None,
              }]}]
    input_file_hash, input_row_hash = "5" * 64, "6" * 64
    population = pd.DataFrame([{"transcript_key": "t", "question": "Q", "transcript": transcript,
                                "transcript_parts_json": json.dumps(parts),
                                "split": "representation_train", "item_key": "i",
                                "input_file_sha256": input_file_hash,
                                "input_row_hash": input_row_hash}])
    bundle = FrozenBundle({"canonical_content_hash": "a" * 64,
                           "files": {"transcript_store": {"sha256": input_file_hash},
                                     "item_split": {"sha256": "b" * 64},
                                     "verdict_reconciliation": {"sha256": "4" * 64}}},
                          "c" * 64, population)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 0\nsegmentation:\n  primary:\n    checkpoint_revision: r\n  comparison: r\n")
    config_hash = sha256_file(config_path)
    boundaries = pd.DataFrame([{"transcript_key": "t", "part_id": 0, "subpart_id": 0,
                                "start": start, "end": len(transcript),
                                "source_text_hash": __import__("hashlib").sha256(
                                    b"One sentence.").hexdigest(),
                                "input_file_sha256": input_file_hash,
                                "input_row_hash": input_row_hash,
                                "segmenter_revision": "r", "segmenter_config_hash": config_hash}])
    boundaries.to_parquet(input_dir / "boundaries.parquet", index=False)
    boundary_schema = stable_id([{"name": column, "dtype": str(boundaries[column].dtype)}
                                 for column in boundaries.columns])
    (input_dir / "boundary_manifest.json").write_text(json.dumps({
        "file_sha256": sha256_file(input_dir / "boundaries.parquet"),
        "selected_system": "isanlp_rst", "revision": "r", "config_hash": config_hash,
        "blind_gate_selected": True, "canonical_content_hash": "a" * 64,
        "freeze_manifest_sha256": "c" * 64, "approval_sha256": "1" * 64,
        "assets_manifest_sha256": "2" * 64,
        "input_sha256": input_file_hash, "input_file_sha256": input_file_hash,
        "input_row_hashes_sha256": stable_id([input_row_hash]),
        "adapter_code_sha256": sha256_file(Path(__file__).with_name("adapters.py")),
        "gate_metrics": {"overall_f1": 0.91, "subset_min_f1": 0.86,
                         "selected_system": "isanlp_rst"}, "mode": "canonical",
        "purpose": "representation_discovery",
        "gate_metrics_sha256": "3" * 64,
        "gate_metrics_content_hash": stable_id({"overall_f1": 0.91, "subset_min_f1": 0.86,
                                                 "selected_system": "isanlp_rst"}),
        "rows": len(boundaries), "schema_hash": boundary_schema,
    }))
    metadata = {"code_commit": "test", "package_lock_hash": "e" * 64,
                "judge_util_sha256": "f" * 64, "seed": 0, "model_revisions": {},
                "approval_sha256": "1" * 64}
    context = StageContext(bundle, {"seed": 0, "segmentation": {
        "primary": {"checkpoint_revision": "r"}, "comparison": "r"}},
        config_path, input_dir, tmp_path / "nodes", metadata, stage="nodes")
    manifest = dispatch_stage("nodes", context=context)
    assert manifest["freeze_manifest_sha256"] == "c" * 64
    assert manifest["input_mask"] == {"correction": False, "verification": False}
    assert (tmp_path / "nodes" / "nodes.parquet").exists()
    load_c_manifest(tmp_path / "nodes" / "artifact_manifest.json", expected_stage="nodes",
                    freeze_manifest_sha256="c" * 64, config_hash=sha256_file(config_path),
                    code_hash=c_code_hash(), upstream_c_manifest_sha256=None)
    confirmation = StageContext(bundle, {}, config_path, input_dir, tmp_path / "attachments",
                                metadata, confirmation_apply=True)
    with pytest.raises(ValueError, match="apply-only"):
        attachments_stage(confirmation)
