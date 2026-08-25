import hashlib
import json
import sys
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.adapters import (FrozenEmbeddingAdapter, FrozenSegmenterAdapter,
                                         require_heavy_runtime, validate_embedding_alignment)
from discourse_network.artifacts import load_c_manifest
from discourse_network.circuits import extract_circuits, marker_input_mask
from discourse_network.edge_views import TopKSAE, bootstrap_group_indices, resolve_view_mask
from discourse_network.gates import (validate_attachment_gate, validate_circuit_encoder_gate,
                                      validate_dense_retrieval_gate)
from discourse_network.nodes import recompute_freeze_contract_hash, stable_id


def test_contract_hash_matches_sorted_snapshot_names(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    files = {}
    for name, text in (("config", "c"), ("main_code", "m"),
                       ("surprisal_code", "s"), ("judge_util_code", "u")):
        path = code / name
        path.write_text(text)
        files[name] = {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    expected = hashlib.sha256(b"config\0cjudge_util_code\0umain_code\0msurprisal_code\0s").hexdigest()
    assert recompute_freeze_contract_hash(files, code) == expected
    (code / "config").write_text("changed")
    with pytest.raises(ValueError, match="snapshot hash"):
        recompute_freeze_contract_hash(files, code)


def test_chained_c_manifest_detects_tamper(tmp_path):
    frame = pd.DataFrame({"x": [1, 2]})
    frame.to_parquet(tmp_path / "rows.parquet", index=False)
    file_hash = hashlib.sha256((tmp_path / "rows.parquet").read_bytes()).hexdigest()
    manifest = {"schema_version": "workstream-c-v1", "stage": "nodes",
                "purpose": "representation_discovery",
                "reconciliation_sha256": "d" * 64,
                "freeze_manifest_sha256": "a" * 64, "config_hash": "b" * 64,
                "code_hash": "c" * 64, "upstream_c_manifest_sha256": None,
                "row_counts": {"rows.parquet": 2},
                "schema_hashes": {"rows.parquet": stable_id([
                    {"name": "x", "dtype": str(frame.x.dtype)}])},
                "files": {"rows.parquet": {"path": "rows.parquet", "sha256": file_hash,
                                                "size": (tmp_path / "rows.parquet").stat().st_size}}}
    path = tmp_path / "artifact_manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = load_c_manifest(path, expected_stage="nodes", freeze_manifest_sha256="a" * 64,
                             config_hash="b" * 64, code_hash="c" * 64,
                             upstream_c_manifest_sha256=None)
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "rows.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_c_manifest(path, expected_stage="nodes", freeze_manifest_sha256="a" * 64,
                        config_hash="b" * 64, code_hash="c" * 64,
                        upstream_c_manifest_sha256=None)


def test_attachment_gate_is_frozen_and_calibration_disjoint():
    gate = {"purpose": "attachment_gate", "freeze_manifest_sha256": "a" * 64, "config_hash": "b" * 64,
            "attachment_manifest_sha256": "c" * 64, "candidate_recall": 0.96,
            "baselines": ["recency", "tfidf", "shared_symbol", "shared_entity", "no_text"],
            "bootstrap_unit": "opaque_item",
            "bootstrap_improvement_ci_low": 0.01, "precision": 0.70, "recall": 0.55,
            "subset_metrics": {"gpqa": {"precision": 0.60, "recall": 0.45}},
            "over_parent_cap": 0.04, "synthetic_macro_f1": 0.81,
            "synthetic_family_min_f1": 0.71, "paraphrase_f1": 0.76,
            "calibration_unit_ids": ["c1"], "gate_unit_ids": ["g1"]}
    gate |= {"topology_grid_cells": 27, "nonconstant_headline_statistics": ["depth"],
             "topology_spearman": {"depth": .81}}
    assert validate_attachment_gate(gate, "a" * 64, "b" * 64, "c" * 64)["status"] == "pass"
    with pytest.raises(ValueError, match="disjoint"):
        validate_attachment_gate(gate | {"gate_unit_ids": ["c1"]}, "a" * 64, "b" * 64,
                                 "c" * 64)


def test_views_fail_independently_and_freeze_mask():
    statuses, mask = resolve_view_mask(dense_pass=True, cluster_pass=False, sae_pass=True)
    assert statuses == {"dense": "pass", "cluster": "failed", "sae": "pass"}
    assert mask == {"dense": True, "cluster": False, "sae": True}
    assert resolve_view_mask(dense_pass=False, cluster_pass=True, sae_pass=True)[0] == {
        "dense": "failed", "cluster": "null", "sae": "null"}
    assert validate_dense_retrieval_gate({"purpose": "dense_retrieval_gate",
                                          "split": "representation_validation",
                                          "bootstrap_ci_low": 0.01})
    assert not validate_dense_retrieval_gate({"purpose": "dense_retrieval_gate",
                                              "split": "representation_train",
                                              "bootstrap_ci_low": 0.01})
    assert validate_circuit_encoder_gate({"purpose": "circuit_encoder_gate",
        "split": "representation_validation", "input_mask": {"edge_dense": True},
        "synthetic_macro_f1": 0.81, "retrieval_ci_low": 0.01,
        "cv": "leave_base_trace_group_out", "explicit_motifs_removed": True,
        "baselines": ["length", "recency", "benchmark"], "baseline_improvement_ci_low": .01})


def test_group_bootstrap_keeps_whole_opaque_items():
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    indices = bootstrap_group_indices(groups, seed=3)
    for group in set(groups):
        selected = indices[groups[indices] == group]
        assert len(selected) % np.sum(groups == group) == 0


def test_torch_topk_sae_runs_minibatches():
    values = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    sae = TopKSAE.fit(values, width=8, top_k=2, seed=0, iterations=3, batch_size=5,
                      device="cpu", dtype="float32")
    assert sae.trainer == "torch_minibatch_topk"
    assert np.all((sae.transform(values) > 0).sum(1) <= 2)


def test_segmenter_and_embedding_adapters_are_lazy_and_testable_with_fakes():
    calls = []

    class Segmenter:
        def segment(self, text):
            return [(0, len(text))]

    segmenter = FrozenSegmenterAdapter(Path("/staged"), lambda _: calls.append("segment") or Segmenter())
    rows = segmenter.segment_rows([{"transcript_key": "t", "part_id": 0, "subpart_id": 0,
                                    "start": 10, "text": "hello"}], revision="r",
                                   config_hash="a" * 64)
    assert calls == ["segment"] and rows[0]["start"] == 10 and rows[0]["end"] == 15

    class Encoder:
        def encode_nodes(self, texts, **_):
            return np.array([[1., 0.], [0., 1.]], dtype=np.float32), [1, 2], [1, 1]

    embedding = FrozenEmbeddingAdapter(Path("/model"), lambda _: Encoder())
    vectors, metadata = embedding.encode([{"transcript_key": "t", "node_id": 0, "semantic_text": "a"},
                                           {"transcript_key": "t", "node_id": 1, "semantic_text": "b"}],
                                          revision="r")
    aligned = validate_embedding_alignment(pd.DataFrame({"transcript_key": ["t", "t"],
                                                          "node_id": [0, 1]}), metadata, vectors)
    assert aligned["embedding_row"].tolist() == [0, 1]


def test_heavy_runtime_refuses_local_and_gateway():
    with pytest.raises(RuntimeError, match="Slurm"):
        require_heavy_runtime(env={}, hostname="worker1")
    with pytest.raises(RuntimeError, match="gateway"):
        require_heavy_runtime(env={"SLURM_JOB_ID": "1"}, hostname="gateway2")
    require_heavy_runtime(env={"SLURM_JOB_ID": "1"}, hostname="gpu42")


def test_stage_cli_refuses_when_wrapper_is_bypassed(tmp_path):
    command = [sys.executable, "-m", "scripts.discourse_network.stages", "nodes"]
    for name in ("freeze", "store", "split", "reconciliation", "judge-util", "approval",
                 "config", "input", "output"):
        command += [f"--{name}", str(tmp_path / name)]
    command += ["--code-commit", "a" * 40]
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "Slurm-only" in result.stderr


def test_chorded_path_keeps_only_consecutive_edges_and_radius_is_capped():
    nodes = [{"transcript_key": "t", "node_id": i, "kind": "assistant_edu",
              "rendered_text": str(i), "target_eligible": True, "is_final_sink": i == 14}
             for i in range(15)]
    nodes[0]["kind"] = "question"
    edges = [{"edge_id": stable_id("e", a, b), "source_node_id": a, "target_node_id": b}
             for a, b in [(0, 1), (1, 2), (0, 2)] + [(i, 14) for i in range(2, 14)]]
    circuits = extract_circuits(nodes, edges, config_hash="f" * 64, max_local_nodes=12,
                                max_learned=128)
    path = next(row for row in circuits if row["extractor_kind"] == "path" and
                row["node_ids"] == [0, 1, 2])
    assert path["edge_count"] == 2 and stable_id("e", 0, 2) not in path["edge_ids"]
    radius = next(row for row in circuits if row["extractor_kind"] == "conclusion_radius_two")
    assert radius["node_count"] <= 12


def test_missing_markers_disable_extractors_explicitly():
    assert marker_input_mask([{"backtracking_marker_available": False,
                               "verification_marker_available": False}]) == {
        "correction": False, "verification": False}
