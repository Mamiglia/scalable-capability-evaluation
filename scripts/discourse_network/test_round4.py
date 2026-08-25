import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.adapters import (FrozenEmbeddingAdapter, FrozenSegmenterAdapter,
                                         load_segmentation_spans)
from discourse_network.artifacts import c_code_hash, load_c_manifest, load_c_manifest_chain
from discourse_network.attachments import build_candidates
from discourse_network.gates import apply_shortcut_recurrence_gate, validate_attachment_gate
from discourse_network.network_outputs import (aggregate_circuit_transcripts,
                                                aggregate_edge_transcripts,
                                                build_topology_outputs,
                                                topology_robustness)
from discourse_network.nodes import FrozenBundle, sha256_file, stable_id
from discourse_network.stages import (StageContext, attachments_stage, circuits_stage,
                                      edge_views_stage, nodes_stage, _load_feature_inventory)
from discourse_network.test_pipeline import frozen_inputs


class _FakeHead:
    C = .1

    def decision_function(self, values):
        return np.asarray(values)[:, -1] * 2 - 1


def test_edge_aggregates_zero_retained_cluster_when_a_transcript_loses_all_assignments():
    nodes = pd.DataFrame([
        {"transcript_key": key, "node_id": node, "is_final_sink": node == 1}
        for key in ("a", "b") for node in (0, 1)
    ])
    edges = pd.DataFrame([
        {"transcript_key": "a", "edge_id": "a-edge", "source_node_id": 0,
         "target_node_id": 1, "dense_row": 0, "edge_source": "inferred",
         "cluster_id": np.nan, "item_key": "a-item", "split": "representation_validation"},
        {"transcript_key": "b", "edge_id": "b-edge", "source_node_id": 0,
         "target_node_id": 1, "dense_row": 1, "edge_source": "inferred",
         "cluster_id": 0, "item_key": "b-item", "split": "representation_validation"},
    ])

    summaries, _ = aggregate_edge_transcripts(
        edges, np.ones((2, 2)), pd.DataFrame(columns=["edge_id", "feature_id", "activation"]),
        nodes, feature_ids=[], cluster_ids=[0])
    combined = summaries[summaries.edge_scope == "combined"].set_index("transcript_key")

    assert combined.loc["a", "cluster_0_proportion"] == 0
    assert combined.loc["b", "cluster_0_proportion"] == 1


class _FakePlatt:
    def transform(self, logits):
        return np.where(np.asarray(logits) > 0, .9, .1)


class _FakeEncoder:
    def encode_nodes(self, texts, **_):
        vectors = np.array([[1., (index % 5) + 1.] for index in range(len(texts))])
        return vectors, [len(text) for text in texts], [1] * len(texts)


class _PositiveSAE:
    def __init__(self, input_width):
        self.decoder = np.zeros((4, input_width), dtype=np.float32)
        self.top_k = 2
        self.fit_id = stable_id("positive-sae-fit", input_width)
        self.dictionary_id = stable_id("positive-sae-dictionary", input_width)
        self.calls = 0

    def transform(self, values):
        self.calls += 1
        result = np.zeros((len(values), 4), dtype=np.float32)
        if self.calls == 1:
            result[:, :2] = 1
        else:
            result[:, 2] = 1
        return result

    def reconstruct(self, values):
        return self.transform(values) @ self.decoder


def _positive_views(train, validation, *_args, **_kwargs):
    sae = _PositiveSAE(train.shape[1])
    return {"clusters": None, "sae": sae,
            "statuses": {"dense": "pass", "cluster": "failed", "sae": "pass"},
            "input_mask": {"dense": True, "cluster": False, "sae": True},
            "errors": {"cluster": "test fixture"}, "stable_features": {0, 1},
            "sae_refits": [], "matches": [], "cluster_stability": []}


def test_frozen_sae_feature_inventory_rejects_reordered_model_ids():
    sae = _PositiveSAE(1)
    digest = stable_id({"sae_fit_id": sae.fit_id, "dictionary_id": sae.dictionary_id,
                        "feature_ids": [0, 1]})
    models = {"retained_features": [1, 0], "retained_features_sha256": digest}
    manifest = {"retained_edge_sae_feature_ids": [0, 1],
                "retained_edge_sae_feature_ids_sha256": digest}
    with pytest.raises(ValueError, match="inventory mismatch"):
        _load_feature_inventory(models, manifest, "edge", sae)


def test_100_node_reservoir_has_distinct_16_32_64_prefixes_and_gate_populations():
    nodes = [{"transcript_key": "t", "item_key": "i", "split": "representation_validation",
              "node_id": i, "kind": ("question" if i == 0 else
                                       "tool_output" if 60 <= i < 68 else "assistant_edu"),
              "rendered_text": f"node {i}" + (" SHARED" if 40 <= i < 56 or i == 99 else ""),
              "target_eligible": i > 0, "is_final_sink": i == 99}
             for i in range(100)]
    embeddings = np.zeros((100, 3))
    embeddings[:, 1] = 1
    embeddings[20:36] = [1, 0, 0]
    embeddings[99] = [1, 0, 0]
    rows = build_candidates(nodes, embeddings, 99, cap=64, preceding=8, semantic=8,
                            identifiers=8, structural=4, reservoir_multiplier=2)
    assert len(rows) == 57
    assert [sum(row["candidate_rank"] <= cap for row in rows) for cap in (16, 32, 64)] == [16, 32, 57]

    candidates = pd.DataFrame(rows)
    candidates["probability"] = np.linspace(.99, .01, len(candidates))
    primary, grid = build_topology_outputs(
        pd.DataFrame(nodes), candidates, [], threshold=.05, primary_parents=64,
        primary_budget=32, parent_grid=(64,), threshold_offsets=(0,),
        budget_grid=(16, 32, 64), config_hash="f" * 64)
    metrics = topology_robustness(primary, grid, ["edge_count"])
    assert metrics["candidate_budget_populations"] == {"16": 16, "32": 32, "64": 57}
    assert metrics["max_realized_candidate_size"] == 57


def test_attachment_gate_rejects_fake_budget_labels_without_distinct_populations():
    gate = {"purpose": "attachment_gate", "freeze_manifest_sha256": "a" * 64,
            "config_hash": "b" * 64, "attachment_manifest_sha256": "c" * 64,
            "baselines": ["recency", "tfidf", "shared_symbol", "shared_entity", "no_text"],
            "bootstrap_unit": "opaque_item", "candidate_recall": .96,
            "bootstrap_improvement_ci_low": .01, "precision": .7, "recall": .55,
            "subset_metrics": {"gpqa": {"precision": .6, "recall": .45}},
            "over_parent_cap": .04, "synthetic_macro_f1": .81,
            "synthetic_family_min_f1": .71, "paraphrase_f1": .76,
            "calibration_unit_ids": ["c"], "gate_unit_ids": ["g"],
            "topology_grid_cells": 27, "nonconstant_headline_statistics": ["depth"],
            "topology_spearman": {"depth": .81}, "max_realized_candidate_size": 64,
            "candidate_budget_populations": {"16": 100, "32": 100, "64": 100}}
    with pytest.raises(ValueError, match="candidate-budget populations"):
        validate_attachment_gate(gate, "a" * 64, "b" * 64, "c" * 64)


def test_segmentation_input_is_exact_frozen_store_and_hashes_rows_internally(tmp_path):
    store, split, freeze, _ = frozen_inputs(tmp_path)
    spans, bundle = load_segmentation_spans(
        freeze, store, split_path=split,
        judge_util_path=json.loads(freeze.read_text())["files"]["judge_util_code"]["path"])
    assert spans and {"input_file_sha256", "input_row_hash", "source_text_hash"} <= set(spans[0])
    assert spans[0]["input_file_sha256"] == bundle.manifest["files"]["transcript_store"]["sha256"]
    assert spans[0]["text"] in spans[0]["source_text"]

    copied = tmp_path / "copy.parquet"
    copied.write_bytes(store.read_bytes())
    with pytest.raises(ValueError, match="exact frozen transcript store"):
        load_segmentation_spans(
            freeze, copied, split_path=split,
            judge_util_path=json.loads(freeze.read_text())["files"]["judge_util_code"]["path"])


def test_adapter_wrapper_allows_hash_checked_partial_resume(tmp_path):
    output = tmp_path / "adapter-output"
    Path(f"{output}.partial").mkdir()
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python"
    python.write_text("#!/usr/bin/env bash\nexit 0\n")
    python.chmod(0o755)
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    env = os.environ | {
        "SLURM_JOB_ID": "1", "C_CODE_DIR": str(Path(__file__).resolve().parents[2]),
        "C_ADAPTER_VENV": str(tmp_path / "venv"), "C_ADAPTER_INPUT": str(tmp_path / "store"),
        "C_ADAPTER_OUTPUT": str(output), "C_ADAPTER_ASSETS": str(tmp_path / "assets"),
        "C_ADAPTER_REVISION": "r", "C_CONFIG_HASH": "a" * 64,
        "C_FREEZE_SHA256": "b" * 64, "C_CONTENT_HASH": "c" * 64,
        "C_APPROVAL": str(tmp_path / "approval"), "C_GATE_METRICS": str(gate),
        "C_FREEZE": str(tmp_path / "freeze"), "C_SPLIT": str(tmp_path / "split"),
        "C_JUDGE_UTIL": str(tmp_path / "judge_util"),
        "C_RECONCILIATION": str(tmp_path / "reconciliation"),
        "C_CONFIG": str(tmp_path / "config"),
    }
    result = subprocess.run(["bash", "scripts/discourse_network/adapters.sbatch", "segment"],
                            cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shortcut_gate_is_unit_level_separates_probes_and_preserves_surface():
    statuses = {"dense": "pass", "cluster": "pass", "sae": "pass"}
    probes = {name: "pass" for name in ("distance", "adjacency", "lexical", "benchmark",
              "provider", "node_kind_pair", "edge_source", "transcript_length",
              "a1_marker_density")}
    probes["node_kind_pair"] = "surface"
    gate = {"purpose": "shortcut_recurrence_gate", "selection_changed": False,
            "probes": probes, "recurrence": {
                "cluster": {"0": {"items": 12, "provider_groups": 3, "scope": "global"},
                            "1": {"items": 2, "provider_groups": 1,
                                  "scope": "benchmark:gpqa"}},
                "sae": {"4": {"items": 12, "provider_groups": 3, "scope": "global"}}}}
    audit = apply_shortcut_recurrence_gate(
        statuses, gate, retained_units={"cluster": {0, 1}, "sae": {4}})
    assert audit["view_statuses"] == {"dense": "surface", "cluster": "surface", "sae": "surface"}
    assert audit["unit_statuses"] == {"cluster": {"0": "surface", "1": "failed"},
                                      "sae": {"4": "surface"}}
    assert audit["surface_labels"] == ["node_kind_pair"]


def test_edge_and_circuit_aggregates_are_zero_aware_and_estimands_are_labeled():
    edges = pd.DataFrame([
        {"transcript_key": "t", "edge_id": "a", "edge_source": "inferred",
         "source_node_id": 0, "target_node_id": 1, "dense_row": 0, "cluster_id": 0},
        {"transcript_key": "t", "edge_id": "b", "edge_source": "structural",
         "source_node_id": 1, "target_node_id": 2, "dense_row": 1, "cluster_id": 1},
    ])
    nodes = pd.DataFrame([{"transcript_key": "t", "node_id": i, "is_final_sink": i == 2}
                          for i in range(3)])
    edge_acts = pd.DataFrame([{"edge_id": "a", "feature_id": 4, "activation": 2.,
                               "sae_fit_id": "f", "dictionary_id": "d"}])
    edge_summary, edge_features = aggregate_edge_transcripts(
        edges, np.array([[1., 1.], [3., 3.]]), edge_acts, nodes, cluster_ids={1})
    assert set(edge_summary["edge_scope"]) == {"combined", "inferred", "structural"}
    assert "cluster_0_proportion" not in edge_summary
    assert edge_summary.loc[edge_summary["edge_scope"].eq("combined"),
                            "cluster_1_proportion"].iloc[0] == .5
    combined_feature = edge_features[(edge_features["edge_scope"] == "combined") &
                                     (edge_features["feature_id"] == 4)].iloc[0]
    assert combined_feature["mean"] == 1.0 and combined_feature["proportion_active"] == .5

    circuits = pd.DataFrame([
        {"transcript_key": "t", "circuit_id": "a", "extractor_kind": "path",
         "precap_count": 3, "true_kind_proportion": .75, "final_ancestry": True,
         "cluster_id": 0, "dense_row": 0},
        {"transcript_key": "t", "circuit_id": "b", "extractor_kind": "tool",
         "precap_count": 1, "true_kind_proportion": .25, "final_ancestry": False,
         "cluster_id": 1, "dense_row": 1},
    ])
    circuit_summary, circuit_features = aggregate_circuit_transcripts(
        circuits, np.array([[1., 1.], [3., 3.]]), edge_acts.rename(columns={"edge_id": "circuit_id"}))
    assert set(circuit_summary["summary_scope"]) == {"combined_weighted", "extractor_kind"}
    combined = circuit_summary[circuit_summary["summary_scope"] == "combined_weighted"].iloc[0]
    assert combined["estimand_basis"] == "exact_precap_kind_weighted_means"
    assert np.isnan(combined["dense_median_0"])
    assert np.isnan(combined["dense_ancestry_contrast_0"])
    combined_feature = circuit_features[circuit_features["summary_scope"] == "combined_weighted"].iloc[0]
    assert (combined_feature["mean"] == 1.5 and np.isnan(combined_feature["median"])
            and np.isnan(combined_feature["ancestry_contrast"]))


def _population_row(name, split, input_hash):
    transcript = "[Assistant]: Step. Finish."
    start = len("[Assistant]: ")
    content = transcript[start:]
    parts = [{"role": "assistant", "content": content, "start": start, "end": len(transcript),
              "bare": False, "subparts": [{"kind": "content", "content": content,
              "start": start, "end": len(transcript), "call_id": None, "function": None,
              "raw_payload": None}]}]
    return {"transcript_key": name, "item_key": f"item-{name}", "question": "Q",
            "transcript": transcript, "transcript_parts_json": json.dumps(parts), "split": split,
            "input_row_hash": hashlib.sha256(name.encode()).hexdigest(),
            "input_file_sha256": input_hash}


def _boundary_artifact(directory, population, config, config_path, bundle, approval_hash, purpose):
    directory.mkdir()
    spans = []
    for record in population.to_dict("records"):
        parts = json.loads(record["transcript_parts_json"])
        spans.append({"transcript_key": record["transcript_key"], "part_id": 0, "subpart_id": 0,
                      "start": parts[0]["subparts"][0]["start"], "text": "Step. Finish.",
                      "source_text": "Step. Finish.",
                      "source_text_hash": hashlib.sha256(b"Step. Finish.").hexdigest(),
                      "input_file_sha256": record["input_file_sha256"],
                      "input_row_hash": record["input_row_hash"]})

    class Segmenter:
        def segment(self, _):
            return [(0, 5), (6, 13)]

    rows = FrozenSegmenterAdapter(directory, lambda _: Segmenter()).segment_rows(
        spans, revision="r", config_hash=sha256_file(config_path))
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / "boundaries.parquet", index=False)
    gate = {"overall_f1": .91, "subset_min_f1": .86, "selected_system": "isanlp_rst"}
    manifest = {"file_sha256": sha256_file(directory / "boundaries.parquet"),
                "selected_system": "isanlp_rst", "revision": "r",
                "config_hash": sha256_file(config_path), "blind_gate_selected": True,
                "canonical_content_hash": bundle.manifest["canonical_content_hash"],
                "freeze_manifest_sha256": bundle.manifest_sha256,
                "approval_sha256": approval_hash, "assets_manifest_sha256": "2" * 64,
                "input_sha256": bundle.manifest["files"]["transcript_store"]["sha256"],
                "input_file_sha256": bundle.manifest["files"]["transcript_store"]["sha256"],
                "input_row_hashes_sha256": stable_id(sorted(population["input_row_hash"])),
                "adapter_code_sha256": sha256_file(Path(__file__).with_name("adapters.py")),
                "gate_metrics": gate, "mode": "canonical", "purpose": purpose,
                "gate_metrics_sha256": "3" * 64, "gate_metrics_content_hash": stable_id(gate),
                "rows": len(frame), "schema_hash": stable_id([
                    {"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns])}
    (directory / "boundary_manifest.json").write_text(json.dumps(manifest))


def _embedding_artifact(directory, nodes_artifact, config, config_path, bundle, approval_hash, purpose):
    nodes = pd.read_parquet(nodes_artifact.path.parent / "nodes.parquet")
    vectors, metadata = FrozenEmbeddingAdapter(directory, lambda _: _FakeEncoder()).encode(
        nodes.to_dict("records"), revision=config["relations"]["encoder_revision"])
    np.save(directory / "node_embeddings.npy", vectors)
    metadata.to_parquet(directory / "node_embedding_metadata.parquet", index=False)
    manifest = {"mode": "canonical", "purpose": purpose,
                "vectors_sha256": sha256_file(directory / "node_embeddings.npy"),
                "metadata_sha256": sha256_file(directory / "node_embedding_metadata.parquet"),
                "revision": config["relations"]["encoder_revision"],
                "config_hash": sha256_file(config_path), "approval_sha256": approval_hash,
                "adapter_code_sha256": sha256_file(Path(__file__).with_name("adapters.py")),
                "assets_manifest_sha256": "4" * 64,
                "nodes_manifest_sha256": nodes_artifact.sha256,
                "nodes_file_sha256": nodes_artifact.manifest["files"]["nodes.parquet"]["sha256"],
                "canonical_content_hash": bundle.manifest["canonical_content_hash"],
                "freeze_manifest_sha256": bundle.manifest_sha256, "rows": len(nodes),
                "coverage": 1.0, "vector_schema_hash": stable_id(
                    {"dtype": str(vectors.dtype), "shape_tail": list(vectors.shape[1:])}),
                "metadata_schema_hash": stable_id([
                    {"name": column, "dtype": str(metadata[column].dtype)}
                    for column in metadata.columns])}
    (directory / "embedding_manifest.json").write_text(json.dumps(manifest))


def _artifact(path, stage, bundle, config_path, upstream=None):
    manifest = path / "artifact_manifest.json"
    if upstream is None:
        return load_c_manifest(manifest, expected_stage=stage,
            freeze_manifest_sha256=bundle.manifest_sha256, config_hash=sha256_file(config_path),
            code_hash=c_code_hash(), upstream_c_manifest_sha256=None)
    return load_c_manifest_chain(manifest, expected_sha256=sha256_file(manifest),
        expected_stage=stage, freeze_manifest_sha256=bundle.manifest_sha256,
        config_hash=sha256_file(config_path), code_hash=c_code_hash())


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_fake_assets_execute_full_chain_and_confirmation_reuses_development_gates(tmp_path,
                                                                                  monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    monkeypatch.setattr("discourse_network.stages.fit_independent_views", _positive_views)
    monkeypatch.setattr("discourse_network.stages.support_floor", lambda *_: 1)
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/workstream_c.yaml").read_text())
    config["segmentation"]["primary"]["checkpoint_revision"] = "r"
    config["relations"] |= {"pca_components": 1, "minimum_positive_edges": 1}
    for section in ("edge_views", "circuits"):
        config[section] |= {"cluster_counts": [2], "sae_width": 4, "sae_top_k": 2,
                            "sae_epochs": 1, "sae_batch_size": 8, "sae_device": "cpu",
                            "sae_dtype": "float32"}
    config["circuits"]["dense_pca_components"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    input_hash = "1" * 64
    population = pd.DataFrame([_population_row(f"t{index}", split, input_hash)
                               for index, split in enumerate([
                                   "representation_train", "representation_train",
                                   "representation_train", "representation_validation",
                                   "confirmation"])])
    manifest = {"canonical_content_hash": "a" * 64,
                "files": {"transcript_store": {"sha256": input_hash},
                          "item_split": {"sha256": "b" * 64},
                          "verdict_reconciliation": {"sha256": "c" * 64}}}
    dev_bundle = FrozenBundle(manifest, "d" * 64,
                              population[~population["split"].eq("confirmation")].copy())
    confirmation_bundle = FrozenBundle(manifest, "d" * 64,
                                       population[population["split"].eq("confirmation")].copy())
    approval_hash = "e" * 64
    metadata = {"code_commit": "test", "package_lock_hash": "f" * 64,
                "judge_util_sha256": "9" * 64, "seed": 0, "model_revisions": {},
                "approval_sha256": approval_hash}

    dev_adapter = tmp_path / "dev-adapter"
    _boundary_artifact(dev_adapter, dev_bundle.population, config, config_path, dev_bundle,
                       approval_hash, "representation_discovery")
    dev_nodes_dir = tmp_path / "dev-nodes"
    nodes_stage(StageContext(dev_bundle, config, config_path, dev_adapter, dev_nodes_dir,
                             metadata, stage="nodes"))
    dev_nodes = _artifact(dev_nodes_dir, "nodes", dev_bundle, config_path)
    _embedding_artifact(dev_adapter, dev_nodes, config, config_path, dev_bundle,
                        approval_hash, "representation_discovery")
    node_rows = pd.read_parquet(dev_nodes_dir / "nodes.parquet")
    labels = []
    for transcript_key in ("t0", "t1", "t2"):
        purpose = "train" if transcript_key in {"t0", "t1"} else "calibration"
        for source, label in ((0, 0), (1, 1)):
            labels.append({"transcript_key": transcript_key, "source_node_id": source,
                           "target_node_id": 2, "label": label, "purpose": purpose,
                           "item_group": f"item-{transcript_key}", "subset": "gpqa"})
    pd.DataFrame(labels).to_parquet(dev_adapter / "attachment_labels.parquet", index=False)
    monkeypatch.setattr("discourse_network.stages.fit_attachment_head", lambda *_a, **_k: _FakeHead())
    monkeypatch.setattr("discourse_network.stages.fit_platt_scaler", lambda *_a, **_k: _FakePlatt())

    dev_attachments_dir = tmp_path / "dev-attachments"
    attachments_stage(StageContext(dev_bundle, config, config_path, dev_nodes_dir,
        dev_attachments_dir, metadata, stage="attachments", upstream=dev_nodes,
        adapter_dir=dev_adapter))
    dev_attachments = _artifact(dev_attachments_dir, "attachments", dev_bundle, config_path,
                                dev_nodes)

    dev_edge_dir = tmp_path / "dev-edge-views"
    edge_context = StageContext(dev_bundle, config, config_path, dev_attachments_dir,
        dev_edge_dir, metadata, stage="edge-views", upstream=dev_attachments,
        adapter_dir=dev_adapter)
    with pytest.raises(FileNotFoundError):
        edge_views_stage(edge_context)  # topology/attachment gate is load-bearing and precedes semantics
    robustness = json.loads((dev_attachments_dir / "topology_robustness.json").read_text())
    gate = {"purpose": "attachment_gate", "freeze_manifest_sha256": dev_bundle.manifest_sha256,
            "config_hash": sha256_file(config_path),
            "attachment_manifest_sha256": dev_attachments.sha256,
            "baselines": ["recency", "tfidf", "shared_symbol", "shared_entity", "no_text"],
            "bootstrap_unit": "opaque_item", "candidate_recall": .96,
            "bootstrap_improvement_ci_low": .01, "precision": .7, "recall": .55,
            "subset_metrics": {"gpqa": {"precision": .6, "recall": .45}},
            "over_parent_cap": 0., "synthetic_macro_f1": .81,
            "synthetic_family_min_f1": .71, "paraphrase_f1": .76,
            "calibration_unit_ids": ["c"], "gate_unit_ids": ["g"],
            "input_mask": dev_attachments.manifest["input_mask"],
            "dense_retrieval": {"purpose": "dense_retrieval_gate",
                                "split": "representation_validation", "bootstrap_ci_low": .01}} | robustness
    (dev_adapter / "attachment_gate.json").write_text(json.dumps(gate))
    edge_views_stage(edge_context)
    dev_edges = _artifact(dev_edge_dir, "edge-views", dev_bundle, config_path, dev_attachments)
    assert dev_edges.manifest["view_statuses"] == {"dense": "pass", "cluster": "failed", "sae": "pass"}
    assert dev_edges.manifest["retained_edge_sae_feature_ids"] == [0, 1]
    edge_models = joblib.load(dev_edge_dir / "edge_view_models.joblib")
    assert edge_models["retained_features"] == [0, 1]
    assert (edge_models["retained_features_sha256"] ==
            dev_edges.manifest["retained_edge_sae_feature_ids_sha256"])
    assert edge_models["retained_features_sha256"] == stable_id({
        "sae_fit_id": edge_models["sae"].fit_id,
        "dictionary_id": edge_models["sae"].dictionary_id, "feature_ids": [0, 1]})
    assert dev_edges.manifest["edge_summary_status"] == "pre_audit_non_final"
    assert set(pd.read_parquet(dev_edge_dir / "edge_feature_aggregates.parquet")["feature_id"]) == {0, 1}
    assert {"edge_transcript_aggregates.parquet", "edge_feature_aggregates.parquet"} <= set(
        dev_edges.manifest["files"])

    edge_frame = pd.read_parquet(dev_edge_dir / "edges.parquet")
    shortcut = {"purpose": "shortcut_recurrence_gate", "selection_changed": False,
                "freeze_manifest_sha256": dev_bundle.manifest_sha256,
                "config_hash": sha256_file(config_path), "edge_manifest_sha256": dev_edges.sha256,
                "edge_order_hash": stable_id(edge_frame["edge_id"].astype(str).tolist()),
                "probes": {name: "pass" for name in ("distance", "adjacency", "lexical",
                    "benchmark", "provider", "node_kind_pair", "edge_source",
                    "transcript_length", "a1_marker_density")}, "recurrence": {
                    "sae": {"0": {"items": 12, "provider_groups": 3, "scope": "global"},
                            "1": {"items": 2, "provider_groups": 1,
                                  "scope": "benchmark:gpqa"}}}}
    shortcut["probes"]["node_kind_pair"] = "surface"
    (dev_adapter / "shortcut_recurrence_gate.json").write_text(json.dumps(shortcut))
    effective_mask = dev_edges.manifest["input_mask"]
    encoder_gate = {"purpose": "circuit_encoder_gate", "split": "representation_validation",
                    "freeze_manifest_sha256": dev_bundle.manifest_sha256,
                    "config_hash": sha256_file(config_path), "edge_manifest_sha256": dev_edges.sha256,
                    "input_mask": effective_mask, "synthetic_macro_f1": .81,
                    "retrieval_ci_low": .01, "cv": "leave_base_trace_group_out",
                    "explicit_motifs_removed": True,
                    "baselines": ["length", "recency", "benchmark"],
                    "baseline_improvement_ci_low": .01}
    (dev_adapter / "circuit_encoder_gate.json").write_text(json.dumps(encoder_gate))
    dev_circuits_dir = tmp_path / "dev-circuits"
    circuits_stage(StageContext(dev_bundle, config, config_path, dev_edge_dir,
        dev_circuits_dir, metadata, stage="circuits", upstream=dev_edges,
        adapter_dir=dev_adapter))
    dev_circuits = _artifact(dev_circuits_dir, "circuits", dev_bundle, config_path, dev_edges)
    assert dev_circuits.manifest["upstream_c_manifest_sha256"] == dev_edges.sha256
    assert dev_circuits.manifest["surface_labels"] == ["node_kind_pair"]
    assert dev_circuits.manifest["audited_edge_view_statuses"]["dense"] == "surface"
    assert dev_circuits.manifest["audited_edge_view_statuses"]["sae"] == "surface"
    assert dev_circuits.manifest["edge_unit_statuses"]["sae"] == {
        "0": "surface", "1": "failed"}
    assert dev_circuits.manifest["retained_circuit_sae_feature_ids"] == [0, 1]
    circuit_models = joblib.load(dev_circuits_dir / "circuit_view_models.joblib")
    assert circuit_models["retained_features"] == [0, 1]
    assert (circuit_models["retained_features_sha256"] ==
            dev_circuits.manifest["retained_circuit_sae_feature_ids_sha256"])
    assert circuit_models["retained_features_sha256"] == stable_id({
        "sae_fit_id": circuit_models["sae"].fit_id,
        "dictionary_id": circuit_models["sae"].dictionary_id, "feature_ids": [0, 1]})
    final_edge = dev_circuits.manifest["final_audited_edge_summaries"]
    assert final_edge["source_edge_manifest_sha256"] == dev_edges.sha256
    assert final_edge["shortcut_recurrence_gate_sha256"] == sha256_file(
        dev_adapter / "shortcut_recurrence_gate.json")
    assert final_edge["feature_sha256"] == dev_circuits.manifest["files"][
        "final_edge_feature_aggregates.parquet"]["sha256"]
    final_features = pd.read_parquet(dev_circuits_dir / "final_edge_feature_aggregates.parquet")
    assert set(final_features["feature_id"]) == {0}
    assert set(final_features["edge_summary_status"]) == {"final_audited"}
    assert set(final_features["surface_labels"]) == {'["node_kind_pair"]'}
    frozen_status = json.loads((dev_circuits_dir / "edge_shortcut_recurrence_status.json").read_text())
    assert frozen_status["surface_labels"] == ["node_kind_pair"]
    assert (dev_circuits_dir / "shortcut_recurrence_gate.json").read_bytes() == (
        dev_adapter / "shortcut_recurrence_gate.json").read_bytes()

    confirmation_adapter = tmp_path / "confirmation-adapter"
    _boundary_artifact(confirmation_adapter, confirmation_bundle.population, config, config_path,
                       confirmation_bundle, approval_hash, "confirmation_apply")
    confirmation_nodes_dir = tmp_path / "confirmation-nodes"
    nodes_stage(StageContext(confirmation_bundle, config, config_path, confirmation_adapter,
        confirmation_nodes_dir, metadata, confirmation_apply=True, stage="nodes"))
    confirmation_nodes = _artifact(confirmation_nodes_dir, "nodes", confirmation_bundle, config_path)
    _embedding_artifact(confirmation_adapter, confirmation_nodes, config, config_path,
                        confirmation_bundle, approval_hash, "confirmation_apply")
    confirmation_attachments_dir = tmp_path / "confirmation-attachments"
    attachments_stage(StageContext(confirmation_bundle, config, config_path, confirmation_nodes_dir,
        confirmation_attachments_dir, metadata, confirmation_apply=True,
        apply_from=dev_attachments_dir, stage="attachments", upstream=confirmation_nodes,
        adapter_dir=confirmation_adapter))
    confirmation_attachments = _artifact(confirmation_attachments_dir, "attachments",
                                         confirmation_bundle, config_path, confirmation_nodes)
    confirmation_edge_dir = tmp_path / "confirmation-edge"
    edge_views_stage(StageContext(confirmation_bundle, config, config_path,
        confirmation_attachments_dir, confirmation_edge_dir, metadata, confirmation_apply=True,
        apply_from=dev_edge_dir, stage="edge-views", upstream=confirmation_attachments,
        adapter_dir=confirmation_adapter))
    confirmation_edges = _artifact(confirmation_edge_dir, "edge-views", confirmation_bundle,
                                   config_path, confirmation_attachments)
    assert confirmation_edges.manifest["retained_edge_sae_feature_ids"] == [0, 1]
    assert pd.read_parquet(confirmation_edge_dir / "edge_sae_activations.parquet").empty
    confirmation_edge_features = pd.read_parquet(
        confirmation_edge_dir / "edge_feature_aggregates.parquet")
    assert set(confirmation_edge_features["feature_id"]) == {0, 1}
    assert (confirmation_edge_features[["mean", "proportion_active"]] == 0).all().all()

    rogue = tmp_path / "rogue-gates"
    rogue.mkdir()
    (rogue / "shortcut_recurrence_gate.json").write_text(json.dumps(shortcut | {"selection_changed": True}))
    with pytest.raises(ValueError, match="newly supplied"):
        circuits_stage(StageContext(confirmation_bundle, config, config_path, confirmation_edge_dir,
            tmp_path / "rejected-confirmation-circuits", metadata, confirmation_apply=True,
            apply_from=dev_circuits_dir, stage="circuits", upstream=confirmation_edges,
            adapter_dir=rogue))
    confirmation_circuits_dir = tmp_path / "confirmation-circuits"
    circuits_stage(StageContext(confirmation_bundle, config, config_path, confirmation_edge_dir,
        confirmation_circuits_dir, metadata, confirmation_apply=True,
        apply_from=dev_circuits_dir, stage="circuits", upstream=confirmation_edges,
        adapter_dir=confirmation_adapter))
    confirmation_circuits = _artifact(confirmation_circuits_dir, "circuits",
                                      confirmation_bundle, config_path, confirmation_edges)
    assert confirmation_circuits.manifest["purpose"] == "confirmation_apply"
    assert confirmation_circuits.manifest["circuit_encoder_gate_sha256"] is None
    assert confirmation_circuits.manifest["retained_circuit_sae_feature_ids"] == [0, 1]
    assert pd.read_parquet(
        confirmation_circuits_dir / "circuit_sae_activations.parquet").empty
    confirmation_circuit_features = pd.read_parquet(
        confirmation_circuits_dir / "circuit_feature_aggregates.parquet")
    assert set(confirmation_circuit_features["feature_id"]) == {0, 1}
    assert (confirmation_circuit_features[["mean", "proportion_active"]] == 0).all().all()
    confirmation_final_features = pd.read_parquet(
        confirmation_circuits_dir / "final_edge_feature_aggregates.parquet")
    assert set(confirmation_final_features["feature_id"]) == {0}
    assert (confirmation_final_features[["mean", "proportion_active"]] == 0).all().all()
    assert confirmation_circuits.manifest["final_audited_edge_summaries"][
        "source_edge_manifest_sha256"] == confirmation_edges.sha256
    assert (confirmation_circuits_dir / "shortcut_recurrence_gate.json").read_bytes() == (
        dev_circuits_dir / "shortcut_recurrence_gate.json").read_bytes()
