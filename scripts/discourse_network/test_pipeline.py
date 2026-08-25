import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.attachments import (
    attachment_features, build_candidates, fit_attachment_head, fit_platt_scaler,
    fit_relation_geometry, link_tool_edges, select_parents, select_threshold,
)
from discourse_network.circuits import extract_circuits, topology_features
from discourse_network.edge_views import (
    aggregate_activations, aggregate_dense, retained_across_splits, retained_support,
    support_floor, topk_nonnegative, weighted_kind_summary,
)
from discourse_network.nodes import (
    build_nodes, guard_canonical_stage, load_frozen_population, sha256_file, stable_id,
    validate_artifact_rows, write_artifact_manifest,
)


ROW_KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]


def _part(role, subparts, cursor):
    prefix = f"[{role.capitalize()}]: "
    content, offset = "\n".join(part["content"] for part in subparts), cursor + len(prefix)
    cooked = []
    for part in subparts:
        cooked.append(part | {"start": offset, "end": offset + len(part["content"])})
        offset += len(part["content"]) + 1
    return ({"role": role, "content": content, "start": cursor + len(prefix),
             "end": cursor + len(prefix) + len(content), "bare": False,
             "subparts": cooked}, prefix + content)


def structured_transcript():
    call = {"kind": "tool_call", "content": "[Tool Call: py({'x': 1})]",
            "call_id": "c1", "function": "py", "raw_payload": {"x": 1}}
    text = {"kind": "content",
            "content": "Plan. ```py\nx = 1\n``` $$x=1$$ Then continue. [User]: not a role",
            "call_id": None, "function": None, "raw_payload": None}
    first, rendered_first = _part("assistant", [text, call], 0)
    output = {"kind": "tool_output", "content": "[Tool Output (py)]: 1",
              "call_id": "c1", "function": "py", "raw_payload": "1"}
    second_start = len(rendered_first) + 2
    second, rendered_second = _part("tool", [output], second_start)
    feedback = {"kind": "content", "content": "Check the sign.",
                "call_id": None, "function": None, "raw_payload": None}
    third_start = second_start + len(rendered_second) + 2
    third, rendered_third = _part("user", [feedback], third_start)
    final = {"kind": "content", "content": "Revised. Therefore 1.",
             "call_id": None, "function": None, "raw_payload": None}
    fourth_start = third_start + len(rendered_third) + 2
    fourth, rendered_fourth = _part("assistant", [final], fourth_start)
    return "\n\n".join((rendered_first, rendered_second, rendered_third, rendered_fourth)), [
        first, second, third, fourth,
    ]


def frozen_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    transcript, parts = structured_transcript()
    rows = pd.DataFrame([
        ["gpqa", "secret-model-a", "1", 0, "run", 0, "a" * 64,
         "Question?", transcript, json.dumps(parts), True, True],
        ["gpqa", "secret-model-b", "2", 0, "run", 0, "b" * 64,
         "Other?", "[Assistant]: no", json.dumps([]), False, True],
    ], columns=ROW_KEYS + ["question", "transcript", "transcript_parts_json",
                           "correct", "is_canonical"])
    live_judge_util = Path(__file__).resolve().parents[2] / "src" / "judge" / "util.py"
    reconciliation_data = {"max_length_unresolved_logs": 0,
                           "unmatched_transcript_logs": 0, "input_mismatch_logs": 0}
    schema = [{"name": column, "dtype": str(rows[column].dtype)} for column in rows.columns]
    schema_hash = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
    population = rows[rows.is_canonical & rows.correct][ROW_KEYS].sort_values(ROW_KEYS)
    content_hash = hashlib.sha256(population.to_json(orient="records").encode()).hexdigest()
    snapshot_text = {"config": "frozen: true\n", "main_code": "# frozen A code\n",
                     "surprisal_code": "# frozen surprisal code\n",
                     "judge_util_code": live_judge_util.read_text()}
    contract_digest = hashlib.sha256()
    for name, text in sorted(snapshot_text.items()):
        contract_digest.update(name.encode() + b"\0" + text.encode())
    contract_hash = contract_digest.hexdigest()
    freeze_id = f"v1-{content_hash[:12]}-{contract_hash[:12]}"
    freeze_dir = tmp_path / "freezes" / freeze_id
    freeze_dir.mkdir(parents=True)
    code_dir = freeze_dir / "code"
    code_dir.mkdir()
    judge_util = code_dir / "judge_util.py"
    judge_util.write_text(snapshot_text["judge_util_code"])
    config_snapshot, main_snapshot = code_dir / "verdict_dissection.yaml", code_dir / "main.py"
    surprisal_snapshot = code_dir / "llm_surprisal.py"
    config_snapshot.write_text(snapshot_text["config"])
    main_snapshot.write_text(snapshot_text["main_code"])
    surprisal_snapshot.write_text(snapshot_text["surprisal_code"])
    store, split = freeze_dir / "store.parquet", freeze_dir / "split.csv"
    rows.to_parquet(store, index=False)
    pd.DataFrame({"bench": ["gpqa", "gpqa"], "item": ["1", "2"],
                  "split": ["representation_train", "confirmation"]}).to_csv(split, index=False)
    reconciliation = freeze_dir / "reconciliation.json"
    reconciliation.write_text(json.dumps(reconciliation_data))
    verdicts = freeze_dir / "verdicts.parquet"
    pd.DataFrame({
        "bench": ["gpqa", "aime2024", "aime2025", "aime2026", "intercode", "swe_bench_mini"],
        "draw_id": list(range(6)), "max_length_resolution": ["recorded"] * 6,
    }).to_parquet(verdicts, index=False)
    manifest = freeze_dir / "freeze_manifest.json"
    manifest.write_text(json.dumps({
        "manifest_version": 1,
        "freeze_contract_hash": contract_hash,
        "scope": {"benchmarks": ["gpqa", "aime2024", "aime2025", "aime2026",
                                  "intercode", "swe_bench_mini"]},
        "row_keys": {"transcript_store": ROW_KEYS,
                     "item_split": ["bench", "item", "split"],
                     "verdicts": ["bench", "draw_id"], "pairs": ["opaque-pair-key"]},
        "correctness_rule": "judge.util.extract_score",
        "population_rule": "is_canonical & correct",
        "transcript_schema_hash": schema_hash,
        "canonical_content_hash": content_hash,
        "judge_config_hashes": ["e" * 64],
        "split": {"seed": 0, "representation_train": 0.64,
                  "representation_validation": 0.16, "confirmation": 0.20},
        "transcript_store": {"path": str(store), "sha256": sha256_file(store)},
        "item_split": {"path": str(split), "sha256": sha256_file(split)},
        "files": {
            "transcript_store": {"path": str(store), "sha256": sha256_file(store)},
            "item_split": {"path": str(split), "sha256": sha256_file(split)},
            "judge_util_code": {"path": str(judge_util), "sha256": sha256_file(judge_util)},
            "verdict_reconciliation": {"path": str(reconciliation),
                                       "sha256": sha256_file(reconciliation)},
            "verdicts": {"path": str(verdicts), "sha256": sha256_file(verdicts)},
            "config": {"path": str(config_snapshot), "sha256": sha256_file(config_snapshot)},
            "main_code": {"path": str(main_snapshot), "sha256": sha256_file(main_snapshot)},
            "surprisal_code": {"path": str(surprisal_snapshot),
                               "sha256": sha256_file(surprisal_snapshot)},
        },
    }))
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "design_approved": True,
        "annotation_budget_approved": True,
        "annotation_budget_hours": 350,
        "named_annotators": ["annotator-1", "annotator-2"],
    }))
    return store, split, manifest, approval


def test_loader_verifies_nested_freeze_and_projects_opaque_identity(tmp_path):
    store, split, manifest, _ = frozen_inputs(tmp_path)
    rows = load_frozen_population(store, split, manifest, {"representation_train"})
    assert rows.columns.tolist() == [
        "transcript_key", "item_key", "question", "transcript", "transcript_parts_json", "split",
        "input_row_hash", "input_file_sha256",
    ]
    assert rows["transcript_key"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not ({"bench", "model", "item", "correct", "is_canonical"} & set(rows))

    data = json.loads(manifest.read_text())
    data["files"]["item_split"]["sha256"] = "0" * 64
    data["item_split"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="item_split SHA-256"):
        load_frozen_population(store, split, manifest, {"representation_train"})


def test_loader_refuses_confirmation_discovery_and_missing_population_contract(tmp_path):
    store, split, manifest, _ = frozen_inputs(tmp_path)
    with pytest.raises(ValueError, match="apply-only"):
        load_frozen_population(store, split, manifest, {"confirmation"})
    data = json.loads(manifest.read_text())
    del data["canonical_content_hash"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="canonical_content_hash"):
        load_frozen_population(store, split, manifest, {"representation_train"})


def test_loader_recomputes_schema_and_population_hashes(tmp_path):
    store, split, manifest, _ = frozen_inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["transcript_schema_hash"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="transcript schema hash"):
        load_frozen_population(store, split, manifest, {"representation_train"})


def test_loader_requires_seed_zero_split_contract(tmp_path):
    store, split, manifest, _ = frozen_inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["split"]["seed"] = 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed must be zero"):
        load_frozen_population(store, split, manifest, {"representation_train"})


def test_loader_rejects_invalid_freeze_and_nonzero_reconciliation(tmp_path):
    store, split, manifest, _ = frozen_inputs(tmp_path)
    invalid_manifest = tmp_path / "mutable-latest.json"
    invalid_manifest.write_text(manifest.read_text())
    with pytest.raises(ValueError, match="immutable versioned"):
        load_frozen_population(store, split, invalid_manifest, {"representation_train"})

    store, split, manifest, _ = frozen_inputs(tmp_path / "second")
    data = json.loads(manifest.read_text())
    reconciliation = Path(data["files"]["verdict_reconciliation"]["path"])
    reconciliation.write_text(json.dumps({"max_length_unresolved_logs": 5,
                                          "unmatched_transcript_logs": 0,
                                          "input_mismatch_logs": 0}))
    digest = sha256_file(reconciliation)
    data["files"]["verdict_reconciliation"]["sha256"] = digest
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="max_length_unresolved_logs"):
        load_frozen_population(store, split, manifest, {"representation_train"},
                               purpose="verdict_join")


def test_freeze_allows_four_excluded_gpqa_length_logs_but_not_unmatched(tmp_path):
    store, split, manifest, _ = frozen_inputs(tmp_path)
    data = json.loads(manifest.read_text())
    reconciliation = Path(data["files"]["verdict_reconciliation"]["path"])
    counters = {"max_length_unresolved_logs": 4, "max_length_unresolved_rows": 973,
                "unmatched_transcript_logs": 0, "input_mismatch_logs": 0}
    reconciliation.write_text(json.dumps(counters))
    data["files"]["verdict_reconciliation"]["sha256"] = sha256_file(reconciliation)
    manifest.write_text(json.dumps(data))
    c_config = tmp_path / "c.yaml"
    c_config.write_text(json.dumps({"approved_exclusions": [{
        "canonical_content_hash": data["canonical_content_hash"],
        "freeze_contract_hash": data["freeze_contract_hash"], "benchmark": "gpqa",
        "reason": "max_length_unresolved", "excluded_logs": 4, "excluded_rows": 973}]}))
    load_frozen_population(store, split, manifest, {"representation_train"}, c_config_path=c_config,
                           purpose="verdict_join")
    counters["unmatched_transcript_logs"] = 1
    reconciliation.write_text(json.dumps(counters))
    data["files"]["verdict_reconciliation"]["sha256"] = sha256_file(reconciliation)
    manifest.write_text(json.dumps(data))
    load_frozen_population(store, split, manifest, {"representation_train"}, c_config_path=c_config)
    with pytest.raises(ValueError, match="unmatched_transcript_logs"):
        load_frozen_population(store, split, manifest, {"representation_train"},
                               c_config_path=c_config, purpose="verdict_join")


def test_canonical_stage_guard_requires_freeze_annotators_and_budget(tmp_path):
    _, _, manifest, approval = frozen_inputs(tmp_path)
    guard_canonical_stage(manifest, approval, "nodes")
    data = json.loads(approval.read_text())
    data["named_annotators"] = ["only-one"]
    approval.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="two named annotators"):
        guard_canonical_stage(manifest, approval, "nodes")


def test_nodes_use_only_structured_roles_and_preserve_protected_spans():
    transcript, parts = structured_transcript()
    nodes = build_nodes("opaque", "Question?", transcript, parts)
    kinds = [node["kind"] for node in nodes]
    assert kinds[0] == "question"
    assert {"code", "equation", "tool_call", "tool_output", "user_feedback"} <= set(kinds)
    assert len([node for node in nodes if node["kind"] == "user_feedback"]) == 1
    assert any("[User]: not a role" in node["rendered_text"] for node in nodes)
    assert next(node for node in nodes if node["kind"] == "code")["rendered_text"] == "```py\nx = 1\n```"
    assert next(node for node in nodes if node["kind"] == "equation")["rendered_text"] == "$$x=1$$"
    assert sum(node["is_final_sink"] for node in nodes) == 1
    assert nodes[-1]["is_final_sink"]
    assert [node["node_id"] for node in nodes] == list(range(len(nodes)))


def graph_nodes():
    return [
        {"transcript_key": "t", "node_id": 0, "kind": "question", "rendered_text": "find x", "target_eligible": False},
        {"transcript_key": "t", "node_id": 1, "kind": "assistant_edu", "rendered_text": "let x=1", "target_eligible": True},
        {"transcript_key": "t", "node_id": 2, "kind": "tool_call", "rendered_text": "calculate x", "call_id": "c", "target_eligible": True},
        {"transcript_key": "t", "node_id": 3, "kind": "tool_output", "rendered_text": "x=1", "call_id": "c", "target_eligible": False},
        {"transcript_key": "t", "node_id": 4, "kind": "assistant_edu", "rendered_text": "thus x is 1", "target_eligible": True},
        {"transcript_key": "t", "node_id": 5, "kind": "assistant_edu", "rendered_text": "verify x=1",
         "is_final_sink": True, "target_eligible": True},
    ]


def test_structural_tool_link_and_candidates_are_chronological_capped_and_audited():
    nodes = graph_nodes()
    edges = link_tool_edges(nodes, "f" * 64)
    assert [(edge["source_node_id"], edge["target_node_id"], edge["edge_source"])
            for edge in edges] == [(2, 3, "structural")]
    embeddings = np.array([[1, 0], [0.8, 0.2], [0, 1], [0.1, 0.9], [0.7, 0.3], [1, 0]])
    candidates = build_candidates(nodes, embeddings, target_node_id=5, cap=4,
                                  preceding=2, semantic=2, identifiers=2, structural=2)
    assert len(candidates) == 4
    assert all(row["source_node_id"] < row["target_node_id"] for row in candidates)
    assert any("question" in row["reasons"] for row in candidates)
    assert all(row["reasons"] == sorted(set(row["reasons"])) for row in candidates)


def test_relation_geometry_is_directed_outcome_blind_and_attachment_is_regularized():
    source = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 2.], [2., 2.]])
    target = np.array([[0., 1.], [1., 0.], [2., 1.], [1., 1.], [2., 2.], [1., 2.]])
    geometry = fit_relation_geometry(source, target, n_components=3, seed=0)
    forward = geometry.transform(source, target)
    reverse = geometry.transform(target, source)
    assert forward.shape == (6, 3)
    assert not np.allclose(forward, reverse)

    x = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = fit_attachment_head(x, y, groups=np.arange(6), cs=(0.01, 0.1, 1.0),
                                min_positive=1)
    assert model.C in (0.01, 0.1, 1.0)
    assert model.predict_proba(np.array([[-4.], [4.]]))[:, 1].tolist()[0] < 0.5
    calibrated = fit_platt_scaler(np.array([-3., -1., 1., 3.]), np.array([0, 0, 1, 1]))
    assert np.all(np.diff(calibrated.transform(np.array([-2., 0., 2.]))) > 0)


def test_attachment_feature_modes_keep_text_controls_out_of_full_scorer():
    candidates = [
        {"source_node_id": 0, "target_node_id": 2, "node_distance": 2, "adjacent": False,
         "lexical_overlap": 0.9, "symbol_overlap": 2, "entity_overlap": 1},
        {"source_node_id": 1, "target_node_id": 2, "node_distance": 1, "adjacent": True,
         "lexical_overlap": 0.1, "symbol_overlap": 0, "entity_overlap": 0},
    ]
    nodes = [{"kind": "question"}, {"kind": "assistant_edu"}, {"kind": "assistant_edu"}]
    dense = np.array([[10., 11.], [12., 13.]])
    full = attachment_features(dense, candidates, nodes, mode="full")
    dense_only = attachment_features(dense, candidates, nodes, mode="dense")
    no_text = attachment_features(dense, candidates, nodes, mode="no_text")
    assert np.array_equal(dense_only, dense)
    assert np.array_equal(full[:, :2], dense)
    assert no_text.shape[1] + 2 == full.shape[1] + 3
    assert 0.9 in no_text[0] and 0.9 not in full[0, 2:]


def test_threshold_and_parent_selection_apply_floors_ties_and_cap():
    threshold = select_threshold(
        np.array([0.95, 0.8, 0.7, 0.2]), np.array([1, 1, 0, 0]),
        precision_floor=0.65, recall_floor=0.5,
    )
    assert threshold == 0.8
    parents = select_parents([
        {"source_node_id": 1, "probability": 0.91},
        {"source_node_id": 2, "probability": 0.91},
        {"source_node_id": 3, "probability": 0.89},
        {"source_node_id": 4, "probability": 0.88},
    ], threshold=0.8, max_parents=3)
    assert [row["source_node_id"] for row in parents] == [1, 2, 3]


def test_topology_and_bounded_circuits_are_stable():
    nodes = graph_nodes()
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4), (4, 5)]
    edges = [{"edge_id": stable_id("e", s, t), "source_node_id": s,
              "target_node_id": t} for s, t in pairs]
    stats = topology_features(nodes, edges)
    assert stats["longest_path_edges"] == 5
    assert stats["branch_nodes"] == 1
    assert stats["convergence_nodes"] == 1
    assert [stats[f"path_{length}_count"] for length in (2, 3, 4)] == [6, 4, 2]
    assert stats["tool_chain_motifs"] == 1
    assert stats["final_ancestor_fraction"] == 1.0
    first = extract_circuits(nodes, edges, config_hash="f" * 64, max_local_nodes=4,
                             max_final_nodes=5, max_learned=8)
    second = extract_circuits(nodes, edges, config_hash="f" * 64, max_local_nodes=4,
                              max_final_nodes=5, max_learned=8)
    assert [row["circuit_id"] for row in first] == [row["circuit_id"] for row in second]
    assert all(row["node_count"] <= (5 if row["extractor_kind"] == "final_ancestry" else 4)
               for row in first)
    assert {"path", "convergence", "tool", "final_ancestry"} <= {
        row["extractor_kind"] for row in first
    }


def test_tool_circuit_uses_graph_paths_not_nearest_chronological_assistants():
    nodes = [
        {"transcript_key": "t", "node_id": 0, "kind": "question"},
        {"transcript_key": "t", "node_id": 1, "kind": "assistant_edu"},
        {"transcript_key": "t", "node_id": 2, "kind": "assistant_edu"},
        {"transcript_key": "t", "node_id": 3, "kind": "tool_call", "call_id": "c"},
        {"transcript_key": "t", "node_id": 4, "kind": "tool_output", "call_id": "c"},
        {"transcript_key": "t", "node_id": 5, "kind": "assistant_edu"},
        {"transcript_key": "t", "node_id": 6, "kind": "assistant_edu", "is_final_sink": True},
    ]
    pairs = [(1, 3), (3, 4), (4, 6)]
    edges = [{"edge_id": stable_id("e", source, target), "source_node_id": source,
              "target_node_id": target} for source, target in pairs]
    circuits = extract_circuits(nodes, edges, config_hash="f" * 64, max_learned=32)
    tool = next(row for row in circuits if row["extractor_kind"] == "tool")
    assert tool["node_ids"] == [1, 3, 4, 6]


def test_parallel_view_helpers_use_topk_support_and_exposure_adjusted_aggregation():
    assert topk_nonnegative(np.array([-1., 3., 2., 0.]), 2) == [(1, 3.0), (2, 2.0)]
    activation_rows = [
        {"unit_id": "a", "feature_id": 1, "activation": 2.0, "in_final_ancestry": True},
        {"unit_id": "b", "feature_id": 1, "activation": 0.0, "in_final_ancestry": False},
        {"unit_id": "c", "feature_id": 2, "activation": 4.0, "in_final_ancestry": False},
    ]
    summary = aggregate_activations(activation_rows)
    assert summary[1]["mean"] == pytest.approx(2 / 3)
    assert summary[1]["proportion_active"] == pytest.approx(1 / 3)
    assert summary[1]["ancestry_contrast"] == pytest.approx(2.0)
    assert "sum" not in summary[1] and "max" not in summary[1]
    assert retained_support([0, 0, 1, 1, 1, 2], minimum=2) == {0: 2, 1: 3}
    assert support_floor(100_000, 0.0005) == 50
    assert support_floor(200_000, 0.001) == 200
    assert retained_across_splits([0, 0, 1, 1], [0, 0, 0, 1], 2, 2) == {0: (2, 3)}
    dense = aggregate_dense(np.array([[1., 2.], [3., 4.], [5., 6.]]), [True, False, False])
    assert dense["mean"].tolist() == [3.0, 4.0]
    assert dense["ancestry_contrast"].tolist() == [-3.0, -3.0]
    assert np.allclose(weighted_kind_summary({"a": np.array([1., 3.]),
                                              "b": np.array([5., 7.])},
                                             {"a": 0.25, "b": 0.75}), [4., 6.])


def test_artifact_ids_schemas_hashes_and_no_overwrite(tmp_path):
    assert stable_id("a", 1, {"x": 2}) == stable_id("a", 1, {"x": 2})
    assert stable_id("a", 1, {"x": 2}) != stable_id("a", 2, {"x": 2})
    validate_artifact_rows("edges", [{
        "edge_id": "e", "transcript_key": "t", "source_node_id": 0,
            "target_node_id": 1, "edge_source": "inferred", "dense_row": 0,
            "cluster_id": 1, "sae_fit_id": "fit", "dictionary_id": "dict",
            "cluster_probability": 0.8,
            "probability": 0.9, "accepted": True, "threshold": 0.5, "maximum_parents": 3,
        }])
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = artifact_dir / "payload.json"
    payload.write_text("{}")
    provenance = {"code_commit": "dirty-test", "package_lock_hash": "d" * 64,
                  "judge_util_sha256": "e" * 64, "seed": 0, "model_revisions": {}}
    manifest = write_artifact_manifest(artifact_dir, {"payload.json": payload},
                                       upstream_hash="a" * 64, split_hash="b" * 64,
                                       config_hash="c" * 64, metadata=provenance,
                                       freeze_manifest_sha256="f" * 64,
                                       row_counts={"payload.json": 1},
                                       schema_hashes={"payload.json": "1" * 64})
    assert manifest["files"]["payload.json"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert manifest["freeze_manifest_sha256"] == "f" * 64
    with pytest.raises(FileExistsError):
        write_artifact_manifest(artifact_dir, {"payload.json": payload},
                                upstream_hash="a" * 64, split_hash="b" * 64,
                                config_hash="c" * 64, metadata=provenance,
                                freeze_manifest_sha256="f" * 64,
                                row_counts={"payload.json": 1},
                                schema_hashes={"payload.json": "1" * 64})
    with pytest.raises(ValueError, match="judge_util_sha256"):
        write_artifact_manifest(tmp_path / "other", {"payload.json": payload},
                                upstream_hash="a" * 64, split_hash="b" * 64,
                                config_hash="c" * 64, metadata=provenance | {
                                    "judge_util_sha256": "missing"},
                                freeze_manifest_sha256="f" * 64,
                                row_counts={"payload.json": 1},
                                schema_hashes={"payload.json": "1" * 64})
