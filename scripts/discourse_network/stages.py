"""Four plain, guarded, immutable Workstream C stages."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from .adapters import require_heavy_runtime, validate_embedding_alignment

from .attachments import (attachment_features, build_candidates, candidate_recall,
                          fit_attachment_head, fit_platt_scaler, fit_relation_geometry,
                          full_earlier_baselines, link_tool_edges, parent_cap_audit,
                          select_parents, select_threshold, transfer_metrics)
from .circuits import extract_circuits, marker_input_mask
from .artifacts import CArtifact, c_code_hash, load_c_manifest_chain
from .gates import (apply_shortcut_recurrence_gate, validate_attachment_gate, validate_circuit_encoder_gate,
                    validate_dense_retrieval_gate)
from .edge_views import (TopKSAE, fit_scaled_pca, fit_spherical_clusters,
                         cluster_metrics, fit_cluster_refits, fit_independent_views, fit_sae_refits,
                         parallel_view_fields, resolve_view_mask, retained_across_splits,
                         sae_metrics, support_floor)
from .nodes import (FrozenBundle, build_nodes, guard_canonical_stage, sha256_file,
                    stable_id, validate_artifact_rows, write_artifact_manifest)
from .network_outputs import (aggregate_circuit_transcripts, build_topology_outputs,
                              aggregate_edge_transcripts, topology_robustness)


STAGES = ("nodes", "attachments", "edge-views", "circuits")


@dataclass(frozen=True)
class StageContext:
    bundle: FrozenBundle
    config: dict[str, Any]
    config_path: Path
    input_dir: Path
    output_dir: Path
    metadata: dict[str, Any]
    confirmation_apply: bool = False
    apply_from: Path | None = None
    stage: str = ""
    upstream: CArtifact | None = None
    input_mask: dict[str, bool] | None = None
    adapter_dir: Path | None = None


def dispatch_stage(stage: str, *, context: StageContext | None = None,
                   handlers: dict[str, Callable[[], object]] | None = None):
    handlers = handlers or {
        "nodes": lambda: nodes_stage(context),
        "attachments": lambda: attachments_stage(context),
        "edge-views": lambda: edge_views_stage(context),
        "circuits": lambda: circuits_stage(context),
    }
    if stage not in STAGES or stage not in handlers:
        raise ValueError(f"unknown or unavailable Workstream C stage: {stage}")
    return handlers[stage]()


def _schema_hash(value: pd.DataFrame | np.ndarray | dict[str, Any]) -> str:
    if isinstance(value, pd.DataFrame):
        value = [{"name": column, "dtype": str(value[column].dtype)} for column in value.columns]
    elif isinstance(value, np.ndarray):
        value = {"dtype": str(value.dtype), "shape_tail": list(value.shape[1:])}
    return stable_id(value)


def _write_parquet_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_like=False)
    else:
        frame.to_parquet(path, index=False)


def _write_npy_once(values: np.ndarray, path: Path) -> None:
    if path.exists():
        if not np.array_equal(np.load(path), values):
            raise ValueError(f"checkpoint mismatch at {path}")
    else:
        np.save(path, values)


def _write_json_once(value: dict[str, Any], path: Path) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"checkpoint mismatch at {path}")
    else:
        path.write_text(text)


def _write_bytes_once(value: bytes, path: Path) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"checkpoint mismatch at {path}")
    else:
        path.write_bytes(value)


def _atomic_stage(context: StageContext,
                  producer: Callable[[Path], tuple[dict[str, Path], dict[str, Any]]]):
    output, partial = context.output_dir, Path(f"{context.output_dir}.partial")
    if (output / "artifact_manifest.json").exists():
        return json.loads((output / "artifact_manifest.json").read_text())
    if output.exists():
        raise FileExistsError(f"refusing incomplete immutable output {output}")
    partial.mkdir(parents=True, exist_ok=True)
    if (partial / "artifact_manifest.json").exists():
        partial.rename(output)
        return json.loads((output / "artifact_manifest.json").read_text())
    produced = producer(partial)
    files, values = produced[:2]
    manifest_fields = produced[2] if len(produced) == 3 else {}
    row_counts = {name: len(value) if not isinstance(value, dict) else 1
                  for name, value in values.items()}
    metadata = context.metadata | {"stage": context.stage,
                                   "purpose": "confirmation_apply" if context.confirmation_apply
                                   else "representation_discovery",
                                   "reconciliation_sha256": context.bundle.manifest["files"][
                                       "verdict_reconciliation"]["sha256"],
                                   "code_hash": c_code_hash(),
                                   "upstream_c_manifest_sha256": context.upstream.sha256
                                   if context.upstream else None,
                                   "upstream_c_manifest_path": str(context.upstream.path)
                                   if context.upstream else None,
                                   "input_mask": context.input_mask or {}} | manifest_fields
    manifest = write_artifact_manifest(
        partial, files, upstream_hash=context.bundle.manifest["canonical_content_hash"],
        split_hash=context.bundle.manifest["files"]["item_split"]["sha256"],
        config_hash=sha256_file(context.config_path),
        freeze_manifest_sha256=context.bundle.manifest_sha256,
        row_counts=row_counts, schema_hashes={name: _schema_hash(value)
                                              for name, value in values.items()},
        metadata=metadata,
    )
    partial.rename(output)
    return manifest


def _boundary_artifact(context: StageContext):
    path = context.input_dir / "boundaries.parquet"
    manifest = json.loads((context.input_dir / "boundary_manifest.json").read_text())
    required = {"file_sha256", "selected_system", "revision", "config_hash",
                "blind_gate_selected", "canonical_content_hash", "freeze_manifest_sha256",
                "approval_sha256", "assets_manifest_sha256", "gate_metrics",
                "gate_metrics_sha256", "gate_metrics_content_hash", "adapter_code_sha256",
                "input_sha256", "input_file_sha256", "input_row_hashes_sha256",
                "mode", "purpose"}
    if missing := required - set(manifest):
        raise ValueError(f"boundary manifest missing {sorted(missing)}")
    if manifest["file_sha256"] != sha256_file(path):
        raise ValueError("boundary-offset artifact hash mismatch")
    if manifest["canonical_content_hash"] != context.bundle.manifest["canonical_content_hash"]:
        raise ValueError("boundary artifact belongs to another transcript population")
    if manifest["freeze_manifest_sha256"] != context.bundle.manifest_sha256:
        raise ValueError("boundary artifact belongs to another A freeze")
    if manifest["blind_gate_selected"] is not True:
        raise ValueError("boundary system was not selected by the blind gate")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["assets_manifest_sha256"]):
        raise ValueError("boundary staged-assets provenance is invalid")
    expected_input = context.bundle.manifest["files"]["transcript_store"]["sha256"]
    if manifest["input_sha256"] != expected_input or manifest["input_file_sha256"] != expected_input:
        raise ValueError("boundary input hash does not match the frozen transcript store")
    if manifest["adapter_code_sha256"] != sha256_file(Path(__file__).with_name("adapters.py")):
        raise ValueError("boundary adapter code hash mismatch")
    if manifest["selected_system"] not in {"isanlp_rst", "punctuation_fallback"}:
        raise ValueError("unknown frozen segmenter selection")
    expected_revision = (context.config["segmentation"]["primary"]["checkpoint_revision"]
                         if manifest["selected_system"] == "isanlp_rst"
                         else context.config["segmentation"]["comparison"])
    if manifest["revision"] != expected_revision:
        raise ValueError("boundary segmenter revision does not match the frozen config")
    expected_purpose = "confirmation_apply" if context.confirmation_apply else "representation_discovery"
    if (manifest["mode"] != "canonical" or manifest["purpose"] != expected_purpose or
            manifest["approval_sha256"] !=
            context.metadata["approval_sha256"] or manifest["config_hash"] != sha256_file(
                context.config_path)):
        raise ValueError("boundary approval/config provenance mismatch")
    if manifest["gate_metrics"].get("overall_f1", 0) < 0.90 or manifest[
            "gate_metrics"].get("subset_min_f1", 0) < 0.85:
        raise ValueError("boundary blind gate metrics fail the frozen floors")
    if stable_id(manifest["gate_metrics"]) != manifest["gate_metrics_content_hash"]:
        raise ValueError("boundary embedded gate metrics hash mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["gate_metrics_sha256"]):
        raise ValueError("boundary gate-metrics file hash is invalid")
    if manifest["gate_metrics"].get("selected_system") != manifest["selected_system"]:
        raise ValueError("boundary system disagrees with its blind gate metrics")
    boundaries = pd.read_parquet(path)
    boundary_fields = {"transcript_key", "part_id", "subpart_id", "start", "end",
                       "source_text_hash", "segmenter_revision", "segmenter_config_hash",
                       "input_file_sha256", "input_row_hash"}
    if missing := boundary_fields - set(boundaries):
        raise ValueError(f"boundary-offset artifact missing {sorted(missing)}")
    if boundaries.duplicated(["transcript_key", "part_id", "subpart_id", "start", "end"]).any():
        raise ValueError("boundary-offset artifact has duplicate span keys")
    boundary_schema = stable_id([{"name": column, "dtype": str(boundaries[column].dtype)}
                                 for column in boundaries.columns])
    if manifest.get("rows") != len(boundaries) or manifest.get("schema_hash") != boundary_schema:
        raise ValueError("boundary row count/schema mismatch")
    if stable_id(sorted(set(boundaries["input_row_hash"]))) != manifest["input_row_hashes_sha256"]:
        raise ValueError("boundary input-row hash inventory mismatch")
    return boundaries, {"system": manifest["selected_system"],
                                  "revision": manifest["revision"],
                                  "config_hash": manifest["config_hash"],
                                  "blind_gate_selected": True}


def _embedding_artifact(context: StageContext, directory: Path) -> str:
    manifest_path = directory / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {"mode": "canonical", "freeze_manifest_sha256": context.bundle.manifest_sha256,
                "purpose": "confirmation_apply" if context.confirmation_apply else "representation_discovery",
                "canonical_content_hash": context.bundle.manifest["canonical_content_hash"],
                "config_hash": sha256_file(context.config_path),
                "approval_sha256": context.metadata["approval_sha256"],
                "nodes_manifest_sha256": context.upstream.sha256 if context.upstream else None,
                "nodes_file_sha256": context.upstream.manifest["files"]["nodes.parquet"]["sha256"]
                if context.upstream else None,
                "revision": context.config["relations"]["encoder_revision"]}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"embedding manifest {field} mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("assets_manifest_sha256", "")):
        raise ValueError("embedding staged-assets provenance is invalid")
    if manifest.get("adapter_code_sha256") != sha256_file(Path(__file__).with_name("adapters.py")):
        raise ValueError("embedding adapter code hash mismatch")
    for name, field in (("node_embeddings.npy", "vectors_sha256"),
                        ("node_embedding_metadata.parquet", "metadata_sha256")):
        if sha256_file(directory / name) != manifest[field]:
            raise ValueError(f"embedding artifact hash mismatch for {name}")
    if manifest.get("coverage") != 1.0:
        raise ValueError("embedding artifact coverage must be complete")
    vectors = np.load(directory / "node_embeddings.npy", mmap_mode="r")
    metadata = pd.read_parquet(directory / "node_embedding_metadata.parquet")
    if (manifest.get("rows") != len(vectors) or manifest.get("rows") != len(metadata) or
            manifest.get("vector_schema_hash") != stable_id({"dtype": str(vectors.dtype),
                                                              "shape_tail": list(vectors.shape[1:])}) or
            manifest.get("metadata_schema_hash") != stable_id([
                {"name": column, "dtype": str(metadata[column].dtype)} for column in metadata.columns])):
        raise ValueError("embedding row count/schema mismatch")
    return sha256_file(manifest_path)


def nodes_stage(context: StageContext | None):
    if context is None:
        raise ValueError("nodes stage inputs are required")

    def produce(directory: Path):
        boundaries, provenance = _boundary_artifact(context)
        rows = []
        for record in context.bundle.population.to_dict("records"):
            selected = boundaries[boundaries["transcript_key"].eq(record["transcript_key"])]
            nodes = build_nodes(record["transcript_key"], record["question"], record["transcript"],
                                record["transcript_parts_json"],
                                prose_boundaries=selected.to_dict("records"),
                                segmenter_provenance=provenance,
                                input_row_hash=record["input_row_hash"],
                                input_file_sha256=record["input_file_sha256"])
            rows.extend(node | {"split": record["split"], "item_key": record["item_key"]}
                        for node in nodes)
        validate_artifact_rows("nodes", rows)
        frame, path = pd.DataFrame(rows), directory / "nodes.parquet"
        _write_parquet_once(frame, path)
        return {path.name: path}, {path.name: frame}, {
            "input_mask": marker_input_mask(rows),
            "boundary_manifest_sha256": sha256_file(context.input_dir / "boundary_manifest.json")}

    return _atomic_stage(context, produce)


def _attachment_pairs(nodes: pd.DataFrame, embeddings: np.ndarray, config: dict[str, Any]):
    candidates, groups, source_vectors, target_vectors, structural = [], [], [], [], []
    for _, group in nodes.groupby("transcript_key", sort=True):
        group = group.sort_values("node_id")
        positions = group.index.to_numpy()
        local_nodes, local_embeddings = group.to_dict("records"), embeddings[positions]
        structural.extend(row | {"split": group.iloc[0]["split"], "item_key": group.iloc[0]["item_key"]}
                          for row in link_tool_edges(local_nodes, config["config_hash"]))
        for target in [node["node_id"] for node in local_nodes if node["target_eligible"]]:
            rows = build_candidates(local_nodes, local_embeddings, target,
                                    cap=config["cap"], preceding=config["preceding"],
                                    semantic=config["semantic"], identifiers=config["identifiers"],
                                    structural=config["structural_context"],
                                    reservoir_multiplier=config.get("reservoir_multiplier", 1))
            rows = [row | {"split": group.iloc[0]["split"], "item_key": group.iloc[0]["item_key"]}
                    for row in rows]
            offset = len(candidates)
            candidates.extend(rows)
            groups.append((offset, local_nodes, rows))
            source_vectors.extend(local_embeddings[row["source_node_id"]] for row in rows)
            target_vectors.extend(local_embeddings[row["target_node_id"]] for row in rows)
    return candidates, groups, np.asarray(source_vectors), np.asarray(target_vectors), structural


def attachments_stage(context: StageContext | None):
    if context is None:
        raise ValueError("attachments stage inputs are required")
    if context.confirmation_apply and context.apply_from is None:
        raise ValueError("confirmation attachments are apply-only and require --apply-from")

    def produce(directory: Path):
        upstream_dir = context.upstream.path.parent if context.upstream else context.input_dir
        adapter_dir = context.adapter_dir or context.input_dir
        embedding_manifest_sha256 = _embedding_artifact(context, adapter_dir)
        nodes = pd.read_parquet(upstream_dir / "nodes.parquet").reset_index(drop=True)
        embeddings = np.load(adapter_dir / "node_embeddings.npy")
        embedding_meta = pd.read_parquet(adapter_dir / "node_embedding_metadata.parquet")
        required_meta = {"transcript_key", "node_id", "embedding_row", "token_count", "chunk_count",
                         "coverage", "encoder_revision"}
        if missing := required_meta - set(embedding_meta):
            raise ValueError(f"node embedding metadata missing {sorted(missing)}")
        aligned = nodes[["transcript_key", "node_id"]].merge(
            embedding_meta, on=["transcript_key", "node_id"], how="left", validate="one_to_one")
        if aligned["embedding_row"].isna().any() or not aligned["embedding_row"].astype(int).between(
                0, len(embeddings) - 1).all() or not np.allclose(aligned["coverage"], 1.0):
            raise ValueError("node embeddings require complete row alignment and full coverage")
        if set(aligned["encoder_revision"].astype(str)) != {context.config["relations"]["encoder_revision"]}:
            raise ValueError("node embedding revision does not match the frozen C config")
        embeddings = embeddings[aligned["embedding_row"].astype(int).to_numpy()]
        nodes[["token_count", "chunk_count", "coverage"]] = aligned[
            ["token_count", "chunk_count", "coverage"]]
        nodes["embedding_row"] = aligned["embedding_row"].astype(int)
        validate_artifact_rows("nodes", nodes.to_dict("records"))
        cfg = context.config["candidates"] | {"config_hash": sha256_file(context.config_path)}
        primary_candidate_cap = cfg["cap"]
        cfg["cap"] = max(context.config["relations"]["sensitivity"]["candidate_caps"])
        candidates, feature_groups, source, target, structural = _attachment_pairs(nodes, embeddings, cfg)
        candidate_frame = pd.DataFrame(candidates)
        candidate_frame["primary_candidate"] = candidate_frame["candidate_rank"].le(primary_candidate_cap)
        baseline_rows = []
        for _, group in nodes.groupby("transcript_key", sort=True):
            local = group.sort_values("node_id").to_dict("records")
            for target_id in [node["node_id"] for node in local if node["target_eligible"]]:
                baseline_rows.extend(row | {"transcript_key": local[0]["transcript_key"],
                                            "split": local[0]["split"]}
                                     for row in full_earlier_baselines(local, target_id))
        baseline_frame = pd.DataFrame(baseline_rows)
        lookup = {(row.transcript_key, row.node_id): i for i, row in enumerate(nodes.itertuples())}
        structural_source = np.asarray([embeddings[lookup[row["transcript_key"], row["source_node_id"]]]
                                        for row in structural])
        structural_target = np.asarray([embeddings[lookup[row["transcript_key"], row["target_node_id"]]]
                                        for row in structural])
        if context.confirmation_apply:
            frozen = joblib.load(context.apply_from / "attachment_models.joblib")
            relation, head, platt, threshold = (frozen[key] for key in
                                                ("relation", "head", "platt", "threshold"))
        else:
            candidate_train = np.array([row["split"] == "representation_train" and
                                        row["candidate_rank"] <= primary_candidate_cap for row in candidates])
            structural_train = np.array([row["split"] == "representation_train" for row in structural])
            fit_source = np.vstack([source[candidate_train], structural_source[structural_train]]) \
                if structural_train.any() else source[candidate_train]
            fit_target = np.vstack([target[candidate_train], structural_target[structural_train]]) \
                if structural_train.any() else target[candidate_train]
            fit_rows = [row for row in candidates if row["split"] == "representation_train" and
                        row["candidate_rank"] <= primary_candidate_cap] + [
                            row for row in structural if row["split"] == "representation_train"]
            pair_keys = [stable_id(row["transcript_key"], row["source_node_id"], row["target_node_id"])
                         for row in fit_rows]
            relation = fit_relation_geometry(fit_source, fit_target,
                                             n_components=context.config["relations"]["pca_components"],
                                             seed=0, pair_keys=pair_keys,
                                             max_pairs=context.config["relations"]["pca_max_pairs"])
        dense = relation.transform(source, target)
        full_width = dense.shape[1] + 16
        full = np.empty((len(candidates), full_width))
        for offset, local_nodes, rows in feature_groups:
            full[offset:offset + len(rows)] = attachment_features(
                dense[offset:offset + len(rows)], rows, local_nodes, mode="full")
        if not context.confirmation_apply:
            labels = pd.read_parquet(adapter_dir / "attachment_labels.parquet")
            keys = ["transcript_key", "source_node_id", "target_node_id"]
            if labels.duplicated(keys).any():
                raise ValueError("attachment labels contain duplicate edge keys")
            positive = labels[labels["label"].eq(1)]
            gold_sets, candidate_sets = [], []
            for key, group in positive.groupby(["transcript_key", "target_node_id"]):
                gold_sets.append(set(group["source_node_id"]))
                candidate_sets.append(set(candidate_frame[
                    candidate_frame["transcript_key"].eq(key[0]) &
                    candidate_frame["target_node_id"].eq(key[1])]["source_node_id"]))
            if gold_sets and np.mean([candidate_recall(gold, candidate) for gold, candidate in
                                      zip(gold_sets, candidate_sets)]) < context.config["gates"]["candidate_recall"]:
                raise ValueError("candidate recall gate failed")
            if gold_sets and parent_cap_audit(gold_sets, context.config["relations"]["maximum_parents"]) > 0.05:
                raise ValueError("gold parent-cap audit failed")
            target_fields = [field for field in ("purpose", "item_group", "subset") if field in labels]
            target_meta = labels[["transcript_key", "target_node_id", *target_fields]].drop_duplicates()
            if target_meta.duplicated(["transcript_key", "target_node_id"]).any():
                raise ValueError("attachment target metadata is inconsistent")
            joined = candidate_frame.reset_index(names="candidate_row").merge(
                target_meta, on=["transcript_key", "target_node_id"], how="inner").merge(
                    labels[keys + ["label"]], on=keys, how="left")
            joined["label"] = joined["label"].fillna(0).astype(int)
            training = joined[joined["purpose"].eq("train") & joined["primary_candidate"]]
            calibration = joined[joined["purpose"].eq("calibration") & joined["primary_candidate"]]
            head = fit_attachment_head(full[training["candidate_row"]], training["label"].to_numpy(),
                                       training["item_group"].to_numpy(),
                                       cs=context.config["relations"]["attachment_cs"],
                                       min_positive=context.config["relations"]["minimum_positive_edges"])
            logits = head.decision_function(full[calibration["candidate_row"]])
            platt = fit_platt_scaler(logits, calibration["label"].to_numpy())
            threshold = select_threshold(platt.transform(logits), calibration["label"].to_numpy())
            if "subset" in calibration:
                metrics = transfer_metrics(calibration["label"].to_numpy(),
                                           (platt.transform(logits) >= threshold).astype(int),
                                           calibration["subset"].astype(str).to_numpy())
                if any(row["precision"] < context.config["gates"]["subset_precision"] or
                       row["recall"] < context.config["gates"]["subset_recall"]
                       for name, row in metrics.items() if name != "overall"):
                    raise ValueError("per-subset attachment transfer gate failed")
        candidate_frame["probability"] = platt.transform(head.decision_function(full))
        maximum = int(context.config["relations"]["maximum_parents"])
        accepted = []
        for _, group in candidate_frame.groupby(["transcript_key", "target_node_id"], sort=True):
            accepted.extend(select_parents(group[group["primary_candidate"]].to_dict("records"), threshold=threshold,
                                           max_parents=maximum))
        accepted_keys = {(row["transcript_key"], row["source_node_id"], row["target_node_id"])
                         for row in accepted}
        candidate_frame["accepted"] = [(row.transcript_key, row.source_node_id, row.target_node_id)
                                        in accepted_keys for row in candidate_frame.itertuples()]
        candidate_frame["threshold"], candidate_frame["maximum_parents"] = threshold, maximum
        relation_rows = {tuple(row[key] for key in ("transcript_key", "source_node_id", "target_node_id")): i
                         for i, row in enumerate(candidates)}
        edges, accepted_dense = [], []
        for row in accepted:
            accepted_dense.append(dense[relation_rows[row["transcript_key"], row["source_node_id"],
                                                       row["target_node_id"]]])
            edges.append(row | {"edge_id": stable_id(row["transcript_key"], row["source_node_id"],
                                                       row["target_node_id"], "inferred", cfg["config_hash"]),
                                "edge_source": "inferred", "accepted": True, "threshold": threshold,
                                "maximum_parents": maximum,
                                **parallel_view_fields(len(accepted_dense) - 1, None, None, None)})
        structural_dense = []
        for row in structural:
            s, t = embeddings[lookup[row["transcript_key"], row["source_node_id"]]], embeddings[
                lookup[row["transcript_key"], row["target_node_id"]]]
            structural_dense.append(relation.transform(s[None], t[None])[0])
            edges.append(row | {"probability": None, "threshold": None, "maximum_parents": None,
                                **parallel_view_fields(len(accepted_dense) + len(structural_dense) - 1,
                                                       None, None, None)})
        if not accepted_dense and not structural_dense:
            raise ValueError("attachment stage produced no accepted or structural edges")
        blocks = [np.asarray(accepted_dense)] + ([np.asarray(structural_dense)] if structural_dense else [])
        edge_dense = np.vstack(blocks).astype(np.float16)
        edge_frame = pd.DataFrame(edges)
        validate_artifact_rows("edges", edge_frame.to_dict("records"))
        sensitivity = context.config["relations"]["sensitivity"]
        topology_primary, topology_grid = build_topology_outputs(
            nodes, candidate_frame, structural, threshold=threshold,
            primary_parents=maximum, primary_budget=primary_candidate_cap,
            parent_grid=sensitivity["maximum_parents"],
            threshold_offsets=sensitivity["threshold_offsets"],
            budget_grid=sensitivity["candidate_caps"], config_hash=cfg["config_hash"])
        headline = context.config["relations"]["headline_topology_statistics"]
        topology_gate_metrics = topology_robustness(topology_primary, topology_grid, headline)
        paths = {"attachment_candidates.parquet": directory / "attachment_candidates.parquet",
                 "attachment_baselines.parquet": directory / "attachment_baselines.parquet",
                 "edges.parquet": directory / "edges.parquet", "dense_edges.npy": directory / "dense_edges.npy",
                 "attachment_models.joblib": directory / "attachment_models.joblib",
                 "nodes.parquet": directory / "nodes.parquet",
                 "node_embeddings.npy": directory / "node_embeddings.npy",
                 "node_embedding_metadata.parquet": directory / "node_embedding_metadata.parquet",
                 "topology_primary.parquet": directory / "topology_primary.parquet",
                 "topology_sensitivity.parquet": directory / "topology_sensitivity.parquet",
                 "topology_robustness.json": directory / "topology_robustness.json"}
        _write_parquet_once(candidate_frame, paths["attachment_candidates.parquet"])
        _write_parquet_once(baseline_frame, paths["attachment_baselines.parquet"])
        _write_parquet_once(edge_frame, paths["edges.parquet"])
        _write_npy_once(edge_dense, paths["dense_edges.npy"])
        _write_parquet_once(nodes, paths["nodes.parquet"])
        _write_npy_once(embeddings.astype(np.float32), paths["node_embeddings.npy"])
        _write_parquet_once(embedding_meta, paths["node_embedding_metadata.parquet"])
        _write_parquet_once(topology_primary, paths["topology_primary.parquet"])
        _write_parquet_once(topology_grid, paths["topology_sensitivity.parquet"])
        _write_json_once(topology_gate_metrics, paths["topology_robustness.json"])
        models = {"relation": relation, "head": head, "platt": platt, "threshold": threshold}
        if not paths["attachment_models.joblib"].exists():
            joblib.dump(models, paths["attachment_models.joblib"])
        return paths, {"attachment_candidates.parquet": candidate_frame,
                       "attachment_baselines.parquet": baseline_frame, "edges.parquet": edge_frame,
                       "dense_edges.npy": edge_dense,
                       "nodes.parquet": nodes, "node_embeddings.npy": embeddings.astype(np.float32),
                       "node_embedding_metadata.parquet": embedding_meta,
                       "topology_primary.parquet": topology_primary,
                       "topology_sensitivity.parquet": topology_grid,
                       "topology_robustness.json": topology_gate_metrics,
                       "attachment_models.joblib": {"keys": sorted(models)}}, {
                           "input_mask": context.upstream.manifest.get("input_mask", {})
                           if context.upstream else {},
                           "embedding_manifest_sha256": embedding_manifest_sha256}

    return _atomic_stage(context, produce)


def _activation_frame(ids: list[str], activations: np.ndarray, sae: TopKSAE, key: str) -> pd.DataFrame:
    return pd.DataFrame([{key: unit, "sae_fit_id": sae.fit_id, "dictionary_id": sae.dictionary_id,
                          "feature_id": feature, "activation": float(value)}
                         for unit, vector in zip(ids, activations)
                         for feature, value in enumerate(vector) if value > 0])


def _feature_inventory(sae: TopKSAE | None, features) -> tuple[list[int], str]:
    features = sorted({int(feature) for feature in features})
    if sae is None:
        if features:
            raise ValueError("retained SAE features require a dictionary")
    elif any(feature < 0 or feature >= len(sae.decoder) for feature in features):
        raise ValueError("retained SAE feature is outside the frozen dictionary")
    return features, stable_id({"sae_fit_id": sae.fit_id if sae is not None else None,
                                "dictionary_id": sae.dictionary_id if sae is not None else None,
                                "feature_ids": features})


def _load_feature_inventory(models: dict[str, Any], manifest: dict[str, Any], prefix: str,
                            sae: TopKSAE | None) -> tuple[list[int], str]:
    recorded = models.get("retained_features", [])
    features, digest = _feature_inventory(sae, recorded)
    if (recorded != features or models.get("retained_features_sha256") != digest or
            manifest.get(f"retained_{prefix}_sae_feature_ids") != features or
            manifest.get(f"retained_{prefix}_sae_feature_ids_sha256") != digest):
        raise ValueError(f"frozen {prefix} SAE feature inventory mismatch")
    return features, digest


def edge_views_stage(context: StageContext | None):
    if context is None:
        raise ValueError("edge-views stage inputs are required")
    if context.confirmation_apply and context.apply_from is None:
        raise ValueError("confirmation edge views are apply-only and require --apply-from")

    def produce(directory: Path):
        edges = pd.read_parquet(context.input_dir / "edges.parquet")
        dense = np.load(context.input_dir / "dense_edges.npy").astype(np.float32)
        nodes = pd.read_parquet(context.input_dir / "nodes.parquet")
        node_embeddings = np.load(context.input_dir / "node_embeddings.npy")
        node_meta = pd.read_parquet(context.input_dir / "node_embedding_metadata.parquet")
        gate_dir = context.adapter_dir or context.input_dir
        if context.confirmation_apply:
            models = joblib.load(context.apply_from / "edge_view_models.joblib")
            development_manifest = json.loads(
                (context.apply_from / "artifact_manifest.json").read_text())
            clusters, sae = models["clusters"], models["sae"]
            statuses, input_mask = models["statuses"], models["fit_input_mask"]
            fit = models
            gate_sha256 = None
        else:
            gate = json.loads((gate_dir / "attachment_gate.json").read_text())
            validate_attachment_gate(gate, context.bundle.manifest_sha256, sha256_file(context.config_path),
                                     context.upstream.sha256 if context.upstream else "")
            robustness = json.loads((context.input_dir / "topology_robustness.json").read_text())
            for field in ("topology_grid_cells", "nonconstant_headline_statistics", "topology_spearman",
                          "candidate_budget_populations", "max_realized_candidate_size"):
                if gate.get(field) != robustness.get(field):
                    raise ValueError("attachment gate topology robustness disagrees with pipeline output")
            if gate.get("input_mask") != (context.upstream.manifest.get("input_mask", {})
                                           if context.upstream else {}):
                raise ValueError("attachment gate input mask mismatch")
            dense_pass = validate_dense_retrieval_gate(gate.get("dense_retrieval", {}))
            gate_sha256 = sha256_file(gate_dir / "attachment_gate.json")
            train_mask = edges["split"].eq("representation_train").to_numpy()
            validation_mask = edges["split"].eq("representation_validation").to_numpy()
            train, validation = dense[train_mask], dense[validation_mask]
            train_groups = edges.loc[train_mask, "item_key"].astype(str).to_numpy()
            fit = fit_independent_views(train, validation, train_groups,
                                        context.config["edge_views"], context.config["gates"],
                                        dense_pass=dense_pass)
            clusters, sae = fit["clusters"], fit["sae"]
            statuses, input_mask = fit["statuses"], fit["input_mask"]
        if context.confirmation_apply:
            retained, retained_sha256 = _load_feature_inventory(
                models, development_manifest, "edge", sae)
        else:
            retained, retained_sha256 = _feature_inventory(sae, [])
        edges = edges.copy()
        if clusters is not None:
            probabilities = clusters.predict_proba(dense)
            assignments = probabilities.argmax(1)
            edges["cluster_id"] = assignments
            edges["cluster_probability"] = probabilities[np.arange(len(edges)), assignments]
        else:
            edges["cluster_id"], edges["cluster_probability"] = None, None
        if sae is not None:
            activations = sae.transform(dense)
            edges["sae_fit_id"], edges["dictionary_id"] = sae.fit_id, sae.dictionary_id
            activation = _activation_frame(edges["edge_id"].astype(str).tolist(), activations, sae, "edge_id")
        else:
            edges["sae_fit_id"], edges["dictionary_id"] = None, None
            activation = pd.DataFrame(columns=["edge_id", "sae_fit_id", "dictionary_id",
                                               "feature_id", "activation"])
        if not context.confirmation_apply and sae is not None:
            train_features = [feature for vector in activations[train_mask] for feature in np.flatnonzero(vector)]
            validation_features = [feature for vector in activations[validation_mask]
                                   for feature in np.flatnonzero(vector)]
            retained = retained_across_splits(
                train_features, validation_features,
                support_floor(train_mask.sum(), context.config["edge_views"]["sae_support_rate"]),
                support_floor(validation_mask.sum(), context.config["edge_views"]["sae_support_rate"]))
            retained, retained_sha256 = _feature_inventory(
                sae, set(retained) & fit["stable_features"])
        if sae is not None:
            activation = activation[activation["feature_id"].isin(retained)].reset_index(drop=True)
        validate_artifact_rows("edges", edges.to_dict("records"))
        validate_artifact_rows("edge_sae_activations", activation.to_dict("records"))
        candidate_audit = pd.read_parquet(context.input_dir / "attachment_candidates.parquet")
        audit = edges.merge(candidate_audit[["transcript_key", "source_node_id", "target_node_id",
                                              "node_distance", "adjacent", "lexical_overlap"]],
                            on=["transcript_key", "source_node_id", "target_node_id"], how="left",
                            suffixes=("", "_candidate"))
        for field in ("node_distance", "adjacent", "lexical_overlap"):
            candidate_field = f"{field}_candidate"
            if candidate_field in audit:
                audit[field] = audit[field].combine_first(audit[candidate_field])
        source_kinds = nodes[["transcript_key", "node_id", "kind"]].rename(
            columns={"node_id": "source_node_id", "kind": "source_kind"})
        target_kinds = nodes[["transcript_key", "node_id", "kind"]].rename(
            columns={"node_id": "target_node_id", "kind": "target_kind"})
        audit = audit.merge(source_kinds, on=["transcript_key", "source_node_id"], how="left",
                            validate="many_to_one").merge(
            target_kinds, on=["transcript_key", "target_node_id"], how="left",
            validate="many_to_one")
        nodes = nodes.copy()
        nodes["_a1_marker"] = nodes["verification_marker"].astype(bool) | nodes[
            "backtracking_marker"].astype(bool)
        transcript_exposure = nodes.groupby("transcript_key").agg(
            transcript_length=("rendered_text", lambda values: sum(len(str(value)) for value in values)),
            a1_marker_density=("_a1_marker", "mean")
        ).reset_index()
        shortcut_inputs = audit.merge(transcript_exposure, on="transcript_key", how="left")[[
            "edge_id", "transcript_key", "item_key", "node_distance", "adjacent", "lexical_overlap",
            "source_kind", "target_kind", "edge_source", "transcript_length", "a1_marker_density"]]
        shortcut_inputs["node_kind_pair"] = (shortcut_inputs["source_kind"].astype(str) + "->" +
                                                   shortcut_inputs["target_kind"].astype(str))
        shortcut_inputs["benchmark_join_required"] = True
        shortcut_inputs["provider_join_required"] = True
        if input_mask["dense"]:
            transcript_aggregates, feature_aggregates = aggregate_edge_transcripts(
                edges.reset_index(drop=True), dense, activation, nodes,
                feature_ids=retained if input_mask["sae"] else [])
        else:
            count_rows = []
            for transcript_key, group in edges.groupby("transcript_key", sort=True):
                for scope, scoped in [("combined", group)] + [
                        (str(source), source_group)
                        for source, source_group in group.groupby("edge_source", sort=True)]:
                    count_rows.append({"transcript_key": transcript_key, "edge_scope": scope,
                                       "edge_count": len(scoped),
                                       "item_key": scoped.iloc[0].get("item_key"),
                                       "split": scoped.iloc[0].get("split")})
            transcript_aggregates = pd.DataFrame(count_rows)
            feature_aggregates = pd.DataFrame(columns=[
                "transcript_key", "edge_scope", "item_key", "split", "feature_id", "mean",
                "proportion_active", "median", "q90", "ancestry_contrast"])
        for name, status in statuses.items():
            transcript_aggregates[f"{name}_status"] = status
            feature_aggregates[f"{name}_status"] = status
        transcript_aggregates["edge_summary_status"] = "pre_audit_non_final"
        feature_aggregates["edge_summary_status"] = "pre_audit_non_final"
        paths = {"edges.parquet": directory / "edges.parquet", "dense_edges.npy": directory / "dense_edges.npy",
                 "edge_sae_activations.parquet": directory / "edge_sae_activations.parquet",
                 "edge_view_models.joblib": directory / "edge_view_models.joblib",
                 "edge_view_metrics.json": directory / "edge_view_metrics.json",
                 "nodes.parquet": directory / "nodes.parquet",
                 "node_embeddings.npy": directory / "node_embeddings.npy",
                 "node_embedding_metadata.parquet": directory / "node_embedding_metadata.parquet",
                 "shortcut_audit_inputs.parquet": directory / "shortcut_audit_inputs.parquet",
                 "edge_transcript_aggregates.parquet": directory / "edge_transcript_aggregates.parquet",
                 "edge_feature_aggregates.parquet": directory / "edge_feature_aggregates.parquet"}
        _write_parquet_once(edges, paths["edges.parquet"])
        _write_npy_once(dense.astype(np.float16), paths["dense_edges.npy"])
        _write_parquet_once(activation, paths["edge_sae_activations.parquet"])
        _write_parquet_once(nodes, paths["nodes.parquet"])
        _write_npy_once(node_embeddings, paths["node_embeddings.npy"])
        _write_parquet_once(node_meta, paths["node_embedding_metadata.parquet"])
        _write_parquet_once(shortcut_inputs, paths["shortcut_audit_inputs.parquet"])
        _write_parquet_once(transcript_aggregates, paths["edge_transcript_aggregates.parquet"])
        _write_parquet_once(feature_aggregates, paths["edge_feature_aggregates.parquet"])
        if not paths["edge_view_models.joblib"].exists():
            joblib.dump({"clusters": clusters, "sae": sae, "statuses": statuses,
                         "fit_input_mask": input_mask,
                         "retained_features": retained,
                         "retained_features_sha256": retained_sha256,
                         "sae_refits": fit.get("sae_refits", []), "matches": fit.get("matches", [])},
                        paths["edge_view_models.joblib"])
        metrics = ({"metric_split": "representation_validation", "view_statuses": statuses,
                    "errors": fit.get("errors", {}), "cluster_stability": fit.get("cluster_stability", []),
                    "sae": sae_metrics(validation, sae, [match["cosines"] for match in fit.get("matches", [])])
                    if sae is not None else None,
                    "retained_features": retained}
                   if not context.confirmation_apply else
                   {"applied_edges": len(edges), "view_statuses": statuses})
        _write_json_once(metrics, paths["edge_view_metrics.json"])
        frozen_edge_mask = (context.upstream.manifest.get("input_mask", {})
                            if context.upstream else {}) | {
                                f"edge_{name}": enabled for name, enabled in input_mask.items()}
        return paths, {"edges.parquet": edges, "dense_edges.npy": dense.astype(np.float16),
                       "edge_sae_activations.parquet": activation,
                       "edge_view_models.joblib": {"clusters": len(clusters.estimator.cluster_centers_)
                                                   if clusters is not None else 0,
                                                   "sae_width": len(sae.decoder) if sae is not None else 0},
                       "edge_view_metrics.json": metrics, "nodes.parquet": nodes,
                       "node_embeddings.npy": node_embeddings,
                       "node_embedding_metadata.parquet": node_meta,
                       "shortcut_audit_inputs.parquet": shortcut_inputs,
                       "edge_transcript_aggregates.parquet": transcript_aggregates,
                       "edge_feature_aggregates.parquet": feature_aggregates}, {
                           "view_statuses": statuses,
                           "input_mask": frozen_edge_mask,
                           "retained_edge_sae_feature_ids": retained,
                           "retained_edge_sae_feature_ids_sha256": retained_sha256,
                           "edge_summary_status": "pre_audit_non_final",
                           "attachment_gate_sha256": gate_sha256}

    return _atomic_stage(context, produce)


def circuits_stage(context: StageContext | None):
    if context is None:
        raise ValueError("circuits stage inputs are required")
    if context.confirmation_apply and context.apply_from is None:
        raise ValueError("confirmation circuit views are apply-only and require --apply-from")

    def produce(directory: Path):
        nodes = pd.read_parquet(context.input_dir / "nodes.parquet")
        edges = pd.read_parquet(context.input_dir / "edges.parquet")
        node_embeddings = np.load(context.input_dir / "node_embeddings.npy").astype(np.float32)
        node_meta = pd.read_parquet(context.input_dir / "node_embedding_metadata.parquet")
        aligned_nodes = validate_embedding_alignment(nodes, node_meta, node_embeddings)
        node_embeddings = node_embeddings[aligned_nodes["embedding_row"].astype(int).to_numpy()]
        edge_dense = np.load(context.input_dir / "dense_edges.npy").astype(np.float32)
        if len(nodes) != len(node_embeddings):
            raise ValueError("circuit node embeddings must align one-to-one with nodes")
        edge_activation_path = context.input_dir / "edge_sae_activations.parquet"
        edge_activations = pd.read_parquet(edge_activation_path) if edge_activation_path.exists() else pd.DataFrame()
        gate_dir = context.adapter_dir or context.input_dir
        upstream_statuses = (context.upstream.manifest.get("view_statuses", {})
                             if context.upstream else {})
        edge_order_hash = stable_id(edges["edge_id"].astype(str).tolist())
        development_models = None
        if context.confirmation_apply:
            development_models = joblib.load(context.apply_from / "circuit_view_models.joblib")
            shortcut_gate_path = context.apply_from / "shortcut_recurrence_gate.json"
            supplied_gate = gate_dir / "shortcut_recurrence_gate.json"
            if supplied_gate.exists() and supplied_gate.resolve() != shortcut_gate_path.resolve():
                raise ValueError("confirmation refuses a newly supplied shortcut/recurrence gate")
            if sha256_file(shortcut_gate_path) != development_models.get("shortcut_gate_sha256"):
                raise ValueError("development shortcut/recurrence gate hash mismatch")
            shortcut_gate = json.loads(shortcut_gate_path.read_text())
            if (shortcut_gate.get("edge_manifest_sha256") !=
                    development_models.get("source_edge_manifest_sha256") or
                    shortcut_gate.get("edge_order_hash") !=
                    development_models.get("source_edge_order_hash")):
                raise ValueError("development shortcut gate source hash/order mismatch")
            edge_audit = development_models["edge_audit"]
            effective_edge_mask = development_models["effective_edge_mask"]
            if upstream_statuses != development_models["source_edge_view_statuses"]:
                raise ValueError("confirmation edge views disagree with development-frozen statuses")
            if (context.upstream.manifest.get("retained_edge_sae_feature_ids_sha256") !=
                    development_models.get("source_edge_sae_feature_ids_sha256")):
                raise ValueError("confirmation edge SAE inventory disagrees with development")
        else:
            shortcut_gate_path = gate_dir / "shortcut_recurrence_gate.json"
            shortcut_gate = json.loads(shortcut_gate_path.read_text())
            for field, expected in {
                    "freeze_manifest_sha256": context.bundle.manifest_sha256,
                    "config_hash": sha256_file(context.config_path),
                    "edge_manifest_sha256": context.upstream.sha256 if context.upstream else "",
                    "edge_order_hash": edge_order_hash}.items():
                if shortcut_gate.get(field) != expected:
                    raise ValueError(f"shortcut/recurrence gate {field} mismatch")
            retained_units = {
                "cluster": set(edges["cluster_id"].dropna().astype(int)),
                "sae": set(edge_activations.get("feature_id", pd.Series(dtype=int)).astype(int))}
            edge_audit = apply_shortcut_recurrence_gate(
                upstream_statuses, shortcut_gate, retained_units=retained_units)
            effective_edge_mask = (context.upstream.manifest.get("input_mask", {}).copy()
                                   if context.upstream else {})
            for view, status in edge_audit["view_statuses"].items():
                effective_edge_mask[f"edge_{view}"] = status in {"pass", "surface"}
        unit_statuses = edge_audit["unit_statuses"]
        if effective_edge_mask.get("edge_cluster", False):
            retained_clusters = {int(unit) for unit, status in unit_statuses.get("cluster", {}).items()
                                 if status in {"pass", "surface"}}
            failed = ~edges["cluster_id"].isin(retained_clusters) & edges["cluster_id"].notna()
            edges = edges.copy()
            edges.loc[failed, ["cluster_id", "cluster_probability"]] = None
        if not effective_edge_mask.get("edge_cluster", False):
            edges = edges.copy()
            edges["cluster_id"], edges["cluster_probability"] = None, None
        if effective_edge_mask.get("edge_sae", False) and not edge_activations.empty:
            retained_features = {int(unit) for unit, status in unit_statuses.get("sae", {}).items()
                                 if status in {"pass", "surface"}}
            edge_activations = edge_activations[
                edge_activations["feature_id"].isin(retained_features)].reset_index(drop=True)
        if not effective_edge_mask.get("edge_sae", False):
            edge_activations = pd.DataFrame()
        retained_edge_features = sorted(
            int(unit) for unit, status in unit_statuses.get("sae", {}).items()
            if effective_edge_mask.get("edge_sae", False) and status in {"pass", "surface"})
        retained_edge_clusters = sorted(
            int(unit) for unit, status in unit_statuses.get("cluster", {}).items()
            if effective_edge_mask.get("edge_cluster", False) and status in {"pass", "surface"})
        if effective_edge_mask.get("edge_dense", False):
            final_edge_transcripts, final_edge_features = aggregate_edge_transcripts(
                edges.reset_index(drop=True), edge_dense, edge_activations, nodes,
                feature_ids=retained_edge_features, cluster_ids=retained_edge_clusters)
        else:
            final_edge_transcripts = pd.read_parquet(
                context.input_dir / "edge_transcript_aggregates.parquet")[
                    ["transcript_key", "edge_scope", "edge_count", "item_key", "split"]]
            final_edge_features = pd.DataFrame(columns=[
                "transcript_key", "edge_scope", "item_key", "split", "feature_id", "mean",
                "proportion_active", "median", "q90", "ancestry_contrast"])
        for name, status in edge_audit["view_statuses"].items():
            final_edge_transcripts[f"{name}_status"] = status
            final_edge_features[f"{name}_status"] = status
        for summary in (final_edge_transcripts, final_edge_features):
            summary["edge_summary_status"] = "final_audited"
            summary["surface_labels"] = json.dumps(edge_audit["surface_labels"])
        circuits = []
        for transcript_key, node_group in nodes.groupby("transcript_key", sort=True):
            extracted = extract_circuits(
                node_group.sort_values("node_id").to_dict("records"),
                edges[edges["transcript_key"].eq(transcript_key)].to_dict("records"),
                config_hash=sha256_file(context.config_path),
                max_local_nodes=context.config["circuits"]["local_node_cap"],
                max_final_nodes=context.config["circuits"]["final_ancestry_node_cap"],
                max_learned=context.config["circuits"]["learned_cap_per_transcript"])
            circuits.extend(row | {"split": node_group.iloc[0]["split"],
                                   "item_key": node_group.iloc[0]["item_key"]} for row in extracted)
        edge_rows = {row.edge_id: int(row.dense_row) for row in edges.itertuples()}
        node_rows = {(row.transcript_key, row.node_id): i for i, row in enumerate(nodes.itertuples())}
        edge_records = edges.set_index("edge_id").to_dict("index")
        kinds = ("question", "assistant_edu", "equation", "code", "tool_call", "tool_output",
                 "user_feedback")
        raw = []
        for row in circuits:
            topology = [row["node_count"], row["edge_count"], row["depth"], row["branch_count"],
                        row["convergence_count"], row["truncated"], row["final_ancestry"],
                        row["abandoned_sibling"], row["tool_motif"], row["correction_motif"],
                        row["verification_motif"]]
            members = nodes[nodes["transcript_key"].eq(row["transcript_key"]) &
                            nodes["node_id"].isin(row["node_ids"])]
            kind_proportions = [float(np.mean(members["kind"].eq(kind))) for kind in kinds]
            pooled_nodes = np.mean([node_embeddings[node_rows[row["transcript_key"], node_id]]
                                    for node_id in row["node_ids"]], axis=0)
            pooled_edges = (np.mean([edge_dense[edge_rows[e]] for e in row["edge_ids"]], axis=0)
                            if row["edge_ids"] else np.zeros(edge_dense.shape[1]))
            cluster_proportions = np.zeros(max(context.config["edge_views"]["cluster_counts"]))
            for edge_id in row["edge_ids"]:
                cluster = edge_records[edge_id].get("cluster_id")
                if pd.notna(cluster):
                    cluster_proportions[int(cluster)] += 1
            if row["edge_ids"]:
                cluster_proportions /= len(row["edge_ids"])
            sae_means = np.zeros(context.config["edge_views"]["sae_width"])
            if not edge_activations.empty and row["edge_ids"]:
                selected = edge_activations[edge_activations["edge_id"].isin(row["edge_ids"])]
                for feature, group in selected.groupby("feature_id"):
                    sae_means[int(feature)] = group["activation"].sum() / len(row["edge_ids"])
            raw.append(np.r_[topology, kind_proportions, pooled_nodes, pooled_edges,
                             cluster_proportions, sae_means])
        raw = np.asarray(raw, dtype=np.float32)
        train_mask = np.array([row["split"] == "representation_train" for row in circuits])
        validation_mask = np.array([row["split"] == "representation_validation" for row in circuits])
        if context.confirmation_apply:
            models = development_models
            geometry, clusters, sae = models["geometry"], models["clusters"], models["sae"]
            statuses, input_mask, fit = models["statuses"], models["fit_input_mask"], models
            dense = geometry.transform(raw) if geometry is not None else np.empty((len(raw), 0))
            gate_sha256 = None
        else:
            encoder_gate = json.loads((gate_dir / "circuit_encoder_gate.json").read_text())
            for field, expected in {
                    "freeze_manifest_sha256": context.bundle.manifest_sha256,
                    "config_hash": sha256_file(context.config_path),
                    "edge_manifest_sha256": context.upstream.sha256 if context.upstream else "",
                    "input_mask": effective_edge_mask}.items():
                if encoder_gate.get(field) != expected:
                    raise ValueError(f"circuit encoder gate {field} mismatch")
            encoder_pass = validate_circuit_encoder_gate(
                encoder_gate, expected_input_mask=effective_edge_mask)
            gate_sha256 = sha256_file(gate_dir / "circuit_encoder_gate.json")
            try:
                if not encoder_pass:
                    raise ValueError("circuit encoder validation gate failed")
                geometry = fit_scaled_pca(raw[train_mask], context.config["circuits"]["dense_pca_components"])
                dense = geometry.transform(raw)
                train, validation = dense[train_mask], dense[validation_mask]
                train_groups = np.array([row["item_key"] for row, keep in zip(circuits, train_mask) if keep])
                fit = fit_independent_views(train, validation, train_groups,
                                            context.config["circuits"], context.config["gates"],
                                            dense_pass=True)
            except ValueError as error:
                geometry, dense = None, np.empty((len(raw), 0))
                statuses, input_mask = resolve_view_mask(dense_pass=False, cluster_pass=None, sae_pass=None)
                fit = {"clusters": None, "sae": None, "statuses": statuses,
                       "input_mask": input_mask, "errors": {"dense": str(error)}}
            clusters, sae = fit["clusters"], fit["sae"]
            statuses, input_mask = fit["statuses"], fit["input_mask"]
        if context.confirmation_apply:
            development_manifest = json.loads(
                (context.apply_from / "artifact_manifest.json").read_text())
            retained, retained_sha256 = _load_feature_inventory(
                models, development_manifest, "circuit", sae)
        else:
            retained, retained_sha256 = _feature_inventory(sae, [])
        probabilities = clusters.predict_proba(dense) if clusters is not None else None
        activations = sae.transform(dense) if sae is not None else None
        assignments = probabilities.argmax(1) if probabilities is not None else [None] * len(circuits)
        for dense_row, (row, cluster) in enumerate(zip(circuits, assignments)):
            row.update(parallel_view_fields(dense_row if input_mask["dense"] else None,
                                            int(cluster) if cluster is not None else None,
                                            sae.fit_id if sae is not None else None,
                                            sae.dictionary_id if sae is not None else None,
                                            float(probabilities[dense_row, cluster])
                                            if cluster is not None else None))
        frame = pd.DataFrame(circuits)
        activation = (_activation_frame(frame["circuit_id"].astype(str).tolist(), activations, sae, "circuit_id")
                      if sae is not None else pd.DataFrame(columns=["circuit_id", "sae_fit_id",
                                                                   "dictionary_id", "feature_id",
                                                                   "activation"]))
        if not context.confirmation_apply and sae is not None:
            train_features = [feature for vector in activations[train_mask] for feature in np.flatnonzero(vector)]
            validation_features = [feature for vector in activations[validation_mask]
                                   for feature in np.flatnonzero(vector)]
            retained = retained_across_splits(
                train_features, validation_features,
                support_floor(train_mask.sum(), context.config["circuits"]["sae_support_rate"]),
                support_floor(validation_mask.sum(), context.config["circuits"]["sae_support_rate"]))
            retained, retained_sha256 = _feature_inventory(
                sae, set(retained) & fit["stable_features"])
        if sae is not None:
            activation = activation[activation["feature_id"].isin(retained)].reset_index(drop=True)
        transcript_aggregates, feature_aggregates = aggregate_circuit_transcripts(
            frame.reset_index(drop=True), dense, activation,
            feature_ids=retained if input_mask["sae"] else [])
        validate_artifact_rows("circuits", frame.to_dict("records"))
        validate_artifact_rows("circuit_sae_activations", activation.to_dict("records"))
        paths = {"circuits.parquet": directory / "circuits.parquet",
                 "dense_circuits.npy": directory / "dense_circuits.npy",
                 "circuit_sae_activations.parquet": directory / "circuit_sae_activations.parquet",
                 "circuit_view_models.joblib": directory / "circuit_view_models.joblib",
                 "circuit_view_metrics.json": directory / "circuit_view_metrics.json",
                 "circuit_transcript_aggregates.parquet": directory / "circuit_transcript_aggregates.parquet",
                 "circuit_feature_aggregates.parquet": directory / "circuit_feature_aggregates.parquet",
                 "final_edge_transcript_aggregates.parquet": directory / "final_edge_transcript_aggregates.parquet",
                 "final_edge_feature_aggregates.parquet": directory / "final_edge_feature_aggregates.parquet",
                 "shortcut_recurrence_gate.json": directory / "shortcut_recurrence_gate.json",
                 "edge_shortcut_recurrence_status.json": directory / "edge_shortcut_recurrence_status.json"}
        _write_parquet_once(frame, paths["circuits.parquet"])
        _write_npy_once(dense.astype(np.float16), paths["dense_circuits.npy"])
        _write_parquet_once(activation, paths["circuit_sae_activations.parquet"])
        _write_parquet_once(transcript_aggregates, paths["circuit_transcript_aggregates.parquet"])
        _write_parquet_once(feature_aggregates, paths["circuit_feature_aggregates.parquet"])
        _write_parquet_once(final_edge_transcripts, paths["final_edge_transcript_aggregates.parquet"])
        _write_parquet_once(final_edge_features, paths["final_edge_feature_aggregates.parquet"])
        _write_bytes_once(shortcut_gate_path.read_bytes(), paths["shortcut_recurrence_gate.json"])
        _write_json_once(edge_audit, paths["edge_shortcut_recurrence_status.json"])
        if not paths["circuit_view_models.joblib"].exists():
            joblib.dump({"geometry": geometry, "clusters": clusters, "sae": sae,
                         "statuses": statuses, "fit_input_mask": input_mask,
                         "retained_features": retained,
                         "retained_features_sha256": retained_sha256,
                         "sae_refits": fit.get("sae_refits", []), "matches": fit.get("matches", []),
                         "edge_audit": edge_audit, "effective_edge_mask": effective_edge_mask,
                         "shortcut_gate_sha256": sha256_file(shortcut_gate_path),
                         "source_edge_manifest_sha256": shortcut_gate["edge_manifest_sha256"],
                         "source_edge_order_hash": shortcut_gate["edge_order_hash"],
                         "source_edge_sae_feature_ids_sha256": context.upstream.manifest.get(
                             "retained_edge_sae_feature_ids_sha256") if context.upstream else None,
                         "source_edge_view_statuses": upstream_statuses},
                        paths["circuit_view_models.joblib"])
        metrics = ({"metric_split": "representation_validation", "view_statuses": statuses,
                    "errors": fit.get("errors", {}), "cluster_stability": fit.get("cluster_stability", []),
                    "sae": sae_metrics(dense[validation_mask], sae,
                                       [match["cosines"] for match in fit.get("matches", [])])
                    if sae is not None else None,
                    "retained_features": retained}
                   if not context.confirmation_apply else
                   {"applied_circuits": len(frame), "view_statuses": statuses})
        _write_json_once(metrics, paths["circuit_view_metrics.json"])
        frozen_mask = effective_edge_mask | {
            f"circuit_{name}": enabled for name, enabled in input_mask.items()}
        return paths, {"circuits.parquet": frame, "dense_circuits.npy": dense.astype(np.float16),
                       "circuit_sae_activations.parquet": activation,
                       "circuit_view_models.joblib": {"sae_width": len(sae.decoder)
                                                      if sae is not None else 0},
                       "circuit_view_metrics.json": metrics,
                       "circuit_transcript_aggregates.parquet": transcript_aggregates,
                       "circuit_feature_aggregates.parquet": feature_aggregates,
                       "final_edge_transcript_aggregates.parquet": final_edge_transcripts,
                       "final_edge_feature_aggregates.parquet": final_edge_features,
                       "shortcut_recurrence_gate.json": json.loads(
                           paths["shortcut_recurrence_gate.json"].read_text()),
                       "edge_shortcut_recurrence_status.json": edge_audit}, {
                           "view_statuses": statuses, "input_mask": frozen_mask,
                           "audited_edge_view_statuses": edge_audit["view_statuses"],
                           "edge_unit_statuses": edge_audit["unit_statuses"],
                           "surface_labels": edge_audit["surface_labels"],
                           "retained_circuit_sae_feature_ids": retained,
                           "retained_circuit_sae_feature_ids_sha256": retained_sha256,
                           "final_audited_edge_summaries": {
                               "status": "final_audited",
                               "transcript_path": "final_edge_transcript_aggregates.parquet",
                               "transcript_sha256": sha256_file(
                                   paths["final_edge_transcript_aggregates.parquet"]),
                               "feature_path": "final_edge_feature_aggregates.parquet",
                               "feature_sha256": sha256_file(
                                   paths["final_edge_feature_aggregates.parquet"]),
                               "source_edge_manifest_sha256": context.upstream.sha256
                               if context.upstream else None,
                               "audited_development_edge_manifest_sha256": shortcut_gate[
                                   "edge_manifest_sha256"],
                               "shortcut_recurrence_gate_sha256": sha256_file(shortcut_gate_path)},
                           "circuit_encoder_gate_sha256": gate_sha256,
                           "shortcut_recurrence_gate_sha256": sha256_file(shortcut_gate_path)}

    return _atomic_stage(context, produce)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    for name in ("freeze", "store", "split", "reconciliation", "judge-util",
                 "approval", "config", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--confirmation-apply", action="store_true")
    parser.add_argument("--apply-from", type=Path)
    parser.add_argument("--adapter-input", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--parent-manifest-sha256")
    parser.add_argument("--apply-from-manifest-sha256")
    args = parser.parse_args()
    require_heavy_runtime()
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        raise ValueError("--code-commit must be an exact 40-character Git commit")
    bundle = guard_canonical_stage(args.freeze, args.approval, args.stage, store_path=args.store,
                                   split_path=args.split, judge_util_path=args.judge_util,
                                   reconciliation_path=args.reconciliation, verdicts_path=args.verdicts,
                                   c_config_path=args.config)
    config = yaml.safe_load(args.config.read_text())
    role = "confirmation" if args.confirmation_apply else None
    population = bundle.population[bundle.population["split"].eq(role)] if role else bundle.population[
        ~bundle.population["split"].eq("confirmation")]
    bundle = FrozenBundle(bundle.manifest, bundle.manifest_sha256, population.copy())
    root = Path(__file__).resolve().parents[2]
    metadata = {"code_commit": args.code_commit, "package_lock_hash": sha256_file(root / "uv.lock"),
                "judge_util_sha256": sha256_file(args.judge_util), "seed": config["seed"],
                "approval_sha256": sha256_file(args.approval),
                "model_revisions": {"segmenter": config["segmentation"]["primary"]["checkpoint_revision"],
                                    "relation_encoder": config["relations"]["encoder_revision"]}}
    prior = {"attachments": "nodes", "edge-views": "attachments", "circuits": "edge-views"}.get(args.stage)
    upstream = None
    if prior:
        if not args.parent_manifest or not args.parent_manifest_sha256:
            raise ValueError("stage requires an explicitly configured parent manifest path and SHA")
        upstream = load_c_manifest_chain(
            args.parent_manifest, expected_sha256=args.parent_manifest_sha256,
            expected_stage=prior, freeze_manifest_sha256=bundle.manifest_sha256,
            config_hash=sha256_file(args.config), code_hash=c_code_hash())
        if args.input.resolve() != args.parent_manifest.resolve().parent:
            raise ValueError("--input must be the verified parent manifest directory")
        expected_purpose = "confirmation_apply" if args.confirmation_apply else "representation_discovery"
        if upstream.manifest["purpose"] != expected_purpose:
            raise ValueError("upstream C stage purpose does not match this run")
    if args.apply_from:
        if not args.apply_from_manifest_sha256:
            raise ValueError("apply_from requires its configured manifest SHA")
        apply_manifest = args.apply_from / "artifact_manifest.json"
        frozen_apply = load_c_manifest_chain(
            apply_manifest, expected_sha256=args.apply_from_manifest_sha256,
            expected_stage=args.stage,
            freeze_manifest_sha256=bundle.manifest_sha256,
            config_hash=sha256_file(args.config), code_hash=c_code_hash())
        if frozen_apply.manifest.get("purpose") != "representation_discovery":
            raise ValueError("confirmation apply_from must be a frozen development-stage artifact")
    dispatch_stage(args.stage, context=StageContext(bundle, config, args.config, args.input,
                                                     args.output, metadata, args.confirmation_apply,
                                                     args.apply_from, args.stage, upstream, None,
                                                     args.adapter_input))


if __name__ == "__main__":
    main()
