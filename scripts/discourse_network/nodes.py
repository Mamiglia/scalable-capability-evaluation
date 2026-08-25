"""Outcome-blind Workstream C input boundary, node construction, and artifact identity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from numbers import Real
from typing import Any, Iterable

import pandas as pd
import numpy as np
import yaml


ROW_KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
SPLITS = {"representation_train", "representation_validation", "confirmation"}
BENCHMARKS = {"gpqa", "aime2024", "aime2025", "aime2026", "intercode", "swe_bench_mini"}
ZERO_RECONCILIATION = {"max_length_unresolved_logs", "unmatched_transcript_logs",
                       "input_mismatch_logs"}
HEX64 = re.compile(r"[0-9a-f]{64}")
PROTECTED = re.compile(r"```[\s\S]*?```|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")
BOUNDARY = re.compile(r"(?<=[.!?;])\s+")
DEFAULT_JUDGE_UTIL = Path(__file__).resolve().parents[2] / "src" / "judge" / "util.py"
DEFAULT_C_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "workstream_c.yaml"

ARTIFACT_REQUIRED = {
    "nodes": {"transcript_key", "node_id", "part_id", "role", "kind", "rendered_text",
              "semantic_text", "target_eligible", "is_final_response", "is_final_sink",
              "rendered_start", "rendered_end", "part_start", "part_end",
              "rendered_text_hash", "semantic_text_hash",
              "source_span_id", "segmenter_name", "segmenter_revision",
              "segmenter_config_hash", "protected_span_kind", "chunk_count", "token_count",
              "coverage", "embedding_row", "input_row_hash", "input_file_sha256"},
    "edges": {"edge_id", "transcript_key", "source_node_id", "target_node_id", "edge_source",
              "dense_row", "cluster_id", "sae_fit_id", "dictionary_id", "probability",
              "cluster_probability", "accepted", "threshold", "maximum_parents"},
    "circuits": {"circuit_id", "transcript_key", "extractor_kind", "node_ids", "edge_ids",
                 "node_count", "edge_count", "dense_row", "cluster_id", "sae_fit_id",
                 "dictionary_id", "precap_count", "true_kind_proportion", "truncated",
                 "cluster_probability", "final_ancestry", "abandoned_sibling", "path_length",
                 "depth", "branch_count", "convergence_count", "tool_motif",
                 "correction_motif", "verification_motif", "start_node_id", "end_node_id"},
    "edge_sae_activations": {"edge_id", "sae_fit_id", "dictionary_id", "feature_id",
                             "activation"},
    "circuit_sae_activations": {"circuit_id", "sae_fit_id", "dictionary_id", "feature_id",
                                "activation"},
}
FORBIDDEN_ARTIFACT_FIELDS = {"correct", "is_canonical", "model", "judge", "capability",
                             "verdict", "pick_left"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"invalid or missing {field}")
    return value


def _record_suffix(record_path: str | Path, freeze_basename: str) -> Path:
    parts = Path(record_path).parts
    positions = [index for index, part in enumerate(parts) if part == freeze_basename]
    if not positions or positions[-1] == len(parts) - 1:
        raise ValueError("freeze file record does not point below its declared freeze directory")
    return Path(*parts[positions[-1] + 1:])


def _snapshot_path(freeze_path: str | Path, freeze: dict[str, Any], name: str) -> Path:
    suffix = _record_suffix(freeze["files"][name]["path"], Path(freeze_path).resolve().parent.name)
    return Path(freeze_path).resolve().parent / suffix


def recompute_freeze_contract_hash(files: dict[str, dict[str, Any]], code_dir: str | Path) -> str:
    code_dir, digest = Path(code_dir), hashlib.sha256()
    names = ("config", "main_code", "surprisal_code", "judge_util_code")
    for name in sorted(names):
        path = code_dir / Path(files[name]["path"]).name
        if sha256_file(path) != files[name]["sha256"]:
            raise ValueError(f"freeze contract snapshot hash mismatch for {name}")
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def read_freeze(path: str | Path) -> dict[str, Any]:
    freeze = json.loads(Path(path).read_text())
    required = {
        "manifest_version", "freeze_contract_hash", "scope", "row_keys",
        "correctness_rule", "population_rule",
        "transcript_schema_hash", "canonical_content_hash", "judge_config_hashes",
        "transcript_store", "item_split", "files",
    }
    if missing := required - set(freeze):
        raise ValueError(f"freeze manifest missing {sorted(missing)}")
    if freeze["manifest_version"] != 1:
        raise ValueError("unsupported Phase 1 manifest_version")
    if set(freeze["scope"].get("benchmarks", ())) != BENCHMARKS:
        raise ValueError("Phase 1 freeze must contain exactly the six Workstream C benchmarks")
    row_keys = freeze["row_keys"]
    if not isinstance(row_keys, dict) or row_keys.get("transcript_store") != ROW_KEYS:
        raise ValueError("Phase 1 row_keys do not match Workstream C")
    if row_keys.get("item_split") != ["bench", "item", "split"]:
        raise ValueError("Phase 1 item_split row keys do not match Workstream C")
    if freeze["correctness_rule"] != "judge.util.extract_score":
        raise ValueError("correctness_rule must be judge.util.extract_score")
    if freeze["population_rule"] != "is_canonical & correct":
        raise ValueError("population_rule must be is_canonical & correct")
    _hex(freeze["transcript_schema_hash"], "transcript_schema_hash")
    _hex(freeze["canonical_content_hash"], "canonical_content_hash")
    _hex(freeze["freeze_contract_hash"], "freeze_contract_hash")
    manifest_path = Path(path).resolve()
    freeze_basename = f"v1-{freeze['canonical_content_hash'][:12]}-{freeze['freeze_contract_hash'][:12]}"
    if (manifest_path.name != "freeze_manifest.json" or manifest_path.parent.name != freeze_basename
            or manifest_path.parent.parent.name != "freezes"):
        raise ValueError("Phase 1 manifest must be an explicitly configured immutable versioned freeze")
    hashes = freeze["judge_config_hashes"]
    if not isinstance(hashes, list) or not hashes or hashes != sorted(set(hashes)):
        raise ValueError("judge_config_hashes must be a nonempty sorted unique list")
    for value in hashes:
        _hex(value, "judge_config_hashes entry")
    split = freeze.get("split", {})
    if {key: split.get(key) for key in SPLITS} != {
            "representation_train": 0.64, "representation_validation": 0.16,
            "confirmation": 0.20}:
        raise ValueError("Phase 1 split must be frozen at 64/16/20")
    if not isinstance(freeze["files"], dict) or not freeze["files"]:
        raise ValueError("freeze manifest files must be a nonempty mapping")
    for name, record in freeze["files"].items():
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"freeze manifest missing files.{name}.path")
        _hex(record.get("sha256"), f"files.{name}.sha256")
    for name in ("transcript_store", "item_split", "judge_util_code", "verdict_reconciliation",
                 "verdicts", "config", "main_code", "surprisal_code"):
        record = freeze["files"].get(name, {})
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"freeze manifest missing files.{name}")
        _hex(record.get("sha256"), f"files.{name}.sha256")
        if name in {"transcript_store", "item_split"} and freeze[name] != record:
            raise ValueError(f"top-level {name} record disagrees with files.{name}")
        suffix = _record_suffix(record["path"], freeze_basename)
        if name in {"config", "main_code", "surprisal_code", "judge_util_code"} and suffix.parent != Path("code"):
            raise ValueError(f"files.{name} must live directly under the freeze code snapshot")
    return freeze


def _verify_file(path: str | Path, freeze: dict[str, Any], name: str) -> None:
    expected = freeze["files"][name]["sha256"]
    record = freeze["files"][name]
    if sha256_file(path) != expected or ("size" in record and Path(path).stat().st_size != record["size"]):
        raise ValueError(f"{name} SHA-256 does not match Phase 1 freeze")


def _verify_reconciliation(path: str | Path, freeze: dict[str, Any], c_config: dict[str, Any],
                           purpose: str) -> None:
    _verify_file(path, freeze, "verdict_reconciliation")
    reconciliation = json.loads(Path(path).read_text())
    if not isinstance(reconciliation, dict):
        raise ValueError("hashed verdict reconciliation must be a counter mapping")
    if purpose == "representation_discovery":
        return
    if purpose != "verdict_join":
        raise ValueError("freeze purpose must be representation_discovery or verdict_join")
    for counter in {"unmatched_transcript_logs", "input_mismatch_logs"}:
        if reconciliation.get(counter, 0) != 0:
            raise ValueError(f"verdict_join blocks unresolved {counter}")
    unresolved = reconciliation.get("max_length_unresolved_logs", 0)
    if unresolved:
        approved = [row for row in c_config.get("approved_exclusions", ())
                    if row.get("canonical_content_hash") == freeze["canonical_content_hash"]
                    and row.get("freeze_contract_hash") == freeze["freeze_contract_hash"]
                    and row.get("benchmark") == "gpqa"
                    and row.get("reason") == "max_length_unresolved"
                    and row.get("excluded_logs") == unresolved == 4
                    and row.get("excluded_rows") == reconciliation.get("max_length_unresolved_rows")]
        if len(approved) != 1:
            raise ValueError("max_length_unresolved_logs lack the exact approved-exclusion pin")


@dataclass(frozen=True)
class FrozenBundle:
    manifest: dict[str, Any]
    manifest_sha256: str
    population: pd.DataFrame


def validate_freeze(freeze_path: str | Path, *, store_path: str | Path | None = None,
                    split_path: str | Path | None = None,
                    judge_util_path: str | Path = DEFAULT_JUDGE_UTIL,
                    reconciliation_path: str | Path | None = None,
                    verdicts_path: str | Path | None = None,
                    c_config_path: str | Path = DEFAULT_C_CONFIG,
                    purpose: str = "representation_discovery") -> FrozenBundle:
    """One fail-closed A boundary shared by all four canonical C stages."""
    freeze = read_freeze(freeze_path)
    store_path = Path(store_path or freeze["files"]["transcript_store"]["path"])
    split_path = Path(split_path or freeze["files"]["item_split"]["path"])
    verdicts_path = Path(verdicts_path or freeze["files"]["verdicts"]["path"])
    _verify_file(store_path, freeze, "transcript_store")
    _verify_file(split_path, freeze, "item_split")
    _verify_file(judge_util_path, freeze, "judge_util_code")
    facet_files = {"transcript_store", "item_split", "judge_util_code", "verdict_reconciliation",
                   "config", "main_code", "surprisal_code"}
    if purpose == "verdict_join":
        facet_files.add("verdicts")
    for name in facet_files:
        snapshot = _snapshot_path(freeze_path, freeze, name)
        if not snapshot.exists():
            raise ValueError(f"immutable freeze snapshot is missing files.{name}")
        _verify_file(snapshot, freeze, name)
    _verify_file(_snapshot_path(freeze_path, freeze, "config"), freeze, "config")
    _verify_file(_snapshot_path(freeze_path, freeze, "main_code"), freeze, "main_code")
    _verify_file(_snapshot_path(freeze_path, freeze, "surprisal_code"), freeze, "surprisal_code")
    c_config = yaml.safe_load(Path(c_config_path).read_text())
    _verify_reconciliation(reconciliation_path or freeze["files"]["verdict_reconciliation"]["path"],
                           freeze, c_config, purpose)
    contract = recompute_freeze_contract_hash(freeze["files"], Path(freeze_path).resolve().parent / "code")
    if contract != freeze["freeze_contract_hash"]:
        raise ValueError("freeze_contract_hash does not match its four code/config snapshots")

    frozen_store = pd.read_parquet(store_path)
    structure = ["question", "transcript", "transcript_parts_json", "correct", "is_canonical"]
    if missing := set(ROW_KEYS + structure) - set(frozen_store):
        raise ValueError(f"transcript store lacks the frozen C schema: {sorted(missing)}")
    schema = [{"name": column, "dtype": str(frozen_store[column].dtype)}
              for column in frozen_store.columns]
    schema_hash = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
    if schema_hash != freeze["transcript_schema_hash"]:
        raise ValueError("transcript schema hash does not match Phase 1 freeze")
    selected = frozen_store[frozen_store["correct"].eq(True) & frozen_store["is_canonical"].eq(True)]
    content_rows = selected[ROW_KEYS].sort_values(ROW_KEYS)
    if hashlib.sha256(content_rows.to_json(orient="records").encode()).hexdigest() != freeze[
            "canonical_content_hash"]:
        raise ValueError("canonical content hash does not match Phase 1 freeze")
    if frozen_store.duplicated(ROW_KEYS).any():
        raise ValueError("transcript store has duplicate frozen row keys")

    if freeze["split"].get("seed") != 0:
        raise ValueError("Phase 1 split seed must be zero")
    item_split = pd.read_csv(split_path, dtype={"bench": str, "item": str})
    if item_split.columns.tolist() != ["bench", "item", "split"]:
        raise ValueError("item split must contain exactly bench,item,split")
    if item_split.duplicated(["bench", "item"]).any() or not set(item_split["split"]) <= SPLITS:
        raise ValueError("item split keys or roles are invalid")
    selected = selected.copy()
    selected["item"] = selected["item"].astype(str)
    selected["input_row_hash"] = [stable_id("canonical_input_row", row)
                                   for row in selected.to_dict("records")]
    population = selected.merge(item_split, on=["bench", "item"], how="left",
                                validate="many_to_one", indicator=True)
    if not (population.pop("_merge") == "both").all():
        raise ValueError("frozen population contains items absent from the item split")

    if purpose == "verdict_join":
        verdicts = pd.read_parquet(verdicts_path)
        verdict_keys = freeze["row_keys"].get("verdicts")
        if not isinstance(verdict_keys, list) or not verdict_keys or not set(verdict_keys) <= set(verdicts):
            raise ValueError("verdict rows do not satisfy their frozen key schema")
        if verdicts.duplicated(verdict_keys).any():
            raise ValueError("verdict store has duplicate draw keys")
        if set(verdicts["bench"].astype(str)) != BENCHMARKS:
            raise ValueError("verdict store must contain all six judged subsets")
        status_columns = [column for column in verdicts if "status" in column.casefold()
                          or "resolution" in column.casefold()]
        if "max_length_resolution" not in verdicts or any(
                verdicts[column].astype(str).str.contains("unresolved", case=False).any()
                for column in status_columns):
            raise ValueError("accepted verdict rows contain unresolved max-length status")

    source_records = population.to_dict("records")
    population["transcript_key"] = [stable_id("transcript", *(row[key] for key in ROW_KEYS))
                                      for row in source_records]
    population["item_key"] = [stable_id("item", row["bench"], row["item"])
                                for row in source_records]
    population["input_file_sha256"] = freeze["files"]["transcript_store"]["sha256"]
    keep = ["transcript_key", "item_key", "question", "transcript", "transcript_parts_json", "split",
            "input_row_hash", "input_file_sha256"]
    return FrozenBundle(freeze, sha256_file(freeze_path),
                        population[keep].sort_values("transcript_key").reset_index(drop=True))


def load_frozen_population(store_path: str | Path, split_path: str | Path,
                           freeze_path: str | Path, splits: set[str], *,
                           confirmation_apply: bool = False,
                           judge_util_path: str | Path = DEFAULT_JUDGE_UTIL,
                           reconciliation_path: str | Path | None = None,
                           c_config_path: str | Path = DEFAULT_C_CONFIG,
                           purpose: str = "representation_discovery") -> pd.DataFrame:
    """Verify A's freeze, then expose structure under opaque transcript identities only."""
    if not splits or not splits <= SPLITS:
        raise ValueError(f"splits must be a nonempty subset of {sorted(SPLITS)}")
    if "confirmation" in splits and not confirmation_apply:
        raise ValueError("confirmation is apply-only and unavailable to discovery")
    bundle = validate_freeze(freeze_path, store_path=store_path, split_path=split_path,
                             judge_util_path=judge_util_path,
                             reconciliation_path=reconciliation_path, c_config_path=c_config_path,
                             purpose=purpose)
    return bundle.population[bundle.population["split"].isin(splits)].reset_index(drop=True)


def guard_canonical_stage(freeze_path: str | Path, approval_path: str | Path,
                          stage: str, *, store_path: str | Path | None = None,
                          split_path: str | Path | None = None,
                          judge_util_path: str | Path = DEFAULT_JUDGE_UTIL,
                          reconciliation_path: str | Path | None = None,
                          verdicts_path: str | Path | None = None,
                          c_config_path: str | Path = DEFAULT_C_CONFIG,
                          purpose: str = "representation_discovery") -> FrozenBundle:
    bundle = validate_freeze(freeze_path, store_path=store_path, split_path=split_path,
                             judge_util_path=judge_util_path,
                             reconciliation_path=reconciliation_path,
                             verdicts_path=verdicts_path, c_config_path=c_config_path, purpose=purpose)
    approval = json.loads(Path(approval_path).read_text())
    if not approval.get("design_approved"):
        raise ValueError("Workstream C design approval is required")
    annotators = approval.get("named_annotators", [])
    if len(set(filter(None, annotators))) < 2:
        raise ValueError("two named annotators are required")
    if not approval.get("annotation_budget_approved"):
        raise ValueError("annotation budget approval is required")
    hours = approval.get("annotation_budget_hours", 0)
    if not isinstance(hours, (int, float)) or not 150 <= hours <= 350:
        raise ValueError("approved annotation budget must be within 150--350 hours")
    if stage not in {"nodes", "attachments", "edge-views", "circuits"}:
        raise ValueError(f"unknown Workstream C stage: {stage}")
    return bundle


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _assistant_spans(text: str) -> list[tuple[int, int, str]]:
    protected = [(match.start(), match.end(), "code" if match.group().startswith("```") else "equation")
                 for match in PROTECTED.finditer(text)]
    spans, cursor = [], 0
    for start, end, kind in protected:
        for piece in BOUNDARY.split(text[cursor:start]):
            local = text.find(piece, cursor, start) if piece else cursor
            trimmed = _trimmed_span(text, local, local + len(piece))
            if trimmed:
                spans.append((*trimmed, "assistant_edu"))
            cursor = local + len(piece)
        spans.append((start, end, kind))
        cursor = end
    for match in re.finditer(r"[^.!?;]+(?:[.!?;]+|$)", text[cursor:]):
        trimmed = _trimmed_span(text, cursor + match.start(), cursor + match.end())
        if trimmed:
            spans.append((*trimmed, "assistant_edu"))
    return sorted(spans)


def _semantic_text(subpart: dict[str, Any], rendered: str) -> str:
    value = subpart.get("semantic_text", subpart.get("raw_payload"))
    if value is None:
        return rendered
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_nodes(transcript_key: str, question: str, transcript: str,
                parts: list[dict[str, Any]] | str, *,
                prose_boundaries: list[dict[str, Any]] | None = None,
                segmenter_provenance: dict[str, Any] | None = None,
                input_row_hash: str | None = None,
                input_file_sha256: str | None = None) -> list[dict[str, Any]]:
    """Build chronological nodes strictly from A roles/subparts and exact offsets."""
    if isinstance(parts, str):
        parts = json.loads(parts)
    provenance = segmenter_provenance or {
        "system": "punctuation_fallback", "revision": "v1",
        "config_hash": stable_id("punctuation_fallback", "v1"), "blind_gate_selected": True,
    }
    if provenance.get("blind_gate_selected") is not True:
        raise ValueError("segmenter provenance must record blind gate selection")
    for field in ("system", "revision", "config_hash"):
        if not provenance.get(field):
            raise ValueError(f"segmenter provenance missing {field}")
    if prose_boundaries is not None:
        ordered_boundaries = sorted(prose_boundaries, key=lambda row: (row["start"], row["end"]))
        if any(a["end"] > b["start"] for a, b in zip(ordered_boundaries, ordered_boundaries[1:])):
            raise ValueError("segmenter boundaries overlap")
    nodes = [{
        "transcript_key": transcript_key, "node_id": 0, "part_id": None,
        "role": "question", "kind": "question", "rendered_text": question,
        "semantic_text": question, "rendered_start": None, "rendered_end": None,
        "part_start": None, "part_end": None,
        "call_id": None, "function": None, "target_eligible": False,
        "is_final_response": False, "is_final_sink": False,
        "rendered_text_hash": _text_hash(question), "semantic_text_hash": _text_hash(question),
        "source_span_id": "question", "segmenter_name": "structural",
        "segmenter_revision": "question-v1", "segmenter_config_hash": provenance["config_hash"],
        "protected_span_kind": None, "chunk_count": None, "token_count": None, "coverage": None,
        "embedding_row": None,
        "input_row_hash": input_row_hash, "input_file_sha256": input_file_sha256,
        "backtracking_marker": False, "verification_marker": False,
        "backtracking_marker_available": False, "verification_marker_available": False,
    }]
    allowed_roles = {"assistant", "tool", "user"}
    final_assistant = max((i for i, part in enumerate(parts) if part.get("role") == "assistant"),
                          default=None)
    for part_id, part in enumerate(parts):
        role = part.get("role")
        if role not in allowed_roles:
            raise ValueError(f"unmapped A role: {role!r}")
        for subpart_id, subpart in enumerate(part.get("subparts", [])):
            subtype = subpart.get("kind")
            start, end = subpart.get("start"), subpart.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or transcript[start:end] != subpart.get("content"):
                raise ValueError("A subpart offsets do not reconcile with the rendered transcript")
            pieces: Iterable[tuple[int, int, str]]
            if subtype == "content" and role == "assistant":
                protected = _assistant_spans(subpart["content"])
                protected = [piece for piece in protected if piece[2] in {"code", "equation"}]
                if prose_boundaries is None:
                    pieces = _assistant_spans(subpart["content"])
                else:
                    external = []
                    for boundary in prose_boundaries:
                        if boundary.get("part_id") != part_id or boundary.get("subpart_id") != subpart_id:
                            continue
                        if (boundary.get("source_text_hash") != _text_hash(subpart["content"]) or
                                boundary.get("segmenter_revision") != provenance["revision"] or
                                boundary.get("segmenter_config_hash") != provenance["config_hash"]):
                            raise ValueError("segmenter boundary provenance does not match its source span")
                        if input_row_hash is not None and boundary.get("input_row_hash") != input_row_hash:
                            raise ValueError("segmenter boundary input-row hash mismatch")
                        if input_file_sha256 is not None and boundary.get(
                                "input_file_sha256") != input_file_sha256:
                            raise ValueError("segmenter boundary input-file hash mismatch")
                        local_start, local_end = boundary["start"] - start, boundary["end"] - start
                        if not 0 <= local_start < local_end <= len(subpart["content"]):
                            raise ValueError("segmenter boundary lies outside its A subpart")
                        if any(local_start < p_end and local_end > p_start
                               for p_start, p_end, _ in protected):
                            raise ValueError("segmenter boundary crosses a protected span")
                        external.append((local_start, local_end, "assistant_edu", boundary))
                    covered = np.zeros(len(subpart["content"]), dtype=bool)
                    for p_start, p_end, _ in protected:
                        covered[p_start:p_end] = True
                    for p_start, p_end, _, _ in external:
                        covered[p_start:p_end] = True
                    if any(not covered[i] and not char.isspace()
                           for i, char in enumerate(subpart["content"])):
                        raise ValueError("segmenter boundaries do not cover all ordinary prose")
                    pieces = [(*piece, {}) for piece in protected] + external
            elif subtype == "content" and role == "user":
                pieces = [(0, len(subpart["content"]), "user_feedback", {})]
            elif subtype in {"tool_call", "tool_output"}:
                pieces = [(0, len(subpart["content"]), subtype, {})]
            else:
                raise ValueError(f"unmapped A subpart subtype {subtype!r} for role {role!r}")
            for piece in pieces:
                local_start, local_end, kind = piece[:3]
                boundary = piece[3] if len(piece) == 4 else {}
                rendered = subpart["content"][local_start:local_end]
                if not rendered.strip():
                    continue
                semantic = _semantic_text(subpart, rendered)
                structural = kind in {"question", "tool_call", "tool_output", "user_feedback"}
                node = {
                    "transcript_key": transcript_key, "node_id": len(nodes), "part_id": part_id,
                    "role": role, "kind": kind, "rendered_text": rendered,
                    "semantic_text": semantic,
                    "rendered_start": start + local_start, "rendered_end": start + local_end,
                    "part_start": start + local_start - part["start"],
                    "part_end": start + local_end - part["start"],
                    "call_id": subpart.get("call_id"), "function": subpart.get("function"),
                    "target_eligible": kind in {"assistant_edu", "equation", "code", "tool_call"},
                    "is_final_response": part_id == final_assistant,
                    "is_final_sink": False,
                    "rendered_text_hash": _text_hash(rendered),
                    "semantic_text_hash": _text_hash(semantic),
                    "source_span_id": f"{part_id}:{subpart_id}",
                    "segmenter_name": "structural" if structural or kind in {"code", "equation"}
                    else provenance["system"],
                    "segmenter_revision": provenance["revision"],
                    "segmenter_config_hash": provenance["config_hash"],
                    "protected_span_kind": kind if kind in {"code", "equation"} else None,
                    "chunk_count": None, "token_count": None, "coverage": None,
                    "embedding_row": None,
                    "input_row_hash": input_row_hash, "input_file_sha256": input_file_sha256,
                    "backtracking_marker": bool(boundary.get(
                        "backtracking_marker", subpart.get("backtracking_marker", False))),
                    "verification_marker": bool(boundary.get(
                        "verification_marker", subpart.get("verification_marker", False))),
                    "backtracking_marker_available": bool(boundary.get(
                        "backtracking_marker_available", "backtracking_marker" in boundary or
                        "backtracking_marker" in subpart)),
                    "verification_marker_available": bool(boundary.get(
                        "verification_marker_available", "verification_marker" in boundary or
                        "verification_marker" in subpart)),
                }
                nodes.append(node)
    assistant = [node for node in nodes if node["is_final_response"]]
    if assistant:
        assistant[-1]["is_final_sink"] = True
    return nodes


def validate_artifact_rows(kind: str, rows: Iterable[dict[str, Any]]) -> None:
    required = ARTIFACT_REQUIRED.get(kind)
    if required is None:
        raise ValueError(f"unknown artifact schema: {kind}")
    rows = list(rows)
    seen = set()
    key_fields = {"nodes": ("transcript_key", "node_id"), "edges": ("edge_id",),
                  "circuits": ("circuit_id",),
                  "edge_sae_activations": ("edge_id", "sae_fit_id", "dictionary_id", "feature_id"),
                  "circuit_sae_activations": ("circuit_id", "sae_fit_id", "dictionary_id", "feature_id")}[kind]
    for number, row in enumerate(rows):
        forbidden = [field for field in row if any(re.search(
            rf"(?:^|_){re.escape(term)}(?:$|_)", field.casefold())
            for term in FORBIDDEN_ARTIFACT_FIELDS)]
        if forbidden:
            raise ValueError(f"{kind} row {number} contains forbidden fields {sorted(forbidden)}")
        if missing := required - set(row):
            raise ValueError(f"{kind} row {number} missing {sorted(missing)}")
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"{kind} contains a duplicate primary/activation key")
        seen.add(key)
        if kind == "nodes" and row["target_eligible"] != (
                row["kind"] in {"assistant_edu", "equation", "code", "tool_call"}):
            raise ValueError("node target_eligible disagrees with kind")
        if kind == "nodes":
            if row["rendered_text_hash"] != _text_hash(row["rendered_text"]):
                raise ValueError("node rendered-text hash mismatch")
            if row["semantic_text_hash"] != _text_hash(row["semantic_text"]):
                raise ValueError("node semantic-text hash mismatch")
            offsets = (row["rendered_start"], row["rendered_end"])
            if row["kind"] != "question" and not (all(
                    isinstance(value, Real) and np.isfinite(value) and float(value).is_integer()
                    for value in offsets) and offsets[0] < offsets[1]):
                raise ValueError("non-question node needs an exact nonempty rendered offset")
            if row["kind"] != "question" and not (0 <= row["part_start"] < row["part_end"]):
                raise ValueError("non-question node needs exact part-relative offsets")
            if row["coverage"] is not None and not 0 <= float(row["coverage"]) <= 1:
                raise ValueError("node representation coverage must be within [0,1]")
            if row["coverage"] is not None and row["embedding_row"] is None:
                raise ValueError("represented node requires embedding_row provenance")
            for field in ("input_row_hash", "input_file_sha256"):
                if row[field] is not None:
                    _hex(row[field], field)
        if kind == "edges" and row["edge_source"] == "inferred":
            if row["probability"] is None or not 0 <= float(row["probability"]) <= 1:
                raise ValueError("inferred edge probability must be within [0,1]")
        if kind in {"edges", "circuits"} and row["cluster_probability"] is not None and not 0 <= float(
                row["cluster_probability"]) <= 1:
            raise ValueError("cluster probability must be within [0,1]")


def write_artifact_manifest(output_dir: str | Path, files: dict[str, str | Path], *,
                            upstream_hash: str, split_hash: str, config_hash: str,
                            freeze_manifest_sha256: str, row_counts: dict[str, int],
                            schema_hashes: dict[str, str],
                            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "artifact_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    for value, field in ((upstream_hash, "upstream_hash"), (split_hash, "split_hash"),
                         (config_hash, "config_hash"),
                         (freeze_manifest_sha256, "freeze_manifest_sha256")):
        _hex(value, field)
    if set(row_counts) != set(files) or set(schema_hashes) != set(files):
        raise ValueError("row counts and schema hashes must cover every artifact file")
    if any(not isinstance(count, int) or count < 0 for count in row_counts.values()):
        raise ValueError("artifact row counts must be nonnegative integers")
    for name, value in schema_hashes.items():
        _hex(value, f"schema_hashes.{name}")
    required_metadata = {"code_commit", "package_lock_hash", "judge_util_sha256", "seed",
                         "model_revisions"}
    if not isinstance(metadata, dict) or (missing := required_metadata - set(metadata)):
        raise ValueError(f"artifact metadata missing {sorted(missing) if isinstance(metadata, dict) else sorted(required_metadata)}")
    _hex(metadata["package_lock_hash"], "package_lock_hash")
    _hex(metadata["judge_util_sha256"], "judge_util_sha256")
    if not metadata["code_commit"] or not isinstance(metadata["seed"], int) or not isinstance(
            metadata["model_revisions"], dict):
        raise ValueError("invalid code_commit, seed, or model_revisions")
    if any(not Path(path).resolve().is_relative_to(output_dir.resolve()) for path in files.values()):
        raise ValueError("artifact files must live below their manifest directory")
    records = {name: {"path": str(Path(path).resolve().relative_to(output_dir.resolve())),
                      "sha256": sha256_file(path),
                      "size": Path(path).stat().st_size} for name, path in sorted(files.items())}
    manifest = {"schema_version": "workstream-c-v1", "upstream_hash": upstream_hash,
                "split_hash": split_hash, "config_hash": config_hash,
                "freeze_manifest_sha256": freeze_manifest_sha256,
                "row_counts": row_counts, "schema_hashes": schema_hashes, "files": records}
    manifest.update(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
