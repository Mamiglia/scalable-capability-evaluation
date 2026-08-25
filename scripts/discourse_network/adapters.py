"""Offline staged adapters for the isolated segmenter and frozen GTE encoder."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import socket
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .nodes import (DEFAULT_C_CONFIG, DEFAULT_JUDGE_UTIL, PROTECTED, FrozenBundle,
                    _snapshot_path, read_freeze, sha256_file, stable_id, validate_freeze)


def require_heavy_runtime(*, env: dict[str, str] | os._Environ[str] = os.environ,
                          hostname: str | None = None) -> None:
    if not env.get("SLURM_JOB_ID"):
        raise RuntimeError("canonical heavy stages are Slurm-only and require SLURM_JOB_ID")
    hostname = hostname or socket.gethostname()
    if hostname.startswith(("gateway", "sgw")):
        raise RuntimeError("canonical heavy stages refuse gateway execution")


def _factory_from_assets(asset_dir: Path):
    manifest = json.loads((asset_dir / "adapter.json").read_text())
    module, function = manifest["factory"].split(":", 1)
    return getattr(importlib.import_module(module), function)(asset_dir)


def verify_asset_bundle(asset_dir: Path) -> str:
    manifest_path = asset_dir / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.get("files", {}).items():
        path = asset_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"staged asset hash mismatch for {name}")
    if not manifest.get("files"):
        raise ValueError("staged asset manifest is empty")
    return sha256_file(manifest_path)


class FrozenSegmenterAdapter:
    def __init__(self, asset_dir: Path, loader: Callable[[Path], Any] = _factory_from_assets):
        self.asset_dir, self.loader, self._segmenter = Path(asset_dir), loader, None

    def _load(self):
        if self._segmenter is None:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._segmenter = self.loader(self.asset_dir)
        return self._segmenter

    def segment_rows(self, spans: list[dict[str, Any]], *, revision: str,
                     config_hash: str) -> list[dict[str, Any]]:
        segmenter, rows = self._load(), []
        for span in spans:
            text = str(span["text"])
            source_text = str(span.get("source_text", text))
            source_hash = hashlib.sha256(source_text.encode()).hexdigest()
            if span.get("source_text_hash", source_hash) != source_hash:
                raise ValueError("segmenter source text/hash mismatch")
            boundaries = segmenter.segment(text)
            cursor = 0
            for local_start, local_end in boundaries:
                if not 0 <= local_start < local_end <= len(text) or local_start < cursor:
                    raise ValueError("segmenter returned invalid or overlapping offsets")
                cursor = local_end
                rows.append({key: span[key] for key in
                             ("transcript_key", "part_id", "subpart_id")} | {
                                 "start": span["start"] + local_start,
                                 "end": span["start"] + local_end,
                                 "source_text_hash": source_hash,
                                 "segmenter_revision": revision,
                                 "segmenter_config_hash": config_hash,
                             } | {key: span[key] for key in ("input_file_sha256", "input_row_hash")
                                  if key in span})
        return rows


def extract_prose_spans(population: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map A assistant subparts to ordinary-prose spans without reparsing role text."""
    rows = []
    for record in population:
        parts = json.loads(record["transcript_parts_json"]) if isinstance(
            record["transcript_parts_json"], str) else record["transcript_parts_json"]
        for part_id, part in enumerate(parts):
            if part.get("role") != "assistant":
                continue
            for subpart_id, subpart in enumerate(part.get("subparts", ())):
                if subpart.get("kind") != "content":
                    continue
                text, cursor = subpart["content"], 0
                source_hash = hashlib.sha256(text.encode()).hexdigest()
                for protected in [*PROTECTED.finditer(text), None]:
                    end = protected.start() if protected else len(text)
                    if text[cursor:end].strip():
                        rows.append({"transcript_key": record["transcript_key"],
                                     "part_id": part_id, "subpart_id": subpart_id,
                                     "start": subpart["start"] + cursor,
                                     "text": text[cursor:end], "source_text": text,
                                     "source_text_hash": source_hash,
                                     "input_file_sha256": record["input_file_sha256"],
                                     "input_row_hash": record["input_row_hash"]})
                    cursor = protected.end() if protected else end
    return rows


def load_segmentation_spans(freeze_path: str | Path, input_path: str | Path, *,
                            split_path: str | Path | None = None,
                            judge_util_path: str | Path = DEFAULT_JUDGE_UTIL,
                            reconciliation_path: str | Path | None = None,
                            c_config_path: str | Path = DEFAULT_C_CONFIG,
                            confirmation_apply: bool = False
                            ) -> tuple[list[dict[str, Any]], FrozenBundle]:
    """Load prose only from the exact hash-verified A transcript-store record."""
    freeze_path, input_path = Path(freeze_path), Path(input_path)
    freeze = read_freeze(freeze_path)
    recorded = _snapshot_path(freeze_path, freeze, "transcript_store")
    if input_path.resolve() != recorded.resolve():
        raise ValueError("segmentation input must be the exact frozen transcript store")
    bundle = validate_freeze(
        freeze_path, store_path=input_path, split_path=split_path,
        judge_util_path=judge_util_path, reconciliation_path=reconciliation_path,
        c_config_path=c_config_path, purpose="representation_discovery")
    wanted = {"confirmation"} if confirmation_apply else {
        "representation_train", "representation_validation"}
    population = bundle.population[bundle.population["split"].isin(wanted)].reset_index(drop=True)
    bound = FrozenBundle(bundle.manifest, bundle.manifest_sha256, population)
    return extract_prose_spans(population.to_dict("records")), bound


class _PunctuationSegmenter:
    def segment(self, text: str):
        rows, cursor = [], 0
        for match in re.finditer(r"[^.!?;]+(?:[.!?;]+|$)", text):
            start, end = match.span()
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                rows.append((start, end))
            cursor = match.end()
        return rows


class _SentenceTransformerEncoder:
    def __init__(self, model_path: Path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(model_path), local_files_only=True,
                                         trust_remote_code=True)

    def encode_nodes(self, texts, *, chunk_tokens=4096, overlap_tokens=256,
                     device=None, dtype=None):
        tokenizer = self.model.tokenizer
        vectors, token_counts, chunk_counts = [], [], []
        stride = chunk_tokens - overlap_tokens
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            spans = []
            for start in range(0, max(1, len(token_ids)), stride):
                spans.append((start, min(start + chunk_tokens, len(token_ids))))
                if spans[-1][1] == len(token_ids):
                    break
            chunks = [token_ids[start:end] for start, end in spans]
            chunk_text = [tokenizer.decode(chunk) for chunk in chunks]
            encoded = np.asarray(self.model.encode(chunk_text, normalize_embeddings=True,
                                                   convert_to_numpy=True, device=device), dtype=np.float32)
            if token_ids:
                coverage = np.zeros(len(token_ids), dtype=int)
                for start, end in spans:
                    coverage[start:end] += 1
                weights = np.array([(1.0 / coverage[start:end]).sum() for start, end in spans])
            else:
                weights = np.ones(1)
            pooled = np.average(encoded, axis=0, weights=weights)
            pooled /= np.linalg.norm(pooled)
            vectors.append(pooled)
            token_counts.append(len(token_ids))
            chunk_counts.append(len(chunks))
        return np.asarray(vectors, dtype=np.float32), token_counts, chunk_counts


class FrozenEmbeddingAdapter:
    def __init__(self, model_path: Path,
                 loader: Callable[[Path], Any] = _SentenceTransformerEncoder):
        self.model_path, self.loader, self._encoder = Path(model_path), loader, None

    def _load(self):
        if self._encoder is None:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._encoder = self.loader(self.model_path)
        return self._encoder

    def encode(self, nodes: list[dict[str, Any]], *, revision: str, chunk_tokens: int = 4096,
               overlap_tokens: int = 256, device: str | None = None,
               dtype: str = "float32") -> tuple[np.ndarray, pd.DataFrame]:
        vectors, token_counts, chunk_counts = self._load().encode_nodes(
            [str(node["semantic_text"]) for node in nodes], chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens, device=device, dtype=dtype)
        if len(vectors) != len(nodes) or not np.isfinite(vectors).all():
            raise ValueError("embedding adapter returned invalid row coverage")
        metadata = pd.DataFrame([{
            "transcript_key": node["transcript_key"], "node_id": node["node_id"],
            "embedding_row": row, "token_count": int(token_counts[row]),
            "chunk_count": int(chunk_counts[row]), "coverage": 1.0,
            "encoder_revision": revision,
        } for row, node in enumerate(nodes)])
        return vectors.astype(np.float32), metadata


def validate_embedding_alignment(nodes: pd.DataFrame, metadata: pd.DataFrame,
                                 vectors: np.ndarray) -> pd.DataFrame:
    keys = ["transcript_key", "node_id"]
    if metadata.duplicated(keys).any() or metadata["embedding_row"].duplicated().any():
        raise ValueError("embedding metadata has duplicate keys or rows")
    aligned = nodes[keys].merge(metadata, on=keys, how="left", validate="one_to_one")
    if (aligned["embedding_row"].isna().any() or sorted(aligned["embedding_row"].astype(int)) !=
            list(range(len(vectors))) or not np.allclose(aligned["coverage"], 1.0)):
        raise ValueError("embedding_row alignment or coverage is incomplete")
    return aligned


def verify_embedding_input(nodes_path: Path, nodes_manifest_path: Path,
                           expected_manifest_sha256: str, freeze_manifest_sha256: str,
                           config_hash: str, code_hash: str):
    from .artifacts import load_c_manifest_chain
    artifact = load_c_manifest_chain(
        nodes_manifest_path, expected_sha256=expected_manifest_sha256, expected_stage="nodes",
        freeze_manifest_sha256=freeze_manifest_sha256, config_hash=config_hash, code_hash=code_hash)
    record = artifact.manifest["files"].get("nodes.parquet")
    if record is None or sha256_file(nodes_path) != record["sha256"]:
        raise ValueError("embedding input nodes.parquet hash mismatch")
    expected_path = nodes_manifest_path.parent / record["path"]
    if nodes_path.resolve() != expected_path.resolve():
        raise ValueError("embedding input path is not the verified nodes artifact")
    return artifact


def write_bytes_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable checkpoint mismatch at {path}")
    else:
        path.write_bytes(content)


def _write_parquet_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    else:
        frame.to_parquet(path, index=False)


def _write_npy_once(array: np.ndarray, path: Path) -> None:
    if path.exists():
        if not np.array_equal(np.load(path), array):
            raise ValueError(f"immutable checkpoint mismatch at {path}")
    else:
        np.save(path, array)


def _write_json_once(value: dict[str, Any], path: Path) -> None:
    write_bytes_once(path, (json.dumps(value, indent=2) + "\n").encode())


def _atomic_adapter(output: Path, writer: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    partial = Path(f"{output}.partial")
    if output.exists():
        raise FileExistsError(f"refusing immutable adapter output {output}")
    partial.mkdir(parents=True, exist_ok=True)
    manifest = writer(partial)
    partial.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("segment", "embed"))
    parser.add_argument("--system", choices=("isanlp_rst", "punctuation_fallback"),
                        default="isanlp_rst")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--freeze-manifest-sha256", required=True)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--judge-util", type=Path, default=DEFAULT_JUDGE_UTIL)
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--c-config", type=Path, default=DEFAULT_C_CONFIG)
    parser.add_argument("--canonical-content-hash", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--nodes-manifest-sha256")
    parser.add_argument("--nodes-manifest", type=Path)
    parser.add_argument("--c-code-hash")
    parser.add_argument("--gate-metrics", type=Path)
    parser.add_argument("--public-smoke", action="store_true")
    parser.add_argument("--confirmation-apply", action="store_true")
    args = parser.parse_args()
    if not args.public_smoke:
        require_heavy_runtime()
    approval_hash = sha256_file(args.approval)
    assets_manifest_sha256 = verify_asset_bundle(args.assets)
    approval = json.loads(args.approval.read_text())
    if not args.public_smoke and (approval.get("design_approved") is not True or
            approval.get("annotation_budget_approved") is not True or
            len(set(filter(None, approval.get("named_annotators", ())))) < 2 or
            not 150 <= approval.get("annotation_budget_hours", 0) <= 350):
        raise ValueError("canonical adapter requires the frozen design/annotator/budget approval")
    if args.kind == "segment":
        if args.freeze_manifest is None:
            raise ValueError("segmentation requires the immutable A freeze manifest")
        if args.gate_metrics is None:
            raise ValueError("canonical segmentation requires frozen gate metrics")
        gate = json.loads(args.gate_metrics.read_text())
        if not args.public_smoke and (gate.get("overall_f1", 0) < 0.90 or gate.get("subset_min_f1", 0) < 0.85):
            raise ValueError("segmenter blind gate failed")
        if not args.public_smoke and gate.get("selected_system") != args.system:
            raise ValueError("segmenter does not match the blind gate selection")
        spans, frozen_input = load_segmentation_spans(
            args.freeze_manifest, args.input, split_path=args.split,
            judge_util_path=args.judge_util, reconciliation_path=args.reconciliation,
            c_config_path=args.c_config, confirmation_apply=args.confirmation_apply)
        if frozen_input.manifest_sha256 != args.freeze_manifest_sha256:
            raise ValueError("configured freeze-manifest SHA does not match the verified A manifest")
        if frozen_input.manifest["canonical_content_hash"] != args.canonical_content_hash:
            raise ValueError("configured canonical-content hash does not match the verified A manifest")

        def write(directory):
            loader = (lambda _: _PunctuationSegmenter()) if args.system == "punctuation_fallback" else _factory_from_assets
            rows = FrozenSegmenterAdapter(args.assets, loader).segment_rows(
                spans, revision=args.revision, config_hash=args.config_hash)
            path = directory / "boundaries.parquet"
            frame = pd.DataFrame(rows)
            _write_parquet_once(frame, path)
            purpose = "public_smoke" if args.public_smoke else (
                "confirmation_apply" if args.confirmation_apply else "representation_discovery")
            manifest = {"mode": "public_smoke" if args.public_smoke else "canonical",
                        "purpose": purpose,
                        "file_sha256": sha256_file(path), "selected_system": args.system,
                        "revision": args.revision, "config_hash": args.config_hash,
                        "blind_gate_selected": True, "gate_metrics": gate,
                        "gate_metrics_sha256": sha256_file(args.gate_metrics),
                        "gate_metrics_content_hash": stable_id(gate),
                        "approval_sha256": approval_hash,
                        "input_sha256": sha256_file(args.input),
                        "input_file_sha256": frozen_input.manifest["files"]["transcript_store"]["sha256"],
                        "input_row_hashes_sha256": stable_id(sorted(
                            {row["input_row_hash"] for row in spans})),
                        "adapter_code_sha256": sha256_file(__file__),
                        "assets_manifest_sha256": assets_manifest_sha256,
                        "canonical_content_hash": args.canonical_content_hash,
                        "freeze_manifest_sha256": args.freeze_manifest_sha256,
                        "rows": len(frame), "schema_hash": stable_id([
                            {"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns])}
            _write_json_once(manifest, directory / "boundary_manifest.json")
            return manifest
    else:
        if not args.nodes_manifest or not args.nodes_manifest_sha256 or not args.c_code_hash:
            raise ValueError("embedding requires the verified nodes manifest path/SHA and C code hash")
        verify_embedding_input(args.input, args.nodes_manifest, args.nodes_manifest_sha256,
                               args.freeze_manifest_sha256, args.config_hash, args.c_code_hash)
        nodes = pd.read_parquet(args.input)

        def write(directory):
            vectors, metadata = FrozenEmbeddingAdapter(args.assets).encode(
                nodes.to_dict("records"), revision=args.revision)
            _write_npy_once(vectors, directory / "node_embeddings.npy")
            _write_parquet_once(metadata, directory / "node_embedding_metadata.parquet")
            purpose = "public_smoke" if args.public_smoke else (
                "confirmation_apply" if args.confirmation_apply else "representation_discovery")
            manifest = {"mode": "public_smoke" if args.public_smoke else "canonical",
                        "purpose": purpose,
                        "vectors_sha256": sha256_file(directory / "node_embeddings.npy"),
                        "metadata_sha256": sha256_file(directory / "node_embedding_metadata.parquet"),
                        "revision": args.revision, "config_hash": args.config_hash,
                        "approval_sha256": approval_hash,
                        "adapter_code_sha256": sha256_file(__file__),
                        "assets_manifest_sha256": assets_manifest_sha256,
                        "nodes_manifest_sha256": args.nodes_manifest_sha256,
                        "nodes_file_sha256": sha256_file(args.input),
                        "canonical_content_hash": args.canonical_content_hash,
                        "freeze_manifest_sha256": args.freeze_manifest_sha256,
                        "rows": len(nodes), "coverage": 1.0,
                        "vector_schema_hash": stable_id({"dtype": str(vectors.dtype),
                                                         "shape_tail": list(vectors.shape[1:])}),
                        "metadata_schema_hash": stable_id([
                            {"name": column, "dtype": str(metadata[column].dtype)}
                            for column in metadata.columns])}
            _write_json_once(manifest, directory / "embedding_manifest.json")
            return manifest
    _atomic_adapter(args.output, write)


if __name__ == "__main__":
    main()
