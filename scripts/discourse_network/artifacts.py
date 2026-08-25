"""Verified Workstream C manifest chaining."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .nodes import sha256_file, stable_id


@dataclass(frozen=True)
class CArtifact:
    path: Path
    manifest: dict[str, Any]
    sha256: str


def c_code_hash(directory: str | Path | None = None) -> str:
    directory = Path(directory or Path(__file__).parent)
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py"), key=lambda item: item.name):
        if path.name.startswith("test_"):
            continue
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _shape(path: Path) -> tuple[int, str] | None:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        schema = [{"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns]
        return len(frame), stable_id(schema)
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        return len(array), stable_id({"dtype": str(array.dtype), "shape_tail": list(array.shape[1:])})
    if path.suffix == ".json":
        return 1, stable_id(json.loads(path.read_text()))
    return None


def load_c_manifest(path: str | Path, *, expected_stage: str,
                    freeze_manifest_sha256: str, config_hash: str, code_hash: str,
                    upstream_c_manifest_sha256: str | None) -> CArtifact:
    path = Path(path)
    manifest = json.loads(path.read_text())
    expected = {"schema_version": "workstream-c-v1", "stage": expected_stage,
                "freeze_manifest_sha256": freeze_manifest_sha256, "config_hash": config_hash,
                "code_hash": code_hash,
                "upstream_c_manifest_sha256": upstream_c_manifest_sha256}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"C manifest {field} mismatch")
    if manifest.get("purpose") not in {"representation_discovery", "confirmation_apply"}:
        raise ValueError("C manifest purpose is missing or invalid")
    if not isinstance(manifest.get("reconciliation_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", manifest["reconciliation_sha256"]):
        raise ValueError("C manifest reconciliation hash is missing")
    files, counts, schemas = manifest.get("files", {}), manifest.get("row_counts", {}), manifest.get(
        "schema_hashes", {})
    if not files or set(files) != set(counts) or set(files) != set(schemas):
        raise ValueError("C manifest files/counts/schemas do not align")
    for name, record in files.items():
        file_path = path.parent / record["path"]
        if not file_path.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("C artifact path escapes its immutable directory")
        if (not file_path.is_file() or sha256_file(file_path) != record["sha256"] or
                file_path.stat().st_size != record["size"]):
            raise ValueError(f"C artifact hash/size mismatch for {name}")
        shape = _shape(file_path)
        if shape is not None and shape != (counts[name], schemas[name]):
            raise ValueError(f"C artifact row/schema mismatch for {name}")
        if shape is None and counts[name] != 1:
            raise ValueError(f"opaque C artifact {name} must have row_count=1")
    return CArtifact(path, manifest, sha256_file(path))


def load_c_manifest_chain(path: str | Path, *, expected_sha256: str, expected_stage: str,
                          freeze_manifest_sha256: str, config_hash: str,
                          code_hash: str) -> CArtifact:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("actual parent manifest SHA does not match the configured parent SHA")
    manifest = json.loads(path.read_text())
    upstream_sha = manifest.get("upstream_c_manifest_sha256")
    upstream_path = manifest.get("upstream_c_manifest_path")
    parent_stage = {"attachments": "nodes", "edge-views": "attachments",
                    "circuits": "edge-views"}.get(expected_stage)
    if upstream_sha is None:
        if upstream_path is not None or parent_stage is not None:
            raise ValueError("parent manifest chain terminates at the wrong stage")
    else:
        if not upstream_path or parent_stage is None:
            raise ValueError("parent manifest path/stage identity is missing")
        parent_path = Path(upstream_path)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent = load_c_manifest_chain(
            parent_path, expected_sha256=upstream_sha, expected_stage=parent_stage,
            freeze_manifest_sha256=freeze_manifest_sha256, config_hash=config_hash,
            code_hash=code_hash)
        if parent.sha256 != upstream_sha:
            raise ValueError("recorded upstream SHA disagrees with the actual parent manifest")
    return load_c_manifest(
        path, expected_stage=expected_stage, freeze_manifest_sha256=freeze_manifest_sha256,
        config_hash=config_hash, code_hash=code_hash,
        upstream_c_manifest_sha256=upstream_sha)
