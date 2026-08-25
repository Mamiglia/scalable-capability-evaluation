"""Frozen, graph-free transcript embedding helpers for Workstream B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
SPLIT_ROLES = {"representation_train", "representation_validation", "confirmation"}
EXPECTED_BENCHMARKS = ["gpqa", "aime2024", "aime2025", "aime2026",
                       "intercode", "swe_bench_mini"]
EXPECTED_MODELS = sorted([
    "claude-opus-4.7", "claude-sonnet-4.5", "deepseek-r1-0528", "deepseek-v3.2",
    "gemini-3.5-flash", "gemma-3-12b-it", "gemma-4-e2b-it", "granite-4.1-8b",
    "ling-2.6-flash", "muse-glimmer-30b", "mistral-medium-3", "kimi-k2.6",
    "kimi-k3", "nvidia-nemotron-3.5-lightning-30b-a3b-bf16",
    "nvidia-nemotron-3-ultra-550b-a55b-bf16", "gpt-4.1", "gpt-5", "o3",
    "qwen3-32b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.8-2.4t-a95b",
    "qwen3.8-27b", "glm-4.6", "glm-5.2", "glm-5.3",
])
EXPECTED_SPLIT = {"seed": 0, "representation_train": 0.64,
                  "representation_validation": 0.16, "confirmation": 0.20}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    context: int
    chunk_length: int
    overlap: int
    question_envelope: int
    pool: str
    hidden_layers: tuple[int, ...]
    padding_side: str
    dimension: int | None = None
    license: str | None = None


@dataclass(frozen=True)
class ChunkInputs:
    start: int
    end: int
    transcript_chunk_ids: list[int]
    transcript_ids: list[int]
    conditioned_ids: list[int]
    question_ids: list[int]
    question_original_tokens: int
    question_retained_tokens: int
    question_truncated: bool
    empty_transcript: bool


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_key_hash(keys) -> str:
    return hashlib.sha256(json.dumps(
        list(keys), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(path).resolve().parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _require_sha256(value, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid frozen {field}")


def verify_artifact_files(directory: str | Path, manifest: dict) -> None:
    """Fail closed before consuming files named by an artifact manifest."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    checksum_path = directory / "manifest.sha256"
    if (not manifest_path.is_file() or not checksum_path.is_file()
            or checksum_path.read_text().strip() != sha256_file(manifest_path)):
        raise ValueError("artifact manifest checksum mismatch")
    if manifest != json.loads(manifest_path.read_text()):
        raise ValueError("artifact manifest object does not match its checksummed file")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("artifact manifest has no output hashes")
    for name, expected in outputs.items():
        _require_sha256(expected, f"outputs.{name}")
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")


def atomic_artifact_directory(output_dir: str | Path, build) -> dict:
    """Build beside the final path, write manifest last, then atomically rename."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.staging"
    if stage.exists():
        raise FileExistsError(f"failed staging directory requires inspection: {stage}")
    stage.mkdir()
    frozen = json_ready(build(stage))
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(
        json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    (stage / "manifest.sha256").write_text(sha256_file(manifest_path) + "\n")
    stage.rename(output_dir)
    return frozen


def _frozen_file(record: dict, freeze_path: Path) -> Path:
    """Resolve A's recorded path, or the immutable staged artifact by basename."""
    path = Path(record.get("path", ""))
    if path.is_file():
        return path
    sibling = freeze_path.parent / path.name
    if sibling.is_file():
        return sibling
    raise ValueError(f"frozen file is unavailable: {path.name}")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def chunk_spans(n_tokens: int, length: int, overlap: int) -> list[tuple[int, int]]:
    if n_tokens < 0 or length <= 0 or not 0 <= overlap < length:
        raise ValueError("require n_tokens >= 0 and 0 <= overlap < length")
    if n_tokens == 0:
        return [(0, 0)]
    stride = length - overlap
    spans = []
    for start in range(0, n_tokens, stride):
        spans.append((start, min(start + length, n_tokens)))
        if spans[-1][1] == n_tokens:
            break
    return spans


def overlap_weights(spans: list[tuple[int, int]], n_tokens: int) -> np.ndarray:
    if n_tokens == 0:
        if spans != [(0, 0)]:
            raise ValueError("empty transcript must have one empty chunk")
        return np.ones(1)
    coverage = np.zeros(n_tokens, dtype=np.int32)
    for start, end in spans:
        if not 0 <= start < end <= n_tokens:
            raise ValueError("invalid chunk span")
        coverage[start:end] += 1
    if np.any(coverage == 0):
        raise ValueError("chunk spans leave token-coverage gaps")
    return np.array([(1.0 / coverage[start:end]).sum() for start, end in spans])


def pool_chunks(vectors: np.ndarray, spans: list[tuple[int, int]], n_tokens: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(spans):
        raise ValueError("one vector is required per chunk span")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("chunk vectors must be finite and nonzero")
    weights = overlap_weights(spans, n_tokens)
    pooled = np.average(vectors / norms[:, None], axis=0, weights=weights)
    norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("pooled vector must be finite and nonzero")
    return (pooled / norm).astype(np.float32)


def special_token_frame(tokenizer) -> tuple[list[int], list[int]]:
    """Prefix and suffix special tokens the tokenizer puts around one sequence."""
    probe = tokenizer.encode("a", add_special_tokens=False)
    framed = tokenizer("a")["input_ids"]
    position = framed.index(probe[0])
    return framed[:position], framed[position + len(probe):]


def assemble_inputs(tokenizer, question: str, transcript: str, spec: ModelSpec) -> list[ChunkInputs]:
    transcript_ids = tokenizer.encode(transcript, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    header = tokenizer.encode("Question:\n", add_special_tokens=False)
    footer = tokenizer.encode("\n\nTranscript:\n", add_special_tokens=False)
    prefix, suffix = special_token_frame(tokenizer)
    available = spec.question_envelope - len(header) - len(footer) - len(prefix) - len(suffix)
    if available < 0:
        raise ValueError("conditioning labels exceed the question envelope")
    if len(question_ids) > available:
        head = available // 2
        tail = available - head
        kept_question = question_ids[:head] + (question_ids[-tail:] if tail else [])
    else:
        kept_question = question_ids

    chunks = []
    for start, end in chunk_spans(len(transcript_ids), spec.chunk_length, spec.overlap):
        chunk = transcript_ids[start:end]
        empty = len(transcript_ids) == 0
        plain = prefix + chunk + suffix
        conditioned = prefix + header + kept_question + footer + chunk + suffix
        if not plain:
            raise ValueError("empty transcript did not produce a model-valid input token")
        if len(plain) > spec.context or len(conditioned) > spec.context:
            raise ValueError("assembled input exceeds native context")
        chunks.append(ChunkInputs(
            start, end, chunk, plain, conditioned, kept_question,
            len(question_ids), len(kept_question), len(kept_question) < len(question_ids), empty,
        ))
    return chunks


class FrozenEncoder:
    """Small wrapper for deterministic native pooling and selected-layer hooks."""

    def __init__(self, model, pad_token_id: int, pool: str, padding_side: str,
                 hidden_layers: tuple[int, ...]):
        if pool not in {"last", "cls"} or padding_side not in {"left", "right"}:
            raise ValueError("unsupported pooling or padding rule")
        self.model = model.eval()
        self.pad_token_id = pad_token_id
        self.pool = pool
        self.padding_side = padding_side
        self.hidden_layers = hidden_layers

    def _layers(self):
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return self.model.encoder.layer
        raise TypeError("unsupported encoder layer layout")

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            pooled = hidden[:, 0]
        elif self.padding_side == "left":
            pooled = hidden[:, -1]
        else:
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(1) - 1]
        return torch.nn.functional.normalize(pooled.float(), dim=1)

    @torch.inference_mode()
    def encode_batch(self, input_ids: list[list[int]]) -> dict[str | int, np.ndarray]:
        if not input_ids:
            raise ValueError("cannot encode an empty batch")
        width = max(map(len, input_ids))
        rows, masks = [], []
        for ids in input_ids:
            padding = [self.pad_token_id] * (width - len(ids))
            rows.append(padding + ids if self.padding_side == "left" else ids + padding)
            masks.append([0] * len(padding) + [1] * len(ids) if self.padding_side == "left"
                         else [1] * len(ids) + [0] * len(padding))
        device = next(self.model.parameters()).device
        tokens = torch.tensor(rows, dtype=torch.long, device=device)
        mask = torch.tensor(masks, dtype=torch.long, device=device)

        captured, handles = {}, []
        layers = self._layers()
        for number in self.hidden_layers:
            if not 1 <= number <= len(layers):
                raise ValueError(f"hidden layer {number} is outside 1..{len(layers)}")

            def save(_, __, output, number=number):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[number] = self._pool(hidden, mask)

            handles.append(layers[number - 1].register_forward_hook(save))
        try:
            final = self.model(input_ids=tokens, attention_mask=mask).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()

        result = {"final": self._pool(final, mask).cpu().numpy()}
        result.update({number: captured[number].cpu().numpy()
                       for number in self.hidden_layers})
        return result


def extract_dense(rows: pd.DataFrame, tokenizer, encoder: FrozenEncoder, spec: ModelSpec,
                  batch_size: int, max_batch_tokens: int | None = None
                  ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Encode all chunks in deterministic length buckets and pool them by row."""
    if batch_size <= 0 or not {"question", "transcript"} <= set(rows):
        raise ValueError("positive batch_size and question/transcript columns are required")
    jobs, chunk_metadata = [], []
    for row_number, row in enumerate(rows.itertuples(index=False)):
        chunks = assemble_inputs(tokenizer, row.question, row.transcript, spec)
        spans = [(chunk.start, chunk.end) for chunk in chunks]
        n_tokens = chunks[-1].end
        weights = overlap_weights(spans, n_tokens)
        chunk_metadata.append({
            "transcript_tokens": n_tokens,
            "chunk_count": len(chunks),
            "question_original_tokens": chunks[0].question_original_tokens,
            "question_retained_tokens": chunks[0].question_retained_tokens,
            "question_truncated": chunks[0].question_truncated,
            "empty_transcript": chunks[0].empty_transcript,
            "inference_tokens": sum(
                len(chunk.transcript_ids) + len(chunk.conditioned_ids) for chunk in chunks),
            "chunk_spans_json": json.dumps(spans),
            "pooling_weights_json": json.dumps(weights.tolist()),
            "transcript_input_hashes_json": json.dumps([
                hashlib.sha256(np.asarray(chunk.transcript_ids, dtype=np.int64).tobytes()).hexdigest()
                for chunk in chunks]),
            "conditioned_input_hashes_json": json.dumps([
                hashlib.sha256(np.asarray(chunk.conditioned_ids, dtype=np.int64).tobytes()).hexdigest()
                for chunk in chunks]),
        })
        for chunk_number, chunk in enumerate(chunks):
            for view, ids in (("transcript_only", chunk.transcript_ids),
                              ("question_conditioned", chunk.conditioned_ids)):
                job = {"row": row_number, "chunk": chunk_number, "view": view,
                       "ids": ids, "weight": weights[chunk_number]}
                jobs.append(job)

    accumulated = {
        layer: {view: [None] * len(rows)
                for view in ("transcript_only", "question_conditioned")}
        for layer in ("final", *spec.hidden_layers)
    }
    ordered = sorted(jobs, key=lambda job: (len(job["ids"]), job["row"], job["view"], job["chunk"]))
    for batch in batch_jobs(ordered, batch_size, max_batch_tokens):
        outputs = encoder.encode_batch([job["ids"] for job in batch])
        for layer, values in outputs.items():
            for job, value in zip(batch, values, strict=True):
                value = np.asarray(value, dtype=np.float64)
                norm = np.linalg.norm(value)
                if not np.isfinite(value).all() or norm == 0:
                    raise ValueError("chunk vectors must be finite and nonzero")
                current = accumulated[layer][job["view"]][job["row"]]
                weighted = job["weight"] * value / norm
                accumulated[layer][job["view"]][job["row"]] = (
                    weighted if current is None else current + weighted)

    arrays = {}
    for view in ("transcript_only", "question_conditioned"):
        for layer in ("final", *spec.hidden_layers):
            pooled = np.stack(accumulated[layer][view])
            norms = np.linalg.norm(pooled, axis=1)
            if not np.isfinite(pooled).all() or np.any(norms == 0):
                raise ValueError("pooled vectors must be finite and nonzero")
            suffix = "final" if layer == "final" else f"layer_{layer}"
            arrays[f"{view}__{suffix}"] = (pooled / norms[:, None]).astype(np.float32)
    if spec.dimension is not None and any(
            values.shape[1] != spec.dimension for values in arrays.values()):
        raise ValueError("encoder output does not match the frozen dimension")

    metadata = rows[[column for column in (
        "transcript_key", "item_group_key", "benchmark", "split_role", "judge_truncated",
    ) if column in rows]].copy().reset_index(drop=True)
    for column in chunk_metadata[0]:
        metadata[column] = [record[column] for record in chunk_metadata]
    return arrays, metadata


def batch_jobs(jobs: list[dict], batch_size: int,
               max_batch_tokens: int | None) -> list[list[dict]]:
    if batch_size <= 0 or (max_batch_tokens is not None and max_batch_tokens <= 0):
        raise ValueError("batch limits must be positive")
    batches, batch = [], []
    for job in jobs:
        longest = max((len(item["ids"]) for item in (*batch, job)), default=0)
        padded_tokens = longest * (len(batch) + 1)
        if batch and (len(batch) == batch_size or
                      max_batch_tokens is not None and padded_tokens > max_batch_tokens):
            batches.append(batch)
            batch = []
        batch.append(job)
    if batch:
        batches.append(batch)
    return batches


def _row_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denominator, out=np.full(len(left), np.nan),
                     where=denominator > 0)


def _probe_coverage(metadata: pd.DataFrame, arrays: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    failures = []
    required = {"transcript_only__final", "question_conditioned__final"}
    if not required <= set(arrays):
        failures.append("missing_input_view")
    if "transcript_key" not in metadata or metadata["transcript_key"].duplicated().any():
        failures.append("duplicate_or_missing_transcript_key")
    for name, values in arrays.items():
        values = np.asarray(values)
        if values.ndim != 2 or len(values) != len(metadata):
            failures.append(f"{name}:row_alignment")
            continue
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or np.any(norms == 0):
            failures.append(f"{name}:nonfinite_or_zero")
        elif not np.allclose(norms, 1.0, atol=1e-6, rtol=0):
            failures.append(f"{name}:not_unit_norm")
    required_metadata = {
        "transcript_tokens", "chunk_count", "chunk_spans_json",
        "transcript_input_hashes_json", "conditioned_input_hashes_json",
    }
    if not required_metadata <= set(metadata):
        failures.append("missing_chunk_metadata")
        return not failures, failures
    for row in metadata.itertuples(index=False):
        spans = json.loads(row.chunk_spans_json)
        n_tokens = int(row.transcript_tokens)
        cursor = 0
        for start, end in spans:
            if start > cursor or start < 0 or end < start or end > n_tokens:
                failures.append("token_coverage_gap")
                break
            cursor = max(cursor, end)
        if cursor != n_tokens or len(spans) != int(row.chunk_count):
            failures.append("token_coverage_gap")
        if any(len(json.loads(getattr(row, field))) != len(spans) for field in (
                "transcript_input_hashes_json", "conditioned_input_hashes_json")):
            failures.append("input_hash_coverage")
    return not failures, sorted(set(failures))


def evaluate_dense_gate(
    metadata: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    repeated_metadata: pd.DataFrame,
    repeated_arrays: dict[str, np.ndarray],
    overlap_arrays: dict[str, dict[str, np.ndarray] | tuple[pd.DataFrame, dict[str, np.ndarray]]],
    timing: dict[str, float],
    thresholds: dict[str, float],
) -> dict:
    """Evaluate the frozen outcome-blind dense probe gates without model execution."""
    coverage_passed, coverage_failures = _probe_coverage(metadata, arrays)
    if len(metadata) != int(thresholds["determinism_sample"]):
        coverage_passed = False
        coverage_failures.append("wrong_probe_row_count")
    same_metadata = metadata.reset_index(drop=True).equals(
        repeated_metadata.reset_index(drop=True))
    same_arrays = set(arrays) == set(repeated_arrays)
    maximum_difference, minimum_cosine = 0.0, 1.0
    if same_arrays:
        for name in arrays:
            left, right = np.asarray(arrays[name]), np.asarray(repeated_arrays[name])
            if left.shape != right.shape:
                same_arrays = False
                break
            maximum_difference = max(maximum_difference, float(np.max(np.abs(left - right))))
            minimum_cosine = min(minimum_cosine, float(np.nanmin(_row_cosines(left, right))))
    determinism_passed = bool(
        same_metadata and same_arrays
        and maximum_difference <= thresholds["max_abs_difference"]
        and minimum_cosine >= thresholds["min_repeat_cosine"])

    eligible = metadata["chunk_count"].to_numpy() > 1
    overlap_report = {}
    overlap_passed = True
    for label, value in overlap_arrays.items():
        if isinstance(value, tuple):
            sensitivity_metadata, sensitivity = value
            aligned = metadata["transcript_key"].reset_index(drop=True).equals(
                sensitivity_metadata["transcript_key"].reset_index(drop=True))
        else:
            sensitivity, aligned = value, True
        if int(eligible.sum()) < thresholds["min_overlap_rows"]:
            overlap_report[label] = {"status": "not_exercised", "eligible_rows": int(eligible.sum())}
            continue
        cosines = [] if aligned else None
        for name in ("transcript_only__final", "question_conditioned__final"):
            if cosines is None:
                break
            if name not in sensitivity or np.asarray(sensitivity[name]).shape != np.asarray(arrays[name]).shape:
                cosines = []
                break
            cosines.extend(_row_cosines(
                np.asarray(arrays[name])[eligible], np.asarray(sensitivity[name])[eligible]).tolist())
        finite = np.asarray(cosines or [])[np.isfinite(cosines or [])]
        if not len(finite):
            item = {"status": "failed", "eligible_rows": int(eligible.sum())}
        else:
            median, fifth = float(np.median(finite)), float(np.percentile(finite, 5))
            passed = (median >= thresholds["overlap_median_cosine"]
                      and fifth >= thresholds["overlap_fifth_percentile_cosine"])
            item = {"status": "passed" if passed else "failed",
                    "eligible_rows": int(eligible.sum()), "median_cosine": median,
                    "fifth_percentile_cosine": fifth}
        overlap_report[label] = item
        overlap_passed &= item["status"] == "passed"

    if timing["elapsed_seconds"] <= 0 or timing["sample_tokens"] <= 0 or timing["gpus"] <= 0:
        raise ValueError("timing values must be positive")
    projected = (timing["elapsed_seconds"] * timing["gpus"]
                 * timing["total_tokens"] / timing["sample_tokens"] / 3600)
    approval_required = False
    passed = coverage_passed and determinism_passed and overlap_passed
    return json_ready({
        "status": "passed" if passed else "failed",
        "coverage": {"passed": coverage_passed, "failures": coverage_failures,
                     "rows": len(metadata)},
        "determinism": {"passed": determinism_passed, "metadata_identical": same_metadata,
                        "max_abs_difference": maximum_difference,
                        "minimum_cosine": minimum_cosine},
        "overlap": overlap_report,
        "projected_gpu_hours": projected,
        "approval_required": approval_required,
    })


def select_probe_rows(rows: pd.DataFrame, qwen_chunk_counts: np.ndarray,
                      qwen_token_counts: np.ndarray, n: int = 256,
                      seed: int = 0) -> pd.DataFrame:
    """Balance Qwen chunk class by benchmark, then fill proportionally."""
    counts = np.asarray(qwen_chunk_counts)
    token_counts = np.asarray(qwen_token_counts)
    if (len(rows) != len(counts) or len(rows) != len(token_counts)
            or not 0 < n <= len(rows) or not rows.index.is_unique or "benchmark" not in rows
            or np.any(counts < 1) or np.any(token_counts < 0)):
        raise ValueError("probe selection requires aligned unique rows and 0 < n <= row count")
    data = rows.copy()
    data["_chunk_class"] = np.minimum(counts, 3)
    deciles = min(10, len(data))
    data["_length_decile"] = pd.qcut(
        pd.Series(token_counts, index=data.index).rank(method="first"),
        deciles, labels=False,
    ).astype(int)
    benchmarks = sorted(data["benchmark"].unique())
    target = n // (len(benchmarks) * 3)
    rng, selected = np.random.default_rng(seed), []
    deficits = []
    for bench in benchmarks:
        for chunk_class in (1, 2, 3):
            indices = np.sort(data.index[
                data["benchmark"].eq(bench) & data["_chunk_class"].eq(chunk_class)
            ].to_numpy())
            take = min(target, len(indices))
            selected.extend(rng.choice(indices, size=take, replace=False).tolist())
            deficits.append((bench, chunk_class, target - take))

    selected_set = set(selected)
    for bench, chunk_class, deficit in deficits:
        candidates = data[
            data["benchmark"].eq(bench) & ~data.index.isin(selected_set)].copy()
        candidates["_distance"] = (candidates["_chunk_class"] - chunk_class).abs()
        for distance in sorted(candidates["_distance"].unique()):
            pool = candidates[candidates["_distance"] == distance].index.to_numpy()
            take = min(deficit, len(pool))
            chosen = rng.choice(pool, size=take, replace=False).tolist()
            selected.extend(chosen)
            selected_set.update(chosen)
            deficit -= take
            if deficit == 0:
                break

    remaining = n - len(selected)
    population = data.groupby(["benchmark", "_length_decile"], sort=True).size()
    while remaining > 0:
        available = data[~data.index.isin(selected_set)]
        selected_groups = data.loc[selected].groupby(
            ["benchmark", "_length_decile"], sort=True).size() if selected else pd.Series(dtype=int)
        groups = sorted(available.groupby(
            ["benchmark", "_length_decile"], sort=True).groups)
        group = max(groups, key=lambda key: (
            n * population[key] / len(data) - selected_groups.get(key, 0),
            tuple(map(str, key)),
        ))
        pool = available.index[
            available["benchmark"].eq(group[0]) & available["_length_decile"].eq(group[1])
        ].to_numpy()
        chosen = rng.choice(pool).item()
        selected.append(chosen)
        selected_set.add(chosen)
        remaining -= 1
    return rows.loc[selected]


def write_dense_artifacts(output_dir: str | Path, rows: pd.DataFrame,
                          arrays: dict[str, np.ndarray], manifest: dict,
                          *, audit_rows: pd.DataFrame | None = None,
                          report_rows: pd.DataFrame | None = None) -> dict:
    if report_rows is not None:
        forbidden = {"model", "bench", "item", "epoch", "log", "sample_epoch",
                     "content_hash", "correct", "is_canonical", "judge", "verdict",
                     "capability", "pick_left"}
        required = {"transcript_key", "item_group_key", "question", "transcript", "split_role"}
        if forbidden & set(report_rows) or not required <= set(report_rows):
            raise ValueError("outcome-free report rows contain forbidden or missing columns")
        if ("transcript_key" not in rows
                or list(report_rows["transcript_key"]) != list(rows["transcript_key"])):
            raise ValueError("report rows must match dense transcript-key order")

    def build(stage: Path) -> dict:
        rows.to_parquet(stage / "rows.parquet", index=False)
        outputs = {"rows.parquet": sha256_file(stage / "rows.parquet")}
        if audit_rows is not None:
            audit_rows.to_parquet(stage / "audit_rows.parquet", index=False)
            outputs["audit_rows.parquet"] = sha256_file(stage / "audit_rows.parquet")
        if report_rows is not None:
            report_rows.to_parquet(stage / "report_rows.parquet", index=False)
            outputs["report_rows.parquet"] = sha256_file(stage / "report_rows.parquet")
        for name, values in arrays.items():
            path = stage / f"{name}.npy"
            np.save(path, np.asarray(values, dtype=np.float32), allow_pickle=False)
            outputs[path.name] = sha256_file(path)
        result = dict(manifest) | {"outputs": outputs}
        if "transcript_key" in rows:
            result["row_order_hash"] = ordered_key_hash(rows["transcript_key"])
        if report_rows is not None:
            result["report_row_order_hash"] = ordered_key_hash(report_rows["transcript_key"])
        return result

    return atomic_artifact_directory(output_dir, build)


def write_dense_gate_report(output_dir: str | Path, report: dict,
                            source_hashes: dict[str, str], *, family: str | None = None,
                            identity: dict | None = None) -> dict:
    def build(stage: Path) -> dict:
        path = stage / "dense_gate.json"
        path.write_text(json.dumps(json_ready(report), sort_keys=True) + "\n")
        return {"kind": "dense_gate", "family": family,
                "gate_status": report.get("status"),
                "dense_identity": identity,
                "source_hashes": source_hashes,
                "outputs": {path.name: sha256_file(path)}}

    return atomic_artifact_directory(output_dir, build)


def load_passed_dense_gate(directory: str | Path, family: str,
                           expected_identity: dict | None = None) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_artifact_files(directory, manifest)
    if (manifest.get("kind") != "dense_gate" or manifest.get("family") != family
            or manifest.get("gate_status") != "passed"):
        raise ValueError(f"full {family} extraction requires a passed dense gate")
    report = json.loads((directory / "dense_gate.json").read_text())
    if report.get("status") != "passed" or report.get("approval_required"):
        raise ValueError(f"full {family} extraction requires a passed dense gate")
    if expected_identity is not None and manifest.get("dense_identity") != json_ready(expected_identity):
        raise ValueError(f"full {family} extraction requires a matching dense gate identity")
    return manifest


def load_dense_probe(directory: str | Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_artifact_files(directory, manifest)
    rows = pd.read_parquet(directory / "rows.parquet")
    arrays = {path.stem: np.load(path, allow_pickle=False)
              for path in sorted(directory.glob("*.npy"))}
    return rows, arrays, manifest


def load_spec(config_path: str | Path, family: str) -> tuple[ModelSpec, dict]:
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("scope", {}).get("benchmarks") != EXPECTED_BENCHMARKS:
        raise ValueError("Workstream B config must freeze the exact six benchmarks")
    if config.get("scope", {}).get("models") != EXPECTED_MODELS:
        raise ValueError("Workstream B config must freeze the exact 26 models")
    if config.get("scope", {}).get("split") != EXPECTED_SPLIT:
        raise ValueError("Workstream B config split does not match the frozen design")
    try:
        values = config["models"][family]
    except KeyError as error:
        raise ValueError(f"unknown embedding family {family!r}") from error
    return ModelSpec(
        values["model_id"], values["revision"], values["context"],
        values["chunk_length"], values["overlap"], values["question_envelope"],
        values["pool"], tuple(values["hidden_layers"]), values["padding_side"],
        values["dimension"], values["license"],
    ), config


def load_encoder(spec: ModelSpec, model_path: str | Path | None, dtype: str,
                 attention_backend: str):
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("embedding extraction requires a CUDA GPU")
    if model_path and Path(model_path).resolve().name != spec.revision:
        raise ValueError("staged model directory name must equal the pinned revision")
    source = str(model_path or spec.model_id)
    revision = None if model_path else spec.revision
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=revision, local_files_only=True, trust_remote_code=False,
        padding_side=spec.padding_side,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = AutoModel.from_pretrained(
        source, revision=revision, local_files_only=True, trust_remote_code=False,
        dtype=torch_dtype, attn_implementation=attention_backend,
    ).to("cuda")
    return tokenizer, FrozenEncoder(model, tokenizer.pad_token_id, spec.pool,
                                    spec.padding_side, spec.hidden_layers)


def _model_file_hashes(model_path: Path) -> dict[str, str]:
    return {str(path.relative_to(model_path)): sha256_file(path)
            for path in sorted(model_path.rglob("*")) if path.is_file()}


def _runtime_manifest(config_path: Path, spec: ModelSpec, family: str,
                      freeze_path: Path, model_path: Path, extraction: dict,
                      timing: dict | None = None) -> dict:
    import transformers
    import pyarrow

    return {
        "family": family,
        "model": spec.__dict__,
        "model_files": _model_file_hashes(model_path),
        "config_sha256": sha256_file(config_path),
        "embedding_code_sha256": sha256_file(__file__),
        "git_commit": git_commit(__file__),
        "phase1_freeze_sha256": sha256_file(freeze_path),
        "phase1_freeze": json.loads(freeze_path.read_text()),
        "purpose": "representation_discovery",
        "extraction": extraction,
        "timing": timing,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "pyarrow": pyarrow.__version__,
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "attention_backend": extraction["attention_backend"],
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "tf32": False},
        "argv": sys.argv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/workstream_b.yaml"))
    parser.add_argument("--family", choices=("qwen", "bge"), required=True)
    parser.add_argument("--model-path", type=Path,
                        help="Staged immutable model directory; no downloads are attempted")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=sorted(SPLIT_ROLES),
                        default=["representation_train", "representation_validation"])
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="Select the frozen 256-row Qwen-stratified development probe")
    parser.add_argument("--probe-keys", type=Path,
                        help="Use row keys from a completed Qwen probe (for BGE)")
    parser.add_argument("--selected-keys", type=Path,
                        help="Use an already-frozen probe population for overlap sensitivity")
    parser.add_argument("--overlap", type=int, help="Predeclared overlap sensitivity")
    parser.add_argument("--dense-gate", type=Path,
                        help="Checksum-verified passed family gate required for full extraction")
    parser.add_argument("--gate-from", type=Path, nargs=4,
                        metavar=("FIRST", "REPEAT", "HALF", "DOUBLE"))
    parser.add_argument("--timing-json", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("embedding extraction is Slurm-only")

    spec, config = load_spec(args.config, args.family)
    if args.gate_from:
        first, repeated, half, double = [load_dense_probe(path) for path in args.gate_from]
        if any(manifest.get("family") != args.family
               for _, _, manifest in (first, repeated, half, double)):
            parser.error("dense gate inputs must use one embedding family")
        report = evaluate_dense_gate(
            first[0], first[1], repeated[0], repeated[1],
            {"half": (half[0], half[1]), "double": (double[0], double[1])},
            (json.loads(args.timing_json.read_text()) if args.timing_json
             else first[2].get("timing")), config["extraction"],
        )
        identity_fields = ("family", "model_files", "config_sha256",
                           "embedding_code_sha256", "phase1_freeze_sha256", "extraction")
        repeat_identity = (
            all(first[2].get(name) == repeated[2].get(name) for name in identity_fields)
            and first[2].get("model") == repeated[2].get("model"))
        sensitivity_identity = True
        for artifact, expected_overlap in zip(
                (half, double), config["models"][args.family]["overlap_sensitivity"], strict=True):
            expected_model = dict(first[2].get("model", {})) | {"overlap": expected_overlap}
            sensitivity_identity &= all(first[2].get(name) == artifact[2].get(name)
                                        for name in identity_fields)
            sensitivity_identity &= artifact[2].get("model") == expected_model
        report["identity"] = {"repeat_matches": repeat_identity,
                              "sensitivities_match": sensitivity_identity}
        if not repeat_identity or not sensitivity_identity:
            report["status"] = "failed"
        sources = {label: sha256_file(path / "manifest.json") for label, path in zip(
            ("first", "repeat", "half", "double"), args.gate_from, strict=True)}
        identity = {name: first[2].get(name) for name in (
            "family", "model", "model_files", "config_sha256",
            "embedding_code_sha256", "phase1_freeze_sha256", "extraction")}
        write_dense_gate_report(
            args.output, report, sources, family=args.family, identity=identity)
        return
    if any(path is None for path in (args.model_path, args.store, args.split, args.freeze)):
        parser.error("extraction requires --model-path, --store, --split, and --freeze")
    if args.overlap is not None:
        if args.overlap not in config["models"][args.family]["overlap_sensitivity"]:
            parser.error("--overlap must be a predeclared family sensitivity")
        spec = replace(spec, overlap=args.overlap)
    if args.probe_keys and args.family != "bge":
        parser.error("--probe-keys is BGE-only")
    if sum(value is not None and value is not False for value in (
            args.probe, args.probe_keys, args.selected_keys)) > 1:
        parser.error("--probe, --probe-keys, and --selected-keys are mutually exclusive")
    roles = set(args.roles)
    rows = load_frozen_rows(args.store, args.split, args.freeze, roles,
                            confirmation_mode=args.confirmation)
    audit_rows = rows.attrs.pop("audit_map")
    tokenizer, encoder = load_encoder(
        spec, args.model_path, config["extraction"]["dtype"],
        config["extraction"]["attention_backend"],
    )
    total_inference_tokens = None
    if args.probe or args.probe_keys or args.selected_keys:
        census = [assemble_inputs(tokenizer, row.question, row.transcript, spec)
                  for row in rows.itertuples(index=False)]
        total_inference_tokens = sum(
            len(chunk.transcript_ids) + len(chunk.conditioned_ids)
            for row_chunks in census for chunk in row_chunks)
    if args.probe:
        if args.family != "qwen" or args.confirmation:
            parser.error("--probe is Qwen development-only")
        counts = np.array([len(row_chunks) for row_chunks in census])
        token_counts = np.array([row_chunks[-1].end for row_chunks in census])
        rows = select_probe_rows(
            rows, counts, token_counts, config["extraction"]["determinism_sample"],
            config["extraction"]["seed"])
        audit_rows = audit_rows[
            audit_rows["transcript_key"].isin(rows["transcript_key"])]
    elif args.probe_keys or args.selected_keys:
        keys = pd.read_parquet(args.probe_keys or args.selected_keys)[["transcript_key"]]
        rows = keys.merge(rows, on="transcript_key", how="left", validate="one_to_one",
                          indicator=True)
        if not (rows.pop("_merge") == "both").all():
            raise ValueError("Qwen probe keys do not match the frozen BGE population")
        audit_rows = audit_rows[
            audit_rows["transcript_key"].isin(rows["transcript_key"])]
    else:
        if args.dense_gate is None:
            parser.error("full extraction requires --dense-gate")
        expected_identity = {
            "family": args.family, "model": spec.__dict__,
            "model_files": _model_file_hashes(args.model_path),
            "config_sha256": sha256_file(args.config),
            "embedding_code_sha256": sha256_file(__file__),
            "phase1_freeze_sha256": sha256_file(args.freeze),
            "extraction": config["extraction"],
        }
        load_passed_dense_gate(args.dense_gate, args.family, expected_identity)
    started = time.perf_counter()
    arrays, metadata = extract_dense(rows, tokenizer, encoder, spec,
                                     config["extraction"]["batch_size"],
                                     config["extraction"]["max_batch_tokens"])
    elapsed = time.perf_counter() - started
    timing = ({"elapsed_seconds": elapsed,
               "sample_tokens": int(metadata["inference_tokens"].sum()),
               "total_tokens": int(total_inference_tokens), "gpus": 1}
              if total_inference_tokens is not None else None)
    manifest = _runtime_manifest(
        args.config, spec, args.family, args.freeze, args.model_path,
        config["extraction"], timing)
    coverage_passed, coverage_failures = _probe_coverage(metadata, arrays)
    manifest["coverage_gate"] = {"passed": coverage_passed,
                                 "failures": coverage_failures, "rows": len(metadata)}
    if not coverage_passed:
        raise ValueError(f"dense coverage gate failed: {coverage_failures}")
    report_columns = [column for column in (
        "transcript_key", "item_group_key", "benchmark", "question", "transcript",
        "split_role", "judge_truncated",
    ) if column in rows]
    report_rows = metadata.merge(
        rows[report_columns],
        on=[column for column in ("transcript_key", "item_group_key", "benchmark",
                                  "split_role", "judge_truncated")
            if column in metadata and column in report_columns],
        how="left", validate="one_to_one", sort=False,
    )
    write_dense_artifacts(args.output, metadata, arrays, manifest,
                          audit_rows=audit_rows, report_rows=report_rows)


def load_frozen_rows(
    store_path: str | Path,
    split_path: str | Path,
    freeze_path: str | Path,
    roles: set[str],
    *,
    confirmation_mode: bool = False,
    validation_mode: str = "representation",
) -> pd.DataFrame:
    """Load hashed canonical-correct rows for selected item roles, then strip flags."""
    if not roles or not roles <= SPLIT_ROLES:
        raise ValueError(f"roles must be a nonempty subset of {sorted(SPLIT_ROLES)}")
    if confirmation_mode != (roles == {"confirmation"}):
        raise ValueError("confirmation mode must select only confirmation rows")
    if validation_mode not in {"representation", "verdict_join"}:
        raise ValueError("validation_mode must be representation or verdict_join")

    store_path, split_path = Path(store_path), Path(split_path)
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("manifest_version") != 1:
        raise ValueError("unsupported or missing manifest_version")
    _require_sha256(freeze.get("freeze_contract_hash"), "freeze_contract_hash")
    try:
        store_sha256 = freeze["files"]["transcript_store"]["sha256"]
        split_sha256 = freeze["files"]["item_split"]["sha256"]
        reconciliation_record = freeze["files"]["verdict_reconciliation"]
        code_records = {name: freeze["files"][name] for name in (
            "config", "main_code", "surprisal_code", "judge_util_code")}
        reconciliation_sha256 = reconciliation_record["sha256"]
    except (KeyError, TypeError) as error:
        name = str(error).strip("'")
        raise ValueError(f"Phase 1 manifest is missing nested {name} hash") from error
    _require_sha256(reconciliation_sha256, "files.verdict_reconciliation.sha256")
    code_root = (freeze_path.parent / "code").resolve()
    contract = hashlib.sha256()
    for name, record in code_records.items():
        _require_sha256(record.get("sha256"), f"files.{name}.sha256")
        path = Path(record.get("path", ""))
        if not path.is_file():
            path = code_root / path.name
        if (not path.is_file() or not path.resolve().is_relative_to(code_root)
                or sha256_file(path) != record["sha256"]):
            raise ValueError(f"{name} SHA-256/path does not match immutable freeze code snapshot")
    for name in sorted(code_records):
        path = Path(code_records[name]["path"])
        if not path.is_file():
            path = code_root / path.name
        contract.update(name.encode() + b"\0" + path.read_bytes())
    if contract.hexdigest() != freeze["freeze_contract_hash"]:
        raise ValueError("freeze_contract_hash does not match immutable code snapshots")
    reconciliation_path = _frozen_file(reconciliation_record, freeze_path)
    if sha256_file(reconciliation_path) != reconciliation_sha256:
        raise ValueError("verdict reconciliation SHA-256 does not match Phase 1 freeze")
    reconciliation = json.loads(reconciliation_path.read_text())
    verdicts = pairs = None
    if validation_mode == "verdict_join":
        for reason in ("unmatched_transcript_logs", "input_mismatch_logs"):
            if reconciliation.get(reason, 0) != 0:
                raise ValueError(f"Phase 1 reconciliation has nonzero {reason}")
        verdicts_record, pairs_record = (freeze["files"]["verdicts"],
                                         freeze["files"]["pairs"])
        for name, record in (("verdicts", verdicts_record), ("pairs", pairs_record)):
            _require_sha256(record.get("sha256"), f"files.{name}.sha256")
        verdicts_path = _frozen_file(verdicts_record, freeze_path)
        pairs_path = _frozen_file(pairs_record, freeze_path)
        if sha256_file(verdicts_path) != verdicts_record["sha256"]:
            raise ValueError("accepted verdicts SHA-256 does not match Phase 1 freeze")
        if sha256_file(pairs_path) != pairs_record["sha256"]:
            raise ValueError("pairs SHA-256 does not match Phase 1 freeze")
        verdicts = pd.read_parquet(
            verdicts_path, columns=["bench", "max_length_resolution"])
        allowed = {"recorded", "non_binding", "identified_65536", "identified_262144"}
        if set(verdicts["bench"]) != set(EXPECTED_BENCHMARKS):
            raise ValueError("accepted verdicts do not cover the exact six benchmarks")
        if (not verdicts["max_length_resolution"].notna().all()
                or not set(verdicts["max_length_resolution"]) <= allowed):
            raise ValueError("accepted verdicts contain an unresolved accepted status")
    if store_sha256 != sha256_file(store_path):
        raise ValueError("transcript-store SHA-256 does not match Phase 1 freeze")
    if split_sha256 != sha256_file(split_path):
        raise ValueError("split SHA-256 does not match Phase 1 freeze")
    row_keys = freeze.get("row_keys")
    if not isinstance(row_keys, dict) or row_keys.get("transcript_store") != TRANSCRIPT_KEYS:
        raise ValueError("Phase 1 row_keys do not match Workstream B")
    if row_keys.get("item_split") != ["bench", "item", "split"]:
        raise ValueError("Phase 1 item_split row_keys do not match Workstream B")
    if freeze.get("correctness_rule") != "judge.util.extract_score":
        raise ValueError("Phase 1 correctness must use judge.util.extract_score")
    if freeze.get("population_rule") != "is_canonical & correct":
        raise ValueError("Phase 1 population must be canonical correct transcripts")
    scope = freeze.get("scope", {})
    if scope.get("benchmarks") != EXPECTED_BENCHMARKS:
        raise ValueError("Phase 1 scope must contain the exact six benchmarks")
    if scope.get("models") != EXPECTED_MODELS:
        raise ValueError("Phase 1 scope must contain the exact 26 models")
    if freeze.get("split") != EXPECTED_SPLIT:
        raise ValueError("Phase 1 split seed/fractions do not match Workstream B")
    for field in ("transcript_schema_hash", "canonical_content_hash"):
        _require_sha256(freeze.get(field), field)
    judge_hashes = freeze.get("judge_config_hashes")
    if (not isinstance(judge_hashes, list) or not judge_hashes
            or judge_hashes != sorted(set(judge_hashes))):
        raise ValueError("invalid frozen judge_config_hashes")
    for value in judge_hashes:
        _require_sha256(value, "judge_config_hashes")

    split = pd.read_csv(split_path, dtype={"bench": str, "item": str, "split": str})
    required_split = {"bench", "item", "split"}
    if missing := required_split - set(split):
        raise ValueError(f"item split missing columns: {sorted(missing)}")
    if split.duplicated(["bench", "item"]).any():
        raise ValueError("duplicate item split keys")
    if set(split["split"]) != SPLIT_ROLES:
        raise ValueError("item split must contain all three frozen roles")

    required_store = set(TRANSCRIPT_KEYS) | {"question", "transcript", "correct", "is_canonical"}
    from pyarrow.parquet import ParquetFile
    parquet = ParquetFile(store_path)
    available_store = set(parquet.schema.names)
    if missing := required_store - available_store:
        raise ValueError(f"transcript store missing columns: {sorted(missing)}")
    empty_store = parquet.schema_arrow.empty_table().to_pandas()
    schema = [{"name": column, "dtype": str(empty_store[column].dtype)}
              for column in empty_store.columns]
    schema_hash = hashlib.sha256(json.dumps(
        schema, separators=(",", ":")).encode()).hexdigest()
    if schema_hash != freeze["transcript_schema_hash"]:
        raise ValueError("transcript_schema_hash does not match the frozen store")
    store_columns = pd.read_parquet(
        store_path, columns=TRANSCRIPT_KEYS + ["correct", "is_canonical"])
    if store_columns.duplicated(TRANSCRIPT_KEYS).any():
        raise ValueError("duplicate transcript row keys")
    if set(store_columns["bench"]) != set(EXPECTED_BENCHMARKS):
        raise ValueError("transcript store does not contain the exact six benchmarks")
    if set(store_columns["model"]) != set(EXPECTED_MODELS):
        raise ValueError("transcript store does not contain the exact 26 models")
    frozen_population = store_columns[
        store_columns["correct"].eq(True) & store_columns["is_canonical"].eq(True)
    ][TRANSCRIPT_KEYS].sort_values(TRANSCRIPT_KEYS)
    content_hash = hashlib.sha256(
        frozen_population.to_json(orient="records").encode()).hexdigest()
    if content_hash != freeze["canonical_content_hash"]:
        raise ValueError("canonical_content_hash does not match the frozen population")
    manifest_rows = freeze.get("rows", {})
    realised_rows = {"store": len(store_columns),
                     "canonical_store": int(store_columns["is_canonical"].eq(True).sum()),
                     "canonical_correct": len(frozen_population)}
    if validation_mode == "verdict_join":
        realised_rows |= {"verdicts": len(verdicts),
                          "pairs": len(pd.read_parquet(pairs_path))}
    if any(manifest_rows.get(name) != count for name, count in realised_rows.items()):
        raise ValueError("Phase 1 manifest row counts do not match hashed artifacts")
    expected_directory = f"v1-{content_hash[:12]}-{freeze['freeze_contract_hash'][:12]}"
    if (freeze_path.name != "freeze_manifest.json"
            or freeze_path.parent.name != expected_directory
            or freeze_path.parent.parent.name != "freezes"):
        raise ValueError("Phase 1 manifest path does not match its immutable freeze contract")
    selected = store_columns[
        store_columns["correct"].eq(True) & store_columns["is_canonical"].eq(True)
    ].merge(split, on=["bench", "item"], how="left", validate="many_to_one", indicator=True)
    if not (selected["_merge"] == "both").all():
        raise ValueError("canonical-correct transcript rows are missing an item split")
    selected = selected[selected["split"].isin(roles)].drop(
        columns=["correct", "is_canonical", "_merge"]
    ).rename(columns={"split": "split_role"})
    if selected.empty:
        raise ValueError("selected roles contain no canonical-correct transcripts")

    item_filters = [[("bench", "==", bench), ("item", "==", item)]
                    for bench, item in selected[["bench", "item"]].drop_duplicates().itertuples(index=False)]
    payload_columns = TRANSCRIPT_KEYS + [column for column in (
        "question", "transcript", "options_json", "transcript_parts_json",
        "judge_truncated",
    ) if column in available_store]
    payload = pd.read_parquet(store_path, columns=payload_columns, filters=item_filters)
    rows = selected.merge(payload, on=TRANSCRIPT_KEYS, how="left", validate="one_to_one", indicator=True)
    if not (rows.pop("_merge") == "both").all():
        raise ValueError("selected transcript payload is incomplete")
    if rows[["question", "transcript"]].isna().any().any():
        raise ValueError("missing question or transcript")
    rows["transcript_key"] = [hashlib.sha256(json.dumps(
        json_ready([getattr(row, key) for key in TRANSCRIPT_KEYS]),
        ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest() for row in rows.itertuples(index=False)]
    rows["item_group_key"] = [hashlib.sha256(json.dumps(
        [row.bench, row.item], ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest() for row in rows.itertuples(index=False)]
    audit_map = rows[["transcript_key", "item_group_key", *TRANSCRIPT_KEYS]].copy()
    rows = rows.rename(columns={"bench": "benchmark"})
    keep = ["transcript_key", "item_group_key", "benchmark"] + [column for column in (
        "question", "transcript", "options_json", "transcript_parts_json",
        "judge_truncated", "split_role",
    ) if column in rows]
    learner_rows = rows[keep].sort_values("transcript_key").reset_index(drop=True)
    learner_rows.attrs["audit_map"] = audit_map.sort_values("transcript_key").reset_index(drop=True)
    learner_rows.attrs["freeze_validation"] = {
        "purpose": validation_mode,
        "phase1_manifest_sha256": sha256_file(freeze_path),
        "reconciliation_sha256": reconciliation_sha256,
        "reconciliation_exclusions": reconciliation,
    }
    return learner_rows


if __name__ == "__main__":
    main()
