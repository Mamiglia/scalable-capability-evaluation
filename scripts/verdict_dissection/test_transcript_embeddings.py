import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from transcript_embeddings import (FrozenEncoder, ModelSpec, assemble_inputs,
                                   batch_jobs, chunk_spans, extract_dense,
                                   EXPECTED_BENCHMARKS, EXPECTED_MODELS,
                                   evaluate_combined_dense_gate, evaluate_dense_gate,
                                   load_passed_combined_gate,
                                   load_passed_dense_gate,
                                   load_frozen_rows, overlap_weights, pool_chunks,
                                   select_probe_rows, sha256_file,
                                   verify_artifact_files,
                                   write_combined_dense_gate, write_dense_gate_report,
                                   write_dense_artifacts)


KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]


def test_chunk_spans_cover_every_token_with_declared_overlap():
    assert chunk_spans(14, length=6, overlap=2) == [(0, 6), (4, 10), (8, 14)]
    assert chunk_spans(0, length=6, overlap=2) == [(0, 0)]


def test_overlap_weights_count_each_token_once():
    spans = [(0, 6), (4, 10)]
    assert np.array_equal(overlap_weights(spans, 10), np.array([5.0, 5.0]))


def test_pool_chunks_is_overlap_corrected_and_unit_norm():
    pooled = pool_chunks(np.array([[1.0, 0.0], [0.0, 1.0]]), [(0, 6), (4, 10)], 10)
    assert np.allclose(pooled, np.array([2**-0.5, 2**-0.5]))


def frozen_inputs(tmp_path):
    rows = [[
        EXPECTED_BENCHMARKS[index % len(EXPECTED_BENCHMARKS)], model, str(index + 1),
        0, "run", 0, hashlib.sha256(model.encode()).hexdigest(), f"Q{index + 1}",
        f"T{index + 1}", True, True,
    ] for index, model in enumerate(EXPECTED_MODELS)]
    rows.extend([
        ["gpqa", EXPECTED_MODELS[0], "incorrect", 0, "run", 0, "c" * 64,
         "Q", "T", False, True],
        ["gpqa", EXPECTED_MODELS[0], "noncanonical", 0, "run", 0, "d" * 64,
         "Q", "T", True, False],
    ])
    store = pd.DataFrame(rows, columns=KEYS + ["question", "transcript", "correct", "is_canonical"])
    split = store[["bench", "item"]].drop_duplicates().copy()
    split["split"] = [
        ("confirmation" if index % 5 == 1 else
         "representation_validation" if index % 5 == 2 else "representation_train")
        for index in range(len(split))
    ]
    store_path, split_path = tmp_path / "store.parquet", tmp_path / "item_split.csv"
    store.to_parquet(store_path, index=False)
    split.to_csv(split_path, index=False)
    reconciliation_path = tmp_path / "verdict_reconciliation.json"
    reconciliation_path.write_text(json.dumps({"max_length_unresolved_logs": 2}) + "\n")
    verdicts_path = tmp_path / "verdicts.parquet"
    pd.DataFrame({
        "bench": EXPECTED_BENCHMARKS,
        "max_length_resolution": ["recorded"] * len(EXPECTED_BENCHMARKS),
    }).to_parquet(verdicts_path, index=False)
    pairs_path = tmp_path / "pairs.parquet"
    pd.DataFrame(index=range(0)).to_parquet(pairs_path, index=False)
    schema = [{"name": column, "dtype": str(store[column].dtype)} for column in store.columns]
    canonical = store[store.is_canonical & store.correct][KEYS].sort_values(KEYS)
    canonical_hash = hashlib.sha256(canonical.to_json(orient="records").encode()).hexdigest()
    contract = hashlib.sha256()
    for name in sorted(("config", "main_code", "surprisal_code", "judge_util_code")):
        contract.update(name.encode() + b"\0" + name.encode())
    contract_hash = contract.hexdigest()
    freeze_path = (tmp_path / "freezes"
                   / f"v1-{canonical_hash[:12]}-{contract_hash[:12]}"
                   / "freeze_manifest.json")
    freeze_path.parent.mkdir(parents=True)
    code_dir = freeze_path.parent / "code"
    code_dir.mkdir()
    code_records = {}
    for name in ("config", "main_code", "surprisal_code", "judge_util_code"):
        path = code_dir / f"{name}.txt"
        path.write_text(name)
        code_records[name] = {"path": str(path), "sha256": sha256_file(path)}
    freeze_path.write_text(json.dumps({
        "manifest_version": 1,
        "freeze_contract_hash": contract_hash,
        "files": {
            "transcript_store": {"sha256": sha256_file(store_path)},
            "item_split": {"sha256": sha256_file(split_path)},
            **code_records,
            "verdict_reconciliation": {
                "path": str(reconciliation_path),
                "sha256": sha256_file(reconciliation_path),
            },
            "verdicts": {"path": str(verdicts_path), "sha256": sha256_file(verdicts_path)},
            "pairs": {"path": str(pairs_path), "sha256": sha256_file(pairs_path)},
        },
        "transcript_schema_hash": hashlib.sha256(
            json.dumps(schema, separators=(",", ":")).encode()).hexdigest(),
        "canonical_content_hash": hashlib.sha256(
            canonical.to_json(orient="records").encode()).hexdigest(),
        "judge_config_hashes": [hashlib.sha256(b"judge").hexdigest()],
        "scope": {"benchmarks": EXPECTED_BENCHMARKS, "models": EXPECTED_MODELS,
                  "judges": ["judge"]},
        "split": {"seed": 0, "representation_train": 0.64,
                  "representation_validation": 0.16, "confirmation": 0.20},
        "rows": {"store": len(store), "canonical_store": int(store.is_canonical.sum()),
                 "canonical_correct": int((store.is_canonical & store.correct).sum()),
                 "verdicts": len(EXPECTED_BENCHMARKS), "pairs": 0},
        "correctness_rule": "judge.util.extract_score",
        "population_rule": "is_canonical & correct",
        "row_keys": {
            "transcript_store": KEYS,
            "item_split": ["bench", "item", "split"],
            "verdicts": ["judge_draw"],
            "pairs": ["pair"],
        },
    }))
    return store_path, split_path, freeze_path


def test_load_frozen_rows_checks_hashes_and_hides_outcomes(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    rows = load_frozen_rows(store, split, freeze, {"representation_train"})
    assert rows.iloc[0].transcript_key
    assert "correct" not in rows and "is_canonical" not in rows
    assert not ({"model", "bench", "item", "epoch", "log", "content_hash"} & set(rows))
    assert set(rows) <= {"transcript_key", "item_group_key", "benchmark", "question",
                         "transcript", "options_json", "transcript_parts_json",
                         "judge_truncated", "split_role"}
    audit = rows.attrs["audit_map"]
    assert set(KEYS) <= set(audit) and "model" in audit
    assert len(rows.iloc[0].transcript_key) == 64

    data = json.loads(freeze.read_text())
    data["files"]["item_split"]["sha256"] = "0" * 64
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split SHA-256"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


@pytest.mark.parametrize("field", ["transcript_schema_hash", "canonical_content_hash"])
def test_load_frozen_rows_verifies_aggregate_markers(tmp_path, field):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    data[field] = "0" * 64
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        load_frozen_rows(store, split, freeze, {"representation_train"})


@pytest.mark.parametrize("field", [
    "manifest_version", "row_keys", "transcript_schema_hash",
    "canonical_content_hash", "judge_config_hashes",
])
def test_load_frozen_rows_fails_closed_without_provenance_marker(tmp_path, field):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    data.pop(field)
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        load_frozen_rows(store, split, freeze, {"representation_train"})


@pytest.mark.parametrize("reason", [
    "unmatched_transcript_logs", "input_mismatch_logs",
])
def test_load_frozen_rows_rejects_unresolved_reconciliation(tmp_path, reason):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    path = Path(data["files"]["verdict_reconciliation"]["path"])
    path.write_text(json.dumps({reason: 1}))
    data["files"]["verdict_reconciliation"]["sha256"] = sha256_file(path)
    freeze.write_text(json.dumps(data))
    representation = load_frozen_rows(store, split, freeze, {"representation_train"})
    assert representation.attrs["freeze_validation"]["purpose"] == "representation"
    with pytest.raises(ValueError, match=reason):
        load_frozen_rows(store, split, freeze, {"representation_train"},
                         validation_mode="verdict_join")


def test_load_frozen_rows_rejects_mutable_top_level_manifest(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    mutable = tmp_path / "freeze_manifest.json"
    mutable.write_text(freeze.read_text())
    with pytest.raises(ValueError, match="immutable freeze"):
        load_frozen_rows(store, split, mutable, {"representation_train"})


def test_load_frozen_rows_requires_upstream_extract_score_rule(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    data["correctness_rule"] = "another rule"
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="judge.util.extract_score"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_load_frozen_rows_requires_judge_util_provenance(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    del data["files"]["judge_util_code"]
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="judge_util_code"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_load_frozen_rows_hashes_recorded_judge_util(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    data["files"]["judge_util_code"]["sha256"] = "0" * 64
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="judge_util_code SHA-256"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_load_frozen_rows_recomputes_freeze_contract_before_path_check(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    data["freeze_contract_hash"] = "0" * 64
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="freeze_contract_hash does not match"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_load_frozen_rows_rejects_code_snapshot_outside_freeze(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    outside = tmp_path / "outside.py"
    source = Path(data["files"]["judge_util_code"]["path"])
    outside.write_bytes(source.read_bytes())
    data["files"]["judge_util_code"]["path"] = str(outside)
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="immutable freeze code snapshot"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_load_frozen_rows_checks_accepted_verdict_coverage_and_status(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    data = json.loads(freeze.read_text())
    verdicts = Path(data["files"]["verdicts"]["path"])
    frame = pd.read_parquet(verdicts)
    frame.loc[0, "max_length_resolution"] = "unresolved"
    frame.to_parquet(verdicts, index=False)
    data["files"]["verdicts"]["sha256"] = sha256_file(verdicts)
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="accepted verdict"):
        load_frozen_rows(store, split, freeze, {"representation_train"},
                         validation_mode="verdict_join")


def test_confirmation_requires_explicit_confirmation_mode(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    with pytest.raises(ValueError, match="confirmation mode"):
        load_frozen_rows(store, split, freeze, {"confirmation"})
    with pytest.raises(ValueError, match="confirmation mode"):
        load_frozen_rows(store, split, freeze, {"representation_train"},
                         confirmation_mode=True)


def test_duplicate_transcript_keys_abort(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    frame = pd.read_parquet(store)
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(store, index=False)
    data = json.loads(freeze.read_text())
    data["files"]["transcript_store"]["sha256"] = sha256_file(store)
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate transcript row keys"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_duplicate_item_split_keys_abort(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    frame = pd.read_csv(split)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(split, index=False)
    data = json.loads(freeze.read_text())
    data["files"]["item_split"]["sha256"] = sha256_file(split)
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate item split keys"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


def test_missing_text_is_an_explicit_loader_error(tmp_path):
    store, split, freeze = frozen_inputs(tmp_path)
    frame = pd.read_parquet(store)
    frame.loc[0, "question"] = None
    frame.to_parquet(store, index=False)
    data = json.loads(freeze.read_text())
    data["files"]["transcript_store"]["sha256"] = sha256_file(store)
    freeze.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing question or transcript"):
        load_frozen_rows(store, split, freeze, {"representation_train"})


class WordTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [self.ids.setdefault(word, len(self.ids) + 1) for word in text.split()]

    def num_special_tokens_to_add(self, pair=False):
        return 0

    def build_inputs_with_special_tokens(self, ids):
        return list(ids)


def test_assemble_inputs_reuses_spans_and_truncates_question_head_tail():
    tokenizer = WordTokenizer()
    spec = ModelSpec("fake", "a" * 40, 10, 6, 2, 4, "last", (1,), "left")
    chunks = assemble_inputs(tokenizer, "a b c d e", "t0 t1 t2 t3 t4 t5 t6", spec)
    assert [(c.start, c.end) for c in chunks] == [(0, 6), (4, 7)]
    assert [tokenizer.ids["a"], tokenizer.ids["e"]] == chunks[0].question_ids
    assert chunks[0].question_original_tokens == 5
    assert chunks[0].question_retained_tokens == 2
    assert chunks[0].question_truncated
    assert chunks[0].transcript_ids[-6:] == chunks[0].transcript_chunk_ids
    assert chunks[1].conditioned_ids[-3:] == chunks[1].transcript_chunk_ids


class SpecialTokenizer(WordTokenizer):
    eos_token_id = 99

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101, *ids, 102]


def test_empty_transcript_uses_explicit_qwen_and_bge_model_valid_layouts():
    tokenizer = SpecialTokenizer()
    qwen = ModelSpec("fake", "r", 12, 4, 1, 8, "last", (), "left",
                     empty_input="eos")
    bge = ModelSpec("fake", "r", 12, 4, 1, 8, "cls", (), "right",
                    empty_input="special_tokens_only")
    qwen_chunk = assemble_inputs(tokenizer, "", "", qwen)[0]
    bge_chunk = assemble_inputs(tokenizer, "", "", bge)[0]
    assert qwen_chunk.empty_transcript and qwen_chunk.transcript_ids == [101, 99, 102]
    assert bge_chunk.empty_transcript and bge_chunk.transcript_ids == [101, 102]


def test_assembled_inputs_enforce_native_context_boundary():
    tokenizer = SpecialTokenizer()
    fits = ModelSpec("fake", "r", 8, 4, 1, 5, "last", (), "left")
    assert len(assemble_inputs(tokenizer, "", "a b c d", fits)[0].conditioned_ids) == 8
    too_small = ModelSpec("fake", "r", 7, 4, 1, 5, "last", (), "left")
    with pytest.raises(ValueError, match="native context"):
        assemble_inputs(tokenizer, "", "a b c d", too_small)


class AddOne(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.layers = nn.ModuleList([AddOne(), AddOne()])

    def forward(self, input_ids, attention_mask):
        hidden = torch.stack((input_ids.float(), torch.ones_like(input_ids).float()), dim=-1)
        for layer in self.layers:
            hidden = layer(hidden)
        return type("Output", (), {"last_hidden_state": hidden})


def test_frozen_encoder_uses_last_token_and_selected_layer_hook():
    encoder = FrozenEncoder(TinyEncoder(), pad_token_id=0, pool="last",
                            padding_side="left", hidden_layers=(1,))
    result = encoder.encode_batch([[2, 3], [4]])
    expected_final = np.array([[5.0, 3.0], [6.0, 3.0]])
    assert np.allclose(result["final"], expected_final / np.linalg.norm(expected_final, axis=1)[:, None])
    expected_layer = np.array([[4.0, 2.0], [5.0, 2.0]])
    assert np.allclose(result[1], expected_layer / np.linalg.norm(expected_layer, axis=1)[:, None])


def test_frozen_encoder_uses_cls_for_bge_style_pooling():
    encoder = FrozenEncoder(TinyEncoder(), pad_token_id=0, pool="cls",
                            padding_side="right", hidden_layers=())
    result = encoder.encode_batch([[2, 3], [4]])["final"]
    expected = np.array([[4.0, 3.0], [6.0, 3.0]])
    assert np.allclose(result, expected / np.linalg.norm(expected, axis=1)[:, None])


def test_dense_artifacts_are_hashed_and_never_overwritten(tmp_path):
    rows = pd.DataFrame({"transcript_key": ["a" * 64],
                         "item_group_key": ["b" * 64]})
    report_rows = pd.DataFrame({"transcript_key": ["a" * 64],
                                "item_group_key": ["b" * 64],
                                "transcript": ["text"], "question": ["question"],
                                "split_role": ["representation_train"]})
    arrays = {"transcript_only__final": np.array([[1.0, 0.0]], dtype=np.float32)}
    manifest = write_dense_artifacts(
        tmp_path / "dense", rows, arrays, {"config_sha256": "a" * 64},
        report_rows=report_rows)
    assert manifest["outputs"]["transcript_only__final.npy"] == sha256_file(
        tmp_path / "dense" / "transcript_only__final.npy")
    assert manifest["outputs"]["report_rows.parquet"] == sha256_file(
        tmp_path / "dense" / "report_rows.parquet")
    assert (tmp_path / "dense" / "manifest.sha256").read_text().strip() == sha256_file(
        tmp_path / "dense" / "manifest.json")
    verify_artifact_files(tmp_path / "dense", manifest)
    (tmp_path / "dense" / "transcript_only__final.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        verify_artifact_files(tmp_path / "dense", manifest)
    with pytest.raises(FileExistsError):
        write_dense_artifacts(tmp_path / "dense", rows, arrays, {})


def test_interrupted_dense_write_preserves_ceph_staging_and_blocks_retry(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(np, "save", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_dense_artifacts(
            tmp_path / "dense", pd.DataFrame({"transcript_key": ["a" * 64]}),
            {"transcript_only__final": np.ones((1, 2))}, {})
    assert not (tmp_path / "dense").exists()
    stage = tmp_path / ".dense.staging"
    assert stage.is_dir()
    with pytest.raises(FileExistsError, match=str(stage)):
        write_dense_artifacts(
            tmp_path / "dense", pd.DataFrame({"transcript_key": ["a" * 64]}),
            {"transcript_only__final": np.ones((1, 2))}, {})


def test_embedding_cli_defines_loader_before_calling_main(tmp_path):
    script = Path(__file__).with_name("transcript_embeddings.py")
    result = subprocess.run([
        sys.executable, str(script), "--family", "qwen", "--model-path", str(tmp_path / "model"),
        "--store", str(tmp_path / "missing.parquet"), "--split", str(tmp_path / "split.csv"),
        "--freeze", str(tmp_path / "freeze.json"), "--output", str(tmp_path / "out"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "NameError" not in result.stderr
    assert "Slurm-only" in result.stderr


def test_probe_keys_is_rejected_outside_bge_before_loading_models(tmp_path):
    script = Path(__file__).with_name("transcript_embeddings.py")
    result = subprocess.run([
        sys.executable, str(script), "--family", "qwen", "--model-path", str(tmp_path / "m"),
        "--store", str(tmp_path / "s"), "--split", str(tmp_path / "i"),
        "--freeze", str(tmp_path / "f"), "--probe-keys", str(tmp_path / "k"),
        "--output", str(tmp_path / "o"),
    ], capture_output=True, text=True, env=os.environ | {"SLURM_JOB_ID": "1"})
    assert result.returncode != 0 and "BGE-only" in result.stderr


def test_extract_dense_batches_chunks_and_returns_both_views_and_layers():
    rows = pd.DataFrame({"question": ["q", "long question"],
                         "transcript": ["t0 t1", "t0 t1 t2 t3 t4 t5 t6"],
                         "transcript_key": ["a" * 64, "b" * 64],
                         "judge_truncated": [False, True]})
    tokenizer = WordTokenizer()
    spec = ModelSpec("fake", "a" * 40, 10, 6, 2, 4, "last", (1,), "left")
    arrays, metadata = extract_dense(
        rows, tokenizer, FrozenEncoder(TinyEncoder(), 0, "last", "left", (1,)),
        spec, batch_size=2,
    )
    assert set(arrays) == {
        "transcript_only__final", "transcript_only__layer_1",
        "question_conditioned__final", "question_conditioned__layer_1",
    }
    assert all(values.shape == (2, 2) for values in arrays.values())
    assert np.allclose(np.linalg.norm(arrays["transcript_only__final"], axis=1), 1)
    assert metadata["chunk_count"].tolist() == [1, 2]
    assert metadata["transcript_key"].tolist() == ["a" * 64, "b" * 64]
    assert metadata["judge_truncated"].tolist() == [False, True]


def test_extract_dense_rejects_wrong_frozen_dimension():
    rows = pd.DataFrame({"question": ["q"], "transcript": ["t"]})
    tokenizer = WordTokenizer()
    spec = ModelSpec("fake", "a" * 40, 10, 6, 2, 4, "last", (), "left", dimension=3)
    with pytest.raises(ValueError, match="frozen dimension"):
        extract_dense(rows, tokenizer, FrozenEncoder(TinyEncoder(), 0, "last", "left", ()),
                      spec, batch_size=1)


def test_probe_selection_is_seeded_and_keeps_benchmark_chunk_strata():
    rows = pd.DataFrame({"benchmark": ["a"] * 6 + ["b"] * 6,
                         "model": [f"m{i % 3}" for i in range(12)]})
    counts = np.array([1, 1, 1, 2, 3, 3, 1, 1, 2, 2, 3, 3])
    tokens = np.arange(10, 130, 10)
    first = select_probe_rows(rows, counts, tokens, n=8, seed=0)
    second = select_probe_rows(rows.assign(model="changed"), counts, tokens, n=8, seed=0)
    assert first.index.tolist() == second.index.tolist()
    assert len(first) == 8 and set(first["benchmark"]) == {"a", "b"}
    assert set(np.minimum(counts[first.index], 3)) == {1, 2, 3}


def test_batch_jobs_respects_count_and_padded_token_budget():
    jobs = [{"ids": [1, 2, 3]}, {"ids": [1, 2, 3]}, {"ids": [1, 2, 3, 4, 5]}]
    batches = list(batch_jobs(jobs, batch_size=3, max_batch_tokens=6))
    assert [list(map(lambda job: len(job["ids"]), batch)) for batch in batches] == [[3, 3], [5]]


def dense_gate_inputs(n=32):
    keys = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(n)]
    metadata = pd.DataFrame({
        "transcript_key": keys,
        "transcript_tokens": [6] * n,
        "chunk_count": [2] * n,
        "chunk_spans_json": ["[[0,4],[2,6]]"] * n,
        "transcript_input_hashes_json": [json.dumps(["a" * 64, "b" * 64])] * n,
        "conditioned_input_hashes_json": [json.dumps(["c" * 64, "d" * 64])] * n,
    })
    base = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n, 1))
    arrays = {"transcript_only__final": base.copy(),
              "question_conditioned__final": base.copy()}
    timing = {"elapsed_seconds": 36.0, "sample_tokens": 1_000,
              "total_tokens": 100_000, "gpus": 1}
    thresholds = {"max_abs_difference": 1e-6, "min_repeat_cosine": 0.999999,
                  "overlap_median_cosine": 0.98,
                  "overlap_fifth_percentile_cosine": 0.95,
                  "min_overlap_rows": 32, "max_projected_gpu_hours": 60,
                  "determinism_sample": n}
    return metadata, arrays, timing, thresholds


def test_dense_gate_passes_repeats_overlap_coverage_and_projection():
    metadata, arrays, timing, thresholds = dense_gate_inputs()
    report = evaluate_dense_gate(
        metadata, arrays, metadata.copy(), {key: value.copy() for key, value in arrays.items()},
        {"half": arrays, "double": arrays}, timing, thresholds,
    )
    assert report["status"] == "passed"
    assert report["overlap"]["half"]["status"] == "passed"
    assert report["projected_gpu_hours"] == 1.0


def test_dense_family_gate_fails_nondeterminism_and_defers_compute_ceiling_to_combined_gate():
    metadata, arrays, timing, thresholds = dense_gate_inputs()
    repeated = {key: value.copy() for key, value in arrays.items()}
    repeated["transcript_only__final"][0] = [0.0, 1.0]
    timing["total_tokens"] = 10_000_000
    report = evaluate_dense_gate(
        metadata, arrays, metadata.copy(), repeated,
        {"half": arrays, "double": arrays}, timing, thresholds,
    )
    assert report["status"] == "failed"
    assert not report["approval_required"]
    assert not report["determinism"]["passed"]


def test_dense_gate_marks_underfilled_overlap_as_not_exercised():
    metadata, arrays, timing, thresholds = dense_gate_inputs(n=8)
    report = evaluate_dense_gate(
        metadata, arrays, metadata.copy(), arrays,
        {"half": arrays, "double": arrays}, timing, thresholds,
    )
    assert report["status"] == "passed"
    assert {item["status"] for item in report["overlap"].values()} == {"not_exercised"}


def test_dense_gate_report_is_hashed_and_refuses_failed_as_prerequisite(tmp_path):
    report = {"status": "passed", "approval_required": False}
    write_dense_gate_report(tmp_path / "gate", report, {"probe": "a" * 64}, family="qwen")
    manifest = json.loads((tmp_path / "gate" / "manifest.json").read_text())
    verify_artifact_files(tmp_path / "gate", manifest)
    assert manifest["gate_status"] == "passed"
    assert load_passed_dense_gate(tmp_path / "gate", "qwen") == manifest

    write_dense_gate_report(tmp_path / "failed", {"status": "failed"}, {}, family="qwen")
    with pytest.raises(ValueError, match="passed dense gate"):
        load_passed_dense_gate(tmp_path / "failed", "qwen")


def test_combined_dense_gate_enforces_total_sixty_hour_ceiling_and_hashed_approval(tmp_path):
    qwen = {"status": "passed", "projected_gpu_hours": 31.0}
    bge = {"status": "passed", "projected_gpu_hours": 30.0}
    blocked = evaluate_combined_dense_gate(qwen, bge, max_gpu_hours=60)
    assert blocked == {"status": "failed", "projected_gpu_hours": 61.0,
                       "approval_required": True, "approval": None}

    payload = {"reason": "approved compute extension", "approver": "PI",
               "timestamp": "2026-08-25T12:00:00Z"}
    approval = payload | {"hash": hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    passed = evaluate_combined_dense_gate(qwen, bge, max_gpu_hours=60,
                                          approval=approval)
    assert passed["status"] == "passed" and passed["approval"] == approval

    identity = {"qwen": {"phase1_freeze_sha256": "a"},
                "bge": {"phase1_freeze_sha256": "a"}}
    write_combined_dense_gate(tmp_path / "combined", passed,
                              {"qwen_gate": "b" * 64, "bge_gate": "c" * 64}, identity)
    assert load_passed_combined_gate(
        tmp_path / "combined", "qwen", identity["qwen"])["gate_status"] == "passed"
