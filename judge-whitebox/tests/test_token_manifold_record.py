import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

import token_manifold_record as record


def fake_row(uid="p0"):
    return {"pair_uid": uid, "order": 0, "item": "it0", "label": 1,
            "question": "Q", "options_1": "", "options_2": "",
            "text_1": "abc", "text_2": "de"}


def fake_cfg():
    return {"token_ablate": {"markers": {}}}


class OffsetTokenizer:
    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        ids = [1 + i % 30 for i in range(len(text))]
        out = {"input_ids": torch.tensor([ids]) if return_tensors else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def encode(self, text, add_special_tokens=False):
        return {"1": [2], "2": [3]}[text]


LengthTokenizer = OffsetTokenizer


def make_tiny_model():
    config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=4, max_position_embeddings=32)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def make_pairs_df(n):
    return pd.DataFrame([fake_row(f"p{i}") | {"item": f"it{i}", "order": i % 2}
                         for i in range(n)])


def fake_token_meta(row, tokenizer, c):
    n = len(row["text_1"])
    return {"prompt": row["text_1"], "input_ids": list(range(1, n + 1)),
            "offsets": [(i, i + 1) for i in range(n)],
            "group_id": np.ones(n, dtype=np.int8),
            "group_names": record.STRUCTURAL_GROUPS}


def fake_recorded_row(uid, n_tokens):
    return {"row_idx": int(uid[1:]), "pair_uid": uid, "item": "it0", "order": 0,
            "resid": torch.zeros(n_tokens, 2, 4, dtype=torch.float16),
            "input_ids": list(range(n_tokens)),
            "offsets": [(i, i + 1) for i in range(n_tokens)],
            "group_id": np.ones(n_tokens, dtype=np.int8),
            "l1": 1.0, "l2": 0.0, "gap": 1.0}


def test_balanced_item_folds_are_deterministic_disjoint_and_row_balanced():
    df = pd.DataFrame({"item": ["heavy"] * 8 + ["a"] * 4 + ["b"] * 4 + ["c"] * 2})

    one = record.balanced_item_folds(df, seed=7)
    two = record.balanced_item_folds(df.sample(frac=1, random_state=9), seed=7)

    assert one == two
    assert set(one) == set(df.item)
    assert set(one.values()) == {"A", "B"}
    counts = df.assign(fold=df.item.map(one)).groupby("fold").size()
    assert abs(int(counts.A) - int(counts.B)) <= 2


def test_select_fold_keeps_whole_items_and_original_order():
    df = pd.DataFrame({"item": ["a", "a", "b", "b"], "row": range(4)})

    out = record.select_fold(df, {"a": "B", "b": "A"}, "A")

    assert out.row.tolist() == [2, 3]
    assert set(out.item) == {"b"}


def test_limit_pairs_applies_to_whole_pair_ids_after_fold_selection():
    df = pd.DataFrame({"pair_uid": ["p0", "p0", "p1", "p1"],
                       "item": ["a", "a", "b", "b"], "row": range(4)})

    out = record.limit_pairs(df, 1)

    assert out.row.tolist() == [0, 1]
    assert set(out.pair_uid) == {"p0"}


def test_limit_pairs_round_robins_items_deterministically():
    df = pd.DataFrame([
        {"pair_uid": f"{item}{pair}", "item": item, "order": order}
        for item in ("a", "b", "c") for pair in range(4) for order in (0, 1)
    ])

    one = record.limit_pairs(df, 6)
    two = record.limit_pairs(df.sample(frac=1, random_state=7), 6)

    assert set(one.pair_uid) == set(two.pair_uid)
    counts = one.drop_duplicates("pair_uid").groupby("item").size()
    assert counts.to_dict() == {"a": 2, "b": 2, "c": 2}
    assert record.LIMIT_SAMPLING == "item_round_robin_v1"


def test_layer_indices_include_ends_and_are_unique():
    assert record.layer_indices(33, 4) == (0, 11, 21, 32)
    assert record.layer_indices(3, 12) == (0, 1, 2)


def test_prompt_token_meta_aligns_ids_offsets_and_structural_groups(monkeypatch):
    row, tok, cfg = fake_row(), OffsetTokenizer(), fake_cfg()
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: "QABCD")
    monkeypatch.setattr(record.token_ablate, "token_groups", lambda row, tok, c: {
        "question": [0], "transcript_1": [1], "transcript_2": [2],
        "instr_format": [3], "slot_labels": [], "answer_readout": [4],
    })

    meta = record.prompt_token_meta(row, tok, cfg)

    assert len(meta["input_ids"]) == len(meta["offsets"]) == len(meta["group_id"]) == 5
    assert meta["group_names"][meta["group_id"][1]] == "transcript_1"


def test_estimate_storage_counts_rendered_tokens_and_residual_bytes(monkeypatch):
    df = pd.DataFrame([fake_row("p0"), fake_row("p1")])
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: row["text_1"])

    out = record.estimate_storage(df, LengthTokenizer(), fake_cfg(), layers=(0, 2), d_model=4)

    assert out["n_prompts"] == 2
    assert out["n_pairs"] == 2 and out["n_items"] == 1
    assert out["n_tokens"] == len(df.iloc[0].text_1) + len(df.iloc[1].text_1)
    assert out["residual_bytes"] == out["n_tokens"] * 2 * 4 * 2


def test_manifest_identity_hashes_pairs_prompts_and_model(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.parquet"
    pairs.write_bytes(b"pairs")
    df = pd.DataFrame([fake_row("p0"), fake_row("p1")])
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt",
                        lambda row, tok: row["pair_uid"])

    identity = record.manifest_identity(df, OffsetTokenizer(), pairs, "model-revision")

    assert identity["pairs_sha256"] == record.sha256(pairs)
    assert len(identity["prompt_sha256"]) == 64
    assert identity["model_revision"] == "model-revision"


def test_record_rows_keeps_every_token_at_selected_layers(monkeypatch):
    model, tok, df = make_tiny_model(), OffsetTokenizer(), make_pairs_df(2)
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: row["text_1"])
    monkeypatch.setattr(record, "prompt_token_meta", fake_token_meta)

    rows = list(record.record_rows(model, tok, df, fake_cfg(), layers=(0, 2)))

    assert rows[0]["resid"].shape == (len(df.iloc[0].text_1), 2, model.config.hidden_size)
    assert rows[0]["resid"].dtype == torch.float16
    assert rows[0]["gap"] == pytest.approx(rows[0]["l1"] - rows[0]["l2"])


def test_record_rows_preserves_source_dataframe_index(monkeypatch):
    model, tok, df = make_tiny_model(), OffsetTokenizer(), make_pairs_df(2)
    df.index = [7, 11]
    monkeypatch.setattr(record, "prompt_token_meta", fake_token_meta)

    rows = list(record.record_rows(model, tok, df, fake_cfg(), layers=(0,)))

    assert [row["row_idx"] for row in rows] == [7, 11]


def test_write_shards_round_trips_and_hashes(tmp_path):
    rows = [fake_recorded_row("p0", 3), fake_recorded_row("p1", 2)]

    manifest = record.write_shards(iter(rows), tmp_path / "new", 1, {"model": "dev"})

    data = json.loads(manifest.read_text())
    assert data["n_prompts"] == 2 and data["n_tokens"] == 5
    assert len(data["shards"]) == 2
    for entry in data["shards"]:
        path = manifest.parent / entry["path"]
        assert record.sha256(path) == entry["sha256"]


def test_write_shards_refuses_completed_output(tmp_path):
    out = tmp_path / "completed"
    record.write_shards(iter([fake_recorded_row("p0", 2)]), out, 1, {})

    with pytest.raises(FileExistsError):
        record.write_shards(iter([]), out, 1, {})


def test_write_shards_resumes_verified_completed_prefix(tmp_path):
    out = tmp_path / "resume"
    out.mkdir()
    rows = [fake_recorded_row("p0", 2), fake_recorded_row("p1", 3)]
    record._write_chunk(rows[:1], out / "shard_00000.pt")

    manifest = record.write_shards(iter(rows), out, 1, {"model": "dev"})

    data = json.loads(manifest.read_text())
    assert data["n_prompts"] == 2 and len(data["shards"]) == 2
    assert [entry["path"] for entry in data["shards"]] == ["shard_00000.pt", "shard_00001.pt"]


def test_write_shards_rejects_resume_with_same_pair_but_different_order(tmp_path):
    out = tmp_path / "resume"
    out.mkdir()
    original = fake_recorded_row("p0", 2)
    record._write_chunk([original], out / "shard_00000.pt")
    changed = fake_recorded_row("p0", 2)
    changed["order"] = 1

    with pytest.raises(ValueError, match="does not match"):
        record.write_shards(iter([changed]), out, 1, {})


def test_main_estimate_only_prints_json_and_writes_nothing(tmp_path, monkeypatch, capsys):
    df = make_pairs_df(4)
    monkeypatch.setattr(record.wbconfig, "cfg", lambda: {
        "verdicts": {"eval_split": "eval_mm"}, "token_manifold": {"fold_seed": 7},
        "paths": {"pairs": "/fake/pairs.parquet"},
    })
    monkeypatch.setattr(record.wbpairs, "load_pairs", lambda c, split, limit: df)
    monkeypatch.setattr(record, "load_tokenizer_and_shape",
                        lambda model, c: (OffsetTokenizer(), 4, 3))
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt",
                        lambda row, tok: row["text_1"])
    out = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["token_manifold_record.py", "--model", "dev",
                                       "--fold", "A", "--estimate-only", "--out", str(out)])

    record.main()

    estimate = json.loads(capsys.readouterr().out)
    assert estimate["n_prompts"] > 0 and estimate["n_layers"] == 3
    assert not out.exists()
