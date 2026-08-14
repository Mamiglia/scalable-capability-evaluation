import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

import record_resid
import verdicts


def make_tiny_model():
    """Real (random-weight, untrained) transformers Qwen3 model, small enough for a CPU forward
    pass in a test -- gives a genuine output_hidden_states path, not a hand-rolled stand-in."""
    config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=4, max_position_embeddings=32)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


class StubTokenizer:
    """apply_chat_template isn't exercised here -- record()'s call to wbprompts.render_verdict_prompt
    is monkeypatched to return the row's text_1 directly, so __call__ can map its length straight
    to a token count (1 char ~ 1 token, ids kept inside the tiny model's vocab)."""

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[1 + i % 30 for i in range(len(text))]])
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return {"1": [2], "2": [3]}[text]


def make_pairs_df(n):
    return pd.DataFrame({
        "pair_uid": [f"p{i}" for i in range(n)], "order": [0] * n,
        "item": [f"it{i}" for i in range(n)], "label": ([1, 2] * ((n + 1) // 2))[:n],
        "text_1": ["abc" for _ in range(n)],
    })


def test_record_shapes_and_meta(monkeypatch):
    model, tokenizer, df = make_tiny_model(), StubTokenizer(), make_pairs_df(3)
    monkeypatch.setattr(record_resid.wbprompts, "render_verdict_prompt",
                         lambda row, tok, invert=False: row["text_1"])

    resid, l1, l2, kept_row_idx = record_resid.record(model, tokenizer, df, invert=False, max_tokens=100)

    n_layers_plus_1 = model.config.num_hidden_layers + 1
    assert resid.shape == (3, n_layers_plus_1, model.config.hidden_size)
    assert resid.dtype == torch.float32
    assert l1.shape == (3,) and l2.shape == (3,)
    assert kept_row_idx == [0, 1, 2]


def test_record_skips_over_length_prompts(monkeypatch):
    model, tokenizer, df = make_tiny_model(), StubTokenizer(), make_pairs_df(3)
    df.loc[1, "text_1"] = "x" * 20  # over the cap below; rows 0, 2 stay short ("abc" = 3 chars)
    monkeypatch.setattr(record_resid.wbprompts, "render_verdict_prompt",
                         lambda row, tok, invert=False: row["text_1"])

    resid, l1, l2, kept_row_idx = record_resid.record(model, tokenizer, df, invert=False, max_tokens=5)

    assert kept_row_idx == [0, 2]
    assert resid.shape[0] == 2
    assert l1.shape == (2,)


def test_record_saved_dict_has_meta_fields(tmp_path, monkeypatch):
    model, tokenizer, df = make_tiny_model(), StubTokenizer(), make_pairs_df(2)
    monkeypatch.setattr(record_resid.wbprompts, "render_verdict_prompt",
                         lambda row, tok, invert=False: row["text_1"])
    monkeypatch.setattr(record_resid, "verdicts",
                         SimpleNamespace(load_model=lambda *a, **k: (model, tokenizer, "cpu"),
                                          readout=verdicts.readout))
    monkeypatch.setattr(record_resid.wbpairs, "load_pairs", lambda c, split=None, limit=None: df)
    monkeypatch.setattr(record_resid.wbconfig, "cfg",
                         lambda path=None: {"verdicts": {"max_input_tokens": 100, "eval_split": "eval_mm"},
                                             "paths": {"pairs": "/fake/pairs.parquet"}})
    out_path = tmp_path / "rec.pt"
    monkeypatch.setattr(sys, "argv", ["record_resid.py", "--model", "dev", "--out", str(out_path)])

    record_resid.main()

    blob = torch.load(out_path)
    for key in ("resid", "l1", "l2", "kept_row_idx", "split", "limit", "invert", "model", "adapter",
                "pairs_path"):
        assert key in blob
    assert blob["model"] == "dev" and blob["invert"] is False and blob["adapter"] is None
    assert blob["resid"].shape[0] == 2
    assert blob["pairs_path"] == "/fake/pairs.parquet"


def test_record_invert_flag_threads_through(monkeypatch):
    model, tokenizer, df = make_tiny_model(), StubTokenizer(), make_pairs_df(2)
    seen = {}

    def fake_render(row, tok, invert=False):
        seen["invert"] = invert
        return row["text_1"]

    monkeypatch.setattr(record_resid.wbprompts, "render_verdict_prompt", fake_render)
    record_resid.record(model, tokenizer, df, invert=True, max_tokens=100)
    assert seen["invert"] is True


# ---- --single-transcript mode ----

def make_transcript_pairs_df(rows):
    """rows: list of (item, model_1, cap_1, text_1, model_2, cap_2, text_2)."""
    return pd.DataFrame([{
        "item": r[0], "model_1": r[1], "cap_1": r[2], "text_1": r[3],
        "model_2": r[4], "cap_2": r[5], "text_2": r[6],
    } for r in rows])


def test_build_units_dedupes_by_text():
    # p1's text_1 ("same answer") recurs as p2's text_2 -- same transcript, different pair_uid.
    df = make_transcript_pairs_df([
        ("it0", "gpt-4", 0.8, "same answer", "claude-3", 0.6, "other answer"),
        ("it1", "llama", 0.5, "third answer", "gpt-4", 0.8, "same answer"),
    ])
    units = record_resid.build_units(df)
    texts = [u["text"] for u in units]
    assert texts.count("same answer") == 1
    assert len(units) == 3  # "same answer", "other answer", "third answer"


def test_build_units_meta_fields():
    df = make_transcript_pairs_df([
        ("it0", "gpt-4", 0.8, "answer A", "claude-3", 0.6, "answer B"),
    ])
    units = record_resid.build_units(df)
    by_text = {u["text"]: u for u in units}
    assert by_text["answer A"] == {"item": "it0", "author": "gpt-4", "cap": 0.8, "text": "answer A"}
    assert by_text["answer B"] == {"item": "it0", "author": "claude-3", "cap": 0.6, "text": "answer B"}


def test_render_transcript_prompt_assembles_messages():
    class T:  # stub tokenizer, mirrors test_wbprompts.py's render assembly check
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert [m["role"] for m in msgs] == ["system", "user"]
            assert add_generation_prompt is True
            return "SYS:" + msgs[0]["content"] + "|USR:" + msgs[1]["content"]

    out = record_resid.render_transcript_prompt("some solution text", T())
    assert "USR:" in out and "some solution text" in out


def test_transcript_frame_has_no_judging_vocabulary():
    rendered = record_resid.TRANSCRIPT_SYSTEM + "\n" + record_resid.TRANSCRIPT_USER.format(text="some solution text")
    for banned in ("capable", "Transcript 1", "Transcript 2", "ANSWER", "verdict", "judge"):
        assert banned.lower() not in rendered.lower()


def test_record_single_transcript_shapes(monkeypatch):
    model, tokenizer = make_tiny_model(), StubTokenizer()
    units = [{"item": "it0", "author": "gpt-4", "cap": 0.8, "text": "abc"},
             {"item": "it1", "author": "claude-3", "cap": 0.6, "text": "de"}]
    monkeypatch.setattr(record_resid, "render_transcript_prompt", lambda text, tok: text)

    resid, kept_unit_idx = record_resid.record_single_transcript(model, tokenizer, units, max_tokens=100)

    n_layers_plus_1 = model.config.num_hidden_layers + 1
    assert resid.shape == (2, n_layers_plus_1, model.config.hidden_size)
    assert resid.dtype == torch.float32
    assert kept_unit_idx == [0, 1]


def test_record_single_transcript_skips_over_length(monkeypatch):
    model, tokenizer = make_tiny_model(), StubTokenizer()
    units = [{"item": "it0", "author": "gpt-4", "cap": 0.8, "text": "abc"},
             {"item": "it1", "author": "claude-3", "cap": 0.6, "text": "x" * 20}]
    monkeypatch.setattr(record_resid, "render_transcript_prompt", lambda text, tok: text)

    resid, kept_unit_idx = record_resid.record_single_transcript(model, tokenizer, units, max_tokens=5)

    assert kept_unit_idx == [0]
    assert resid.shape[0] == 1


def test_main_single_transcript_saved_dict(tmp_path, monkeypatch):
    model, tokenizer = make_tiny_model(), StubTokenizer()
    df = make_transcript_pairs_df([
        ("it0", "gpt-4", 0.8, "answer A", "claude-3", 0.6, "answer B"),
        ("it1", "llama", 0.4, "answer C", "gpt-4", 0.8, "answer A"),  # dupes p0's text_1
    ])
    monkeypatch.setattr(record_resid, "render_transcript_prompt", lambda text, tok: text)
    monkeypatch.setattr(record_resid, "verdicts",
                         SimpleNamespace(load_model=lambda *a, **k: (model, tokenizer, "cpu"),
                                          readout=verdicts.readout))
    monkeypatch.setattr(record_resid.wbpairs, "load_pairs", lambda c, split=None, limit=None: df)
    monkeypatch.setattr(record_resid.wbconfig, "cfg",
                         lambda path=None: {"verdicts": {"max_input_tokens": 100, "eval_split": "eval_mm"},
                                             "paths": {"pairs": "/fake/pairs.parquet"}})
    out_path = tmp_path / "rec.pt"
    monkeypatch.setattr(sys, "argv",
                         ["record_resid.py", "--model", "dev", "--single-transcript", "--out", str(out_path)])

    record_resid.main()

    blob = torch.load(out_path)
    assert blob["mode"] == "single_transcript"
    assert blob["invert"] is False
    assert blob["resid"].shape[0] == 3  # answer A, answer B, answer C (deduped)
    for key in ("item", "author", "cap", "text_hash"):
        assert len(blob[key]) == 3
    assert blob["author"] == ["gpt-4", "claude-3", "llama"]
    assert blob["cap"] == [0.8, 0.6, 0.4]
    assert len(set(blob["text_hash"])) == 3  # distinct hashes for distinct texts
    assert blob["pairs_path"] == "/fake/pairs.parquet"
    assert blob["split"] == "eval_mm" and blob["limit"] is None


def test_main_single_transcript_rejects_invert(monkeypatch):
    monkeypatch.setattr(record_resid.wbconfig, "cfg",
                         lambda path=None: {"verdicts": {"max_input_tokens": 100, "eval_split": "eval_mm"},
                                             "paths": {"pairs": "/fake/pairs.parquet"}})
    monkeypatch.setattr(sys, "argv",
                         ["record_resid.py", "--model", "dev", "--single-transcript", "--invert", "--out", "x.pt"])
    with pytest.raises(AssertionError, match="single-transcript"):
        record_resid.main()


def test_pairwise_saved_dict_has_mode_field(tmp_path, monkeypatch):
    model, tokenizer, df = make_tiny_model(), StubTokenizer(), make_pairs_df(2)
    monkeypatch.setattr(record_resid.wbprompts, "render_verdict_prompt",
                         lambda row, tok, invert=False: row["text_1"])
    monkeypatch.setattr(record_resid, "verdicts",
                         SimpleNamespace(load_model=lambda *a, **k: (model, tokenizer, "cpu"),
                                          readout=verdicts.readout))
    monkeypatch.setattr(record_resid.wbpairs, "load_pairs", lambda c, split=None, limit=None: df)
    monkeypatch.setattr(record_resid.wbconfig, "cfg",
                         lambda path=None: {"verdicts": {"max_input_tokens": 100, "eval_split": "eval_mm"},
                                             "paths": {"pairs": "/fake/pairs.parquet"}})
    out_path = tmp_path / "rec.pt"
    monkeypatch.setattr(sys, "argv", ["record_resid.py", "--model", "dev", "--out", str(out_path)])

    record_resid.main()

    assert torch.load(out_path)["mode"] == "pairwise"
