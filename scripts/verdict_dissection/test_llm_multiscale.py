"""Rolling multiscale surprisal cache checks."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from llm_multiscale import (SETTINGS, multiscale_features, rolling_surprisal,
                            score_and_cache)
from llm_surprisal import SCORERS, cache_file, start_token, tokenise

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEXT = "The cat sat on the mat. The cat was warm. " * 12


def assistant_parts(text=TEXT):
    return [{"role": "assistant", "content": text, "bare": True,
             "start": 0, "end": len(text)}]


@pytest.fixture(scope="module")
def tiny():
    return AutoModelForCausalLM.from_pretrained(TINY).eval(), AutoTokenizer.from_pretrained(TINY)


def test_whole_transcript_short_and_long_contexts(tiny, tmp_path):
    assert SETTINGS == {"short": (32, 32), "long": (7168, 1024)}
    model, tok = tiny
    path = tmp_path / "whole.npz"
    score_and_cache(model, tok, TEXT, "Why?", assistant_parts(), path, "a" * 64)
    with np.load(path) as got:
        n = len(tokenise(tok, TEXT, max_tokens=None)[0])
        assert n > 32 and len(got["offsets"]) == n
        assert all(len(got[name]) == n for name in ("short", "long", "long_question"))
        assert "medium" not in got
        for name, (history, block) in SETTINGS.items():
            assert got[f"{name}_history"].item() == history
            assert got[f"{name}_block"].item() == block


def test_prefix_cache_reuse_and_determinism(tiny, tmp_path):
    model, tok = tiny
    text_hash = hashlib.sha256(TEXT.encode()).hexdigest()
    ids, offsets = tokenise(tok, TEXT, max_tokens=None)
    cap = 12
    prefix = tmp_path / "prefix.npz"
    np.savez(prefix, content_hash=text_hash, offsets=offsets[:cap],
             local=np.full(cap, 3.0, dtype=np.float32),
             noq=np.full(cap, 2.0, dtype=np.float32),
             full=np.full(cap, 1.0, dtype=np.float32))
    reused, fresh = tmp_path / "reused.npz", tmp_path / "fresh.npz"
    score_and_cache(model, tok, TEXT, "Why?", assistant_parts(), reused, text_hash, prefix)
    score_and_cache(model, tok, TEXT, "Why?", assistant_parts(), fresh, text_hash)
    with np.load(reused) as cached, np.load(fresh) as uncached:
        assert len(ids) > cap
        assert np.array_equal(cached["short"][:cap], np.full(cap, 3.0, dtype=np.float32))
        assert np.array_equal(cached["long"][:cap], np.full(cap, 2.0, dtype=np.float32))
        assert np.array_equal(cached["long_question"][:cap], np.full(cap, 1.0, dtype=np.float32))
        assert np.array_equal(cached["short"][cap:], uncached["short"][cap:])
    again = tmp_path / "again.npz"
    score_and_cache(model, tok, TEXT, "Why?", assistant_parts(), again, text_hash)
    with np.load(fresh) as one, np.load(again) as two:
        assert all(np.array_equal(one[name], two[name]) for name in SETTINGS)


def test_complete_prefix_cache_does_not_run_model(tiny, tmp_path):
    _, tok = tiny
    content_hash = hashlib.sha256(TEXT.encode()).hexdigest()
    _, offsets = tokenise(tok, TEXT, max_tokens=None)
    prefix = tmp_path / "prefix.npz"
    np.savez(prefix, content_hash=content_hash, offsets=offsets,
             local=np.ones(len(offsets)), noq=np.ones(len(offsets)),
             full=np.ones(len(offsets)))

    score_and_cache(None, tok, TEXT, "Why?", assistant_parts(), tmp_path / "whole.npz",
                    content_hash, prefix)


def test_context_only_transcript_needs_no_model(tiny, tmp_path):
    _, tok = tiny
    parts = [{"role": "tool", "content": TEXT, "bare": True,
              "start": 0, "end": len(TEXT)}]
    path = tmp_path / "whole.npz"
    score_and_cache(None, tok, TEXT, "Why?", parts, path, "a" * 64)
    with np.load(path) as got:
        assert not got["assistant_mask"].any()
        assert all(np.isnan(got[name]).all()
                   for name in ("short", "long", "long_question"))


def test_batched_short_context_matches_one_block_at_a_time(tiny):
    model, tok = tiny
    ids, _ = tokenise(tok, TEXT, max_tokens=None)
    prefix = [start_token(tok)]
    one = rolling_surprisal(model, ids, prefix, 32, 32, batch_size=1)
    many = rolling_surprisal(model, ids, prefix, 32, 32, batch_size=32)
    assert np.array_equal(one, many)


def test_invalid_existing_cache_is_never_overwritten(tiny, tmp_path):
    _, tok = tiny
    path = tmp_path / "whole.npz"
    np.savez(path, content_hash="wrong")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        score_and_cache(None, tok, TEXT, "Why?", assistant_parts(), path,
                        "a" * 64)


def test_multiscale_feature_summaries(tmp_path):
    text = "x" * 80
    parts = assistant_parts(text)
    offsets = np.column_stack((np.arange(80), np.arange(1, 81)))
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    for scorer in SCORERS:
        path = cache_file(tmp_path, scorer, "gpqa", "model", "item", 0,
                          "log", content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, short=np.full(80, 5.0), long=np.full(80, 3.0),
                 long_question=np.full(80, 2.0), offsets=offsets,
                 assistant_mask=np.ones(80, dtype=bool), content_hash=content_hash,
                 short_history=32, short_block=32, long_history=7168,
                 long_block=1024)

    got = multiscale_features(text, parts, "gpqa", "model", "item", 0,
                              "log", content_hash, tmp_path)
    for scorer in SCORERS:
        assert got[f"surprisal_rolling_short_sum__{scorer}"] == 400.0
        assert got[f"surprisal_rolling_long_mean__{scorer}"] == 3.0
        assert got[f"surprisal_rolling_long_question_median__{scorer}"] == 2.0
        assert got[f"surprisal_rolling_context_gain_count__{scorer}"] == 16.0
        assert got[f"surprisal_rolling_context_gain_mean__{scorer}"] == 2.0
        assert got[f"surprisal_rolling_question_gain_variance__{scorer}"] == 0.0
        assert got[f"surprisal_rolling_chars__{scorer}"] == 80.0
        assert got[f"surprisal_rolling_tail_chars__{scorer}"] == 0.0
        assert got[f"surprisal_rolling_tail_context_gain_count__{scorer}"] == 0.0
        assert np.isnan(got[f"surprisal_rolling_tail_context_gain_mean__{scorer}"])
