"""llm_surprisal checks: offsets, conditioning, windowing, batching, cache, features."""
import sys
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from llm_surprisal import (assistant_token_mask, score_and_cache,
                           surprisal_full_context, surprisal_local_window)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEXT = "The cat sat on the mat. The cat was warm. " * 8


@pytest.fixture(scope="module")
def tiny():
    return AutoModelForCausalLM.from_pretrained(TINY).eval(), AutoTokenizer.from_pretrained(TINY)


def test_offsets_and_nonnegative_bits(tiny):
    model, tok = tiny
    bits, offsets = surprisal_full_context(model, tok, TEXT)
    assert len(bits) == len(offsets) and offsets[-1][1] <= len(TEXT) and np.all(bits >= 0)


def test_question_changes_conditioning_not_tokens(tiny):
    model, tok = tiny
    noq, oa = surprisal_full_context(model, tok, TEXT)
    withq, ob = surprisal_full_context(model, tok, TEXT, question="A question?")
    assert np.array_equal(oa, ob) and len(noq) == len(withq) and not np.allclose(noq, withq)


def test_local_window_bounds_context(tiny):
    model, tok = tiny
    full, _ = surprisal_full_context(model, tok, TEXT)
    local, _ = surprisal_local_window(model, tok, TEXT, window=8)
    assert len(local) == len(full)
    assert np.allclose(local[:4], full[:4], atol=1e-4)     # early tokens see the same context
    assert not np.allclose(local[-16:], full[-16:])         # late tokens do not


def test_wide_window_equals_full_pass(tiny):
    model, tok = tiny
    full, _ = surprisal_full_context(model, tok, TEXT)
    local, _ = surprisal_local_window(model, tok, TEXT, window=4096)
    assert np.array_equal(local, full)


def test_batching_does_not_change_values(tiny):
    model, tok = tiny
    one, _ = surprisal_local_window(model, tok, TEXT, window=8, batch_size=1)
    many, _ = surprisal_local_window(model, tok, TEXT, window=8, batch_size=32)
    assert np.array_equal(one, many)


def test_determinism(tiny):
    model, tok = tiny
    a, _ = surprisal_full_context(model, tok, TEXT)
    b, _ = surprisal_full_context(model, tok, TEXT)
    assert np.array_equal(a, b)


def test_empty_transcript_scores_nothing(tiny):
    model, tok = tiny
    bits, offsets = surprisal_local_window(model, tok, "", window=8)
    assert len(bits) == 0 and offsets.shape == (0, 2)


def test_assistant_mask_excludes_tool_tokens():
    text = "[Assistant]: fine words here.\n\n[Tool]: garbage garbage"
    n = 12
    step = len(text) // n
    offsets = np.array([[i * step, (i + 1) * step] for i in range(n)], dtype=np.int32)
    keep = assistant_token_mask(text, offsets)
    assert keep[4:7].all() and not keep[-3:].any()   # tokens inside the assistant block, not its marker


def test_score_and_cache_writes_once(tiny, tmp_path):
    model, tok = tiny
    path = tmp_path / "x.npz"
    score_and_cache(model, tok, TEXT, "Q?", path)
    z = np.load(path)
    assert len(z["full"]) == len(z["noq"]) == len(z["local"]) == len(z["offsets"])
    assert not path.with_suffix(".tmp.npz").exists()
