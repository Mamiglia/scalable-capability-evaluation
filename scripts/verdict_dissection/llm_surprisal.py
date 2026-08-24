"""Surprisal of a transcript under small frozen base LMs.

Surprisal of a token = -log2 p(token | what the model saw before it). Three
passes per transcript and scorer model:
  full  : sees the question and everything before the token
  noq   : sees everything before the token, but not the question
  local : sees only the previous `window_tokens` tokens
Features are the total bits over the assistant's own tokens for each pass
(tool/user text is context only, never scored) plus the number of characters
scored; normalise later. Per-token results are cached as one .npz per
transcript and scorer, so the expensive part runs once, on a GPU. If the
scorer models or settings change, delete the cache directory.

Fill the cache (scores the correct transcripts of a store parquet; one
scorer per job on a cluster by naming it):
  uv run python scripts/verdict_dissection/llm_surprisal.py transcript_store.parquet [cache_dir] [scorer]
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "verdict_dissection.yaml").read_text())
SCORERS = CFG["surprisal"]["models"]          # short name -> HF id, e.g. llama -> meta-llama/Llama-3.2-3B
WINDOW = CFG["surprisal"]["window_tokens"]    # context length of the local pass
MAX_TOKENS = CFG["surprisal"]["max_tokens"]   # longer transcripts are cut at the end
CACHE_DIR = ROOT / CFG["data"]["out_dir"] / "surprisal_cache"

ROLE_MARKER = re.compile(r"(?:^|\n\n)\[(\w+)\]: ")   # "[Assistant]: ", "[Tool]: " blocks
LN2 = float(np.log(2.0))
LOGSOFTMAX_CHUNK = 1024                              # positions per log-softmax, bounds the float32 copy


def load_scorer(hf_id):
    """Load a base LM on the best available device. Returns (model, tokenizer)."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    return model, AutoTokenizer.from_pretrained(hf_id)


def tokenise(tokenizer, text, max_tokens=MAX_TOKENS):
    """Token ids of `text` (at most max_tokens, no special tokens) and each
    token's (start, end) character offsets. Returns (ids, offsets)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"][:max_tokens]
    offsets = np.array(enc["offset_mapping"][:max_tokens], dtype=np.int32).reshape(-1, 2)
    return ids, offsets


@torch.no_grad()
def surprisal_of_tokens(model, token_rows):
    """Bits of surprisal of every token but the first, per row of a (batch, n)
    tensor of token ids. Returns a (batch, n-1) float32 array."""
    logits = model(token_rows).logits[:, :-1]      # position i predicts token i+1
    targets = token_rows[:, 1:]
    bits = []
    for i in range(0, targets.shape[1], LOGSOFTMAX_CHUNK):
        logp = torch.log_softmax(logits[:, i:i + LOGSOFTMAX_CHUNK].float(), dim=-1)
        picked = logp.gather(2, targets[:, i:i + LOGSOFTMAX_CHUNK].unsqueeze(2)).squeeze(2)
        bits.append(-picked / LN2)
    return torch.cat(bits, dim=1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def surprisal_full_context(model, tokenizer, text, question=None, max_tokens=MAX_TOKENS):
    """Surprisal of each transcript token given the question (if given) and
    all earlier transcript text. Returns (bits per token, char offsets)."""
    ids, offsets = tokenise(tokenizer, text, max_tokens)
    if not ids:
        return np.zeros(0, dtype=np.float32), offsets
    prefix = [tokenizer.bos_token_id]               # context before the transcript: BOS, then the question if given
    if question:
        prefix += tokenizer(question + "\n\n", add_special_tokens=False)["input_ids"]
    bits = surprisal_of_tokens(model, torch.tensor([prefix + ids], device=model.device))[0]
    return bits[len(prefix) - 1:], offsets          # drop the prefix's own bits; transcript tokens only


@torch.no_grad()
def surprisal_local_window(model, tokenizer, text, window=WINDOW, max_tokens=MAX_TOKENS, batch_size=32):
    """Surprisal of each transcript token given only the previous `window` to
    2*window tokens: the transcript is cut into segments of 2*window tokens
    with stride `window`, each is scored and its back half kept. Segments run
    through the model in batches. Returns (bits per token, char offsets)."""
    ids, offsets = tokenise(tokenizer, text, max_tokens)
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=np.float32), offsets
    bos, pad = tokenizer.bos_token_id, tokenizer.pad_token_id or 0
    bits = np.empty(n, dtype=np.float32)
    head = min(2 * window, n)                       # first segment scored whole
    bits[:head] = surprisal_of_tokens(model, torch.tensor([[bos] + ids[:head]], device=model.device))[0]
    starts = list(range(head, n, window))           # segment starting at s scores tokens s .. s+window-1
    segments = [[bos] + ids[s - window:min(s + window, n)] for s in starts]
    for b in range(0, len(segments), batch_size):
        batch = segments[b:b + batch_size]
        width = max(len(seg) for seg in batch)
        rows = torch.tensor([seg + [pad] * (width - len(seg)) for seg in batch], device=model.device)
        scored = surprisal_of_tokens(model, rows)   # padding sits after the tokens we keep
        for seg, s, row in zip(batch, starts[b:b + batch_size], scored):
            n_keep = len(seg) - 1 - window
            bits[s:s + n_keep] = row[window:window + n_keep]
    return bits, offsets


def assistant_token_mask(text, offsets):
    """True for tokens whose middle character lies in an [Assistant] block;
    a transcript without role markers is all assistant text."""
    markers = list(ROLE_MARKER.finditer(text))
    if not markers:
        return np.ones(len(offsets), dtype=bool)
    mid = offsets.mean(axis=1)
    mask = np.zeros(len(offsets), dtype=bool)
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        if m.group(1).lower() == "assistant":
            mask |= (mid >= m.end()) & (mid < end)
    return mask


def cache_file(cache_dir, scorer, bench, model, item, epoch):
    """Where one transcript's passes under one scorer are stored."""
    safe = lambda s: re.sub(r"[^\w.-]", "_", str(s))
    return Path(cache_dir) / scorer / bench / f"{safe(model)}__{safe(item)}__ep{epoch}.npz"


def score_and_cache(model, tokenizer, text, question, path):
    """Run the three passes for one transcript and save them to `path`
    (skipped if the file exists)."""
    if path.exists():
        return
    full, offsets = surprisal_full_context(model, tokenizer, text, question)
    noq, _ = surprisal_full_context(model, tokenizer, text)
    local, _ = surprisal_local_window(model, tokenizer, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".tmp.npz"), full=full, noq=noq, local=local, offsets=offsets)
    os.replace(path.with_suffix(".tmp.npz"), path)   # never a half-written file under the final name


def surprisal_features(text, bench, model, item, epoch, cache_dir=CACHE_DIR):
    """Features for one transcript from the cache, for every scorer:
    surprisal_full__<scorer>, surprisal_noq__<scorer>, surprisal_local__<scorer>
    (total bits over assistant tokens) and surprisal_chars__<scorer> (chars
    scored). Missing cache file = error."""
    out = {}
    for scorer in SCORERS:
        z = np.load(cache_file(cache_dir, scorer, bench, model, item, epoch))   # full, noq, local, offsets
        keep = assistant_token_mask(text, z["offsets"])                         # score the model's own words only
        for pass_name in ("full", "noq", "local"):
            out[f"surprisal_{pass_name}__{scorer}"] = float(z[pass_name][keep].sum())
        out[f"surprisal_chars__{scorer}"] = float((z["offsets"][keep, 1] - z["offsets"][keep, 0]).sum())
    return out


if __name__ == "__main__":
    store = pd.read_parquet(sys.argv[1])
    store = store[store.correct] if "correct" in store else store
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE_DIR
    scorers = {sys.argv[3]: SCORERS[sys.argv[3]]} if len(sys.argv) > 3 else SCORERS
    for scorer, hf_id in scorers.items():
        todo = [r for r in store.itertuples() if not cache_file(cache_dir, scorer, r.bench, r.model, r.item, r.epoch).exists()]
        print(f"{scorer}: {len(todo)} of {len(store)} transcripts to score", flush=True)
        if not todo:
            continue
        model, tokenizer = load_scorer(hf_id)
        t0 = time.time()
        for i, r in enumerate(todo):
            score_and_cache(model, tokenizer, r.transcript, r.question, cache_file(cache_dir, scorer, r.bench, r.model, r.item, r.epoch))
            if i % 50 == 0:
                print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        del model
