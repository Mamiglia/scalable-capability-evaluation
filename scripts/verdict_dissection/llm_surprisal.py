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
import json
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


def start_token(tokenizer):
    """Token that stands for "start of text": BOS where the tokenizer has one,
    otherwise EOS (Qwen has no BOS; its pretraining separates documents with EOS)."""
    return tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id


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
    prefix = [start_token(tokenizer)]               # context before the transcript: BOS, then the question if given
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
    bos, pad = start_token(tokenizer), tokenizer.pad_token_id or 0
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


def assistant_token_mask(text, offsets, parts):
    """True for token midpoints inside structured assistant-message spans."""
    rendered = "\n\n".join(
        part["content"] if part["bare"] else f"[{part['role'].capitalize()}]: {part['content']}"
        for part in parts)
    if rendered != text:
        raise ValueError("structured transcript parts do not reconcile with rendered text")
    mid = offsets.mean(axis=1)
    mask = np.zeros(len(offsets), dtype=bool)
    for part in parts:
        if part["role"] == "assistant":
            mask |= (mid >= part["start"]) & (mid < part["end"])
    return mask


def cache_file(cache_dir, scorer, bench, model, item, epoch, log, content_hash):
    """Content-addressed path for one transcript under one scorer."""
    safe = lambda s: re.sub(r"[^\w.-]", "_", str(s))
    name = f"{safe(model)}__{safe(item)}__ep{epoch}__{safe(log)}__{content_hash[:16]}.npz"
    return Path(cache_dir) / scorer / bench / name


def score_and_cache(model, tokenizer, text, question, path, content_hash):
    """Run the three passes for one transcript and save them to `path`
    (skipped if the file exists)."""
    if path.exists():
        if np.load(path)["content_hash"].item() != content_hash:
            raise ValueError(f"cache content hash mismatch: {path}")
        return
    full, offsets = surprisal_full_context(model, tokenizer, text, question)
    noq, _ = surprisal_full_context(model, tokenizer, text)
    local, _ = surprisal_local_window(model, tokenizer, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".tmp.npz"), full=full, noq=noq, local=local,
             offsets=offsets, content_hash=content_hash)
    os.replace(path.with_suffix(".tmp.npz"), path)   # never a half-written file under the final name


def surprisal_features(text, parts, bench, model, item, epoch, log, content_hash, cache_dir=CACHE_DIR):
    """Features for one transcript from the cache, for every scorer:
    surprisal_full__<scorer>, surprisal_noq__<scorer>, surprisal_local__<scorer>
    (total bits over assistant tokens) and surprisal_chars__<scorer> (chars
    scored). Missing cache file = error."""
    out = {}
    for scorer in SCORERS:
        path = cache_file(cache_dir, scorer, bench, model, item, epoch, log, content_hash)
        z = np.load(path)
        if z["content_hash"].item() != content_hash:
            raise ValueError(f"cache content hash mismatch: {path}")
        offsets = z["offsets"]
        keep = assistant_token_mask(text, offsets, parts)
        chars = float((offsets[keep, 1] - offsets[keep, 0]).sum())
        values = {
            "full": z["full"], "noq": z["noq"], "local": z["local"],
            "question_gain": z["noq"] - z["full"],
            "total_context_gain": z["local"] - z["full"],
            "long_context_gain": z["local"] - z["noq"],
        }
        totals = {name: float(values[name][keep].sum()) for name in ("full", "noq", "local")}
        for pass_name in ("full", "noq", "local"):
            out[f"surprisal_{pass_name}__{scorer}"] = totals[pass_name]
        out[f"surprisal_chars__{scorer}"] = chars
        out[f"surprisal_full_bpc__{scorer}"] = totals["full"] / chars if chars else 0.0
        out[f"surprisal_question_gain_bpc__{scorer}"] = values["question_gain"][keep].sum() / chars if chars else 0.0
        out[f"surprisal_total_context_gain_bpc__{scorer}"] = (
            values["total_context_gain"][keep].sum() / chars if chars else 0.0)
        long_keep = keep & (np.arange(len(offsets)) >= 64)
        long_chars = float((offsets[long_keep, 1] - offsets[long_keep, 0]).sum())
        out[f"surprisal_long_chars__{scorer}"] = long_chars
        out[f"surprisal_long_available__{scorer}"] = float(long_chars > 0)
        out[f"surprisal_long_context_gain_bpc__{scorer}"] = (
            values["long_context_gain"][long_keep].sum() / long_chars if long_chars else 0.0)
        for name, values_ in values.items():
            selected = values_[long_keep if name == "long_context_gain" else keep]
            prefix = f"surprisal_{name}_"
            out[f"{prefix}sum__{scorer}"] = float(selected.sum())
            out[f"{prefix}count__{scorer}"] = float(len(selected))
            if len(selected):
                stats = (("mean", selected.mean()), ("median", np.median(selected)),
                         ("variance", selected.var()), ("min", selected.min()), ("max", selected.max()))
            else:
                stats = (("mean", np.nan), ("median", np.nan), ("variance", np.nan),
                         ("min", np.nan), ("max", np.nan))
            for stat, value in stats:
                out[f"{prefix}{stat}__{scorer}"] = float(value)
    return out


if __name__ == "__main__":
    store = pd.read_parquet(sys.argv[1])
    if {"correct", "is_canonical"} <= set(store):
        store = store[store.correct & store.is_canonical]
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE_DIR
    scorers = {sys.argv[3]: SCORERS[sys.argv[3]]} if len(sys.argv) > 3 else SCORERS
    for scorer, hf_id in scorers.items():
        todo = [
            row for row in store.itertuples()
            if not cache_file(cache_dir, scorer, row.bench, row.model, row.item,
                              row.epoch, row.log, row.content_hash).exists()
        ]
        print(f"{scorer}: {len(todo)} of {len(store)} transcripts to score", flush=True)
        if not todo:
            continue
        model, tokenizer = load_scorer(hf_id)
        t0 = time.time()
        for i, r in enumerate(todo):
            parts = json.loads(r.transcript_parts_json)
            path = cache_file(cache_dir, scorer, r.bench, r.model, r.item, r.epoch, r.log, r.content_hash)
            score_and_cache(model, tokenizer, r.transcript, r.question, path, r.content_hash)
            assistant_token_mask(r.transcript, np.load(path)["offsets"], parts)
            if i % 50 == 0:
                print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        del model
