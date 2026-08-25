"""Whole-transcript rolling surprisal with short and long contexts.

Each target block is teacher-forced with its preceding history, so after the
initial ramp its effective transcript contexts range from history through
history + block - 1 tokens.  Unlike ``llm_surprisal.py``, this never truncates
the transcript.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from llm_surprisal import (SCORERS, assistant_token_mask, cache_file, load_scorer,
                           start_token, surprisal_of_tokens, tokenise)

SETTINGS = {"short": (32, 32), "long": (7168, 1024)}


def question_prefix(tokenizer, question):
    ids = [start_token(tokenizer)]
    if question:
        ids += tokenizer(question + "\n\n", add_special_tokens=False)["input_ids"]
    return ids


def rolling_surprisal(model, ids, prefix, history, block, start=0, keep=None,
                      batch_size=1):
    """Score targets in overlapping context-plus-target blocks from ``start``."""
    values = np.full(len(ids), np.nan, dtype=np.float32)
    if start >= len(ids):
        return values
    keep = np.ones(len(ids), dtype=bool) if keep is None else keep
    chunks = []
    for offset in range(start // block * block, len(ids), block):
        end = min(offset + block, len(ids))
        selected = np.flatnonzero(keep[max(offset, start):end]) + max(offset, start)
        if len(selected):
            context = ids[max(0, offset - history):offset]
            chunks.append((offset, selected, prefix + context + ids[offset:end],
                           len(prefix) + len(context) - 1))
    for first in range(0, len(chunks), batch_size):
        batch = chunks[first:first + batch_size]
        width = max(len(row) for _, _, row, _ in batch)
        rows = torch.tensor([row + [prefix[0]] * (width - len(row))
                             for _, _, row, _ in batch], device=model.device)
        for (offset, selected, _, begin), scored in zip(batch,
                                                        surprisal_of_tokens(model, rows)):
            values[selected] = scored[begin + selected - offset]
    return values


def _prefix_values(path, content_hash, offsets):
    """Validated compatible arrays from an old content-addressed cache, if any."""
    if path is None or not path.exists():
        return None
    try:
        with np.load(path) as source:
            source_offsets = source["offsets"]
            source_n = len(source_offsets)
            cap = min(source_n, 8192)
            if (source["content_hash"].item() != content_hash
                    or source_n > len(offsets)
                    or not np.array_equal(source_offsets, offsets[:source_n])):
                return None
            arrays = {"short": source["local"], "long": source["noq"],
                      "long_question": source["full"]}
            if any(len(values) != source_n or not np.isfinite(values).all()
                   for values in arrays.values()):
                return None
            return {name: values[:cap].astype(np.float32, copy=True)
                    for name, values in arrays.items()}
    except (KeyError, OSError, ValueError):
        return None


def _valid(path, content_hash, offsets, assistant_mask):
    try:
        with np.load(path) as saved:
            return _matches(saved, content_hash, offsets, assistant_mask)
    except (KeyError, OSError, ValueError):
        return False


def _matches(saved, content_hash, offsets, assistant_mask):
    return (saved["content_hash"].item() == content_hash
            and np.array_equal(saved["offsets"], offsets)
            and np.array_equal(saved["assistant_mask"], assistant_mask)
            and all(len(saved[name]) == len(offsets)
                    and np.isfinite(saved[name][assistant_mask]).all()
                    and np.isnan(saved[name][~assistant_mask]).all()
                    for name in ("short", "long", "long_question"))
            and all(saved[f"{name}_history"].item() == history
                    and saved[f"{name}_block"].item() == block
                    for name, (history, block) in SETTINGS.items()))


def _summaries(out, name, values, keep, scorer):
    selected = values[keep]
    prefix = f"surprisal_rolling_{name}_"
    out[f"{prefix}sum__{scorer}"] = float(selected.sum())
    out[f"{prefix}count__{scorer}"] = float(len(selected))
    stats = (("mean", selected.mean()), ("median", np.median(selected)),
             ("variance", selected.var()), ("min", selected.min()),
             ("max", selected.max())) if len(selected) else (
                 ("mean", np.nan), ("median", np.nan), ("variance", np.nan),
                 ("min", np.nan), ("max", np.nan))
    for stat, value in stats:
        out[f"{prefix}{stat}__{scorer}"] = float(value)


def multiscale_features(text, parts, bench, model, item, epoch, log,
                        content_hash, cache_dir):
    """Summarise whole-transcript rolling caches over assistant tokens."""
    out = {}
    for scorer in SCORERS:
        path = cache_file(cache_dir, scorer, bench, model, item, epoch, log,
                          content_hash)
        with np.load(path) as saved:
            offsets = saved["offsets"]
            assistant = assistant_token_mask(text, offsets, parts)
            if not _matches(saved, content_hash, offsets, assistant):
                raise ValueError(f"invalid rolling cache: {path}")
            short, long, long_question = (saved[name] for name in
                                          ("short", "long", "long_question"))
            index = np.arange(len(offsets))
            context = assistant & (index >= 64)
            tail = assistant & (index >= 8192)
            groups = {
                "short": (short, assistant),
                "long": (long, assistant),
                "long_question": (long_question, assistant),
                "context_gain": (short - long, context),
                "question_gain": (long - long_question, assistant),
                "tail_context_gain": (short - long, tail),
                "tail_question_gain": (long - long_question, tail),
            }
            widths = offsets[:, 1] - offsets[:, 0]
            masks = {"": assistant, "context_": context, "tail_": tail}
            for label, mask in masks.items():
                out[f"surprisal_rolling_{label}chars__{scorer}"] = float(widths[mask].sum())
            out[f"surprisal_rolling_tail_available__{scorer}"] = float(tail.any())
            for name, (values, keep) in groups.items():
                _summaries(out, name, values, keep, scorer)
                chars = float(widths[keep].sum())
                out[f"surprisal_rolling_{name}_bpc__{scorer}"] = (
                    float(values[keep].sum() / chars) if chars else np.nan)
    return out


def score_and_cache(model, tokenizer, text, question, parts, path, content_hash,
                    prefix_cache=None):
    """Score assistant tokens and atomically write ``path``; reuse a valid prefix."""
    ids, offsets = tokenise(tokenizer, text, max_tokens=None)
    assistant_mask = assistant_token_mask(text, offsets, parts)
    path = Path(path)
    if path.exists():
        if _valid(path, content_hash, offsets, assistant_mask):
            return path
        raise ValueError(f"refusing to overwrite invalid cache: {path}")
    temporary = path.with_suffix(".tmp.npz")
    if temporary.exists():
        if _valid(temporary, content_hash, offsets, assistant_mask):
            os.rename(temporary, path)
            return path
        raise ValueError(f"refusing to overwrite partial cache: {temporary}")
    reused = _prefix_values(Path(prefix_cache) if prefix_cache else None, content_hash, offsets)
    base, with_question = question_prefix(tokenizer, None), question_prefix(tokenizer, question)
    values = {}
    for name, (history, block) in SETTINGS.items():
        cached = reused.get(name) if reused and name in reused else None
        start = len(cached) if cached is not None else 0
        scored = rolling_surprisal(model, ids, base, history, block, start,
                                   assistant_mask,
                                   batch_size=32 if name == "short" else 1)
        if cached is not None:
            scored[:start] = np.where(assistant_mask[:start], cached, np.nan)
        values[name] = scored
    cached = reused.get("long_question") if reused else None
    start = len(cached) if cached is not None else 0
    scored = rolling_surprisal(model, ids, with_question, *SETTINGS["long"],
                               start=start, keep=assistant_mask)
    if cached is not None:
        scored[:start] = np.where(assistant_mask[:start], cached, np.nan)
    values["long_question"] = scored
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(temporary, **values, offsets=offsets, assistant_mask=assistant_mask,
             content_hash=content_hash,
             **{f"{name}_{field}": value for name, pair in SETTINGS.items()
                for field, value in zip(("history", "block"), pair)})
    os.rename(temporary, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("output_cache", type=Path)
    parser.add_argument("scorer", choices=SCORERS)
    parser.add_argument("--prefix-cache", type=Path)
    args = parser.parse_args(argv)
    store = pd.read_parquet(args.store)
    required = {"is_canonical", "correct"}
    if not required <= set(store):
        raise ValueError(f"store must contain {sorted(required)}")
    rows = store[store.is_canonical & store.correct]
    model, tokenizer = load_scorer(SCORERS[args.scorer])
    started = time.time()
    print(f"{args.scorer}: {len(rows)} canonical-correct transcripts", flush=True)
    for index, row in enumerate(rows.itertuples(index=False), 1):
        path = cache_file(args.output_cache, args.scorer, row.bench, row.model, row.item,
                          row.epoch, row.log, row.content_hash)
        prefix = (cache_file(args.prefix_cache, args.scorer, row.bench, row.model, row.item,
                             row.epoch, row.log, row.content_hash)
                  if args.prefix_cache else None)
        parts = json.loads(row.transcript_parts_json)
        score_and_cache(model, tokenizer, row.transcript, row.question, parts,
                        path, row.content_hash, prefix)
        if index % 25 == 0:
            print(f"{index}/{len(rows)} {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
