#!/usr/bin/env python3
"""Embed collected transcripts with Qwen3-Embedding-8B into the cache protocols.embeddings reads.

    python scripts/embeddings/embed.py --bench gpqa arc_challenge hle [--out /fast/jyeon/smalljudge_embeddings]
        [--variant full answer] [--limit-logs N] [--limit-samples N] [--all-items]

GPU job (scripts/embeddings/embed.sub). For every successful collection log
data/<bench>/<hash>/<model>/<repeat>/*.eval (all repeats) and every text variant it writes
<out>/<bench>/<hash>/<model>/<repeat>/<log>.<variant>.npz (arrays: see protocols.embeddings).
Existing files are skipped, so a re-run only fills gaps. Samples are restricted to the items of the
frozen tables that use the bench (data/meta/full/items.csv, e.g. HLE's gold science+math subset);
a bench absent from every frozen table, or --all-items, embeds every sample.

Text: evals.sample.extract_transcript (the judge-visible transcript), tokenised without truncation,
chunked at --max-tokens with --overlap (at most --max-chunks chunks, the rest is dropped), each chunk
ending in <|endoftext|>, last-token pooled, L2-normalised, token-weighted mean over chunks
(as scripts/hibayes/sae_judge.py), stored as float16.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from configs.paths import collection_logs, is_successful_eval  # noqa: E402
from evals.sample import extract_score, extract_transcript  # noqa: E402
from inspect_ai.log import EvalSample  # noqa: E402
from pydantic import ValidationError  # noqa: E402
import protocols.embeddings as store  # noqa: E402

try:
    import zipfile_zstd  # noqa: F401  registers zstd support for archives that use it
except ImportError:
    pass
import zipfile  # noqa: E402


def iter_samples(log: Path):
    """Every sample of an .eval archive, one member at a time; a sample the installed inspect_ai cannot
    validate (a log written by a newer version, e.g. an unknown tool-error type) is skipped and counted
    instead of aborting the log, unlike inspect's streaming reader."""
    skipped = 0
    with zipfile.ZipFile(log) as archive:
        members = sorted(m for m in archive.namelist() if m.startswith("samples/") and m.endswith(".json"))
        for member in members:
            try:
                yield EvalSample.model_validate_json(archive.read(member))
            except ValidationError:
                skipped += 1
    if skipped:
        print(f"{log.name}: skipped {skipped} samples that failed validation", flush=True)

MODEL = "Qwen/Qwen3-Embedding-8B"
REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INCLUDE_REASONING = {"full": True, "answer": False}


def frozen_items(data: Path) -> dict[str, set[str]]:
    """bench -> item ids that appear in the frozen full/items.csv."""
    table = pd.read_csv(data / "meta" / "full" / "items.csv", dtype={"item": str})
    if os.environ.get("EMBED_TRAIN_ONLY"):  # ponytail: only what protocols.embeddings reads (train items); such
        table = pd.read_csv(data / "meta" / "full" / "task_items.csv", dtype={"item": str}).query("train")  # files are partial -> keep them out of the HF cache
    return {bench: set(ids) for bench, ids in table.groupby("bench").item}


def referenced_logs(data: Path) -> set[str]:
    """Relative repeat-0 log paths in the frozen full/samples.csv: the logs the protocols read."""
    samples = pd.read_csv(data / "meta" / "full" / "samples.csv", usecols=["repeat", "log"], dtype={"log": str})
    return set(samples[samples.repeat == 0].log.dropna())


def chunk(ids: list[int], size: int, overlap: int, max_chunks: int) -> list[list[int]]:
    if len(ids) <= size:
        return [ids]
    return [ids[i:i + size] for i in range(0, len(ids), size - overlap)][:max_chunks]


class Embedder:
    def __init__(self, max_tokens: int, overlap: int, max_chunks: int, tokens_per_batch: int):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if os.environ.get("EMBED_NO_GQA_SDPA"):  # pre-Ampere cards: SDPA's enable_gqa needs the flash kernel; without it
            import transformers.integrations.sdpa_attention as sdpa  # the math kernel materialises N^2 attention.
            sdpa.use_gqa_in_sdpa = lambda *a, **k: False  # repeat K/V instead -> memory-efficient kernel

        self.torch = torch
        self.max_tokens, self.overlap, self.max_chunks, self.tokens_per_batch = max_tokens, overlap, max_chunks, tokens_per_batch
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
        self.eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        if not torch.cuda.is_available():  # a broken GPU slot must not degrade into days of CPU inference
            print("no usable CUDA device (torch.cuda.is_available() is False); exiting 6 for a requeue", flush=True)
            raise SystemExit(6)
        self.device = "cuda"
        # ponytail: EMBED_DTYPE=float16 for Pascal cards (no bf16 kernels); the cache was built in bf16 on H100s
        dtype = getattr(torch, os.environ.get("EMBED_DTYPE", "bfloat16"))
        # ponytail: EMBED_DEVICE_MAP=auto shards the layers over every visible GPU (two 24 GB cards: fp16 weights
        # alone leave too little of one card for an 8192-token chunk); needs accelerate
        if os.environ.get("EMBED_DEVICE_MAP"):
            self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype,
                                                   device_map=os.environ["EMBED_DEVICE_MAP"]).eval()
        else:
            self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype).to(self.device).eval()
        self.dim = self.model.config.hidden_size
        print(f"model {MODEL}@{REVISION[:8]} on {self.device}, dim {self.dim}", flush=True)

    def __call__(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(embedding float16 [n, dim], n_tokens int32 [n], n_chunks int32 [n])."""
        torch = self.torch
        encoded = self.tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
        n_tokens = np.array([len(ids) for ids in encoded], dtype=np.int32)
        flat, owner = [], []
        for i, ids in enumerate(encoded):
            # one chunk of room for the trailing EOS; every chunk ends in <|endoftext|> like a whole text does
            for piece in chunk(ids, self.max_tokens - 1, self.overlap, self.max_chunks):
                flat.append(piece if piece and piece[-1] == self.eos else piece + [self.eos])
                owner.append(i)
        owner = np.asarray(owner)
        embeddings = np.zeros((len(flat), self.dim), dtype=np.float32)
        order = np.argsort([len(x) for x in flat], kind="stable")
        pad = self.tok.pad_token_id or 0
        start = 0
        with torch.inference_mode():
            while start < len(order):
                longest = len(flat[order[start]])  # ascending order: the batch's last member is its longest
                end = start
                while end < len(order) and (end - start + 1) * len(flat[order[end]]) <= self.tokens_per_batch:
                    end += 1
                end = max(end, start + 1)
                batch = [flat[j] for j in order[start:end]]
                length = max(len(x) for x in batch)
                inp = torch.full((len(batch), length), pad, dtype=torch.long)
                att = torch.zeros((len(batch), length), dtype=torch.long)
                for i, ids in enumerate(batch):  # left padding
                    inp[i, length - len(ids):] = torch.tensor(ids)
                    att[i, length - len(ids):] = 1
                # one unpadded sequence needs no mask; passing one makes SDPA fall back to the math kernel, which
                # materialises the 8192^2 attention matrix and OOMs a 24 GB card (--tokens-per-batch 1)
                mask = None if len(batch) == 1 else att.to(self.device)
                out = self.model(input_ids=inp.to(self.device), attention_mask=mask,
                                 use_cache=False).last_hidden_state[:, -1]
                embeddings[order[start:end]] = torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()
                start = end
        weights = np.array([len(x) for x in flat], dtype=np.float32)
        mean = np.zeros((len(texts), self.dim), dtype=np.float32)
        n_chunks = np.zeros(len(texts), dtype=np.int32)
        for i in range(len(texts)):
            member = owner == i
            n_chunks[i] = member.sum()
            mean[i] = (embeddings[member] * weights[member, None]).sum(0) / weights[member].sum()
            mean[i] /= max(np.linalg.norm(mean[i]), 1e-12)  # a mean of unit vectors is shorter than one
        return mean.astype(np.float16), n_tokens, n_chunks


def write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(fd)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", nargs="+", required=True, help="data/<bench> directories")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("/fast/jyeon/smalljudge_embeddings"))
    parser.add_argument("--variant", nargs="+", default=list(store.VARIANTS), choices=store.VARIANTS)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--all-items", action="store_true", help="ignore the frozen tables' item lists")
    parser.add_argument("--limit-logs", type=int, default=0, help="smoke: at most N logs per bench")
    parser.add_argument("--limit-samples", type=int, default=0, help="smoke: at most N samples per log")
    parser.add_argument("--shard", default="0/1", help="k/n: this job takes every n-th log of each bench (n GPU jobs in parallel)")
    parser.add_argument("--reverse", action="store_true",
                        help="walk the shard's logs from the end: a second job per shard that meets the first in the middle")
    parser.add_argument("--referenced", action="store_true",
                        help="only the logs referenced by a frozen models.csv (priority pass for the evaluation)")
    parser.add_argument("--seed", type=int, default=0,
                        help="nonzero: walk the shard's logs in a seeded random order (extra jobs that rarely collide)")
    args = parser.parse_args()
    if not 0 <= args.overlap < args.max_tokens - 1:
        parser.error(f"--overlap {args.overlap} must be smaller than --max-tokens - 1 = {args.max_tokens - 1}")
    shard, shards = map(int, args.shard.split("/"))

    store.EMBEDDINGS = args.out
    items = {} if args.all_items else frozen_items(args.data)
    meta = {"model": MODEL, "revision": REVISION, "max_tokens": args.max_tokens, "overlap": args.overlap,
            "max_chunks": args.max_chunks}
    embedder = None
    written = skipped = 0
    referenced = referenced_logs(args.data) if args.referenced else None
    for bench in args.bench:
        logs = collection_logs(args.data / bench)
        if referenced is not None:
            logs = [p for p in logs if str(p.relative_to(args.data)) in referenced]
        # A frozen table already vetted the logs it references; the success filter only prunes bulk runs.
        logs = [p for p in logs[shard::shards] if referenced is not None or is_successful_eval(p)]
        logs = logs[::-1] if args.reverse else logs
        if args.seed:
            random.Random(args.seed).shuffle(logs)
        logs = logs[:args.limit_logs] if args.limit_logs else logs
        keep = items.get(bench)
        print(f"== {bench}: shard {args.shard}, {len(logs)} successful logs, items: {'all' if keep is None else len(keep)}", flush=True)
        for log in logs:
            rel = log.relative_to(args.data)
            targets = {v: store.npz_path(str(rel), v) for v in args.variant}
            targets = {v: p for v, p in targets.items() if not p.exists()}
            if not targets:
                skipped += 1
                continue
            started = time.time()
            rows = []
            for sample in iter_samples(log):
                if keep is not None and str(sample.id) not in keep:
                    continue
                rows.append((str(sample.id), int(getattr(sample, "epoch", 1) or 1), bool(extract_score(sample)),
                             {v: extract_transcript(sample, include_reasoning=INCLUDE_REASONING[v]) for v in targets}))
                if args.limit_samples and len(rows) >= args.limit_samples:
                    break
            if embedder is None:
                embedder = Embedder(args.max_tokens, args.overlap, args.max_chunks, args.tokens_per_batch)
            for variant, path in targets.items():
                texts = [r[3][variant] for r in rows]
                embedding, n_tokens, n_chunks = embedder(texts) if texts else (
                    np.zeros((0, embedder.dim), np.float16), np.zeros(0, np.int32), np.zeros(0, np.int32))
                write_npz(path, dict(ids=np.array([r[0] for r in rows], dtype=str), epochs=np.array([r[1] for r in rows], np.int32),
                                     correct=np.array([r[2] for r in rows], bool), n_tokens=n_tokens, n_chunks=n_chunks,
                                     embedding=embedding, meta=np.array(json.dumps(meta | {"variant": variant}))))
                written += 1
                print(f"{rel}: {variant} {len(rows)} samples, {int(n_chunks.sum())} chunks, "
                      f"max {int(n_tokens.max()) if len(rows) else 0} tokens, {time.time() - started:.0f}s", flush=True)
    print(f"done: {written} files written, {skipped} logs already complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
