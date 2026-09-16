#!/usr/bin/env python3
"""Validate an embedding cache independently of the job that wrote it.

    python scripts/embeddings/check_store.py [--root /fast/jyeon/smalljudge_embeddings] [--data data] [--verify 20]

Every .npz under --root must load, its arrays must agree in length, ids must be unique, and every
embedding must be a finite unit vector (float16 tolerance). For up to --verify files per bench the ids
and correctness flags are compared with the source log's sample summaries under --data. Prints one
line per bench/variant with file, sample and chunk totals and exits 1 on any failure.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evals.sample import extract_score  # noqa: E402


def check_file(path: Path) -> tuple[dict, list[str]]:
    problems = []
    with np.load(path) as store:
        arrays = {k: store[k] for k in ("ids", "epochs", "correct", "n_tokens", "n_chunks", "embedding")}
        meta = json.loads(str(store["meta"]))
    n = len(arrays["ids"])
    if any(len(v) != n for v in arrays.values()):
        problems.append("array lengths differ")
    if len(set(zip(arrays["ids"].tolist(), arrays["epochs"].tolist()))) != n:
        problems.append("duplicate (id, epoch) rows")
    emb = arrays["embedding"].astype(np.float32)
    if n and (not np.isfinite(emb).all() or np.abs(np.linalg.norm(emb, axis=1) - 1).max() > 0.02):
        problems.append("embeddings are not finite unit vectors")
    if emb.ndim != 2 or emb.shape[1] != 4096:
        problems.append(f"embedding shape {emb.shape}")
    if n and (arrays["n_chunks"] < 1).any():
        problems.append("rows without chunks")  # an empty transcript still embeds as one EOS-only chunk
    first = arrays["epochs"] == 1
    return {"n": n, "chunks": int(arrays["n_chunks"].sum()), "correct": int(arrays["correct"].sum()),
            "max_tokens": int(arrays["n_tokens"].max()) if n else 0, "ids": arrays["ids"][first].tolist(),
            "flags": arrays["correct"][first].tolist(), "meta": meta}, problems


def verify_against_log(info: dict, log: Path) -> list[str]:
    from inspect_ai.log import read_eval_log_sample_summaries

    summaries = {str(s.id): s for s in read_eval_log_sample_summaries(str(log)) if getattr(s, "epoch", 1) == 1}
    problems = []  # info["ids"]/["flags"] hold the epoch-1 rows only, so they compare with the epoch-1 summaries
    unknown = [i for i in info["ids"] if i not in summaries]
    if unknown:
        problems.append(f"{len(unknown)} ids not in the log, e.g. {unknown[:3]}")
    try:
        wrong = [i for i, flag in zip(info["ids"], info["flags"]) if i in summaries and extract_score(summaries[i]) != flag]
        if wrong:
            problems.append(f"{len(wrong)} correctness flags disagree with the log, e.g. {wrong[:3]}")
    except Exception as error:  # noqa: BLE001  summaries lack scores on some inspect versions
        problems.append(f"could not compare correctness: {type(error).__name__}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("/fast/jyeon/smalljudge_embeddings"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--verify", type=int, default=20, help="files per bench to compare with their source log")
    args = parser.parse_args()
    files = sorted(p for p in args.root.rglob("*.npz") if not p.name.startswith("."))
    totals = defaultdict(lambda: {"files": 0, "samples": 0, "correct": 0, "chunks": 0, "max_tokens": 0})
    failures = 0
    verified = defaultdict(int)
    for path in files:
        rel = path.relative_to(args.root)
        bench, variant = rel.parts[0], path.suffixes[-2].lstrip(".")
        info, problems = check_file(path)
        if verified[bench] < args.verify and not problems:
            log = args.data / rel.parent / f"{path.name.removesuffix(f'.{variant}.npz')}.eval"
            problems += verify_against_log(info, log) if log.exists() else [f"source log missing: {log}"]
            verified[bench] += 1
        if problems:
            failures += 1
            print(f"FAIL {rel}: {'; '.join(problems)}")
        t = totals[(bench, variant)]
        t["files"] += 1
        t["samples"] += info["n"]
        t["correct"] += info["correct"]
        t["chunks"] += info["chunks"]
        t["max_tokens"] = max(t["max_tokens"], info["max_tokens"])
    for (bench, variant), t in sorted(totals.items()):
        print(f"{bench:20s} {variant:8s} files {t['files']:5d}  samples {t['samples']:7d}  correct {t['correct']:7d}  "
              f"chunks {t['chunks']:7d}  max_tokens {t['max_tokens']:7d}")
    print(f"{len(files)} files, {failures} failures, {sum(verified.values())} compared with their logs")
    return 1 if failures or not files else 0


if __name__ == "__main__":
    raise SystemExit(main())
