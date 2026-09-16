#!/usr/bin/env python3
"""Mirror the latest main of hf.co/datasets/LASR-G5/benchmarks (the repo's `data` submodule) to /fast.

    /usr/bin/python3 scripts/embeddings/pull_data.py [--dest /fast/jyeon/smalljudge_data/benchmarks] [--workers 8]

Runs as an HTCondor CPU job (network, ~27 GB). Resolves main to a commit, then downloads every
file with the size- and sha-verified urllib downloader of scripts/judging_behavior/hf_download.py
(the path that mirrored this repo before; huggingface_hub.snapshot_download stalled on /fast) over
a thread pool. Files already present with the right size/sha are skipped, so a re-run resumes.
Writes manifest.jsonl, download_report.json and REVISION into --dest.
Point the repo at the mirror with: rmdir data && ln -s <dest> data.
"""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judging_behavior"))
from hf_download import api_request, download_one, full_tree, secret  # noqa: E402  HF_TOKEN from env or /home/jyeon/LASR/.env

REPO = "LASR-G5/benchmarks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dest", type=Path, default=Path("/fast/jyeon/smalljudge_data/benchmarks"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", default="0/1", help="k/n: this job takes every n-th file starting at k (run n jobs in parallel)")
    args = parser.parse_args()
    shard, shards = map(int, args.shard.split("/"))
    token = secret("HF_TOKEN")
    with urllib.request.urlopen(api_request(f"https://huggingface.co/api/datasets/{REPO}/revision/{args.revision}", token), timeout=120) as r:
        sha = json.load(r)["sha"]
    files = sorted((e for e in full_tree(REPO, sha, token) if e.get("type") == "file"), key=lambda e: str(e["path"]))
    files = files[shard::shards]
    total = sum(int(e.get("size", 0)) for e in files)
    print(f"{REPO}@{args.revision} = {sha}: shard {args.shard}, {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    args.dest.mkdir(parents=True, exist_ok=True)
    start, rows, failures = time.time(), [], []

    def fetch(entry):
        try:
            return download_one(REPO, sha, entry, args.dest, token)
        except Exception as error:  # noqa: BLE001
            return {"path": str(entry["path"]), "error": str(error)}

    with ThreadPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(fetch, files), 1):
            (failures if "error" in row else rows).append(row)
            if "error" in row:
                print(f"FAIL {row['path']}: {row['error']}", flush=True)
            if i % 500 == 0 or i == len(files):
                done = sum(r["bytes"] for r in rows)
                print(f"{i}/{len(files)} files, {done / 1e9:.2f} GB, {time.time() - start:.0f}s", flush=True)
    tag = "" if shards == 1 else f".shard{shard}of{shards}"
    with (args.dest / f"manifest{tag}.jsonl").open("w") as out:
        out.writelines(json.dumps(r) + "\n" for r in rows)
    report = {"repo": REPO, "revision": sha, "shard": args.shard, "selected": len(files), "ok": len(rows),
              "failed": failures, "total_bytes": sum(r["bytes"] for r in rows), "seconds": round(time.time() - start)}
    (args.dest / f"download_report{tag}.json").write_text(json.dumps(report, indent=1))
    if failures:
        print(f"{len(failures)} failures; REVISION not written")
        return 1
    if shards == 1:  # a sharded run is complete only when every shard's report says so; see check in README
        (args.dest / "REVISION").write_text(sha + "\n")
    print(f"complete: {len(rows)} files, {report['total_bytes'] / 1e9:.2f} GB; revision {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
