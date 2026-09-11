#!/usr/bin/env python3
"""
Downloads all published SWE-bench Verified .eval log archives from Epoch AI's public S3 bucket.

The bucket is world-readable and supports Range, so this route avoids the aws-waf-token
CAPTCHA that gates logs.epoch.ai (the CDN host scripts/analyses/epoch_irt/fetch.py uses).

Features:
- Parallel downloads with configurable concurrency
- Automatic resume for interrupted downloads (HTTP Range)
- File integrity verification via SHA256 checksum
- Clear progress reporting
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST_URL = "https://huggingface.co/datasets/naderalfares/epoch_ai_swebench_verified/raw/main/conversion_manifest.json"
# 64 GB: must live on /media/hdd, not $HOME (sda2 has a 43 GB per-user quota).
OUTPUT_DIR = os.environ.get(
    "EPOCH_SWEBENCH_DIR",
    "/media/hdd/usr/tao/small-judge/external_data/epoch-swe-bench/logs",
)
MAX_CONCURRENT_DOWNLOADS = 4
CHUNK_SIZE = 1024 * 1024  # 1 MB


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha.update(chunk)
    return sha.hexdigest()


def download_run(run_info, output_dir):
    source_url = run_info.get("source_url")
    if not source_url:
        return {"status": "skipped", "run": run_info, "reason": "No source URL"}

    run_id = run_info["id_runs"]
    model = run_info.get("model", run_id).replace("/", "_")
    filename = f"{model}__{run_id}.eval"
    dest_path = os.path.join(output_dir, filename)
    part_path = dest_path + ".part"
    expected_bytes = run_info.get("bytes", 0)
    expected_sha = run_info.get("sha256", "")

    # Check if already completed and valid
    if os.path.exists(dest_path):
        actual_size = os.path.getsize(dest_path)
        if expected_bytes and actual_size == expected_bytes:
            print(f"✓ [EXISTS] {filename} ({actual_size / (1024**2):.1f} MB)")
            return {"status": "already_done", "run": run_info, "path": dest_path}

    existing_bytes = 0
    if os.path.exists(part_path):
        existing_bytes = os.path.getsize(part_path)
        if existing_bytes > expected_bytes:
            os.remove(part_path)
            existing_bytes = 0

    mode = "ab" if existing_bytes > 0 else "wb"
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"⟳ [RESUMING] {filename} from {existing_bytes / (1024**2):.1f} MB...")
    else:
        print(f"↓ [STARTING] {filename} ({expected_bytes / (1024**2):.1f} MB)...")

    req = urllib.request.Request(source_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, mode) as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)

        os.rename(part_path, dest_path)

        actual_size = os.path.getsize(dest_path)
        if expected_bytes and actual_size != expected_bytes:
            return {"status": "error", "run": run_info, "reason": f"Size mismatch: expected {expected_bytes}, got {actual_size}"}

        if expected_sha:
            actual_sha = compute_sha256(dest_path)
            if actual_sha != expected_sha:
                return {"status": "error", "run": run_info, "reason": "SHA256 mismatch"}

        print(f"✓ [COMPLETED] {filename} ({actual_size / (1024**2):.1f} MB)")
        return {"status": "downloaded", "run": run_info, "path": dest_path}

    except Exception as e:
        return {"status": "error", "run": run_info, "reason": str(e)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Fetching benchmark manifest from Hugging Face...")

    req = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        manifest = json.loads(resp.read().decode("utf-8"))

    runs = manifest["source_manifest"]["runs"]
    valid_runs = [r for r in runs if r.get("source_url")]
    total_bytes = sum(r.get("bytes", 0) for r in valid_runs)

    print(f"Found {len(valid_runs)} valid runs to download.")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB across {MAX_CONCURRENT_DOWNLOADS} worker threads.\n")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        future_to_run = {executor.submit(download_run, run, OUTPUT_DIR): run for run in valid_runs}
        for future in as_completed(future_to_run):
            res = future.result()
            results.append(res)
            if res["status"] == "error":
                print(f"✗ [FAILED] {res['run'].get('filename')}: {res['reason']}", file=sys.stderr)

    elapsed = time.time() - start_time
    success = [r for r in results if r["status"] in ("downloaded", "already_done")]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\n==========================================")
    print(f"Download Summary:")
    print(f"  Total runs: {len(valid_runs)}")
    print(f"  Successful: {len(success)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Time taken: {elapsed / 60:.1f} minutes")
    print(f"  Saved to:   {OUTPUT_DIR}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
