"""Answer pending description prompts with the Codex CLI (subscription), appending replies to the cache.

Reads pending.jsonl written by transcript_sae_describe.py --offline, runs `codex exec` per
prompt with N workers, and appends {key, model, reply, provider} lines to llm_cache.jsonl.
Re-running transcript_sae_describe.py --offline then picks the replies up from the cache.
"""
import argparse
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def codex_reply(prompt: str, model: str, effort: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
        out = Path(handle.name)
    result = subprocess.run(
        ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
         "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "-o", str(out), "-"],
        input=prompt, capture_output=True, text=True, timeout=600)
    reply = out.read_text() if out.exists() else ""
    out.unlink(missing_ok=True)
    if not reply.strip():
        raise RuntimeError(f"codex returned nothing (exit {result.returncode}): {result.stderr[-300:]}")
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="dir holding pending.jsonl and llm_cache.jsonl")
    parser.add_argument("--only-model", help="answer only prompts addressed to this model name (as in pending.jsonl)")
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cache_path = args.output_dir / "llm_cache.jsonl"
    done = set()
    if cache_path.exists():
        done = {json.loads(line)["key"] for line in cache_path.read_text().splitlines()}
    pending = [json.loads(line) for line in (args.output_dir / "pending.jsonl").read_text().splitlines()]
    pending = [item for item in pending if item["key"] not in done
               and (args.only_model is None or item["model"] == args.only_model)]
    if args.limit:
        pending = pending[:args.limit]
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}

    def answer(item):
        try:
            reply = codex_reply(item["prompt"], args.codex_model, args.effort)
        except Exception as error:
            with lock:
                counts["failed"] += 1
                print(f"failed {item['key'][:12]}: {error}", flush=True)
            return
        with lock:
            with cache_path.open("a") as handle:
                handle.write(json.dumps({"key": item["key"], "model": item["model"], "reply": reply,
                                         "provider": f"codex:{args.codex_model}:{args.effort}"}) + "\n")
            counts["ok"] += 1
            if counts["ok"] % 25 == 0:
                print(f"answered {counts['ok']}/{len(pending)}", flush=True)

    print(f"answering {len(pending)} pending prompts with codex {args.codex_model} ({args.effort})", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(answer, pending))
    print(counts)


if __name__ == "__main__":
    main()
