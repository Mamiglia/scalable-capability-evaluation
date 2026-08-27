"""Annotate intermediate errors in correct transcripts with a local model served by vLLM.

    python intermediate_errors.py --input transcripts.parquet --output errors.jsonl --base-url http://localhost:PORT/v1 --model Qwen3-32B

Each transcript reaches the correct final answer; the model is asked to list errors in the
intermediate reasoning only (wrong names or notation, wrong constants or formulas, arithmetic
slips, contradictions between steps, misapplied definitions) and to quote each one. One JSON line
per transcript: content_hash, errors (list of {quote, kind, why}), n_errors, raw. Resumable: rows
already in the output are skipped. Standard library only, so it runs from any venv.
"""
import argparse
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

KINDS = ["nomenclature", "notation", "constant_or_formula", "arithmetic", "contradiction", "definition", "other"]
SYSTEM = ("You are a meticulous scientific referee. You will be shown a problem and a solution whose FINAL ANSWER IS "
          "CORRECT. Your job is to find errors in the intermediate reasoning only: wrong chemical or physical names "
          "(nomenclature), wrong or inconsistent notation, wrong constants or formulas (e.g. a missing factor of 2), "
          "arithmetic slips, statements that contradict an earlier step, misapplied definitions, and claims that are "
          "false even though the answer comes out right. Do not report stylistic issues, omissions, or the final "
          "answer. If there are no errors, say so. Answer with JSON only:\n"
          '{"errors": [{"quote": "<exact words from the solution>", "kind": "<' + "|".join(KINDS) + '>", '
          '"why": "<one sentence>"}]}')
JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def ask(base_url, model, question, transcript, max_tokens, timeout):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"PROBLEM:\n{question}\n\nSOLUTION:\n{transcript}"}],
            "chat_template_kwargs": {"enable_thinking": True}}
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    return reply["choices"][0]["message"]["content"] or ""


def parse(text):
    match = JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        errors = json.loads(match.group(0)).get("errors", [])
    except json.JSONDecodeError:
        return None
    return [e for e in errors if isinstance(e, dict) and e.get("quote")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="parquet with content_hash, question, transcript")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = pd.read_parquet(args.input)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["content_hash"] for line in args.output.open() if line.strip()}
    todo = rows[~rows.content_hash.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} done, {len(todo)} to annotate", flush=True)
    lock = threading.Lock()
    counter = {"n": 0, "start": time.time()}

    def work(row):
        for attempt in range(3):
            try:
                text = ask(args.base_url, args.model, row.question, row.transcript, args.max_tokens, args.timeout)
                break
            except Exception as error:      # server hiccup: retry, then record the failure
                text, failure = "", repr(error)
                time.sleep(5 * (attempt + 1))
        errors = parse(text)
        record = {"content_hash": row.content_hash, "bench": row.bench, "model": row.model, "item": row.item,
                  "errors": errors if errors is not None else [], "n_errors": len(errors) if errors is not None else None,
                  "parsed": errors is not None, "raw": text[-4000:]}
        with lock:
            with args.output.open("a") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counter["n"] += 1
            if counter["n"] % 50 == 0:
                rate = counter["n"] / (time.time() - counter["start"])
                print(f"{counter['n']}/{len(todo)} annotated, {rate * 3600:.0f}/h", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo.itertuples()))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
