"""Annotate intermediate errors in correct transcripts with the Codex CLI (ChatGPT subscription, no API key).

    uv run python scripts/verdict_dissection/annotate/intermediate_errors_codex.py --input transcripts.parquet --output errors.jsonl [--batch 6 --workers 8 --effort medium]

Same task as intermediate_errors.py: each solution reaches the correct final answer; list the slips in
the intermediate reasoning (nomenclature, notation, constants/formulas, arithmetic, contradictions,
misapplied definitions), quoting each one. Several transcripts go in one prompt, numbered; the reply
is JSON with one entry per number. One output line per transcript: content_hash, errors, n_errors,
parsed, raw (the batch reply). Resumable: transcripts already in the output are skipped.
"""
import argparse
import json
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

KINDS = "nomenclature|notation|constant_or_formula|arithmetic|contradiction|definition|other"
INSTRUCTIONS = (
    "You are an expert referee in physics, chemistry, biology and mathematics. Each numbered SOLUTION below reaches "
    "the CORRECT final answer to its PROBLEM. Find errors in the intermediate reasoning only: a misnamed compound or "
    "intermediate, wrong or inconsistent notation, a formula with a wrong or missing factor, a wrong constant, an "
    "arithmetic slip, a statement contradicting an earlier step, a misapplied definition or criterion, a mislabelled "
    "stereocentre or index, a claim that is false even though the answer comes out right. Be exact and conservative: "
    "report only things that are actually wrong, never style, length, omissions, or the final answer. Many solutions "
    "have no errors; then return an empty list for that number.\n"
    'Reply with JSON only, no prose: {"annotations": [{"id": <number>, "errors": [{"quote": "<exact words from the '
    'solution>", "kind": "<' + KINDS + '>", "why": "<one sentence>"}]}, ...]} with one entry per numbered solution.'
)
JSON_BLOCK = re.compile(r"\{.*\}", re.S)
ESCAPE = re.compile(r'\\\\|\\(?![\\"/bfnrtu])')        # a backslash pair (valid) or a lone LaTeX backslash like \( or \frac


def repair_escapes(text):
    """Double every lone backslash that is not a JSON escape, leaving valid pairs alone."""
    return ESCAPE.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "\\\\", text)


def build_prompt(rows):
    parts = [INSTRUCTIONS, ""]
    for number, row in enumerate(rows, start=1):
        parts += [f"===== SOLUTION {number} =====", f"PROBLEM:\n{row.question}", f"\nSOLUTION:\n{row.transcript}", ""]
    return "\n".join(parts)


def codex_reply(prompt, model, effort, timeout):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
        out = Path(handle.name)
    result = subprocess.run(
        ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
         "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "-o", str(out), "-"],
        input=prompt, capture_output=True, text=True, timeout=timeout)
    reply = out.read_text() if out.exists() else ""
    out.unlink(missing_ok=True)
    if not reply.strip():
        raise RuntimeError(f"codex returned nothing (exit {result.returncode}): {result.stderr[-300:]}")
    return reply


def parse(reply, n):
    """{number: errors list} for the numbers present, or None if the reply holds no JSON."""
    match = JSON_BLOCK.search(reply)
    if not match:
        return None
    text = match.group(0)
    try:
        entries = json.loads(text).get("annotations", [])
    except json.JSONDecodeError:
        try:
            entries = json.loads(repair_escapes(text)).get("annotations", [])
        except json.JSONDecodeError:
            return None
    out = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("id"), int) and 1 <= entry["id"] <= n:
            out[entry["id"]] = [e for e in entry.get("errors", []) if isinstance(e, dict) and e.get("quote")]
    if isinstance(entries, list):                 # a number left out of a well-formed reply means no errors found
        for number in range(1, n + 1):
            out.setdefault(number, [])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, help="annotate at most this many transcripts")
    args = parser.parse_args()
    rows = pd.read_parquet(args.input)
    done = set()
    if args.output.exists():
        done = {r["content_hash"] for r in map(json.loads, filter(str.strip, args.output.open())) if r["parsed"]}   # unparsed rows are redone
    todo = rows[~rows.content_hash.isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    batches = [todo.iloc[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(done)} done, {len(todo)} to annotate in {len(batches)} calls", flush=True)
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}

    def work(batch):
        rows = list(batch.itertuples())
        try:
            reply = codex_reply(build_prompt(rows), args.model, args.effort, args.timeout)
        except Exception as error:
            reply, parsed = "", None
            failure = repr(error)
        else:
            parsed = parse(reply, len(rows))
        with lock:
            with args.output.open("a") as out:
                for number, row in enumerate(rows, start=1):
                    errors = parsed.get(number) if parsed else None
                    out.write(json.dumps({"content_hash": row.content_hash, "bench": row.bench, "model": row.model,
                                          "item": row.item, "errors": errors or [], "n_errors": None if errors is None else len(errors),
                                          "parsed": errors is not None, "raw": reply}, ensure_ascii=False) + "\n")
            counts["ok" if parsed else "failed"] += 1
            if (counts["ok"] + counts["failed"]) % 20 == 0:
                print(f"{counts['ok']} calls ok, {counts['failed']} failed, of {len(batches)}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, batches))
    print(f"finished: {counts}", flush=True)


if __name__ == "__main__":
    main()
