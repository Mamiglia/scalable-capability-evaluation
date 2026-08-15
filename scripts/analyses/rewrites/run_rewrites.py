"""Paraphrase every original transcript with a subscription CLI rewriter.

Two rewriters, same instruction: `haiku` (claude CLI) and `gpt` (codex CLI).
Neither costs API credit; both are driven as subprocesses.

Resumable: the output file is append-only JSONL and every (model, item_id)
already in it is skipped, so an interrupted run continues where it stopped.
Batches are written the moment they finish, so a kill costs one batch.

Usage:
  uv run python scripts/analyses/rewrites/run_rewrites.py haiku
  uv run python scripts/analyses/rewrites/run_rewrites.py gpt --limit 10
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import hashlib
import threading
from functools import partial
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGINALS = ROOT / "judge_results/rewrites/originals.jsonl"
OUT_PATHS = {
    "haiku": ROOT / "judge_results/rewrites/paraphrase_haiku.jsonl",
    "gpt": ROOT / "judge_results/rewrites/paraphrase_gpt.jsonl",
}

INSTRUCTION_PATH = Path(__file__).resolve().parent / "rewrite_instruction.md"


def load_instruction(path: Path = INSTRUCTION_PATH) -> tuple[str, str]:
    """The prompt below the horizontal rule, plus a hash of the whole file.

    The instruction is an experiment input the user edits directly, so it lives in a file
    rather than in code, and both rewriters read the same one so they cannot drift apart.
    The hash is logged with every batch: two corpora now exist under different instructions
    and each has to be traceable to the one that produced it.
    """
    text = path.read_text()
    if "\n---\n" not in text:
        raise ValueError(f"{path.name} has no '---' rule separating notes from the prompt")
    prompt = text.split("\n---\n", 1)[1].strip()
    if not prompt:
        raise ValueError(f"{path.name} has no prompt below the rule")
    return prompt, hashlib.sha256(text.encode()).hexdigest()[:12]


BATCH_SIZE = 5
MAX_BATCH_WORDS = 2500  # transcripts run to ~5k words; a batch of 5 long ones truncates the reply
N_WORKERS = 6
CLI_TIMEOUT_S = 900
BATCH_ATTEMPTS = 2  # one retry per batch before its rows are left for a later resume

BLOCK_RE = re.compile(r"<<<REWRITE (\d+)>>>\n(.*?)\n?<<<END \1>>>", re.DOTALL)
MARKER_RE = re.compile(r"<<<|>>>")  # no marker may survive inside a stored rewrite
# codex prints e.g. "tokens used: 12,345"
TOKENS_RE = re.compile(r"tokens used[:\s]+([\d,]+)", re.IGNORECASE)


def build_prompt(batch: list[dict], instruction: str) -> str:
    """Instruction verbatim, then the batch between numbered markers."""
    parts = [
        instruction,
        "",
        f"You are given {len(batch)} transcripts. Apply the instruction above to each one "
        "independently. Output every paraphrase in the same order, each between its marker "
        "lines exactly as shown below, and output nothing else - no preamble, no commentary.",
        "",
        "\n".join(f"<<<REWRITE {i + 1}>>>\n(paraphrase of transcript {i + 1})\n<<<END {i + 1}>>>"
                  for i in range(len(batch))),
        "",
    ]
    for i, row in enumerate(batch, start=1):
        parts += [f"<<<TRANSCRIPT {i}>>>", row["transcript"], f"<<<END {i}>>>", ""]
    return "\n".join(parts)


def call_haiku(prompt: str) -> tuple[str, int]:
    proc = subprocess.run(
        ["claude", "-p", "--model", "haiku", prompt],
        capture_output=True, text=True, timeout=CLI_TIMEOUT_S, cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout, 0


def call_codex(prompt: str, reasoning_effort: str = "") -> tuple[str, int]:
    """-s read-only or it blocks on an approval; --ignore-user-config skips MCP servers."""
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as tmp:
        out_file = Path(tmp.name)
    effort = ["-c", f'model_reasoning_effort="{reasoning_effort}"'] if reasoning_effort else []
    try:
        proc = subprocess.run(
            ["codex", "exec", "-s", "read-only", "--ignore-user-config", *effort,
             "-o", str(out_file), prompt],
            capture_output=True, text=True, timeout=CLI_TIMEOUT_S, cwd=ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"codex exit {proc.returncode}: {proc.stderr[-500:]}")
        tokens = [int(m.replace(",", "")) for m in TOKENS_RE.findall(proc.stdout + proc.stderr)]
        return out_file.read_text(), max(tokens, default=0)
    finally:
        out_file.unlink(missing_ok=True)


CALLERS = {"haiku": call_haiku, "gpt": call_codex}


def parse_blocks(response: str, batch: list[dict]) -> list[dict]:
    """Rows for the transcripts whose block came back; missing ones are left for a resume.

    A block still holding a marker is discarded: when the model misnumbers a closing marker
    the back-reference pairs one block's opener with a later block's closer and swallows
    everything between, which would store two transcripts (and the wrong answer) in one row.
    Dropping it leaves the transcript for the next resume.
    """
    found = {int(n): text.strip() for n, text in BLOCK_RE.findall(response)}
    found = {i: t for i, t in found.items() if not MARKER_RE.search(t)}
    return [
        {"model": row["model"], "item_id": row["item_id"], "rewritten": found[i]}
        for i, row in enumerate(batch, start=1)
        if found.get(i)
    ]


def make_batches(rows: list[dict], max_items: int, max_words: int) -> list[list[dict]]:
    """Greedy packing: cap a batch by item count and by total words. Oversized rows go alone."""
    batches, current, words = [], [], 0
    for row in rows:
        n = row["n_words"]
        if current and (len(current) >= max_items or words + n > max_words):
            batches.append(current)
            current, words = [], 0
        current.append(row)
        words += n
    if current:
        batches.append(current)
    return batches


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((row["model"], row["item_id"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewriter", choices=sorted(CALLERS))
    ap.add_argument("--limit", type=int, help="stop after this many transcripts (pilot runs)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-batch-words", type=int, default=MAX_BATCH_WORDS)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--attempts", type=int, default=BATCH_ATTEMPTS)
    ap.add_argument("--reasoning-effort", default="", help="codex only, e.g. low")
    ap.add_argument("--out", help="write here instead of the default file for this rewriter")
    ap.add_argument("--match", help="only rewrite the (model, item_id) keys present in this jsonl")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else OUT_PATHS[args.rewriter]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    call = CALLERS[args.rewriter]
    if args.reasoning_effort:
        call = partial(call, reasoning_effort=args.reasoning_effort)

    instruction, instruction_hash = load_instruction()
    print(f"instruction {INSTRUCTION_PATH.name} sha256:{instruction_hash}", flush=True)

    originals = [json.loads(l) for l in ORIGINALS.read_text().splitlines() if l.strip()]
    done = done_keys(out_path)
    todo = [r for r in originals if (r["model"], r["item_id"]) not in done]
    if args.match:
        keep = done_keys(Path(args.match))
        todo = [r for r in todo if (r["model"], r["item_id"]) in keep]
    if args.limit:
        todo = todo[: args.limit]
    batches = make_batches(todo, args.batch_size, args.max_batch_words)
    print(f"{args.rewriter}: {len(done)} done, {len(todo)} to do, {len(batches)} batches "
          f"of <={args.batch_size} items / <={args.max_batch_words} words, "
          f"{args.workers} workers", flush=True)

    lock = threading.Lock()
    stats = {"rows": 0, "batches": 0, "failed": 0, "tokens": 0, "seconds": 0.0}
    queue = list(enumerate(batches))
    qlock = threading.Lock()

    def worker() -> None:
        while True:
            with qlock:
                if not queue:
                    return
                idx, batch = queue.pop(0)
            prompt = build_prompt(batch, instruction)
            for attempt in range(1, args.attempts + 1):
                start = time.time()
                try:
                    response, tokens = call(prompt)
                    rows = parse_blocks(response, batch)
                except Exception as exc:  # timeout, non-zero exit, transport error
                    print(f"  batch {idx} attempt {attempt} failed: {exc}", flush=True)
                    continue
                elapsed = time.time() - start
                if not rows:
                    print(f"  batch {idx} attempt {attempt}: no blocks parsed", flush=True)
                    continue
                with lock:
                    with out_path.open("a") as f:  # append + flush per batch
                        for row in rows:
                            f.write(json.dumps(row) + "\n")
                    stats["rows"] += len(rows)
                    stats["batches"] += 1
                    stats["tokens"] += tokens
                    stats["seconds"] += elapsed
                    print(f"  batch {idx}: {len(rows)}/{len(batch)} rows in {elapsed:.0f}s "
                          f"(total {stats['rows']} rows, {stats['batches']} batches) "
                          f"[instr {instruction_hash}]", flush=True)
                break
            else:
                with lock:
                    stats["failed"] += 1

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"DONE {args.rewriter}: {stats['rows']} rows written, {stats['batches']} batches ok, "
          f"{stats['failed']} batches exhausted attempts", flush=True)
    if stats["tokens"]:
        # Undercounts: codex's stdout line omits cached input. The authoritative total is
        # total_token_usage in the matching rollout log under ~/.codex/sessions.
        print(f"tokens (approx, from stdout): {stats['tokens']} over {stats['rows']} transcripts "
              f"= {stats['tokens'] / max(stats['rows'], 1):.0f}/transcript", flush=True)


def _self_check() -> None:
    batch = [{"model": "m", "item_id": f"i{n}", "transcript": "t", "n_words": 1} for n in range(1, 5)]

    prompt, h = load_instruction()
    assert prompt and len(h) == 12, (prompt[:40], h)
    assert "---" not in prompt.splitlines()[0], "notes leaked into the prompt"
    assert build_prompt(batch[:1], prompt).startswith(prompt)

    ok = "<<<REWRITE 1>>>\nAAA\n<<<END 1>>>\n<<<REWRITE 2>>>\nB\nB\n<<<END 2>>>"
    assert [r["rewritten"] for r in parse_blocks(ok, batch)] == ["AAA", "B\nB"]

    # a block missing from the reply is skipped, not shifted onto the wrong transcript
    only2 = "<<<REWRITE 2>>>\nX\n<<<END 2>>>"
    assert parse_blocks(only2, batch) == [{"model": "m", "item_id": "i2", "rewritten": "X"}]

    # the real failure: a misnumbered closing marker makes block 3's opener pair with a
    # stray <<<END 3>>> after block 4, swallowing block 4 whole. That block must be dropped.
    mislabelled = ("<<<REWRITE 3>>>\nfirst\nANSWER: D\n<<<END 4>>>\n\n"
                   "<<<REWRITE 4>>>\nsecond\nANSWER: B\n<<<END 3>>>")
    rows = parse_blocks(mislabelled, batch)
    assert rows == [], rows
    assert all("<<<" not in r["rewritten"] for r in rows)

    # unterminated block yields nothing rather than running to end of reply
    assert parse_blocks("<<<REWRITE 1>>>\nno terminator here", batch) == []

    assert [len(b) for b in make_batches([{"n_words": w} for w in [100] * 6 + [5000, 50, 50]],
                                         5, 2500)] == [5, 1, 1, 2]
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
