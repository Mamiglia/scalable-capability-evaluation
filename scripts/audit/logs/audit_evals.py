"""Content audit of the collected .eval logs: what would silently corrupt the judging study.

Complements the two checks that already exist:
- scripts/audit/logs/validate_data.py  -- structure (repeats, completeness, replayed repeats)
- scripts/audit/logs/report_collection.py -- per-run accuracy, cost, reasoning share

This one reads *samples* and flags per-log defects that only show up in the transcripts,
plus cross-log accuracy outliers. It never modifies data/; it writes a JSONL of findings.

    uv run python scripts/audit/logs/audit_evals.py [data/gpqa ...] [--samples 50] [--jobs 8]
    uv run python scripts/audit/logs/audit_evals.py --report            # re-print from the cache
    uv run python scripts/audit/logs/audit_evals.py data/hle --refresh  # ignore cached rows

Flags (per log, as a rate over the samples read):

  unparsed      scorer stored no answer though the model wrote one -> parser too strict
                (the `ANSWER: $C$` class; see data/collection_runs.md)
  truncated     stop_reason max_tokens -> the transcript is cut, not the reasoning
  no_reasoning  the log asked the provider for reasoning and got none -> broken pin
  reas_summary  reasoning blocks are redacted with a summary -> the judge reads a
                *summary of* the CoT, not the CoT (OpenAI encrypted_content)
  reas_mixed    the log mixes reasoning shapes across its samples (plain / redacted /
                none), so the same model reads as a CoT on some items and a bare answer on
                others. Only `plain` reaches the judge. No rate-based flag catches this:
                the odd shape is a small minority by construction
  reas_lost     usage reports reasoning tokens but no reasoning content was recovered
  reas_thin     recovered reasoning is far shorter than the reasoning tokens billed
                -> summary-like even when the block is not marked redacted
  reas_inline   CoT sits in the text as an un-normalized inline <think> block -> dropped at
                render time; normalize_reasoning.py fixes it
  thought_inline gemma-4 CoT glued into the text: vLLM stripped the <|channel>thought /
                <channel|> tokens, leaving a text that starts with the word `thought` and
                runs into the answer with no separator. The judge reads the CoT in *both*
                arms; normalize_reasoning.py splits it
  think_open    inline <think> never closed -> whole budget on CoT, no answer
  no_answer_field the bench's scorer never records an answer at all (nl2bash, cybashbench,
                math500, the agentic benches) -- structural, not a per-model parser defect
  short_tx      the reasoning-off transcript is an answer with no work (<200 chars): not
                empty, but nothing for the judge to read. Catches models that put every
                token in the CoT, and answer-only logs like claude-opus-4.7 on aime2024
  media         non-text content -> the judge only sees an [Image]/[Audio] placeholder
  self_id       transcript names its own model/vendor -> judge can identify, not assess
  degenerate    repetition loop (few distinct lines over many lines)
  empty_tx      extract_transcript() renders nothing -> unjudgeable item
  render_err    extract_transcript() raised -> unsupported content block
  no_output     the provider returned no choice for the sample
  read_err      samples could not be read at all
  edited        the log carries eval_edits provenance (rescored/patched) -- not a defect,
                but every audit needs to know the log is not as-collected
  acc_outlier   accuracy far below the bench median, or repeats disagreeing wildly
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import collection_logs, is_successful_eval
from eval_edits import edit_history
from evals.sample import extract_transcript

# A summary is a fraction of the CoT it summarises. Raw reasoning runs ~3-4 chars/token,
# so recovered chars below this share of (reasoning_tokens * 3) is summary-like, not CoT.
THIN_RATIO = 0.3

# Below this, the reasoning-off transcript is an answer with no work -- nothing to judge.
SHORT_TX = 200

CACHE = Path("analysis/eval_audit.jsonl")

VENDORS = (
    r"claude|anthropic|gpt-?[0-9]|chatgpt|openai|gemini|gemma|google|deepmind|llama|meta|"
    r"qwen|alibaba|deepseek|mistral|kimi|moonshot|glm|zhipu|grok|xai|nemotron|nvidia|"
    r"granite|ibm|phi-?[0-9]|microsoft"
)
# First-person only: a task prompt may mention a vendor without the model outing itself.
SELF_ID = re.compile(
    rf"(?i)\b(?:I am|I'm|I was|trained by|developed by|created by|made by)\b[^.\n]{{0,80}}\b(?:{VENDORS})\b"
)
THINK_OPEN = re.compile(r"<think>(?!.*</think>)", re.S)
# Gemma-4 served by vLLM with special tokens stripped: `<|channel>thought\n...<channel|>`
# survives only as a leading "thought\n". normalize_reasoning.THOUGHT_PREFIX is the same rule.
THOUGHT_PREFIX = re.compile(r"\A\s*thought\n")


def asked_for_reasoning(header) -> bool:
    """Whether the run itself requested reasoning.

    Read off the log, not off configs/models/: the data dir name does not map back to a
    config for the imported logs, and what the run actually sent is what matters.
    """
    config = header.eval.model_generate_config
    dumped = config.model_dump(exclude_none=True) if config else {}
    # reasoning_effort "none" (the `_none` imports) is a request for *no* reasoning.
    effort = dumped.get("reasoning_effort")
    if (effort and effort != "none") or dumped.get("reasoning_tokens") or dumped.get("reasoning_summary"):
        return True
    extra = dumped.get("extra_body") or {}
    reasoning = extra.get("reasoning") or {}
    if reasoning.get("enabled") or reasoning.get("effort") or reasoning.get("max_tokens"):
        return True
    # The chat-template form (qwen3 / gemma-4 style). Providers can ignore it silently: the
    # gemma-4 aime/gpqa runs asked this way and got answers with no thinking.
    return bool((extra.get("chat_template_kwargs") or {}).get("enable_thinking"))


def scan(path: Path, limit: int) -> dict:
    """Per-sample defect counts for one log. `limit` 0 reads every sample."""
    row = {"path": str(path), "n": 0}
    counts = defaultdict(int)
    try:
        header = read_eval_log(str(path), header_only=True)
        # Search every scorer: hle exposes llm_grader (cerr/original_accuracy) before score (accuracy).
        row["accuracy"] = next(
            (m.value for sc in (header.results.scores if header.results else [])
             for k, m in sc.metrics.items() if k in ("accuracy", "mean", "score")),
            None,
        )
        row["total_samples"] = header.results.total_samples if header.results else 0
        row["model"] = header.eval.model
        row["asked_reasoning"] = asked_for_reasoning(header)
        row["edits"] = edit_history(header)
    except Exception as exc:
        return {**row, "read_err": 1, "error": str(exc).splitlines()[0][:160]}

    try:
        for sample in read_eval_log_samples(str(path), all_samples_required=False, resolve_attachments=True):
            if limit and row["n"] >= limit:
                break
            row["n"] += 1

            scores = list((sample.scores or {}).values())
            # ModelOutput.completion / .stop_reason both index choices[0] and raise when the
            # provider returned no choice at all -- which is itself a defect worth counting.
            choices = getattr(sample.output, "choices", None) if sample.output else None
            if not choices:
                counts["no_output"] += 1
                continue
            completion = sample.output.completion
            if len(scores) > 1:
                counts["multi_scorer"] += 1
            if scores and str(getattr(scores[0], "answer", "") or "").strip():
                counts["answered"] += 1
            elif scores and completion.strip():
                counts["unparsed"] += 1
            if sample.output.stop_reason == "max_tokens":
                counts["truncated"] += 1

            # Reasoning shape, not just presence. ContentReasoning keeps the CoT in
            # .reasoning, a provider-made summary in .summary, and .text is a *property*
            # that falls back to the summary when the block is redacted -- so a redacted
            # block reaches the judge as a summary of the CoT, not the CoT.
            has_reasoning = has_media = redacted = summary = empty_block = False
            # `plain` is the only shape that reaches the judge: util.format_reasoning shows
            # .reasoning when the block is not redacted, and nothing otherwise.
            plain = inline_seen = open_seen = thought_seen = False
            reasoning_chars = 0
            for msg in sample.messages or []:
                content = msg.content if isinstance(msg.content, list) else [msg.content]
                for block in content:
                    kind = getattr(block, "type", "text")
                    if kind == "reasoning":
                        has_reasoning = True
                        is_redacted = bool(getattr(block, "redacted", False))
                        body = getattr(block, "summary", None) if is_redacted else getattr(block, "reasoning", "")
                        reasoning_chars += len(body or "")
                        redacted |= is_redacted
                        summary |= bool(getattr(block, "summary", None))
                        empty_block |= not (body or "").strip()
                        plain |= not is_redacted and bool((getattr(block, "reasoning", "") or "").strip())
                        continue
                    if kind != "text":
                        has_media = True
                        continue
                    text = block if isinstance(block, str) else getattr(block, "text", "") or ""
                    if "<think>" in text:
                        has_reasoning = inline_seen = True
                        reasoning_chars += len(text)
                        open_seen |= bool(THINK_OPEN.search(text))
                    elif msg.role == "assistant" and THOUGHT_PREFIX.match(text):
                        has_reasoning = thought_seen = True
                        reasoning_chars += len(text)
            # Once per sample, not per message: agentic transcripts have dozens of turns.
            counts["reas_inline"] += inline_seen
            counts["think_open"] += open_seen
            counts["thought_inline"] += thought_seen
            # Three shapes, by what the judge is shown for this sample. An un-normalized
            # inline <think> is dropped at render time, so it is not `plain` here -- the
            # reas_inline counter reports it separately, and normalize_reasoning.py fixes it.
            counts["shape_plain" if plain else "shape_redacted" if has_reasoning else "shape_none"] += 1
            counts["reasoning"] += has_reasoning
            counts["reas_empty"] += empty_block
            counts["media"] += has_media
            counts["reas_redacted"] += redacted
            counts["reas_summary_field"] += summary
            counts["reasoning_chars"] += reasoning_chars

            billed = getattr(sample.output.usage, "reasoning_tokens", None) if sample.output.usage else None
            if billed:
                counts["reas_billed_samples"] += 1
                counts["reas_billed_tokens"] += billed
                if not has_reasoning:
                    counts["reas_lost"] += 1
                elif reasoning_chars < THIN_RATIO * billed * 3:
                    counts["reas_thin"] += 1

            try:
                transcript = extract_transcript(sample)
            except Exception:
                counts["render_err"] += 1
                continue
            if not transcript.strip():
                counts["empty_tx"] += 1
                continue
            bare = extract_transcript(sample, include_reasoning=False)
            counts["tx_chars"] += len(transcript)
            counts["tx_chars_noreas"] += len(bare)
            if len(bare.strip()) < SHORT_TX:
                counts["short_tx"] += 1
            if SELF_ID.search(transcript):
                counts["self_id"] += 1
            lines = [ln for ln in transcript.splitlines() if ln.strip()]
            if len(lines) > 50 and len(set(lines)) / len(lines) < 0.3:
                counts["degenerate"] += 1
    except Exception as exc:
        counts["read_err"] = 1
        row["error"] = str(exc).splitlines()[0][:160]

    return {**row, **counts}


def selfcheck() -> int:
    """Assert the flag logic on hand-made rows -- the part that is easy to get subtly wrong."""
    assert SELF_ID.search("I am Claude, made by Anthropic.")
    assert SELF_ID.search("This model was developed by DeepSeek.")
    assert not SELF_ID.search("Compare the OpenAI and Google results in Table 2.")
    assert THINK_OPEN.search("<think> reasoning that never ends")
    assert not THINK_OPEN.search("<think>a</think> ANSWER: B")
    assert THOUGHT_PREFIX.match("thought\n*   Energy levels...")
    assert not THOUGHT_PREFIX.match("The thought\nexperiment shows")
    assert "thought_inline 100%" in flags({"n": 10, "thought_inline": 10, "reasoning": 10}, 0.10)
    assert "reas_inline 50%" in flags({"n": 10, "reas_inline": 5, "reasoning": 10}, 0.10)
    assert flags({"n": 20, "unparsed": 8, "reasoning": 0}, 0.10) == ["unparsed 40%"]
    assert "no_reasoning 0%" in flags({"n": 20, "reasoning": 0, "asked_reasoning": True}, 0.10)
    assert flags({"n": 20, "unparsed": 1, "reasoning": 20, "asked_reasoning": True}, 0.10) == []
    assert flags({"read_err": 1, "n": 0}, 0.10) == ["read_err"]
    # o3/gpt-5 shape: every block redacted, so the judge reads the provider's summary.
    assert "reas_summary 100%" in flags({"n": 15, "reasoning": 15, "reas_redacted": 15}, 0.10)
    # billed for reasoning, none recovered / only a sliver recovered
    assert "reas_lost 100%" in flags({"n": 15, "reasoning": 0, "reas_billed_samples": 15, "reas_lost": 15}, 0.10)
    assert "reas_thin 80%" in flags(
        {"n": 15, "reasoning": 15, "reas_billed_samples": 15, "reas_thin": 12}, 0.10)
    assert flags({"n": 1, "edits": [{"tool": "scripts/judge/rescore_mcq.py"}]}, 0.10) == ["edited rescore_mcq.py"]
    assert flags({"n": 10, "unparsed": 10, "answered": 0}, 0.10) == ["no_answer_field"]
    assert flags({"n": 10, "unparsed": 4, "answered": 6}, 0.10) == ["unparsed 40%"]
    assert "short_tx 100% (~26ch)" in flags({"n": 10, "short_tx": 10, "tx_chars_noreas": 260}, 0.10)
    assert flags({"n": 10, "short_tx": 2, "tx_chars_noreas": 9000}, 0.10) == []
    # One shape throughout is consistent, whichever shape it is.
    for shape in ("plain", "redacted", "none"):
        assert not any(f.startswith("reas_mixed") for f in flags({"n": 30, f"shape_{shape}": 30}, 0.10))
    assert "reas_mixed plain:28+none:2" in flags({"n": 30, "shape_plain": 28, "shape_none": 2}, 0.10)
    # A provider that discloses a summary on some calls and raw CoT on others is mixed too.
    assert "reas_mixed plain:20+redacted:10" in flags(
        {"n": 30, "shape_plain": 20, "shape_redacted": 10}, 0.10)
    print("selfcheck ok")
    return 0


def cached() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    return {r["path"]: r for r in rows}


def flags(row: dict, thresh: float) -> list[str]:
    n = row.get("n") or 0
    out = []
    if row.get("read_err"):
        return ["read_err"]
    if n and not row.get("answered") and row.get("unparsed", 0) / n >= 0.9:
        # The bench's scorer never records an answer (nl2bash, cybashbench, math500,
        # the agentic benches). Structural, not a per-model parser defect.
        out.append("no_answer_field")
        row = {**row, "unparsed": 0}
    if n and row.get("short_tx", 0) / n >= max(thresh, 0.5):
        out.append(f"short_tx {row['short_tx']/n:.0%} (~{row.get('tx_chars_noreas',0)//n}ch)")
    for key in ("unparsed", "truncated", "no_output", "reas_inline", "thought_inline", "think_open",
                "media", "self_id", "degenerate", "empty_tx", "render_err", "multi_scorer", "reas_empty"):
        if n and row.get(key, 0) / n >= thresh:
            out.append(f"{key} {row[key]/n:.0%}")
    if row.get("asked_reasoning") and n and row.get("reasoning", 0) / n < 0.5:
        out.append(f"no_reasoning {row.get('reasoning', 0)/n:.0%}")
    # A log should show the judge one reasoning shape. Where it mixes them, the same model
    # reads as a chain of thought on some items and a bare answer on others, so a pair built
    # from it is not comparing models on those items. Rate flags miss this by construction:
    # the odd shape is a small minority. No threshold -- one mixed item is one bad verdict.
    shapes = {k: row.get(f"shape_{k}", 0) for k in ("plain", "redacted", "none")}
    present = {k: v for k, v in shapes.items() if v}
    if len(present) > 1:
        out.append("reas_mixed " + "+".join(f"{k}:{v}" for k, v in
                                            sorted(present.items(), key=lambda kv: -kv[1])))
    # Shape flags are about *what kind* of reasoning reaches the judge, not how much.
    reasoning = row.get("reasoning", 0)
    if reasoning and row.get("reas_redacted", 0) / reasoning >= 0.5:
        out.append(f"reas_summary {row['reas_redacted']/reasoning:.0%}")
    billed = row.get("reas_billed_samples", 0)
    for key in ("reas_lost", "reas_thin"):
        if billed and row.get(key, 0) / billed >= max(thresh, 0.2):
            out.append(f"{key} {row[key]/billed:.0%}")
    if row.get("edits"):
        out.append("edited " + ",".join(e.get("tool", "?").split("/")[-1] for e in row["edits"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--samples", type=int, default=50, help="samples read per log; 0 = all")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--thresh", type=float, default=0.10, help="rate at which a defect is flagged")
    parser.add_argument("--refresh", action="store_true", help="rescan logs already in the cache")
    parser.add_argument("--report", action="store_true", help="only re-print the cache")
    parser.add_argument("--selfcheck", action="store_true", help="run the flag-logic asserts and exit")
    args = parser.parse_args()

    if args.selfcheck:
        return selfcheck()

    have = cached()
    if not args.report:
        # cache is keyed on path only; pass --refresh after rescoring a log in place.
        paths = [p for root in args.roots for p in collection_logs(root) if is_successful_eval(p)]
        todo = [p for p in paths if args.refresh or str(p) not in have]
        print(f"{len(paths)} logs, {len(todo)} to scan", file=sys.stderr)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.jobs) as pool, CACHE.open("a") as fh:
            for i, row in enumerate(pool.map(scan, todo, [args.samples] * len(todo)), 1):
                have[row["path"]] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"\r{i}/{len(todo)}", end="", file=sys.stderr)
        print(file=sys.stderr)

    by_bench = defaultdict(list)
    for path, row in have.items():
        parts = Path(path).parts
        by_bench[(parts[1], parts[2])].append((parts[3], parts[4], row))

    findings = []
    for (bench, _hash), entries in sorted(by_bench.items()):
        accs = [r.get("accuracy") for _, _, r in entries if r.get("accuracy") is not None]
        median = statistics.median(accs) if accs else 0.0
        per_model = defaultdict(list)
        for model, repeat, row in entries:
            per_model[model].append(row.get("accuracy"))
            found = flags(row, args.thresh)
            acc = row.get("accuracy")
            if acc is not None and median > 0.2 and acc < 0.3 * median:
                found.append(f"acc_outlier {acc:.2f} vs median {median:.2f}")
            if found:
                findings.append((bench, model, repeat, row.get("n", 0), found))
        for model, values in per_model.items():
            vals = [v for v in values if v is not None]
            if len(vals) > 1 and max(vals) - min(vals) > 0.2:
                findings.append((bench, model, "*", 0, [f"repeat_swing {min(vals):.2f}-{max(vals):.2f}"]))

    for bench, model, repeat, n, found in sorted(findings):
        print(f"{bench:16s} {model:50s} r{repeat:2s} n={n:<4d} {', '.join(found)}")
    print(f"\n{len(findings)} flagged / {len(have)} logs scanned -> {CACHE}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
