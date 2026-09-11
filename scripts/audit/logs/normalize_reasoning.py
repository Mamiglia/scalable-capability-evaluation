"""Normalize where reasoning is stored in collected .eval logs.

Two provider quirks put chain-of-thought somewhere the judge pipeline cannot classify, so
the same evidence reaches the judge tagged differently depending on who served the model:

1. **inline `<think>…</think>` in a text block** (DeepSeek-style, when the provider returns
   CoT as plain text instead of a reasoning block) -> split into a real `ContentReasoning`
   plus the answer text.
2. **a signature-only payload in `.reasoning`** -- a JSON envelope like
   ``[{"type": "reasoning.text", "signature": "...", "format": ..., "index": ...}]`` with no
   text field at all (claude-opus-4.7, gemini-3.1-flash-lite). It is not CoT and holds nothing
   recoverable, so it is marked ``redacted`` and renders as nothing. Detected structurally --
   no model list, the block itself is the evidence.
3. **a provider *thought summary* returned in `.reasoning` with `redacted=False`**
   -> moved to `.summary` with `redacted=True`, which is what it actually is: the real CoT
   never left the provider. Nothing in the log marks this, so it is gated on an explicit
   ``--summary-models`` list that **defaults to empty** -- never guessed, never a family-wide
   substring. Verified so far: ``gemini-3.5-flash`` (recovered chars 0.29x the billed
   reasoning tokens, and the text reads as headed third-person prose). Other gemini variants
   measured at ~1.0-1.3x, i.e. genuine CoT -- do not convert them.

After this, `evals/sample.format_reasoning` needs one rule -- show `.reasoning` when the block
is not redacted, never anything else -- and `split_think` is dead code.

    uv run python scripts/audit/logs/normalize_reasoning.py data/                    # dry run (default)
    uv run python scripts/audit/logs/normalize_reasoning.py data/ --write
    uv run python scripts/audit/logs/normalize_reasoning.py data/ --write --summary-models gemini-3.5-flash

Every rewritten log is stamped via ``eval_edits.stamp_edit``. Dry run touches nothing.
"""

import argparse
import json
import sys
from pathlib import Path

from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.log import read_eval_log, write_eval_log

from configs.paths import collection_logs, is_successful_eval
from eval_edits import stamp_edit


def split_think(text: str) -> tuple[str, str]:
    """Split a text block into (inline chain-of-thought, answer).

    Lives here rather than in ``evals/sample.py``: once the logs are normalized the judge
    pipeline must never meet an inline ``<think>`` again -- it raises instead. This is the
    one place that still has to parse it.
    """
    if "</think>" in text:
        head, answer = text.rsplit("</think>", 1)
        return (head.split("<think>", 1)[1] if "<think>" in head else head), answer.lstrip()
    if "<think>" in text:
        answer, thinking = text.split("<think>", 1)
        return thinking, answer.rstrip()
    return "", text


def is_signature_payload(reasoning: str) -> bool:
    """True for a JSON envelope that carries a signature but no reasoning text."""
    text = reasoning.lstrip()
    if not text.startswith("["):
        return False
    try:
        entries = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(entries, list) or not entries:
        return False
    return all(
        isinstance(e, dict) and not (e.get("text") or e.get("reasoning") or "").strip()
        for e in entries
    )


def normalize_message(message, to_summary: bool) -> tuple[int, int, int]:
    """Rewrite one assistant message in place. Returns (inline split, summaries moved, payloads)."""
    content = message.content
    if not isinstance(content, list):
        if not isinstance(content, str) or "<think>" not in content:
            return 0, 0, 0
        content = [ContentText(type="text", text=content)]

    out, split, moved, payloads = [], 0, 0, 0
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text" and "<think>" in (block.text or ""):
            thinking, answer = split_think(block.text)
            if thinking.strip():
                out.append(ContentReasoning(type="reasoning", reasoning=thinking.strip()))
                split += 1
            if answer.strip():
                out.append(ContentText(type="text", text=answer))
            continue
        if kind == "reasoning" and not getattr(block, "redacted", False):
            body = getattr(block, "reasoning", "") or ""
            if is_signature_payload(body):
                out.append(ContentReasoning(
                    type="reasoning", reasoning="", redacted=True,
                    signature=getattr(block, "signature", None),
                ))
                payloads += 1
                continue
            if to_summary and body.strip():
                out.append(ContentReasoning(
                    type="reasoning", reasoning="", summary=body,
                    redacted=True, signature=getattr(block, "signature", None),
                ))
                moved += 1
                continue
        out.append(block)

    if split or moved or payloads:
        message.content = out
    return split, moved, payloads


def normalize_log(path: Path, to_summary: bool, write: bool) -> tuple[int, int, int]:
    log = read_eval_log(str(path))
    split = moved = payloads = 0
    for sample in log.samples or []:
        for message in sample.messages or []:
            if getattr(message, "role", None) != "assistant":
                continue
            s, m, q = normalize_message(message, to_summary)
            split, moved, payloads = split + s, moved + m, payloads + q

    if (split or moved or payloads) and write:
        reason = []
        if split:
            reason.append(f"{split} inline <think> block(s) split into ContentReasoning")
        if moved:
            reason.append(f"{moved} provider summary block(s) moved from .reasoning to .summary")
        if payloads:
            reason.append(f"{payloads} signature-only payload block(s) marked redacted")
        stamp_edit(
            log,
            tool="scripts/normalize_reasoning.py",
            reason="; ".join(reason),
            source=path,
            samples_changed=len(log.samples or []),
            inline_think_split=split,
            summaries_moved=moved,
            signature_payloads_redacted=payloads,
            transcripts_modified=True,
        )
        write_eval_log(log, str(path))
    return split, moved, payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--write", action="store_true", help="rewrite the logs (default: dry run)")
    parser.add_argument("--summary-models", nargs="*", default=[],
                        help="EXACT model dir names whose .reasoning holds a provider thought "
                             "summary. Empty by default: converting a model on a guess destroys "
                             "its CoT. Verify with audit_evals.py `reas_thin` first.")
    args = parser.parse_args()

    paths = [p for root in args.roots for p in collection_logs(root) if is_successful_eval(p)]
    print(f"{len(paths)} logs in scope; {'WRITING' if args.write else 'dry run'}", file=sys.stderr)

    totals, skipped = [0, 0, 0, 0], []
    for path in paths:
        model = path.parts[-3]
        to_summary = model in args.summary_models  # exact match, never a substring
        try:
            split, moved, payloads = normalize_log(path, to_summary, args.write)
        except Exception as exc:
            # A corrupt log must not kill a 4700-log run, but a skip is never "fine": it is
            # counted and the run exits non-zero, so a bug here cannot pass for a clean sweep.
            skipped.append(f"{path}: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
            print(f"{path}: SKIPPED ({type(exc).__name__}: {str(exc).splitlines()[0][:100]})")
            continue
        if split or moved or payloads:
            totals[0] += 1
            totals[1] += split
            totals[2] += moved
            totals[3] += payloads
            print(f"{path}: {split} inline split, {moved} summaries moved, {payloads} payloads redacted")
    verb = "rewritten" if args.write else "would change"
    print(f"\n{totals[0]} logs {verb}: {totals[1]} inline <think> split, "
          f"{totals[2]} summaries moved, {totals[3]} signature payloads redacted", file=sys.stderr)
    if skipped:
        print(f"INCOMPLETE: {len(skipped)} of {len(paths)} logs skipped -- this sweep did not "
              f"cover the data. Fix the cause and re-run (normalizing is idempotent).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
