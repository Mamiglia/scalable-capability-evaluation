"""Rescore AIME logs whose answer was recorded from a truncated completion.

``aime_scorer`` reads ``output.completion``, which in these logs holds only the *last
line* of the assistant's text. A model that finishes ``\\boxed{23}`` and then closes the
display-math block leaves ``completion == '\\]'``, so ``score.answer`` is empty and a
correct sample is scored ``I``. The assistant message itself is intact, so the answer is
recoverable without re-collecting.

This re-reads each sample's full assistant text, and flips a sample to correct **only**
when the recovered answer equals the target. It never flips correct -> incorrect: the
scorer's positive matches are trustworthy, and the regex here is an extractor, not the
equivalence checker matharena applies.

Guarded to exact-match numeric scorers (``aime_scorer``). Do not point it at an
LLM-graded bench (hle's ``llm_grader``, otis' ``model_graded``) -- a numeric coincidence
is not a grading decision there.

    uv run python scripts/judge/rescore_aime.py data/aime2024 data/aime2025 data/aime2026
    uv run python scripts/judge/rescore_aime.py data/aime2024 --write   # rewrites in place
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

from configs.paths import collection_logs, is_successful_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "audit" / "logs"))
from eval_edits import stamp_edit  # noqa: E402

# Scorers whose answer is a bare integer, so an exact string match *is* the grading rule.
EXACT_MATCH_SCORERS = {"aime_scorer"}

BOXED = re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}")
# 'ANSWER: 33', 'ANSWER: $33$', '\text{ANSWER: } 33', 'ANSWER: \(65\)'
ANSWER = re.compile(r"ANSWER[:\s}]*\$?\\?\(?\s*([0-9]{1,4})")


def assistant_text(sample) -> str:
    """Every assistant text block, which is what output.completion should have held."""
    parts = []
    for message in sample.messages:
        if message.role != "assistant":
            continue
        content = message.content if isinstance(message.content, list) else [message.content]
        for block in content:
            if getattr(block, "type", "text") == "text":
                parts.append(block if isinstance(block, str) else block.text)
    return "\n".join(parts)


def normalise(value: str) -> str:
    """AIME answers are integers 0-999; compare without formatting or leading zeros."""
    return re.sub(r"[^0-9A-Za-z]", "", str(value)).lstrip("0") or "0"


def recover_answer(sample) -> str:
    """Last \\boxed{} or ANSWER: in the assistant text, '' if neither is present."""
    text = assistant_text(sample)
    found = BOXED.findall(text) + ANSWER.findall(text)
    return found[-1] if found else ""


def rescore(log) -> tuple[int, str]:
    """Flip recoverable false negatives in place. Returns (samples changed, scorer name)."""
    names = {name for sample in log.samples for name in (sample.scores or {})}
    if len(names) != 1:
        raise ValueError(f"expected one scorer, found {sorted(names)}")
    scorer = names.pop()
    if scorer not in EXACT_MATCH_SCORERS:
        raise ValueError(f"{scorer!r} is not an exact-match scorer; refusing to rescore it")

    fixed = {}
    for sample in log.samples:
        score = sample.scores[scorer]
        if score.value in ("C", 1, 1.0, True):
            continue
        answer = recover_answer(sample)
        if not answer or normalise(answer) != normalise(sample.target):
            continue
        score.value, score.answer = "C", answer
        fixed[sample.id] = answer

    # The reductions carry their own copy of every score (numeric), and the viewer and any
    # reduction-based reader see that one -- leaving it stale would make the log disagree
    # with itself.
    for reduction in log.reductions or []:
        for entry in reduction.samples:
            if entry.sample_id in fixed:
                entry.value, entry.answer = 1.0, fixed[entry.sample_id]

    correct = sum(s.scores[scorer].value in ("C", 1, 1.0, True) for s in log.samples)
    n = len(log.samples)
    accuracy = correct / n
    metrics = log.results.scores[0].metrics
    metrics["accuracy"].value = accuracy
    if "stderr" in metrics:
        metrics["stderr"].value = math.sqrt(accuracy * (1 - accuracy) / n)
    return len(fixed), scorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="benchmark dirs or single .eval files")
    parser.add_argument("--write", action="store_true", help="rewrite the logs (default: dry run)")
    args = parser.parse_args()

    paths = [
        p for root in args.roots
        for p in ([root] if root.suffix == ".eval" else collection_logs(root))
        if is_successful_eval(p)
    ]
    print(f"{len(paths)} logs; {'WRITING' if args.write else 'dry run'}", file=sys.stderr)

    total, touched = 0, 0
    for path in paths:
        # resolve_attachments: large text blocks are stored content-addressed, and without
        # this the assistant text reads back as a bare "attachment://<hash>" placeholder --
        # the answer would be invisible and the sample silently left unrescored.
        log = read_eval_log(str(path), resolve_attachments=True)
        before = log.results.scores[0].metrics["accuracy"].value
        changed, scorer = rescore(log)
        if not changed:
            continue
        after = log.results.scores[0].metrics["accuracy"].value
        total += changed
        touched += 1
        print(f"{path}: {changed} rescored, accuracy {before:.3f} -> {after:.3f}")
        if not args.write:
            continue
        stamp_edit(
            log,
            tool="scripts/judge/rescore_aime.py",
            reason=(
                f"{scorer} read output.completion, which held only the last line of the answer; "
                f"re-extracted from the assistant text with {BOXED.pattern!r} / {ANSWER.pattern!r}"
            ),
            source=path,
            samples_changed=changed,
            before={"accuracy": before},
            after={"accuracy": after},
            transcripts_modified=False,
        )
        # Write beside the target and swap, so a crash cannot leave a half-written log.
        # The temp keeps the .eval suffix (write_eval_log picks its recorder from it) and is
        # dot-prefixed, which collection_logs skips, so a crash cannot leave a stray log that
        # reads as a second successful run in the repeat dir.
        tmp = path.with_name(f".rescore-{path.name}")
        try:
            write_eval_log(log, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    print(f"\n{touched} logs, {total} samples rescored", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
