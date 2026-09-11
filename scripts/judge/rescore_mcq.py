"""Rescore an MCQ collection log with a tolerant ANSWER regex.

The stock inspect_ai multiple-choice parser misses answers like 'ANSWER: $C$'
(some models wrap the letter in $..$, imitating the prompt's '$LETTER'
placeholder), scoring them incorrect. This re-parses each sample's completion,
rewrites the per-sample 'choice' score, recomputes the header accuracy/stderr,
and writes a copy. The input log is never modified.

Usage:
  uv run python scripts/judge/rescore_mcq.py \
      data/gpqa/<hash>/<model>/0/<log>.eval \
      --out data/gpqa/<hash>/<model>-rescored/0/<log>.eval
"""

import argparse
import math
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "audit" / "logs"))
from eval_edits import stamp_edit  # noqa: E402

ANSWER_RE = re.compile(r"ANSWER\s*:\s*[*_$\s]*\(?([A-Za-z])\)?", re.IGNORECASE)


def parse_answer(completion: str, n_choices: int) -> str:
    """Last tolerantly-parsed answer letter, '' if none valid."""
    allowed = {chr(ord("A") + i) for i in range(n_choices)}
    matches = [m.group(1).upper() for m in ANSWER_RE.finditer(completion)]
    matches = [m for m in matches if m in allowed]
    return matches[-1] if matches else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-unparsed", action="store_true",
                        help="re-parse only samples the stock parser left without an answer; never touch a parsed one")
    args = parser.parse_args()

    if args.out.resolve() == args.input.resolve():
        raise SystemExit("refusing to overwrite the input log; pass a different --out")

    log = read_eval_log(args.input)
    changed = correct = 0
    for s in log.samples:
        score = s.scores["choice"]
        if args.only_unparsed and score.answer:
            correct += score.value == "C"
            continue
        answer = parse_answer(s.output.completion, len(s.choices))
        value = "C" if answer and answer == s.target else "I"
        if (score.value, score.answer) != (value, answer):
            changed += 1
        score.value, score.answer = value, answer
        correct += value == "C"

    n = len(log.samples)
    accuracy = correct / n
    metrics = log.results.scores[0].metrics
    old_accuracy = metrics["accuracy"].value
    metrics["accuracy"].value = accuracy
    if "stderr" in metrics:
        metrics["stderr"].value = math.sqrt(accuracy * (1 - accuracy) / n)

    stamp_edit(
        log,
        tool="scripts/judge/rescore_mcq.py",
        reason=f"stock MCQ parser missed {changed} answers; re-parsed {'unparsed samples only ' if args.only_unparsed else ''}with {ANSWER_RE.pattern!r}",
        source=args.input,
        samples_changed=changed,
        before={"accuracy": old_accuracy},
        after={"accuracy": accuracy},
        transcripts_modified=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, args.out)
    print(f"{n} samples, {changed} rescored, accuracy {old_accuracy:.3f} -> {accuracy:.3f}")
    return 0


if __name__ == "__main__":
    main()
