"""Rescore multiple-choice collection logs.

The stock inspect_ai multiple-choice parser only reads a bare letter after
'ANSWER:'. Some models wrap it, imitating the prompt's '$LETTER' placeholder
('ANSWER: $C$', 'ANSWER: **B**'), and get scored incorrect. This re-parses
every sample's completion, rewrites the per-sample 'choice' score, recomputes
the header accuracy/stderr, and writes a copy. Input logs are never modified.

--malformed-answers says what to do with a wrapped letter:
  accept  drop the wrapper and read the letter
  reject  score it incorrect, as the stock parser did
The rule is recorded in the log header metadata and on every rescored score.

Usage:
  uv run python scripts/judge/rescore_mcq.py data/gpqa/<hash> --out <dir>
  uv run python scripts/judge/rescore_mcq.py <log>.eval --out <rescored>.eval --malformed-answers reject
A directory input is mirrored under --out with the same model/epoch layout.
"""

import argparse
import math
import re
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

# Same two patterns as inspect_ai.solver._multiple_choice.parse_answers: a whole
# line first, then anywhere in the text; last match wins.
STRICT_ANSWER = re.compile(r"(?im)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)")
LOOSE_ANSWER = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
# A single letter wrapped in $, * or _ after 'ANSWER:', with markdown allowed round the label
# and the colon too ('**Answer:** A', '**ANSWER**: A').
WRAPPED_LETTER = re.compile(r"(?i)[*_]*(ANSWER)[*_]*\s*(:)[*_]*\s*[$*_]*([A-Za-z])(?![A-Za-z])[$*_]*")
# 'ANSWER: \boxed{A}' or 'Final Answer: ... \boxed{A}'.
BOXED_LETTER = re.compile(r"(?i)(?:final\s+)?answer[^\n]*?\\boxed\{([A-Za-z])\}")
# Main's tolerant behavior: first letter after an ANSWER label, including '(C)' and
# 'C because ...'; invalid choices are dropped before the last valid answer is selected.
ACCEPT_ANSWER = re.compile(r"(?i)ANSWER\s*:\s*[*_$\s]*\(?([A-Za-z])\)?(?![A-Za-z])")


def parse_answer(completion: str, n_choices: int, malformed_answers: str = "accept") -> str:
    """Answer letter as the stock parser reads it, '' if none valid."""
    if malformed_answers == "accept":
        completion = WRAPPED_LETTER.sub(r"\1\2 \3", completion)
        completion = BOXED_LETTER.sub(r"ANSWER: \1", completion)
        allowed = {chr(ord("A") + i) for i in range(n_choices)}
        answers = [answer.upper() for answer in ACCEPT_ANSWER.findall(completion)]
        answers = [answer for answer in answers if answer in allowed]
        return answers[-1] if answers else ""
    matches = STRICT_ANSWER.findall(completion) or LOOSE_ANSWER.findall(completion)
    if not matches:
        return ""
    tokens = [t for t in matches[-1].strip().rstrip(".").upper().split(",") if t]
    allowed = {chr(ord("A") + i) for i in range(n_choices)}
    return tokens[0] if len(tokens) == 1 and tokens[0] in allowed else ""


def finished_cleanly(sample) -> bool:
    """Whether the generation ended on its own (stop_reason 'stop'); cut, empty or unknown
    endings are failures that no parser may promote to a pass."""
    output = sample.output
    return bool(output and output.choices and output.stop_reason == "stop")


def rescore(input_path: Path, out_path: Path, malformed_answers: str = "accept") -> str:
    """Rescore one log into out_path; returns a one-line summary."""
    log = read_eval_log(input_path)
    if log.status != "success" or not log.samples:
        return f"skip ({log.status})"
    changed = 0
    values_by_item = {}
    for sample in log.samples:
        # Samples that errored (timeouts etc.) have no score; leave them alone.
        if not sample.scores or "choice" not in sample.scores:
            continue
        # Only a cleanly finished output is re-parsed under accept; a cut, empty or unknown
        # ending keeps its stock score and is left to a re-run.
        if malformed_answers == "accept" and not finished_cleanly(sample):
            values_by_item.setdefault(sample.id, []).append(float(sample.scores["choice"].value == "C"))
            continue
        answer = parse_answer(sample.output.completion, len(sample.choices), malformed_answers)
        value = "C" if answer and answer == sample.target else "I"
        score = sample.scores["choice"]
        changed += (score.value, score.answer) != (value, answer)
        score.value, score.answer = value, answer
        score.metadata = {**(score.metadata or {}), "malformed_answers": malformed_answers}
        values_by_item.setdefault(sample.id, []).append(float(value == "C"))

    # Header accuracy/stderr as inspect computes them: over the per-item mean across epochs.
    if not values_by_item:
        return "skip (no choice scores)"
    item_means = [sum(item) / len(item) for item in values_by_item.values()]
    accuracy = sum(item_means) / len(item_means)
    metrics = log.results.scores[0].metrics
    old_accuracy = metrics["accuracy"].value
    metrics["accuracy"].value = accuracy
    if "stderr" in metrics:
        metrics["stderr"].value = statistics.stdev(item_means) / math.sqrt(len(item_means)) if len(item_means) > 1 else 0.0
    log.eval.metadata = {**(log.eval.metadata or {}), "malformed_answers": malformed_answers}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, out_path)
    scored = sum(len(item) for item in values_by_item.values())
    return f"{scored} samples, {changed} rescored, accuracy {old_accuracy:.3f} -> {accuracy:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="one .eval log, or a directory of them")
    parser.add_argument("--out", type=Path, required=True, help="output log, or output directory for a directory input")
    parser.add_argument(
        "--malformed-answers",
        choices=["accept", "reject"],
        default="accept",
        help="how to score unambiguous wrapped answers (default: accept)",
    )
    args = parser.parse_args()

    if args.out.resolve() == args.input.resolve():
        raise SystemExit("refusing to overwrite the input; pass a different --out")
    if args.input.is_dir() and args.out.resolve().is_relative_to(args.input.resolve()):
        parser.error("--out must not be inside the input directory")

    if args.input.is_dir():
        for log_path in sorted(args.input.rglob("*.eval")):
            if log_path.name.startswith("._"):  # macOS resource forks
                continue
            relative = log_path.relative_to(args.input)
            print(relative, rescore(log_path, args.out / relative, args.malformed_answers))
    else:
        print(rescore(args.input, args.out, args.malformed_answers))
    return 0


if __name__ == "__main__":
    main()
