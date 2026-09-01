"""Rescore AIME collection logs.

The stock AIME scorer (inspect_evals aime_scorer) reads a number off the last
line only, after dropping \\boxed{}. Some models wrap the answer in LaTeX or
markdown ('**ANSWER:** \\(259\\)', '\\text{ANSWER: }73' inside \\[ \\]) or put
the ANSWER line before a closing '\\]', and get scored incorrect. This re-parses
every sample's completion, rewrites the per-sample score, recomputes the header
accuracy/stderr, and writes a copy. Input logs are never modified.

--malformed-answers says what to do with a wrapped answer:
  accept  read the number off the last ANSWER line anywhere in the completion,
          with \\boxed{}, LaTeX delimiters, braces, markdown emphasis and
          trailing punctuation stripped; no ANSWER line = stock behaviour
  reject  score as the stock scorer did
The rule is recorded in the log header metadata and on every rescored score.

Usage:
  uv run python scripts/judge/rescore_aime.py data/aime2025/<hash> --out <dir>
  uv run python scripts/judge/rescore_aime.py <log>.eval --out <rescored>.eval --malformed-answers reject
A directory input is mirrored under --out with the same model/epoch layout.
"""

import argparse
import math
import re
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer._common import match_str
from inspect_evals.utils.aime_common import remove_boxed_from_ans

# Text after 'ANSWER:' on a line, when a number follows (a bare '### Final Answer:'
# heading does not count); the last such line wins.
ANSWER_LINE = re.compile(r"(?im)^.*ANSWER\s*:(.*\d.*)$")
# LaTeX commands (\boxed, \text, \displaystyle), maths delimiters, braces, markdown emphasis.
WRAPPING = re.compile(r"\\[A-Za-z]+|[\\()\[\]{}$*_]")
TRAILING_PUNCTUATION = re.compile(r"\W+$")


def parse_answer(
    completion: str, target: str, malformed_answers: str = "accept"
) -> tuple[str | None, bool]:
    """(answer, correct): the stock read of the last line, or under 'accept' the last ANSWER line."""
    lines = completion.strip().splitlines()
    if not lines:  # the stock scorer gives an empty completion no answer at all
        return None, False
    answer_lines = ANSWER_LINE.findall(completion) if malformed_answers == "accept" else []
    if answer_lines:
        cleaned = TRAILING_PUNCTUATION.sub("", WRAPPING.sub(" ", answer_lines[-1]))
    else:
        cleaned = remove_boxed_from_ans(lines[-1])
    return match_str(cleaned, target, numeric=True)


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
        if not sample.scores or "aime_scorer" not in sample.scores:
            continue
        # Only a cleanly finished output is re-parsed under accept; a cut, empty or unknown
        # ending keeps its stock score and is left to a re-run.
        if malformed_answers == "accept" and not finished_cleanly(sample):
            values_by_item.setdefault(sample.id, []).append(float(sample.scores["aime_scorer"].value == "C"))
            continue
        score = sample.scores["aime_scorer"]
        # What the model wrote: the stock scorer overwrote the sample completion with
        # the cleaned last line in most logs, its unprocessed_answer metadata is
        # truncated on long outputs, and a few samples have unresolved attachment
        # refs in their messages. The raw text is the longest of the three.
        raw_metadata = (score.metadata or {}).get("unprocessed_answer", "")
        completion = max((sample.messages[-1].text, sample.output.completion, raw_metadata), key=len)
        answer, correct = parse_answer(completion, sample.target, malformed_answers)
        value = "C" if correct else "I"
        changed += (score.value, score.answer) != (value, answer)
        score.value, score.answer = value, answer
        score.metadata = {**(score.metadata or {}), "malformed_answers": malformed_answers}
        values_by_item.setdefault(sample.id, []).append(float(correct))

    # Header accuracy/stderr as inspect computes them: over the per-item mean across epochs.
    if not values_by_item:
        return "skip (no AIME scores)"
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
    n_samples = sum(len(item) for item in values_by_item.values())
    return f"{n_samples} samples, {changed} rescored, accuracy {old_accuracy:.3f} -> {accuracy:.3f}"


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
