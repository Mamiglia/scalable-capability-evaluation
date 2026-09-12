"""Apply the judging rule "both models wrote >= MIN_RESPONSE_CHARS" to existing judge logs.

    uv run python scripts/judge/apply_response_rule.py [logs/judge] [--dry-run]

New runs obey the rule when their dataset is built (judge.dataset.build_dataset). Logs judged
before the rule, and logs imported from older campaigns, are rewritten here: samples where either
shown transcript has fewer response characters (role prefixes excluded) than the rule are dropped,
the kept samples get response_chars_1/2 in their metadata, and the edit is stamped into the log
(scripts/audit/logs/eval_edits.stamp_edit). A log whose samples all pass is only annotated.
"""

import argparse
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit" / "logs"))
from eval_edits import stamp_edit  # noqa: E402

from judge.dataset import MIN_RESPONSE_CHARS, response_chars  # noqa: E402

TRANSCRIPT = re.compile(r"Transcript (1|2):\n---\n\n(.*?)\n\n---", re.S)  # the blocks of judge.dataset.USER_PROMPT


def shown_transcripts(sample) -> tuple[str, str]:
    text = sample.input if isinstance(sample.input, str) else sample.input[-1].text
    blocks = dict(TRANSCRIPT.findall(text))
    return blocks["1"], blocks["2"]


def apply(log) -> tuple[int, int]:
    """Drop the samples that break the rule, annotate the rest. Returns (dropped, annotated)."""
    kept, dropped, annotated = [], 0, 0
    for sample in log.samples or []:
        md = sample.metadata or {}
        if "response_chars_1" not in md:
            one, two = shown_transcripts(sample)
            md = {**md, "response_chars_1": response_chars(one), "response_chars_2": response_chars(two)}
            sample.metadata = md
            annotated += 1
        if min(md["response_chars_1"], md["response_chars_2"]) < MIN_RESPONSE_CHARS:
            dropped += 1
        else:
            kept.append(sample)
    if dropped:
        log.samples = kept
        if log.eval.dataset:
            log.eval.dataset.sample_ids = [s.id for s in kept]
            log.eval.dataset.samples = len(kept)
        if log.results:
            log.results.total_samples = log.results.completed_samples = len(kept)
        stamp_edit(log, tool=__file__, reason=f"dropped samples where a model wrote < {MIN_RESPONSE_CHARS} response chars",
                   samples_changed=dropped, after={"samples": len(kept)})
    return dropped, annotated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path, nargs="?", default=Path("logs/judge"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    n_logs = n_edited = n_dropped = 0
    for path in sorted(args.root.rglob("*.eval")):
        log = read_eval_log(str(path))
        n_logs += 1
        before = len(log.samples or [])
        dropped, annotated = apply(log)
        if dropped or annotated:
            n_edited += dropped > 0
            n_dropped += dropped
            if not args.dry_run:
                write_eval_log(log, str(path))
        if dropped:
            print(f"{path.relative_to(args.root)}: {before} -> {before - dropped} samples")
    print(f"{n_logs} logs, {n_edited} rewritten, {n_dropped} samples dropped{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
