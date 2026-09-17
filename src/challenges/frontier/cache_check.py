"""Did prompt caching work in an agentic log? Per-turn cache reads vs prompt tokens.

    PYTHONPATH=src .venv/bin/python src/challenges/frontier/cache_check.py <log.eval> [...]

PASS when, from the 4th turn of a sample on, at least half of the prompt tokens were served from
cache (a working rolling breakpoint re-reads the whole previous turn). The 2026-08-31
claude-opus-4.7 terminal run FAILs this: a constant ~2.3K reads per turn (system + tools only).
"""

import sys

from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log_sample

THRESHOLD = 0.5
SKIP_TURNS = 3


def share(path: str) -> tuple[float, int, int]:
    """(cache-read share after SKIP_TURNS, turns counted, samples) over every sample of the log."""
    read = prompt = turns = samples = 0
    for s in read_eval_log_sample_summaries(path):
        sample = read_eval_log_sample(path, s.id, s.epoch)
        samples += 1
        usages = [ev.output.usage for ev in sample.events if ev.event == "model" and ev.output and ev.output.usage]
        for u in usages[SKIP_TURNS:]:
            cr = u.input_tokens_cache_read or 0
            read += cr
            prompt += u.input_tokens + cr + (u.input_tokens_cache_write or 0)
            turns += 1
    return (read / prompt if prompt else 0.0), turns, samples


def main() -> int:
    bad = 0
    for path in sys.argv[1:]:
        frac, turns, samples = share(path)
        verdict = "PASS" if frac >= THRESHOLD else "FAIL"
        bad += verdict == "FAIL"
        print(f"{verdict} {frac:.0%} of prompt tokens read from cache after turn {SKIP_TURNS} "
              f"({turns} turns, {samples} samples): {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
