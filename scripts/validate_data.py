"""Validate the collected data: every <bench>/<hash>/<model> should have at least two
complete, successful, mutually distinct repeats.

Flags, per model dir:
- repeat dirs without a full successful log (error/cancelled, --limit runs, missing samples)
- fewer than --min-repeats usable repeats
- more than one successful log in a repeat dir
- pairs of repeats whose completions are mostly identical (until 2026-08-17 collect ran
  with a generate cache, so a repeat could replay another; see AGENTS.md)

    uv run python scripts/validate_data.py [data/aime2024 ...] [--samples 100] [--dup 0.5]

Exit code 1 if anything is flagged.
"""

import argparse
import itertools
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import is_successful_eval


def complete_log(repeat_dir: Path) -> tuple[Path | None, list[str]]:
    """The one full successful log of a repeat dir, plus problems found on the way."""
    problems, good = [], []
    for path in sorted(repeat_dir.glob("*.eval")):
        if not is_successful_eval(path):
            continue  # crashed / partial (--limit) leftovers do not count, by design
        header = read_eval_log(str(path), header_only=True)
        if header.results.completed_samples < header.results.total_samples:
            problems.append(f"{path}: only {header.results.completed_samples}/{header.results.total_samples} samples")
            continue
        good.append(path)
    if len(good) > 1:
        problems.append(f"{repeat_dir}: {len(good)} successful logs; keep one, move the rest to a new repeat dir")
    if not good:
        problems.append(f"{repeat_dir}: no complete successful log")
    return (good[0] if len(good) == 1 else None), problems


def completions(path: Path, limit: int) -> dict:
    samples = read_eval_log_samples(str(path), all_samples_required=False)
    return {(s.id, s.epoch): s.output.completion for s in itertools.islice(samples, limit)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100, help="Samples compared per log for the replay check.")
    parser.add_argument("--dup", type=float, default=0.5, help="Identical share at which two repeats count as replays.")
    args = parser.parse_args()

    problems = []
    # A model dir is any dir holding numbered repeat dirs, so roots can be data/, a bench, or a hash dir.
    model_dirs = sorted(
        d for root in args.roots for d in root.rglob("*")
        if d.is_dir() and "legacy" not in d.parts and any(c.is_dir() and c.name.isdigit() for c in d.iterdir())
    )
    for model_dir in model_dirs:
        for child in model_dir.iterdir():
            if child.is_dir() and not child.name.isdigit():
                problems.append(f"{child}: not a numbered repeat dir")
        usable = {}
        for repeat_dir in sorted(d for d in model_dir.iterdir() if d.name.isdigit()):
            log, found = complete_log(repeat_dir)
            problems += found
            if log:
                usable[int(repeat_dir.name)] = log
        if len(usable) < args.min_repeats:
            problems.append(f"{model_dir}: {len(usable)} usable repeat(s), need {args.min_repeats}")

        texts = {}
        for repeat, path in usable.items():
            try:
                texts[repeat] = completions(path, args.samples)
            except Exception as exc:  # e.g. sandbox plugin missing for cyber benches
                problems.append(f"{path}: cannot read samples ({str(exc).splitlines()[0][:80]})")
        for a, b in itertools.combinations(sorted(texts), 2):
            common = texts[a].keys() & texts[b].keys()
            share = sum(texts[a][k] == texts[b][k] for k in common) / len(common) if common else 0
            if share >= args.dup:
                problems.append(f"{model_dir}: repeats {a} and {b} are {share:.0%} identical ({len(common)} samples)")

    for line in problems:
        print(line)
    print(f"{len(model_dirs)} model dirs checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
