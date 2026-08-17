"""Validate the collected data: every <bench>/<hash>/<model> should have at least two
complete, successful, mutually distinct repeats.

Flags, per model dir:
- repeat dirs without a full successful log (error/cancelled, --limit runs, missing samples)
- fewer than --min-repeats usable repeats (imported single-run logs, tagged ``imported``,
  are exempt: they were never collected in repeats)
- more than one successful log in a repeat dir
- pairs of repeats whose completions are mostly identical (until 2026-08-17 collect ran
  with a generate cache, so a repeat could replay another; see AGENTS.md)

    uv run python scripts/validate_data.py [data/aime2024 ...] [--samples 100] [--dup 0.5] [--delete]

Exit code 1 if anything is flagged. ``--delete`` removes what is flagged as unusable: incomplete
logs, the higher repeat of a replay pair, and repeat dirs left without a log.
"""

import argparse
import itertools
import shutil
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import is_successful_eval


def complete_log(repeat_dir: Path, delete: bool = False) -> tuple[Path | None, bool, list[str]]:
    """The one full successful log of a repeat dir, whether it is imported, and problems found."""
    problems, good = [], []
    for path in sorted(repeat_dir.glob("*.eval")):
        if not is_successful_eval(path):
            if delete:  # crashed / partial (--limit) leftovers do not count, by design
                path.unlink()
            continue
        header = read_eval_log(str(path), header_only=True)
        imported = any("imported" in tag for tag in header.eval.tags or [])
        done, total = header.results.completed_samples, header.results.total_samples
        if done < total - 1 and not imported:  # one failed sample is tolerated; imports are as they come
            problems.append(f"{path}: only {done}/{total} samples")
            if delete:
                path.unlink()
            continue
        good.append((path, imported))
    if len(good) > 1:
        problems.append(f"{repeat_dir}: {len(good)} successful logs; keep one, move the rest to a new repeat dir")
    if not good:
        problems.append(f"{repeat_dir}: no complete successful log")
        if delete:
            shutil.rmtree(repeat_dir)
    path, imported = good[0] if len(good) == 1 else (None, False)
    return path, imported, problems


def completions(path: Path, limit: int) -> dict:
    samples = read_eval_log_samples(str(path), all_samples_required=False)
    return {(s.id, s.epoch): s.output.completion for s in itertools.islice(samples, limit)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100, help="Samples compared per log for the replay check.")
    parser.add_argument("--dup", type=float, default=0.5, help="Identical share at which two repeats count as replays.")
    parser.add_argument("--delete", action="store_true", help="Remove incomplete logs and replayed repeats.")
    args = parser.parse_args()

    problems = []
    # A model dir is any dir holding numbered repeat dirs, so roots can be data/, a bench, or a hash dir.
    model_dirs = sorted(
        d for root in args.roots for d in [root, *root.rglob("*")]
        if d.is_dir() and "legacy" not in d.parts and any(c.is_dir() and c.name.isdigit() for c in d.iterdir())
    )
    for model_dir in model_dirs:
        for child in model_dir.iterdir():
            if child.is_dir() and not child.name.isdigit():
                problems.append(f"{child}: not a numbered repeat dir")
                if args.delete and not any(child.iterdir()):
                    child.rmdir()
        usable, imported = {}, set()
        for repeat_dir in sorted(d for d in model_dir.iterdir() if d.name.isdigit()):
            log, is_imported, found = complete_log(repeat_dir, args.delete)
            problems += found
            if log:
                usable[int(repeat_dir.name)] = log
                if is_imported:
                    imported.add(int(repeat_dir.name))
        own = set(usable) - imported
        if len(usable) < args.min_repeats and (own or not usable):
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
                if args.delete and (model_dir / str(b)).exists():
                    shutil.rmtree(model_dir / str(b))
        if args.delete and not any(model_dir.iterdir()):
            model_dir.rmdir()

    for line in problems:
        print(line)
    print(f"{len(model_dirs)} model dirs checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
