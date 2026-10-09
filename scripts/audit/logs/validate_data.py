"""Validate the collected data: every <bench>/<hash>/<model> should have at least two
complete, successful, mutually distinct repeats.

Flags, per model dir:
- repeat dirs without a full successful log (error/cancelled, --limit runs, missing samples)
- fewer than --min-repeats usable repeats (imported single-run logs, tagged ``imported``,
  are exempt: they were never collected in repeats)
- more than one successful log in a repeat dir
- pairs of repeats whose completions are mostly identical (early collection ran with a
  generate cache, so a repeat could replay another)

    uv run python scripts/audit/logs/validate_data.py [data/aime2024 ...] [--samples 100] [--dup 0.5] [--delete]

Exit code 1 if anything is flagged. ``--delete`` removes what is flagged as unusable: incomplete
logs, the higher repeat of a replay pair, and repeat dirs left without a log.

Per-log reads are cached in ``analysis/validate_cache.json`` keyed by (mtime, size), so a
re-run only reads logs that changed. Delete the file to force a full rescan.
"""

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import is_successful_eval

CACHE_PATH = Path("analysis/validate_cache.json")
_cache: dict = {}


def _cached(path: Path, kind: str, compute) -> dict:
    """compute(path) memoized by the log's (mtime, size); logs only change when stamped."""
    stat = path.stat()
    key = [stat.st_mtime, stat.st_size]
    entry = _cache.get(str(path), {})
    if entry.get("key") != key:
        entry = {"key": key}
    if kind not in entry:
        entry[kind] = compute(path)
        _cache[str(path)] = entry
    return entry[kind]


def log_info(path: Path) -> dict:
    """Header facts needed for validation, cached."""
    def compute(path):
        if not is_successful_eval(path):
            return {"successful": False}
        header = read_eval_log(str(path), header_only=True)
        return {
            "successful": True,
            "imported": any("imported" in tag for tag in header.eval.tags or []),
            "done": header.results.completed_samples,
            "total": header.results.total_samples,
        }
    return _cached(path, "info", compute)


def complete_log(repeat_dir: Path, delete: bool = False) -> tuple[Path | None, bool, list[str]]:
    """The one full successful log of a repeat dir, whether it is imported, and problems found."""
    problems, good = [], []
    for path in sorted(repeat_dir.glob("*.eval")):
        info = log_info(path)
        if not info["successful"]:
            if delete:  # crashed / partial (--limit) leftovers do not count, by design
                path.unlink()
            continue
        imported, done, total = info["imported"], info["done"], info["total"]
        # up to max(1, 2%) failed samples are tolerated (CONTRACT rule 1); imports are as they come
        if done < total - max(1, total // 50) and not imported:
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
    """Per sample: every assistant text and reasoning plus the output token count. The bare
    completion is often just "ANSWER: 204" or a flag, identical across genuine repeats."""
    samples = read_eval_log_samples(str(path), all_samples_required=False)
    out = {}
    for s in itertools.islice(samples, limit):
        parts = []
        for m in s.messages:
            if m.role != "assistant":
                continue
            content = m.content if isinstance(m.content, list) else [m.content]
            parts += [c if isinstance(c, str) else getattr(c, "text", None) or getattr(c, "reasoning", "") for c in content]
        usage = s.output.usage
        out[(s.id, s.epoch)] = ("\n".join(parts), usage.output_tokens if usage else None)
    return out


def completion_hashes(path: Path, limit: int) -> dict:
    """MD5 per sample of (assistant text, output tokens) — equality is all the replay check
    needs, and hashes keep the cache small. Cached; keyed by limit too."""
    def compute(path):
        return {f"{sid}\x00{epoch}": hashlib.md5(repr(val).encode()).hexdigest()
                for (sid, epoch), val in completions(path, limit).items()}
    return _cached(path, f"hashes:{limit}", compute)


SINGLE_REPEAT = {"swe_bench_mini", "terminal_bench_2_1", "terminal_bench_3", "intercode", "cybench", "agent_bench_os", "mmlu_pro_math"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100, help="Samples compared per log for the replay check.")
    parser.add_argument("--dup", type=float, default=0.5, help="Identical share at which two repeats count as replays.")
    parser.add_argument("--delete", action="store_true", help="Remove incomplete logs and replayed repeats.")
    args = parser.parse_args()

    if CACHE_PATH.exists():
        _cache.update(json.loads(CACHE_PATH.read_text()))

    problems = []
    # A model dir is any dir holding numbered repeat dirs, so roots can be data/, a bench, or a hash dir.
    model_dirs = sorted(
        d for root in args.roots for d in [root, *root.rglob("*")]
        # Skip legacy/ (archived) and dot-dirs: data/.cache/huggingface/download is full of
        # numerically-named dirs and otherwise reads as 5800 broken model dirs.
        if d.is_dir() and "legacy" not in d.parts and not any(x.startswith(".") for x in d.parts)
        and any(c.is_dir() and c.name.isdigit() for c in d.iterdir())
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
        # Agentic benches (and the 150-item MMLU-Pro subset) are collected once by design: a
        # sandboxed run costs hours per model and the judging pipeline only reads repeat 0.
        need = 1 if model_dir.parts[1] in SINGLE_REPEAT else args.min_repeats
        if len(usable) < need and (own or not usable):
            problems.append(f"{model_dir}: {len(usable)} usable repeat(s), need {need}")

        texts = {}
        for repeat, path in usable.items():
            try:
                texts[repeat] = completion_hashes(path, args.samples)
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

    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache))

    for line in problems:
        print(line)
    print(f"{len(model_dirs)} model dirs checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
