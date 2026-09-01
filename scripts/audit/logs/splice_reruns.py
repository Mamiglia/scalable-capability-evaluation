"""Fold ``<repeat>/reruns/*.eval`` back into the log they were re-run from.

A targeted re-run collects a handful of samples by ``sample_id`` to replace defective ones
(no reasoning returned, a truncated trace). It is not a repeat of its own: it has a partial
sample set, so ``is_successful_eval`` rejects it -- and ``successful_evals`` raises on it
outright, because the ``isdigit()`` check on the parent directory runs before that. Left in
place a rerun breaks judging; the samples in it are only useful once merged back.

Each spliced sample replaces the one with its id, scores and header metrics are recomputed,
and the edit is recorded inside the log (``eval.metadata["edits"]``) before the ``reruns/``
directory is removed.

    uv run python scripts/audit/logs/splice_reruns.py            # dry run
    uv run python scripts/audit/logs/splice_reruns.py --write
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

from configs.paths import is_successful_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_edits import stamp_edit  # noqa: E402


def created_at(path: Path) -> str:
    """When the run happened, from the log itself."""
    return read_eval_log(str(path), header_only=True).eval.created


def archive(rerun_dir: Path) -> None:
    """Park a folded-in rerun under legacy/, merging when an earlier pass already put one
    there -- the same repeat can be re-run more than once."""
    dest = Path("data") / "legacy" / "reruns" / "/".join(rerun_dir.parts[1:-1]) / "reruns"
    dest.mkdir(parents=True, exist_ok=True)
    for item in rerun_dir.iterdir():
        target = dest / item.name
        if target.exists():
            target = dest / f"{item.stem}-{len(list(dest.iterdir()))}{item.suffix}"
        shutil.move(str(item), str(target))
    rerun_dir.rmdir()


def parent_log(repeat_dir: Path) -> Path | None:
    """The full successful log in this repeat dir; the newest when several are present."""
    full = sorted(p for p in repeat_dir.glob("*.eval") if is_successful_eval(p))
    return full[-1] if full else None


def accuracy_of(log) -> float | None:
    for scorer in log.results.scores if log.results else []:
        metric = scorer.metrics.get("accuracy")
        if metric:
            return metric.value
    return None


def splice(target: Path, reruns: list[Path], write: bool) -> tuple[int, list[str]]:
    log = read_eval_log(str(target), resolve_attachments=True)
    by_id = {str(s.id): i for i, s in enumerate(log.samples)}
    before = accuracy_of(log)
    replaced, sources = [], []
    for rerun in reruns:
        fresh = read_eval_log(str(rerun), resolve_attachments=True)
        for sample in fresh.samples:
            index = by_id.get(str(sample.id))
            if index is None:  # a rerun of a sample this log does not have: leave it alone
                continue
            sample.epoch = log.samples[index].epoch
            log.samples[index] = sample
            replaced.append(str(sample.id))
        sources.append(rerun.name)
    if not replaced:
        return 0, []

    # Scorer names are taken across all samples, not from the first one: a rerun collected
    # under a renamed task can carry a differently-named scorer, and a sample can have none.
    # Several benchmarks legitimately score twice (apex: matharena_correct + matharena_scorer),
    # so every scorer is updated rather than one being singled out.
    names = sorted({n for s in log.samples for n in (s.scores or {})})
    if not names:
        raise ValueError(f"{target}: no scores to update")
    changed_ids = set(replaced)

    def is_correct(score) -> bool:
        return bool(score and score.value in ("C", 1, 1.0, True))

    # The reductions hold their own copy of every score; a reader that goes through them
    # would otherwise still see the pre-splice value. They are keyed per scorer.
    fixed = {
        (name, str(s.id)): s.scores[name]
        for s in log.samples if str(s.id) in changed_ids
        for name in (s.scores or {})
    }
    for reduction in log.reductions or []:
        for entry in reduction.samples:
            score = fixed.get((reduction.scorer, str(entry.sample_id)))
            if score is not None:
                entry.value = 1.0 if is_correct(score) else 0.0
                entry.answer = score.answer
    accuracies = {
        name: sum(is_correct((s.scores or {}).get(name)) for s in log.samples) / len(log.samples)
        for name in names
    }
    for sc in log.results.scores if log.results else []:
        if "accuracy" in sc.metrics and sc.name in accuracies:
            sc.metrics["accuracy"].value = accuracies[sc.name]
    accuracy = accuracies[names[0]]
    if write:
        stamp_edit(
            log,
            tool="scripts/audit/logs/splice_reruns.py",
            reason=f"folded targeted rerun(s) {', '.join(sources)} back into this log; "
                   f"replaced sample(s) {', '.join(sorted(set(replaced)))}",
            source=target,
            samples_changed=len(set(replaced)),
            before={"accuracy": before},
            after={"accuracy": accuracy},
            transcripts_modified=True,
        )
        tmp = target.with_name(f".splice-{target.name}")
        try:
            write_eval_log(log, tmp)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    return len(set(replaced)), sorted(set(replaced))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rerun_dirs = sorted(
        d for root in args.roots for d in root.rglob("reruns")
        if d.is_dir() and "legacy" not in d.parts and d.parent.name.isdigit()
    )
    print(f"{len(rerun_dirs)} reruns/ dirs", file=sys.stderr)
    total, orphaned, superseded = 0, [], []
    for rerun_dir in rerun_dirs:
        target = parent_log(rerun_dir.parent)
        reruns = sorted(rerun_dir.glob("*.eval"))
        if target is None:
            orphaned.append(rerun_dir)
            print(f"NO PARENT  {rerun_dir} ({len(reruns)} rerun log(s)) -- left in place")
            continue
        # A rerun older than the canonical log is already accounted for: either it was folded
        # in when that log was written, or a full re-collection has since replaced the samples
        # outright. Splicing it back would drag stale samples into a newer run -- and, where
        # the newer run is a fresh collection, mix two collections inside one log.
        # eval.created, not the file mtime: a git checkout or merge stamps every file with the
        # time it was written to disk, which says nothing about when the run happened.
        newest = max(created_at(r) for r in reruns)
        if newest < created_at(target):
            superseded.append(rerun_dir)
            print(f"SUPERSEDED {rerun_dir} -- canonical log is newer; archiving")
            if args.write:
                archive(rerun_dir)
            continue
        changed, ids = splice(target, reruns, args.write)
        total += changed
        print(f"{changed:3d} sample(s)  {target}\n              from {[r.name for r in reruns]}\n"
              f"              ids {ids}")
        if args.write and changed:
            archive(rerun_dir)
    print(f"\n{total} samples spliced; {len(superseded)} rerun dir(s) superseded and archived"
          f"{'; ' + str(len(orphaned)) + ' orphaned' if orphaned else ''}", file=sys.stderr)
    if not args.write:
        print("dry run; pass --write to apply", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
