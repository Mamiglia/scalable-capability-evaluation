"""Upload packaged judge logs to LASR-G5/judging as new versions next to the published ones.

For every packaged file (<subset>/<bench>/<judge>/<pair>/<stamp>_<major>|<minor>-<e>_<id>.eval)
the published copies for the same (judge, pair, epoch) are read from the local mirror. A
file whose samples and verdicts (id, epoch, value, answer) equal a published copy is not
uploaded. Otherwise it is added alongside the existing copy. Moving existing copies to
superseded/ requires the explicit --supersede flag. Nothing is ever overwritten.

Usage:
  uv run python scripts/judge/upload_judging.py <packaged dir> --mirror /Volumes/T7/data/hf-judging [--commit] [--message ...]
Without --commit only the plan is printed.
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi
from inspect_ai.log import read_eval_log_samples

from package_judging import published_epoch, published_pair

REPO = "LASR-G5/judging"


def legacy_published(mirror: Path) -> dict[tuple[Path, str], list[Path]]:
    out: dict[tuple[Path, str], list[Path]] = {}
    for path in mirror.glob("aime/*/*/*/*.eval"):
        if path.name.startswith("._") or "-vs-" in path.parent.name:
            continue
        rel = path.relative_to(mirror)
        key = (Path(*rel.parts[:3]) / published_pair(path), published_epoch(path))
        out.setdefault(key, []).append(path)
    return out


def aime_collections(mirror: Path) -> dict[Path, str]:
    out: dict[Path, str] = {}
    for path in mirror.glob("aime/*/*/*/*.eval"):
        if not path.name.startswith("._") and "-vs-" not in path.parent.name:
            rel = path.relative_to(mirror)
            out.setdefault(Path(*rel.parts[:2]), rel.parts[3])
    return out


def judge_directories(mirror: Path) -> dict[tuple[Path, str], str]:
    return {
        (Path(*path.relative_to(mirror).parts[:2]), path.name.casefold()): path.name
        for path in [*mirror.glob("aime/*/*"), *mirror.glob("pilot/*/*")]
        if path.is_dir()
    }


def repo_path(rel: Path, collections: dict[Path, str], judges: dict[tuple[Path, str], str] | None = None) -> Path:
    parts = list(rel.parts)
    parts[2] = (judges or {}).get((Path(*parts[:2]), parts[2].casefold()), parts[2])
    if parts[0] == "aime":
        parts[3] = collections[Path(*parts[:2])]
    return Path(*parts)


def verdicts(path: Path) -> set[tuple]:
    out = set()
    for s in read_eval_log_samples(str(path), all_samples_required=False):
        score = next(iter(s.scores.values()), None) if s.scores else None
        out.add((s.id, s.epoch, score.value if score else None, score.answer if score else None))
    return out


def supersede_operations(old: list[Path], mirror: Path, enabled: bool) -> list:
    if not enabled:
        return []
    ops = []
    for path in old:
        rel = str(path.relative_to(mirror))
        ops.extend([
            CommitOperationCopy(src_path_in_repo=rel, path_in_repo=f"superseded/{rel}"),
            CommitOperationDelete(path_in_repo=rel),
        ])
    return ops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packaged", type=Path)
    parser.add_argument("--mirror", type=Path, required=True, help="local copy of the dataset")
    parser.add_argument("--commit", action="store_true", help="really upload")
    parser.add_argument("--supersede", action="store_true",
                        help="move prior versions into superseded/ (default: additions only)")
    parser.add_argument(
        "--parent-commit", help="abort if the live repository moved from this revision"
    )
    parser.add_argument("--message", default="judging: new versions after the 2026-08 data corrections")
    args = parser.parse_args()
    if args.commit and not args.parent_commit:
        parser.error("--commit requires --parent-commit")

    ops, unchanged, added, superseded = [], 0, 0, 0
    api = HfApi()
    repo_files = (
        set(api.list_repo_files(REPO, repo_type="dataset", revision=args.parent_commit))
        if args.commit
        else set()
    )
    legacy = legacy_published(args.mirror)
    collections = aime_collections(args.mirror)
    judges = judge_directories(args.mirror)
    for new in sorted(args.packaged.rglob("*.eval")):
        if new.name.startswith("._"):
            continue
        rel = new.relative_to(args.packaged)
        pair_dir = args.mirror / rel.parent
        epoch = published_epoch(new)
        old = [p for p in pair_dir.glob("*.eval") if not p.name.startswith("._") and published_epoch(p) == epoch]
        old += legacy.get((rel.parent, epoch), [])
        new_verdicts = verdicts(new)
        if any(verdicts(p) == new_verdicts for p in old):
            unchanged += 1
            continue
        target = repo_path(rel, collections, judges)
        if str(target) in repo_files:
            raise RuntimeError(f"refusing to overwrite existing repository path: {target}")
        ops.append(CommitOperationAdd(path_in_repo=str(target), path_or_fileobj=str(new)))
        added += 1
        if args.supersede and args.commit:
            for path in old:
                archive = f"superseded/{path.relative_to(args.mirror)}"
                if archive in repo_files:
                    raise RuntimeError(
                        f"refusing to overwrite existing supersede path: {archive}"
                    )
        moves = supersede_operations(old, args.mirror, args.supersede)
        ops.extend(moves)
        superseded += len(moves) // 2
        if moves:
            for path in old:
                print(f"supersede {path.relative_to(args.mirror)}")
        print(f"add       {target}")
    print(f"{added} new files, {superseded} published copies moved to superseded/, {unchanged} unchanged")
    if args.commit and ops:
        info = api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=args.message,
            parent_commit=args.parent_commit,
        )
        print("committed", info.oid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
