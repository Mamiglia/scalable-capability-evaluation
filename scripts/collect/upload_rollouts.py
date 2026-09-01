"""Upload finalised rollout logs to LASR-G5/benchmarks next to the originals;
nothing is overwritten or moved.

Finds every log with `finalised_from` in its metadata under data/<bench>/<collection>/ plus
the <epoch>/reruns/ partials only when --include-reruns is explicit, skips paths already in
the repo, and adds the rest in one commit. --note appends a dated entry to collection_runs.md.

Usage:
  uv run python scripts/collect/upload_rollouts.py data/gpqa/b0a0d9c9 data/aime2024/44136fa3 ... [--commit] [--note "..."]
"""

import argparse
import datetime as dt
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from inspect_ai.log import read_eval_log

REPO = "LASR-G5/benchmarks"


def upload_candidate(path: Path, finalised_from: str | None, include_reruns: bool) -> bool:
    return include_reruns if path.parent.name == "reruns" else bool(finalised_from)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collections", type=Path, nargs="+", help="data/<bench>/<collection> dirs")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--include-reruns", action="store_true",
                        help="also upload partial logs under reruns/ (default: finalized logs only)")
    parser.add_argument("--note", help="entry for collection_runs.md")
    parser.add_argument("--message", default="finalised rollout logs after the 2026-08 data corrections")
    args = parser.parse_args()

    api = HfApi()
    parent_commit = api.dataset_info(REPO).sha
    existing = set(
        api.list_repo_files(REPO, repo_type="dataset", revision=parent_commit)
    )
    ops, skipped = [], 0
    for coll in args.collections:
        for path in sorted(coll.glob("*/[0-9]*/**/*.eval")):
            if path.name.startswith("._"):
                continue
            final = (read_eval_log(str(path), header_only=True).eval.metadata or {}).get("finalised_from")
            if not upload_candidate(path, final, args.include_reruns):
                continue
            rel = str(Path(*coll.parts[-2:]) / path.relative_to(coll))  # <bench>/<collection>/<model>/<epoch>/...
            if rel in existing:
                skipped += 1
                continue
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path)))
            print("add", rel)
    if args.note:
        local = hf_hub_download(
            REPO, "collection_runs.md", repo_type="dataset", revision=parent_commit
        )
        text = Path(local).read_text().rstrip("\n") + f"\n\n## {dt.date.today().isoformat()}\n\n{args.note}\n"
        ops.append(CommitOperationAdd(path_in_repo="collection_runs.md", path_or_fileobj=text.encode()))
    print(f"{len(ops)} files to add, {skipped} already in the repo")
    if args.commit and ops:
        info = api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=args.message,
            parent_commit=parent_commit,
        )
        print("committed", info.oid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
