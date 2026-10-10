"""Per-transcript GLM features for every sample of every data/ log -> logs/features/.

    uv run python scripts/glm_features/extract.py [bench ...] [--data data] [--out logs/features]

glm_features/ is a clone of https://huggingface.co/datasets/generality-labs/glm_features and mirrors
data/'s layout: <bench>/<task_args_hash>/<model>/<repeat>/<eval id>.csv, one row per sample
and reasoning view (visible: text + raw CoT; hidden: text only). Values are per-transcript
rates; averaging over a model's selected items happens at fit time. Only successful full-run
logs newer than their CSV are extracted; `legacy/`, `judge/` and crashed/partial logs are skipped.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from configs.paths import is_successful_eval
from protocols.transcript_features.reader import extract_log


SKIP = {"legacy", "judge"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benches", nargs="*", help="default: every bench under --data")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="logs/features", type=Path)
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    jobs = []
    for log in sorted(args.data.glob("*/*/*/*/*.eval")):
        relative = log.relative_to(args.data)
        if relative.parts[0] in SKIP or (args.benches and relative.parts[0] not in args.benches):
            continue
        target = args.out / relative.with_suffix(".csv")
        if target.exists() and target.stat().st_mtime >= log.stat().st_mtime:
            continue
        if log.stat().st_size < 1024 and log.read_bytes().startswith(b"version https://git-lfs"):
            print(f"skip (LFS pointer, not fetched): {log}")
            continue
        if not is_successful_eval(log):  # crashed/partial runs left beside the real one for resume
            print(f"skip (not a successful full run): {log}")
            continue
        jobs.append((log, target))
    # one process per log; big HLE logs cost a few hundred MB each, hence --workers.
    with ProcessPoolExecutor(args.workers) as pool:
        for target in pool.map(_write, jobs):
            print(target, flush=True)


def _write(job: tuple[Path, Path]) -> str:
    log, target = job
    rows = extract_log(log)
    if rows.empty:  # an interrupted start (only _journal/start.json), not a run
        return f"skip (no samples): {log}"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(target, index=False)
    return str(target)


if __name__ == "__main__":
    main()
