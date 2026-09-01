"""Re-run judge_scorer over existing judge logs so old runs carry the current
parser and metrics (answer_rate, position_bias, consistent_accuracy, length_corr).

Omit --out for the existing in-place behavior, or pass one input root and --out
to write corrected copies under a mirrored directory without modifying the input.

Usage:
  uv run python scripts/judge/rescore_judge.py logs/pilot logs/aime [--jobs 8]
  uv run python scripts/judge/rescore_judge.py logs/pilot --out logs-rescored/pilot
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from judge.scorer import judge_scorer


def rescore(paths: tuple[Path, Path]) -> str:
    input_path, out_path = paths
    log = read_eval_log(input_path)
    if log.status != "success" or not log.samples:
        return f"skip {input_path} ({log.status})"
    # the scorer never generates; a mock model avoids the header model needing API keys
    log = score(log, judge_scorer(), model="mockllm/model", action="overwrite", display="none", copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, out_path)
    m = log.results.scores[0].metrics
    return f"{input_path}: " + " ".join(f"{k}={v.value:.3f}" for k, v in m.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="with one input root, mirror corrected logs here")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.out and len(args.roots) != 1:
        ap.error("--out requires exactly one input root")
    if args.out and args.out.resolve().is_relative_to(args.roots[0].resolve()):
        ap.error("--out must not be the input root or one of its descendants")
    paths = sorted(p for root in args.roots for p in root.rglob("*.eval"))
    tasks = [
        (path, args.out / path.relative_to(args.roots[0]) if args.out else path)
        for path in paths
    ]
    print(f"{len(tasks)} logs", file=sys.stderr)
    with ProcessPoolExecutor(args.jobs) as pool:
        for line in pool.map(rescore, tasks, chunksize=4):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
