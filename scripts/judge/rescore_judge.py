"""Re-run judge_scorer over existing judge logs (in place) so old runs carry the
current metrics (answer_rate, position_bias, consistent_accuracy, length_corr).

Usage: uv run python scripts/judge/rescore_judge.py logs/pilot logs/aime [--jobs 8]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from judge.scorer import judge_scorer


def rescore(path: Path) -> str:
    log = read_eval_log(path)
    if log.status != "success" or not log.samples:
        return f"skip {path} ({log.status})"
    # the scorer never generates; a mock model avoids the header model needing API keys
    log = score(log, judge_scorer(), model="mockllm/model", action="overwrite", display="none", copy=False)
    write_eval_log(log, path)
    m = log.results.scores[0].metrics
    return f"{path}: " + " ".join(f"{k}={v.value:.3f}" for k, v in m.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(p for r in args.roots for p in r.rglob("*.eval"))
    print(f"{len(paths)} logs", file=sys.stderr)
    with ProcessPoolExecutor(args.jobs) as pool:
        for line in pool.map(rescore, paths, chunksize=4):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
