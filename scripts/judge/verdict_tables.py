#!/usr/bin/env python3
"""Verdict and rollout tables straight from judge logs, for judged_capability_over_time.py.

The verdict-dissection store only covers the pilot pool; this reads any set of judge-log
roots (flat HF mirror or phased run logs) and writes
  verdicts.parquet  bench, judge, item, model_left, model_right, epoch_left, epoch_right, pick_left
  rollouts.parquet  bench, model, item, correct, is_canonical   (every rollout in every candidate
                    log the judge logs reference; all count, so accuracy averages over epochs)
Candidate logs are named by their pool path data/<bench>/<collection>/<model>/<epoch>/*.eval.

Usage:
  uv run python scripts/judge/verdict_tables.py --out-dir judge_results/frontier/tables \\
      /Volumes/T7/data/hf-judging/pilot/gpqa /Volumes/T7/data/hf-judging/pilot/swe_bench_mini logs/frontier
"""

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import is_successful_eval
from judge.util import extract_score


def rollout_key(path: str) -> tuple[str, str, int]:
    """(bench, model, rollout epoch) of a candidate log path."""
    parts = Path(path).parts
    if parts[0] != "data" or len(parts) < 5:
        raise ValueError(f"not a pool log path: {path}")
    return parts[1], parts[3], int(parts[4])


def read_verdicts(roots: list[Path]) -> tuple[pd.DataFrame, set[str]]:
    rows, candidate_logs = {}, set()
    for root in roots:
        for path in sorted(root.rglob("*.eval")):
            try:
                log = read_eval_log(str(path))
            except Exception as error:  # noqa: BLE001 - a partial or corrupt file is skipped, not fatal
                print(f"skip {path}: {error}")
                continue
            if log.status != "success":
                continue
            meta = log.eval.metadata or {}
            judge = log.eval.model.split("/")[-1]
            bench, major, epoch_major = rollout_key(meta["major_path"])
            _, minor, epoch_minor = rollout_key(meta["minor_path"])
            candidate_logs.update([meta["major_path"], meta["minor_path"]])
            for sample in log.samples or []:
                item, slot = str(sample.id).rsplit("_", 1)
                picked = next(iter(sample.scores.values())).answer if sample.scores else None
                if picked not in ("1", "2"):
                    continue
                major_first = slot == "1"
                left, right = (major, minor) if major_first else (minor, major)
                epochs = (epoch_major, epoch_minor) if major_first else (epoch_minor, epoch_major)
                row = dict(bench=bench, judge=judge, item=item, model_left=left, model_right=right,
                           epoch_left=epochs[0], epoch_right=epochs[1], judge_epoch=int(sample.epoch or 1),
                           pick_left=int(picked == "1"))
                key = (judge, bench, item, left, right, *epochs, row["judge_epoch"])
                if key not in rows or log.eval.created > rows[key][0]:   # a re-run replaces older verdicts
                    rows[key] = (log.eval.created, row)
    return pd.DataFrame([row for _, row in rows.values()]), candidate_logs


def current_log(path: str) -> str:
    """The referenced candidate log, or the successful log now in its directory if it was replaced."""
    if Path(path).exists():
        return path
    moved = sorted(Path(path).parents[3].glob(f"*/*/{Path(path).name}"))   # same log filed under another model dir
    replacements = [str(p) for p in moved] or [
        str(p) for p in sorted(Path(path).parent.glob("*.eval")) if is_successful_eval(p)]
    if not replacements:
        raise FileNotFoundError(f"candidate log {path} is gone and nothing replaced it")
    print(f"{path} replaced by {replacements[0]}")
    return replacements[0]


def read_rollouts(candidate_logs: set[str]) -> pd.DataFrame:
    """Easy-half copies (<collection>-easy-half) point back to the full log: accuracy needs every item."""
    rows = []
    for path in sorted({current_log(p.replace("-easy-half/", "/")) for p in candidate_logs}):
        bench, model, epoch = rollout_key(path)
        for sample in read_eval_log_samples(path, all_samples_required=False):
            rows.append(dict(bench=bench, model=model, item=str(sample.id), epoch=epoch,
                             correct=extract_score(sample), is_canonical=True))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="judge-log directories, searched recursively")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    verdicts, candidate_logs = read_verdicts(args.roots)
    rollouts = read_rollouts(candidate_logs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verdicts.to_parquet(args.out_dir / "verdicts.parquet", index=False)
    rollouts.to_parquet(args.out_dir / "rollouts.parquet", index=False)
    print(verdicts.groupby(["bench", "judge"]).size().unstack(fill_value=0).to_string())
    print(f"{len(verdicts)} verdicts; rollouts for "
          f"{rollouts.groupby('bench').model.nunique().to_dict()} models")


if __name__ == "__main__":
    main()
