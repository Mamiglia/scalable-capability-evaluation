"""Pull the judges' EXPLANATION text out of every judging log.

    uv run python scripts/verdict_dissection/extract_judge_explanations.py [--workers 6]

Reads the judging logs named in the freeze config (the T7 mirror of LASR-G5/judging) and writes
judge_results/verdict_dissection/judge_explanations.parquet: one row per judge sample with the
freeze's verdict keys (bench, judge_log, judge_sample, judge_epoch, judge) plus `explanation`
(the text between EXPLANATION: and ANSWER:) and `completion` (the whole reply). Rows join the
freeze verdicts on the key columns.
"""
import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66"
OUT = ROOT / "judge_results/verdict_dissection/judge_explanations.parquet"
EXPLANATION = re.compile(r"EXPLANATION:\s*(.*?)\s*(?:ANSWER:|$)", re.S)


def explanation_of(completion):
    match = EXPLANATION.search(completion or "")
    return match.group(1).strip() if match else ""


def read_log(args):
    bench, judging_root, path = args
    try:
        log = read_eval_log(str(path))
    except Exception as error:      # a broken file should not stop the sweep
        return [{"bench": bench, "judge_log": str(path.relative_to(judging_root)), "error": repr(error)}]
    if log.status != "success":
        return []
    judge = log.eval.model.split("/")[-1]
    rows = []
    for sample in log.samples or []:
        completion = sample.output.completion if sample.output else ""
        rows.append({"bench": bench, "judge": judge, "judge_log": str(path.relative_to(judging_root)),
                     "judge_sample": str(sample.id), "judge_epoch": int(getattr(sample, "epoch", 0) or 0),
                     "explanation": explanation_of(completion), "completion": completion or ""})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    config = yaml.safe_load((FREEZE / "code/verdict_dissection.yaml").read_text())["data"]
    jobs = []
    for subset in config["judging_subsets"]:
        judging_root = Path(config["judging"]) / subset
        bench = subset.split("/")[-1]
        jobs += [(bench, judging_root, p) for p in sorted(judging_root.rglob("*.eval")) if not p.name.startswith(".")]
    print(f"{len(jobs)} judging logs", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(read_log, jobs, chunksize=4)):
            rows += batch
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} logs, {len(rows)} rows", flush=True)
    table = pd.DataFrame(rows)
    if "error" in table:
        print("unreadable logs:", table.error.dropna().to_dict())
        table = table[table.error.isna()].drop(columns="error")
    table.to_parquet(OUT, index=False)
    verdicts = pd.read_parquet(FREEZE / "verdicts.parquet", columns=["bench", "judge_log", "judge_sample", "judge_epoch", "judge"])
    joined = verdicts.merge(table, on=["bench", "judge_log", "judge_sample", "judge_epoch", "judge"], how="left")
    print(f"{len(table)} rows; freeze verdicts with an explanation: "
          f"{int((joined.explanation.fillna('') != '').sum())}/{len(verdicts)}")
    print(joined.groupby("judge").explanation.apply(lambda s: (s.fillna('') != '').mean()).round(3).to_string())


if __name__ == "__main__":
    main()
