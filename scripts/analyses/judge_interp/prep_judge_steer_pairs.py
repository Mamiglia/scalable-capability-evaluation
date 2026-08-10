"""Pair file for the reader-steering test.

Samples qwen-judged pairs from the pilot verdicts, stratified by |elo
gap| terciles, and joins question/options/text for both slots. Writes
pairs.parquet for pod_judge_steer.py.

Usage:
  uv run python scripts/analyses/judge_interp/prep_judge_steer_pairs.py \
      --inputs <inputs.parquet> --staging <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    js = cfg["interp_pod"]["judge_steer"]

    rows = pd.read_parquet(ROOT / cfg["rows_parquet"])
    rows = rows[rows["judge"] == "qwen3-32b"].reset_index(drop=True)
    elo = pd.read_csv(ROOT / cfg["out_dir"] / "elo.csv")
    elo_of = {(t, i): e for t, i, e in zip(elo["text_id"], elo["item"], elo["elo"])}

    # One row per unordered pair (both slot orders run on the pod).
    seen = set()
    pairs = []
    for r in rows.itertuples():
        key = (r.item, frozenset((r.text_id_slot1, r.text_id_slot2)))
        if key in seen:
            continue
        seen.add(key)
        e1 = elo_of.get((r.text_id_slot1, r.item))
        e2 = elo_of.get((r.text_id_slot2, r.item))
        if e1 is None or e2 is None:
            continue
        pairs.append(dict(item=r.item, text_id_1=r.text_id_slot1,
                          text_id_2=r.text_id_slot2, gap=abs(e1 - e2)))
    pairs = pd.DataFrame(pairs)
    terciles = pairs["gap"].quantile([1 / 3, 2 / 3]).to_numpy()
    pairs["stratum"] = np.digitize(pairs["gap"], terciles)
    rng = np.random.default_rng(js["seed"])
    take = js["n_pairs"] // 3
    sampled = pd.concat([
        g.sample(min(take, len(g)), random_state=rng.integers(2**32))
        for _, g in pairs.groupby("stratum")]).reset_index(drop=True)

    inputs = pd.read_parquet(args.inputs).set_index(["text_id", "item"])
    def col(ids, items, field):
        return [inputs.loc[(t, i), field] for t, i in zip(ids, items)]
    sampled["question"] = col(sampled.text_id_1, sampled.item, "question")
    for n in (1, 2):
        ids = sampled[f"text_id_{n}"]
        sampled[f"options_{n}"] = col(ids, sampled.item, "options")
        sampled[f"text_{n}"] = col(ids, sampled.item, "text")

    args.staging.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.staging / "pairs.parquet", index=False)
    print(f"{len(sampled)} pairs "
          f"(gap terciles at {terciles.round(2)}), -> {args.staging}/pairs.parquet")


if __name__ == "__main__":
    main()
