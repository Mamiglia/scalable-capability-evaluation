#!/usr/bin/env python3
"""Cross-domain selection over the deep sweeps: pick a configuration on the other domains, score it on this one.

    python scripts/embeddings/hillclimb2_cross.py [--domains science math coding cyber]
        [--root /fast/jyeon/smalljudge_embeddings/hillclimb]

Reads <root>/hillclimb2_<domain>.csv (one row per configuration: config, family, variant, preproc,
cv_mse, n, mse_all, ...). A configuration is eligible when it is scored on every target of every
domain. For each domain the "leave-one-domain-out" rule picks the configuration with the lowest mean
relative MSE on the other domains (mse_all divided by that domain's registered emb_knn mse_all, so
domains with different theta spreads weigh equally) and reports its mse_all on the held-out domain.
That number is target-free for the held-out domain. Also prints the 15 configurations with the lowest
mean relative MSE over all domains (an optimistic, all-domain pick) and the registered protocols.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domains", nargs="+", default=["science", "math", "coding", "cyber"])
    parser.add_argument("--root", type=Path, default=Path("/fast/jyeon/smalljudge_embeddings/hillclimb"))
    args = parser.parse_args()

    tables = {}
    for domain in args.domains:
        path = args.root / f"hillclimb2_{domain}.csv"
        if not path.exists():
            print(f"skipping {domain}: no {path.name}")
            continue
        table = pd.read_csv(path).dropna(subset=["mse_all"])
        table = table[table.n == table.n.max()].set_index("config")
        tables[domain] = table.mse_all
    if len(tables) < 2:
        print("need at least two domains")
        return 1
    mse = pd.DataFrame(tables).dropna()  # configs scored in every domain
    relative = mse / mse.loc["registered/emb_knn"]
    print(f"{len(mse)} configurations scored on every target of {list(mse.columns)}\n")

    rows = []
    for domain in mse.columns:
        others = [d for d in mse.columns if d != domain]
        pick = relative[others].mean(axis=1).idxmin()
        rows.append({"held-out domain": domain, "picked on the others": pick,
                     "held-out mse": round(mse.loc[pick, domain], 3),
                     "registered emb_knn": round(mse.loc["registered/emb_knn", domain], 3),
                     "registered emb_kr": round(mse.loc["registered/emb_kr", domain], 3)})
    print("-- leave-one-domain-out selection (target-free for the held-out domain) --")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- 15 best by mean relative MSE over all domains (all-domain pick, optimistic) --")
    summary = mse.assign(mean_relative=relative.mean(axis=1)).sort_values("mean_relative")
    print(summary.head(15).round(3).to_string())
    print("\n-- registered protocols --")
    print(summary.loc[[c for c in summary.index if c.startswith("registered/")]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
