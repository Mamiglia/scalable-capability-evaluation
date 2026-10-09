"""The benchmark score: MSE of theta-hat vs theta* over the 64 targets of the paper's main tasks (2025-II and 2026-I,
every domain), the paper's protocols (data/meta/forecasts.csv, `paper` rows) next to any protocol you evaluated.

    uv run python scripts/leaderboard.py [name ...]   # name: the X of data/meta/<domain>/eval_X.csv (`uv run evaluate`)

Lower is better; the baseline (every target = the references' mean theta*, what saturated benchmarks give) is the bar.
"""

import sys

import pandas as pd

from reference.tables import META

DOMAINS = ["science", "math", "coding", "cyber"]
MAIN = ["2025-II", "2026-I"]
NAMES = {"easy_items": "Baseline (reference mean)", "glm_regex": "Deterministic features", "embedding": "Embedding",
         "direct": "Absolute scoring", "pairwise": "Pairwise judge"}

paper = pd.read_csv(META / "forecasts.csv").query("paper")
rows = [paper.assign(protocol=paper.protocol.map(NAMES))]
for name in sys.argv[1:]:
    for domain in DOMAINS:
        path = META / domain / f"eval_{name}.csv"
        if path.exists():
            rows.append(pd.read_csv(path).assign(domain=domain, protocol=name))
        else:
            print(f"{path} missing: run `uv run evaluate configs/domain/{domain}.yml --semester {' '.join(MAIN)} ...`")
f = pd.concat(rows)
f = f[f.semester.isin(MAIN)].assign(se=lambda f: (f.theta_hat - f.theta) ** 2)
table = f.pivot_table(index="protocol", columns="domain", values="se", aggfunc="mean")[DOMAINS]
table["all"] = f.groupby("protocol").se.mean()
table["n"] = f.groupby("protocol").se.count()  # 64 when every target has a forecast
print(table.sort_values("all").round(2).to_string())
