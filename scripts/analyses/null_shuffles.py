"""Shuffle nulls for the four conditions: are the results reading capability or structure?

Cell script (see scripts/analyses/results.py for the conventions). Writes
analysis/results/null_shuffles.csv (one row per domain x semester x condition with the actual
MSE, the null distribution and p = share of null draws at or below the actual) and prints the
pooled view: the mean actual MSE over tasks against the distribution of the pooled null.
"""

# %% settings
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("SMALL_JUDGE_DATA", "/Volumes/T7/small-judge-remote/data")
os.environ.setdefault("SMALL_JUDGE_EMBEDDINGS", "/Volumes/T7/small-judge-remote/transcript-embeddings")

from analysis import calibration, loading, matching, stats  # noqa: E402
from analysis.challenges import null_shuffles as ns  # noqa: E402
from analysis.embeddings import EmbeddingRegression  # noqa: E402
from protocols import PROTOCOLS  # noqa: E402

DOMAINS = ["science", "math", "coding", "cyber"]
DRAWS, SEED = 20, 0
VARIANTS = {"pairwise": ["linear", "bt"], "direct": ["raw"]}  # every listed conversion is scored
OUT = Path("analysis/results")

# %% the judge each task's results use: best on references, core condition
forecasts = pd.read_csv(OUT / "forecasts.csv")
best = stats.judge_summaries(forecasts)
best = best[best.judge == stats.BEST]  # rows carry condition "core"; recover the judge from the core rows
core = stats.core_condition(forecasts)
core = core[core.protocol.isin(["pairwise", "direct"])]
picked = (core[[(v in VARIANTS[p]) for p, v in zip(core.protocol, core.variant)]]
          .sort_values("judge_loo_mse").drop_duplicates(["domain", "semester", "protocol", "variant"])
          [["domain", "semester", "protocol", "variant", "judge", "condition"]])  # best judge per conversion

# %% run the nulls
rows = []
for name in DOMAINS:
    dom = loading.domain(name)
    features = {"glm_regex": PROTOCOLS["glm_regex"], "embedding": EmbeddingRegression("full")}
    for task in loading.tasks(dom):
        for cond, protocol in features.items():
            actual, nulls = ns.feature_nulls(protocol, task, DRAWS, SEED)
            rows.append(dict(domain=name, semester=task.semester, condition=cond, variant="full" if cond == "embedding" else "-",
                             judge="-", **ns.summarise(actual, nulls)))
        runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
        for protocol, variants in VARIANTS.items():
            for variant in variants:
                sel = picked[(picked.domain == name) & (picked.semester == task.semester) & (picked.protocol == protocol) & (picked.variant == variant)]
                if sel.empty:
                    continue
                judge, condition = sel.judge.iloc[0], sel.condition.iloc[0]
                run = next((r for r in runs if r.protocol == protocol and r.judge == judge and r.condition == condition), None)
                if run is None:
                    continue
                actual, nulls = ns.judge_nulls(run, task, variant, DRAWS, SEED)
                rows.append(dict(domain=name, semester=task.semester, condition=protocol, variant=variant, judge=judge,
                                 **ns.summarise(actual, nulls)))
        print(name, task.semester, "done", flush=True)
table = pd.DataFrame(rows)
table.to_csv(OUT / "null_shuffles.csv", index=False, float_format="%.4f")

# %% pooled view: mean actual over tasks vs the null
print(table.groupby(["condition", "variant"])[["actual", "null_mean", "null_p05", "null_p95", "p_null_le_actual"]].mean().round(3).to_string())
print()
print(table.round(3).to_string(index=False))
