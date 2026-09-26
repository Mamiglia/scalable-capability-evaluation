"""The paper's per-domain error table (Table 3) and its candidate-judge table (Table 8), from the results tables.

Table 3: mean squared error per protocol and domain over the 16 targets of 2025-II and 2026-I (judge protocols: each
task's best judge on references, as figure one), pooled = the mean over domains. Marks: two-sided one-sample t-test
of the targets' log likelihood ratios against 0 (full reference set, t shape), marked only when the mean is positive;
the pooled column uses figure one's test (a target model over its domains is the unit).
Table 8: per task and judge protocol, the three candidate judges ordered by reference leave-one-out error.

Reads analysis/results/{forecasts,likelihood_ratio}.csv; writes analysis/results/paper_tables/{table3,table8}.csv.
Run from the repo root: PYTHONPATH=src uv run python scripts/analyses/paper_tables.py
"""
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

from analysis import stats
from analysis.challenges import generalisation as gen
from analysis.challenges import likelihood_ratio as lr

VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}  # as figure one
SEMESTERS = ["2025-II", "2026-I"]
DOMAINS = ["science", "math", "coding", "cyber"]
PROTOCOLS = {"easy_items": "Baseline protocol", "glm_regex": "Deterministic features", "embedding": "Embedding",
             "direct": "Absolute scoring", "pairwise": "Pairwise judging"}
OUT = Path("analysis/results/paper_tables")


def mark(mean: float, p: float) -> str:
    return "" if mean <= 0 else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


forecasts = pd.read_csv("analysis/results/forecasts.csv")
rows = gen.rows(forecasts, VARIANTS, tuple(SEMESTERS))
llr = lr.select(pd.read_csv("analysis/results/likelihood_ratio.csv"), lr.chosen_judges(forecasts, VARIANTS))
llr = llr[(llr["shape"] == lr.SHAPE) & (llr.dropped == "-") & llr.semester.isin(SEMESTERS)]
pooled = lr.pooled_tests(llr, unit=("semester", "model")).set_index("protocol")
pooled["p"] = 2 * pooled.p.clip(upper=1 - pooled.p)  # two-sided: pooled_tests gives the upper tail, as figure one

table = []
for protocol, label in PROTOCOLS.items():
    mse = rows[rows.protocol == protocol].groupby("domain").err.mean()
    row = {"Protocol": label, "Pooled": f"{mse.reindex(DOMAINS).mean():.2f}"}
    if protocol in pooled.index:
        row["Pooled"] += mark(pooled.loc[protocol, "llr"], pooled.loc[protocol, "p"])
    for domain in DOMAINS:
        cell, g = f"{mse[domain]:.2f}", llr[(llr.protocol == protocol) & (llr.domain == domain)]
        if len(g):
            cell += mark(g.llr.mean(), scipy_stats.ttest_1samp(g.llr, 0).pvalue)
        row[gen.DOMAIN_LABELS[domain]] = cell
    table.append(row)
table = pd.DataFrame(table)

core = stats.core_condition(forecasts)
core = core[core.protocol.isin(["pairwise", "direct"]) & (core.variant == core.protocol.map(VARIANTS))]
loo = core.drop_duplicates(["domain", "semester", "protocol", "judge"]).sort_values("judge_loo_mse")
judges = (loo.groupby(["semester", "domain", "protocol"]).judge.agg(lambda j: list(j)[:3]).apply(pd.Series)
          .rename(columns={0: "Best", 1: "2nd best", 2: "3rd best"}).reset_index())

OUT.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT / "table3.csv", index=False)
judges.to_csv(OUT / "table8.csv", index=False)
print(table.to_string(index=False))
print(judges.to_string(index=False))
