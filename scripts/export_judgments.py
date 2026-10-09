"""The paper's judge outputs and forecasts as flat tables in data/meta/, so a new protocol needs no .eval parsing.

    PYTHONPATH=src uv run python scripts/export_judgments.py

    judgments_pairwise.csv  domain, semester, judge, condition, bench, item, model_left, model_right, pick_left,
                            chars_left, chars_right   one verdict per (item, pair, shown order); pick_left 1 = the
                            left (first shown) transcript judged more capable
    judgments_direct.csv    domain, semester, judge, condition, bench, item, model, epoch, score   1-10 rating
    forecasts.csv           analysis/results/forecasts.csv (scripts/analyses/results.py): theta_hat per target for
                            every protocol, variant, judge and condition, plus `paper` = the row behind the paper's
                            headline (core condition, the task's best judge on reference LOO, pairwise linear,
                            direct raw, embedding full)

Judges = each task's top-3 references (fit.yml `top`), runs as analysis.loading reads them, restricted to the task's
train items and eligible transcripts by analysis.matching. Never launches judging.
"""

from pathlib import Path

import pandas as pd

from analysis import loading, stats
from analysis.challenges import likelihood_ratio as lr
from analysis.matching import match
from reference.tables import META

DOMAINS = ["science", "math", "coding", "cyber"]
HEADLINE = {"pairwise": "linear", "direct": "raw"}

rows = {"pairwise": [], "direct": []}
for name in DOMAINS:
    dom = loading.domain(name)
    for task in loading.tasks(dom):
        for run in loading.judge_runs(dom, task):
            run = match(run, task)
            rows[run.protocol].append(run.table.drop(columns=["judge", "semester", "imported"], errors="ignore")
                                      .assign(domain=name, semester=task.semester, judge=run.judge, condition=run.condition))
        print(name, task.semester, flush=True)
for protocol, frames in rows.items():
    table = pd.concat(frames, ignore_index=True)
    key = ["domain", "semester", "judge", "condition", "bench", "item"]
    table = table[key + [c for c in table.columns if c not in key]].sort_values(key + list(table.columns.drop(key)[:2]))
    table.to_csv(META / f"judgments_{protocol}.csv", index=False)
    print(protocol, len(table))

f = pd.read_csv("analysis/results/forecasts.csv")
chosen = lr.chosen_judges(f, HEADLINE).assign(paper=True)
core = stats.core_condition(f)
core = core[[HEADLINE.get(p) == v for p, v in zip(core.protocol, core.variant)]]
paper = core.merge(chosen)[["domain", "semester", "protocol", "variant", "judge", "condition", "model"]]
paper = pd.concat([paper, f[f.protocol.isin(["easy_items", "glm_regex"]) | ((f.protocol == "embedding") & (f.variant == "full"))]
                   [paper.columns]])
f = f.merge(paper.assign(paper=True), how="left").fillna({"paper": False})
f.to_csv(META / "forecasts.csv", index=False, float_format="%.4f")
print("forecasts", len(f), "paper rows", int(f.paper.sum()))
