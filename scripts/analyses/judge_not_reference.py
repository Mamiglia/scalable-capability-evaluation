"""Control for the judge protocols: figure one (signal_by_protocol) with each task's judge left out of
its own references.

Every task's best judge on references is itself one of the task's references, so the Pairwise judge
compares targets against its own transcripts and Absolute scoring calibrates on its own ratings. Here
the judge's forecasts are refitted on the references without the judge (its verdicts against itself
and its own ratings no longer calibrate anything). The judge choice is unchanged, as are the
transcript protocols and the saturated-item prediction. Stars: the log likelihood ratio rows in
analysis/results/likelihood_ratio.csv whose dropped reference is the judge.

Writes analysis/results/figures/paper/signal_by_protocol_judge_not_reference.{svg,png}; reads logs/, launches nothing.
Run: PYTHONPATH=src uv run python scripts/analyses/judge_not_reference.py
"""

# %% settings
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis import loading, matching, stats  # noqa: E402
from analysis.challenges import generalisation as gen  # noqa: E402
from analysis.challenges import likelihood_ratio as lr  # noqa: E402
from analysis.challenges import signal_glm as sg  # noqa: E402

FINAL = Path("analysis/results/figures/paper")  # the paper figures and their .txt captions
FINAL.mkdir(parents=True, exist_ok=True)
VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}  # as figure one
SEMESTERS = ["2025-II", "2026-I"]

forecasts = pd.read_csv("analysis/results/forecasts.csv")
best = lr.chosen_judges(forecasts, VARIANTS).drop_duplicates(["domain", "semester", "protocol"])
best = best[best.semester.isin(SEMESTERS)]
core = stats.core_condition(forecasts)
runs_used = best.drop(columns="model").merge(
    core[["domain", "semester", "protocol", "variant", "judge", "condition"]].drop_duplicates()
    .assign(keep=lambda d: [VARIANTS.get(p) == v for p, v in zip(d.protocol, d.variant)]).query("keep"))

# %% refit each best judge's forecasts on the references without the judge
for name in sg.DOMAINS:
    dom = loading.domain(name)
    for task in loading.tasks(dom, SEMESTERS):
        runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
        used = runs_used[(runs_used.domain == name) & (runs_used.semester == task.semester)]
        for row in used.itertuples(index=False):
            run = next(r for r in runs if (r.protocol, r.judge, r.condition) == (row.protocol, row.judge, row.condition))
            assert row.judge in task.references.theta.index, (name, task.semester, row.judge)
            refs = task.references.theta.drop(row.judge)
            theta_hat = lr.judge_rule(run.table, row.protocol, row.variant)(refs, list(task.truth.index))
            at = ((forecasts.domain == name) & (forecasts.semester == task.semester) & (forecasts.protocol == row.protocol)
                  & (forecasts.variant == row.variant) & (forecasts.judge == row.judge) & (forecasts.condition == row.condition)
                  & forecasts.model.isin(theta_hat.index))
            assert at.sum() == len(theta_hat), (name, task.semester, row.protocol, at.sum())
            forecasts.loc[at, "theta_hat"] = forecasts.loc[at, "model"].map(theta_hat).to_numpy()
        print(name, task.semester, "done", flush=True)

# %% stars: the log LR with the judge dropped from the references (transcript protocols: full references)
llr = pd.read_csv("analysis/results/likelihood_ratio.csv")
llr = lr.select(llr[(llr["shape"] == lr.SHAPE) & llr.semester.isin(SEMESTERS)], lr.chosen_judges(forecasts, VARIANTS))
llr = llr[(llr.judge == "-") & (llr.dropped == "-") | (llr.judge != "-") & (llr.dropped == llr.judge)].assign(dropped="-")
tests = lr.pooled_tests(llr, unit=("semester", "model"))
tests["p"] = 2 * tests.p.clip(upper=1 - tests.p)  # two-sided, as figure one
print(tests.round(4).to_string(index=False))

pooled = sg.per_model(sg.rows(forecasts, VARIANTS))
print(pooled.agg(["mean", "count"]).round(3).to_string())

fig = gen.model_error_figure(forecasts, VARIANTS, tests)
for ext in ("svg", "png"):
    fig.savefig(FINAL / f"signal_by_protocol_judge_not_reference.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FINAL / "signal_by_protocol_judge_not_reference.svg")
