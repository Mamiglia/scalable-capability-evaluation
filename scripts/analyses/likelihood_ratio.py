"""Per-target log likelihood ratios of the paper protocols against the references' distribution.

Cell script: `PYTHONPATH=src uv run python scripts/analyses/likelihood_ratio.py`, or cell by cell.
Writes analysis/results/likelihood_ratio.csv: one row per (domain, semester, target, protocol,
judge, dropped reference, shape), `dropped` "-" for the full reference set, `judge` "-" for transcript
protocols. Every judge's core-condition run is scored, so a figure can take the task's best judge
on references (likelihood_ratio.chosen_judges) or another rule. Also writes
analysis/results/cross_domain/transfer_likelihood_ratio.csv: the cross-domain transfer cells, maths
and science transcripts scored on coding and cyber theta* (cross_domain.swap), `domain`
"<source>-><scored>", the source domain's best judge as in the cross-domain figures. Every protocol fits on the task's
references only, as the null and the results do. Reads transcripts and embeddings
from data/ and judge runs from logs/; launches nothing. Method: src/analysis/challenges/likelihood_ratio.py.
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

from analysis import loading, matching, stats  # noqa: E402
from analysis.challenges import cross_domain as cd  # noqa: E402
from analysis.challenges import likelihood_ratio as lr  # noqa: E402
from protocols.embeddings import EmbeddingRegression  # noqa: E402
from protocols import PROTOCOLS  # noqa: E402

DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = None  # every task: the time and 2025-I reasoning figures reach back; the rest filter to 2025-II, 2026-I
VARIANTS = {"pairwise": "linear", "direct": "raw"}  # as figure one; embedding is "full"
OUT = Path("analysis/results")
TRANSFER = [(s, d) for s in ("math", "science") for d in ("coding", "cyber")]  # cross-domain transfer cells
TRANSFER_SEMESTERS = ("2025-II", "2026-I")

# %% every judge's core-condition run, and the best on references per task (the results' choice)
forecasts = pd.read_csv(OUT / "forecasts.csv")
core = stats.core_condition(forecasts)
core = core[np.array([VARIANTS.get(p) == v for p, v in zip(core.protocol, core.variant)])]
runs_used = core[["domain", "semester", "protocol", "judge", "condition"]].drop_duplicates()
best = lr.chosen_judges(forecasts, VARIANTS)
best_judge = best.drop_duplicates(["domain", "semester", "protocol"]).set_index(["domain", "semester", "protocol"]).judge

# %% per-target log LR, with every reference dropped in turn
frames, check, transfer = [], [], []
for name in DOMAINS:
    dom = loading.domain(name)
    features = {"glm_regex": PROTOCOLS["glm_regex"], "embedding": EmbeddingRegression("full")}
    for task in loading.tasks(dom, SEMESTERS):
        refs, truth = task.references.theta, task.truth
        rules = {p: lr.feature_rule(f.estimator, pd.concat([f.get_features(task.references), f.get_features(task.test)]))
                 for p, f in features.items()}
        runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
        rules = {(p, "-"): rule for p, rule in rules.items()}
        used = runs_used[(runs_used.domain == name) & (runs_used.semester == task.semester)]
        for protocol, judge, condition in used[["protocol", "judge", "condition"]].itertuples(index=False):
            run = next(r for r in runs if (r.protocol, r.judge, r.condition) == (protocol, judge, condition))
            rules[(protocol, judge)] = lr.judge_rule(run.table, protocol, VARIANTS[protocol])
        for (protocol, judge), rule in rules.items():
            frames.append(lr.task_llr(rule, refs, truth).assign(domain=name, semester=task.semester, protocol=protocol, judge=judge))
            check.append(pd.DataFrame(dict(domain=name, semester=task.semester, protocol=protocol, judge=judge, model=truth.index,
                                           theta_hat=rule(refs, list(truth.index)).reindex(truth.index).to_numpy())))
        for source, scored in TRANSFER:  # the same rules, fitted and scored on another domain's theta*
            if source != name or task.semester not in TRANSFER_SEMESTERS:
                continue
            swapped = cd.swap(task, cd.scored_theta(scored, task.semester))
            for (protocol, judge), rule in rules.items():
                if judge == "-" or judge == best_judge.get((name, task.semester, protocol)):
                    transfer.append(lr.task_llr(rule, swapped.references.theta, swapped.truth).assign(
                        domain=f"{source}->{scored}", semester=task.semester, protocol=protocol, judge=judge))
        print(name, task.semester, "done", flush=True)
llr = pd.concat(frames, ignore_index=True)[["domain", "semester", "model", "protocol", "judge", "shape", "dropped", "llr"]]
llr.to_csv(OUT / "likelihood_ratio.csv", index=False, float_format="%.5f")
if transfer:
    (OUT / "cross_domain").mkdir(parents=True, exist_ok=True)
    pd.concat(transfer, ignore_index=True)[llr.columns].to_csv(OUT / "cross_domain" / "transfer_likelihood_ratio.csv",
                                                               index=False, float_format="%.5f")

# %% check: the full-reference forecasts are the results' own (transcript protocols and each task's best judge)
ours = lr.select(pd.concat(check, ignore_index=True).assign(dropped="-"), best).drop(columns=["judge", "dropped"])
theirs = stats.one_per_target(forecasts)
theirs = theirs[np.array([{"embedding": "full", **VARIANTS}.get(p, "-") == v for p, v in zip(theirs.protocol, theirs.variant)])]
merged = ours.merge(theirs[["domain", "semester", "protocol", "model", "theta_hat"]], on=["domain", "semester", "protocol", "model"],
                    suffixes=("", "_results"))
off = (merged.theta_hat - merged.theta_hat_results).abs()
print("forecasts matched:", len(merged), "of", len(ours), "| largest difference", round(float(off.max()), 4))
assert len(merged) == len(ours) and off.max() < 1e-3, merged[off >= 1e-3]

# %% the mean log LR per task, best judges
full = lr.select(llr, best)
full = full[full.dropped == "-"]
print(full.groupby(["shape", "domain", "semester", "protocol"]).llr.mean().unstack().round(2).to_string())
