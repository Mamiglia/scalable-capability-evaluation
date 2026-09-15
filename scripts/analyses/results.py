"""Results of the forecasting task: the five conditions against easy-item prediction and the oracle.

Cell script: run top to bottom as `PYTHONPATH=src uv run python scripts/analyses/results.py`,
or cell by cell (`# %%`) in PyCharm / VS Code after the cells above it. State lives in the
module-level names each cell defines. Never launches judging: judge runs are read from logs/
and what a campaign still lacks is reported in the `missing` column.

Transcript logs and embeddings are read from the T7 volume unless SMALL_JUDGE_DATA and
SMALL_JUDGE_EMBEDDINGS say otherwise; the frozen tables always come from data/meta.

Design: docs/superpowers/specs/2026-09-15-results-analysis-design.md
"""

# %% settings
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # every path in src/ is repo-relative
os.environ.setdefault("SMALL_JUDGE_DATA", "/Volumes/T7/small-judge-remote/data")
os.environ.setdefault("SMALL_JUDGE_EMBEDDINGS", "/Volumes/T7/small-judge-remote/transcript-embeddings")

from analysis import calibration, loading, matching, plots, stats  # noqa: E402  (after the env is set)

DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = None  # e.g. ["2026-I"]; None = every frozen semester task
OUT = Path("analysis/results")
# Every variant is computed and scored; these pick the ones the figures and tables show.
# Defaults chosen by the reference leave-one-out error (mean over judge runs, four domains):
# pairwise calibrated 0.34 vs linear 0.45; direct raw 0.43 vs item_z 0.59. Never by target MSE.
PAIRWISE_CONVERSION = "calibrated"  # or "linear"
DIRECT_NORMALISATION = "raw"  # or "item_z"
EMBEDDING_VARIANT = "full"  # or "answer"
VARIANTS = {"pairwise": PAIRWISE_CONVERSION, "direct": DIRECT_NORMALISATION, "embedding": EMBEDDING_VARIANT}
(OUT / "tables").mkdir(parents=True, exist_ok=True)

# %% loading: semester tasks and the judge runs on disk
# domains: name -> DomainConfig; tasks: (domain name, semester) -> Task; runs: same key -> [JudgeRun]
domains = {name: loading.domain(name) for name in DOMAINS}
tasks = {(name, t.semester): t for name, dom in domains.items() for t in loading.tasks(dom, SEMESTERS)}
runs = {key: loading.judge_runs(domains[key[0]], task) for key, task in tasks.items()}
print({key: [(r.protocol, r.judge, r.condition) for r in rs] for key, rs in runs.items()})

# %% matching: restrict judge rows to the frozen train items, count the unjudged
runs = {key: [matching.match(r, tasks[key]) for r in rs] for key, rs in runs.items()}
print(pd.DataFrame([(k[0], k[1], r.protocol, r.judge, r.condition, len(r.table), r.missing) for k, rs in runs.items() for r in rs],
                   columns=["domain", "semester", "protocol", "judge", "condition", "rows", "missing"]).to_string(index=False))

# %% calibration: every signal -> theta-hat; one row per target x protocol x variant x judge x condition
protocols = {name: calibration.transcript_protocols(dom) for name, dom in domains.items()}
forecasts = pd.concat([calibration.forecasts(domains[name], task, protocols[name], runs[(name, sem)])
                       for (name, sem), task in tasks.items()], ignore_index=True)
forecasts.to_csv(OUT / "forecasts.csv", index=False, float_format="%.4f")

# %% stats: score tables, judge summaries, skill, and the paired per-task metrics
scores = stats.scores(forecasts)
summary = stats.summary(scores)
metrics = stats.task_metrics(forecasts)
scores.to_csv(OUT / "scores.csv", index=False, float_format="%.4f")
summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
metrics.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
core = summary[summary.judge.isin(["-", stats.BEST, stats.MEAN])]
print(core[["domain", "protocol", "variant", "judge", "n", "missing", "mse", "baseline", "oracle", "skill"]].round(3).to_string(index=False))

# %% tables: semester x condition, every metric, per domain and pooled
for metric in stats.METRICS:
    for domain in [None, *DOMAINS]:
        table = plots.semester_table(metrics, metric, VARIANTS, domain)
        table.to_csv(OUT / "tables" / f"{metric}_{domain or 'pooled'}.csv", float_format="%.3f")
print(plots.semester_table(metrics, "mse", VARIANTS).round(3).to_string())

# %% figures: forecasts of the best condition (all semesters, and the latest), and the bars
pooled = plots.chosen(metrics, VARIANTS)
pooled = pooled[(pooled.domain == "all") & (pooled.semester == "all") & pooled.protocol.isin(plots.CONDITIONS)]
best = pooled.sort_values("mse").iloc[0]
latest = max(sem for _, sem in tasks)
plots.forecast_figure(forecasts, best.protocol, best.variant, OUT / f"forecasts_{best.protocol}.svg")
plots.forecast_figure(forecasts, best.protocol, best.variant, OUT / f"forecasts_{best.protocol}_{latest}.svg", semesters=[latest])
for metric in stats.METRICS:
    plots.bar_figure(metrics, metric, VARIANTS, OUT / f"bars_{metric}.svg")
print("best condition:", best.protocol, best.variant, "pooled mse", round(best.mse, 3))
print("wrote", sorted(p.name for p in OUT.iterdir()))
