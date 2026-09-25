"""Effect of transcript truncation: MSE of the truncated judge runs relative to their full runs.

Cell script (conventions as scripts/analyses/results.py). Writes analysis/results/truncation_forecasts.csv
(per target), truncation.csv (per task x cut) and figures/truncation/truncation.{svg,png}. Coding and cyber only, the semesters
that have truncated runs.
"""

# %% settings
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("SMALL_JUDGE_DATA", "/Volumes/T7/small-judge-remote/data")
os.environ.setdefault("SMALL_JUDGE_EMBEDDINGS", "/Volumes/T7/small-judge-remote/transcript-embeddings")

from analysis import loading, stats  # noqa: E402
from analysis.challenges import truncation  # noqa: E402
from plots import truncation as truncation_plots  # noqa: E402

DOMAINS = ["coding", "cyber"]
CUT = "ends"  # the cut the figure shows against the full transcripts
VARIANTS = {"pairwise": "linear", "direct": "raw"}  # as scripts/analyses/results.py
OUT = Path("analysis/results")
(OUT / "figures/truncation").mkdir(parents=True, exist_ok=True)

# %% load, match and score every truncated run against its full run
frames = []
for name in DOMAINS:
    dom = loading.domain(name)
    for task in loading.tasks(dom):
        truncated = truncation.truncated_runs(dom, task)
        if not truncated:
            continue
        frames.append(truncation.forecasts(dom, task, loading.judge_runs(dom, task), truncated, VARIANTS))
forecasts = pd.concat(frames, ignore_index=True)
forecasts.to_csv(OUT / "truncation_forecasts.csv", index=False)
table = truncation.summary(forecasts)
table.to_csv(OUT / "truncation.csv", index=False)
print(table.round(2).to_string())

# %% figure: the prediction from saturated items (results forecasts) is the reference level
results = pd.read_csv(OUT / "forecasts.csv")
tasks = set(zip(forecasts.domain, forecasts.semester))
baseline = results[results.protocol == stats.BASELINE].merge(pd.DataFrame(sorted(tasks), columns=["domain", "semester"]))
for ext in ("svg", "png"):
    truncation_plots.figure(forecasts, baseline, OUT / "figures/truncation" / f"truncation.{ext}", cut=CUT)
