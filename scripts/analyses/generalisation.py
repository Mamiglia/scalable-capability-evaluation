"""Research question 3 figures: prediction score over time, error against distance from the
references, and the reasoning-model targets of 2025-I.

Cell script: run top to bottom as `PYTHONPATH=src uv run python scripts/analyses/generalisation.py`,
or cell by cell (`# %%`). Reads `analysis/results/forecasts.csv` written by results.py; the frozen
model tables supply the references' theta*. Launches nothing.

Design: docs/superpowers/specs/2026-09-17-generalisation-design.md
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

from analysis.challenges import generalisation as gen  # noqa: E402  (after the path is set)

VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}  # as figure one
OUT = Path("analysis/results/generalisation")
FIG = Path("analysis/results/figures/generalisation")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

forecasts = pd.read_csv("analysis/results/forecasts.csv")
rows = gen.rows(forecasts, VARIANTS)

# %% prediction score over target semesters
scores = gen.task_scores(rows)
scores.to_csv(OUT / "score_over_time.csv", index=False, float_format="%.4f")
# check: the pooled MSE per task matches the results runner's, which computes it by another path
metrics = pd.read_csv("analysis/results/metrics.csv").query("semester != 'all' and domain != 'all'")
metrics = metrics[[VARIANTS.get(p, "-") == v for p, v in zip(metrics.protocol, metrics.variant)]]
merged = scores.merge(metrics[["domain", "semester", "protocol", "mse"]], on=["domain", "semester", "protocol"],
                      suffixes=("", "_results"))
assert len(merged), "no task matched metrics.csv"
# metrics.csv is written to four decimals, so the comparison is to its rounding, not exact. It is a
# cross-check against another script's output, so a disagreement is reported, not fatal: metrics.csv
# is stale whenever results.py has not been re-run since the judge selection or the logs changed.
off = merged[(merged.mse - merged.mse_results).abs() > 5e-4 * merged.mse.abs().clip(lower=1)]
if len(off):
    print("task MSE disagrees with metrics.csv (stale results.py output?):")
    print(off[["domain", "semester", "protocol", "mse", "mse_results"]].round(3).to_string(index=False))
for ext in ("png", "svg"):
    gen.time_figure(scores).savefig(FIG / f"score_over_time.{ext}", bbox_inches="tight")
    gen.time_figure(scores, by="domain").savefig(FIG / f"score_over_time_by_domain.{ext}", bbox_inches="tight")
    gen.time_figure(scores, by="domain", stack=True).savefig(FIG / f"score_over_time_stacked.{ext}", bbox_inches="tight")
    gen.time_figure(scores, by="domain", box=True).savefig(FIG / f"score_over_time_box.{ext}", bbox_inches="tight")
print(scores.pivot_table(index=["domain", "semester"], columns="protocol", values="score").round(0).to_string())

# %% error against distance from the references
distance = gen.distances(rows)
slopes = gen.distance_slopes(distance)
distance.to_csv(OUT / "distance.csv", index=False, float_format="%.4f")
slopes.to_csv(OUT / "distance_slopes.csv", index=False, float_format="%.4f")
for ext in ("png", "svg"):
    for which in gen.DISTANCE_LABELS:
        for metric in ("signed", "score"):
            for points in (True, False):
                name = f"distance_{which}" + ("" if metric == "signed" else "_score") + ("" if points else "_binned")
                gen.distance_figure(distance, which, metric, points).savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
print(slopes.round(2).to_string(index=False))

# %% reasoning-model targets of 2025-I
reasoning = gen.reasoning_scores(rows)
reasoning.to_csv(OUT / "reasoning_2025-I.csv", index=False, float_format="%.4f")
gen.reasoning_model_scores(rows).to_csv(OUT / "reasoning_2025-I_models.csv", index=False, float_format="%.4f")
for ext in ("png", "svg"):
    gen.reasoning_figure(rows).savefig(FIG / f"reasoning_2025-I.{ext}", bbox_inches="tight")
    gen.reasoning_pairwise_figure(rows).savefig(FIG / f"reasoning_2025-I_pairwise_math.{ext}", bbox_inches="tight")
    gen.reasoning_pairwise_figure(rows, metric="error").savefig(FIG / f"reasoning_2025-I_pairwise_math_error.{ext}",
                                                              bbox_inches="tight")
    gen.reasoning_score_figure(rows).savefig(FIG / f"reasoning_2025-I_score.{ext}", bbox_inches="tight")
    gen.reasoning_scatter_figure(rows, "math").savefig(FIG / f"reasoning_2025-I_scatter_math.{ext}", bbox_inches="tight")
print(reasoning.pivot_table(index=["domain", "protocol"], columns="group", values="score").round(0).to_string())
