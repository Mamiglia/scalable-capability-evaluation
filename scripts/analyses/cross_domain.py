"""Transcripts from one domain, calibrated on another domain's theta*: forecasts, the 4 x 4 score
matrix per protocol, and the maths/science -> coding/cyber comparison.

Cell script: `PYTHONPATH=src uv run python scripts/analyses/cross_domain.py`. Reads the judge logs
and transcripts the results runner reads and launches no judging. The diagonal is
checked against analysis/results/forecasts.csv, which it must reproduce.

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

from analysis.challenges import cross_domain as cd  # noqa: E402  (after the env is set)
from plots import cross_domain as cd_plots  # noqa: E402

DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = ("2025-II", "2026-I")  # the semesters every domain has, so every cell holds 16 targets
VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}  # as figure one
RECOMPUTE = False  # True rebuilds the 32 domain pairs (~25 min); False reuses the saved forecasts
OUT = Path("analysis/results/cross_domain")
FIG = Path("analysis/results/figures/cross_domain")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# %% forecasts: every ordered pair of domains
if RECOMPUTE or not (OUT / "cross_domain_forecasts.csv").exists():
    forecasts = cd.forecasts(DOMAINS, SEMESTERS)
    forecasts.to_csv(OUT / "cross_domain_forecasts.csv", index=False, float_format="%.4f")
else:
    forecasts = pd.read_csv(OUT / "cross_domain_forecasts.csv")

# %% check: the diagonal must reproduce the within-domain forecasts of the results runner. The
# comparison is on what the figures use, one forecast per target and protocol with the best judge
# on references, so a difference in which judge wins is caught, not only in what a judge forecasts
from analysis.challenges import generalisation as gen  # noqa: E402

diagonal = gen.rows(forecasts[forecasts.domain == forecasts.scored_domain], VARIANTS)
within = gen.rows(pd.read_csv("analysis/results/forecasts.csv"), VARIANTS, SEMESTERS)
keys = ["domain", "semester", "protocol", "model"]
merged = diagonal.merge(within[[*keys, "theta_hat"]], on=keys, how="outer", suffixes=("", "_within"), indicator=True)
assert (merged._merge == "both").all(), f"rows in one file only: {merged[merged._merge != 'both'][keys].head().to_dict('records')}"
gap = (merged.theta_hat - merged.theta_hat_within).abs()
assert gap.max() < 1e-3, f"diagonal differs from forecasts.csv, worst {gap.max():.4f}: " \
    f"{merged.loc[gap.idxmax(), keys].to_dict()} (rerun results.py if the judge logs changed)"
print(f"diagonal matches forecasts.csv on {len(merged)} best-judge rows, worst gap {gap.max():.5f}")

# %% scores: the 4 x 4 matrix per protocol, and the maths/science -> coding/cyber comparison
scores = cd.cell_scores(forecasts, VARIANTS)
scores.to_csv(OUT / "cell_scores.csv", index=False, float_format="%.4f")
print(scores.pivot_table(index=["protocol", "source"], columns="scored", values="score").round(2).to_string())

TRANSFER = [(s, d) for s in ("math", "science") for d in ("coding", "cyber")]
OWN = [("coding", "coding"), ("cyber", "cyber")]
transferred = cd.model_scores(forecasts, VARIANTS, TRANSFER)
own = cd.model_scores(forecasts, VARIANTS, OWN)
transferred.assign(cells="maths/science -> coding/cyber").to_csv(OUT / "transfer_model_scores.csv", index=False,
                                                                float_format="%.4f")
own.assign(cells="coding/cyber own").to_csv(OUT / "own_model_scores.csv", index=False, float_format="%.4f")
print(transferred.groupby("protocol").score.mean().round(2).to_string())
# the transfer stars: log likelihood ratio of the transferred forecasts (scripts/analyses/likelihood_ratio.py)
from analysis.challenges import likelihood_ratio as lr  # noqa: E402

transfer_llr = pd.read_csv(OUT / "transfer_likelihood_ratio.csv")
transfer_tests = lr.pooled_tests(transfer_llr[transfer_llr["shape"] == lr.SHAPE], unit=("semester", "model"))
transfer_tests.to_csv(OUT / "transfer_llr_tests.csv", index=False, float_format="%.4g")
print(transfer_tests.round(4).to_string(index=False))

# %% figures
for ext in ("png", "svg"):
    for protocol in [p for p in cd.PROTOCOLS[1:] if p != "glm_length"]:
        cd_plots.heatmap_figure(scores, protocol).savefig(FIG / f"cross_domain_heatmap_{protocol}.{ext}", bbox_inches="tight")
        cd_plots.heatmap_error_figure(scores, protocol).savefig(FIG / f"cross_domain_heatmap_{protocol}_error.{ext}",
                                                          bbox_inches="tight")
    cd_plots.transfer_figure(transferred, own, transfer_tests).savefig(FIG / f"cross_domain_transfer.{ext}", bbox_inches="tight")
print("wrote", sorted(p.name for p in FIG.iterdir()))
