"""Transcript thirds for the Embedding protocol: the appendix counterpart of the pairwise thirds figure
(scripts/analyses/position_contributions.py), each third removed only (the kept-alone thirds were not embedded).

Every rendered transcript (reasoning shown) is cut into three equal character spans and one is replaced by "..."
(judge.dataset.cut, truncate thirds-<kept>, the pairwise runs' cut). Qwen3-Embedding-8B vectors of the cut text
(src/challenges/channels/embed.py plan --thirds, variants thirds-23, thirds-13, thirds-12 next to the channel
variants); ridge refitted per condition on the references' cut transcripts, predicting the targets' θ*. 2025-II and
2026-I pooled (16 targets per domain), the 40 train items of the pairwise removal runs
(contributions.removal_items), first attempts, correct answers only.
Statistics and figure: analysis.challenges.contributions (paired change in squared error from the full transcript,
two-sided paired t-test and t-interval).

Run from the repo root (needs the T7 data and embeddings):
  SMALL_JUDGE_DATA=/Volumes/T7/small-judge-remote/data \\
  SMALL_JUDGE_EMBEDDINGS=/Volumes/T7/small-judge-remote/transcript-embeddings \\
  uv run python scripts/analyses/other_protocol_thirds.py [--redraw]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import RidgeCV

from analysis import loading
from protocols import embeddings
from analysis.challenges import contributions
from analysis.challenges import likelihood_ratio as lr
from analysis.challenges.signal_glm import DOMAIN_COLORS

OUT = Path("analysis/results/figures/positions")
ERRORS = OUT / "other_protocol_thirds_errors.csv"
DOMAINS, SEMESTERS = ("coding", "cyber", "math", "science"), ("2025-II", "2026-I")
# position label -> (kept alone, kept when it is removed); the kept-alone runs do not exist and are skipped
PARTS = {"First third": ("thirds-1", "thirds-23"), "Middle third": ("thirds-2", "thirds-13"),
         "Last third": ("thirds-3", "thirds-12")}
CONDITIONS = ["full", "thirds-23", "thirds-13", "thirds-12"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def compute() -> pd.DataFrame:
    """Every target's squared error per domain, semester and condition (plus easy-item prediction, "easy")."""
    errors = []
    for domain in DOMAINS:
        for semester in SEMESTERS:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
            keep = contributions.removal_items(task)
            cohorts = [contributions.on_items(c, keep) for c in (task.references, task.test)]
            for condition in CONDITIONS:
                regression = embeddings.EmbeddingRegression(condition)
                x = pd.concat([regression.get_features(c) for c in cohorts])
                theta = lr.feature_rule(RidgeCV(alphas=embeddings.ALPHAS), x)(refs, targets).reindex(truth.index)
                errors.append(pd.DataFrame({"domain": domain, "semester": semester, "condition": condition,
                                            "model": truth.index, "err": ((theta - truth) ** 2).to_numpy()}))
            errors.append(pd.DataFrame({"domain": domain, "semester": semester, "condition": "easy",
                                        "model": truth.index, "err": ((refs.mean() - truth) ** 2).to_numpy()}))
            print(domain, semester, "done", flush=True)
    return pd.concat(errors, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redraw", action="store_true", help="draw from the saved errors instead of recomputing")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.redraw:
        errors = pd.read_csv(ERRORS)
    else:
        errors = compute()
        errors.to_csv(ERRORS, index=False)
    contributions.save_figures(
        errors, PARTS, DOMAIN_COLORS, OUT, "other_protocol_thirds", final="transcript_thirds_other_protocols",
        measures=("without",),
        caption="Embedding protocol on transcript thirds. Every rendered transcript (reasoning shown) is cut into three "
                "equal character spans and one third is replaced by \"...\"; the ridge is refitted on the references' "
                "embeddings of the cut transcripts and predicts the targets. 2025-II and 2026-I pooled (16 targets per "
                "domain), the same 40 train items (10 per difficulty quartile) as the pairwise removal runs.")
