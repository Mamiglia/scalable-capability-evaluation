"""Transcript thirds for the Deterministic features and Embedding protocols: the appendix counterpart of the pairwise
thirds figure (scripts/analyses/position_contributions.py), one row of domain panels per protocol, each third removed
only (the kept-alone thirds were not run).

Every rendered transcript (reasoning shown) is cut into three equal character spans and one is replaced by "..."
(judge.dataset.cut, truncate thirds-<kept>, the pairwise runs' cut). Qwen3-Embedding-8B vectors of the cut text
(src/challenges/channels/embed.py plan --thirds, variants thirds-23, thirds-13, thirds-12 next to the channel
variants); ridge refitted per condition on the references' cut transcripts, predicting the targets' θ*.
Deterministic features: the 13 features (as other_protocol_channels.py) extracted from the same cut, mapped back
onto the structured messages (src/challenges/channels/features.py --thirds -> positions/features_thirds.csv);
standardised ridge refitted per condition.
Absolute scoring: the main analysis's direct judge of each cell (lowest reference leave-one-out error) rates the cut transcripts of the 40 items
(src/challenges/truncation prepare/run thirds-<kept>, published as logs/direct_truncated-thirds-<kept>/); every condition, the full
transcript included, is calibrated on the ratings judged in all four conditions of its cell (protocols.direct_judge
.scoring.calibrate, as challenges/truncation/compare_truncation.py). 2025-II and
2026-I pooled (16 targets per domain), the 40 train items of the pairwise removal runs
(contributions.removal_items), first attempts, correct answers only.
Statistics and figure: analysis.challenges.contributions (paired change in squared error from the full transcript,
two-sided paired t-test and t-interval).

Run from the repo root (reads data/ and transcript-embeddings/, or SMALL_JUDGE_DATA / SMALL_JUDGE_EMBEDDINGS):
  uv run python scripts/analyses/other_protocol_thirds.py [--redraw]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import loading
from analysis.challenges import contributions
from analysis.challenges import likelihood_ratio as lr
from analysis.challenges.signal_glm import DOMAIN_COLORS
from protocols import embeddings
from protocols.direct_judge.scoring import calibrate
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS
from protocols.transcript_features.reader import rows_table

OUT = Path("analysis/results/figures/positions")
ERRORS = OUT / "other_protocol_thirds_errors.csv"
# the results' forecasts, for the judge each cell uses (lowest reference leave-one-out error); point it at a rerun's
FORECASTS = Path(os.environ.get("SMALL_JUDGE_FORECASTS", "analysis/results/forecasts.csv"))
FEATURES = OUT / "features_thirds.csv"  # src/challenges/channels/features.py --thirds
PROTOCOLS = {"glm_regex": "Deterministic features", "embedding": "Embedding", "direct": "Absolute scoring"}
DOMAINS, SEMESTERS = ("coding", "cyber", "math", "science"), ("2025-II", "2026-I")
# position label -> (kept alone, kept when it is removed); the kept-alone runs do not exist and are skipped
PARTS = {"First third": ("thirds-1", "thirds-23"), "Middle third": ("thirds-2", "thirds-13"),
         "Last third": ("thirds-3", "thirds-12")}
CONDITIONS = ["full", "thirds-23", "thirds-13", "thirds-12", "thirds-1", "thirds-2", "thirds-3"]  # removed, then kept alone

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def compute(protocols, conditions=CONDITIONS) -> pd.DataFrame:
    """Every target's squared error per protocol, domain, semester and condition (plus easy-item prediction, "easy")."""
    features = pd.read_csv(FEATURES, dtype={"item": str}) if "glm_regex" in protocols else None
    refit = [p for p in protocols if p != "direct"]  # the ridge protocols, refitted here; direct reads its ratings
    errors = []
    for domain in DOMAINS:
        for semester in SEMESTERS:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
            keep = contributions.removal_items(task)
            cohorts = [contributions.on_items(c, keep) for c in (task.references, task.test)]
            for protocol in refit:
                for condition in conditions:
                    if protocol == "embedding":
                        regression = embeddings.EmbeddingRegression(condition)
                        x = pd.concat([regression.get_features(c) for c in cohorts])
                        estimator = RidgeCV(alphas=embeddings.ALPHAS)
                    else:
                        rows = features[features.condition == condition]
                        x = pd.concat([rows_table(c, rows, FEATURE_SETS["regex"]) for c in cohorts])
                        estimator = make_pipeline(StandardScaler(), RidgeCV(alphas=FEATURE_ALPHAS))
                    theta = lr.feature_rule(estimator, x)(refs, targets).reindex(truth.index)
                    errors.append(pd.DataFrame({"protocol": protocol, "domain": domain, "semester": semester,
                                                "condition": condition, "model": truth.index,
                                                "err": ((theta - truth) ** 2).to_numpy()}))
                errors.append(pd.DataFrame({"protocol": protocol, "domain": domain, "semester": semester,
                                            "condition": "easy", "model": truth.index,
                                            "err": ((refs.mean() - truth) ** 2).to_numpy()}))
            print(domain, semester, "done", flush=True)
    return pd.concat([*errors, *([direct_errors()] if "direct" in protocols else [])], ignore_index=True)


def direct_errors() -> pd.DataFrame:
    """Absolute scoring's squared error per cell and condition, on the (model, bench, item) ratings in every condition:
    the main analysis's judge of each cell, its published thirds runs (logs/direct_truncated-thirds-<kept>/) and its
    full-transcript run as the main analysis loads it (loading.direct_runs)."""
    key, errors, loo = ["model", "bench", "item"], [], contributions.direct_judges(FORECASTS)
    for domain in DOMAINS:
        for semester in SEMESTERS:
            judge = loo[domain, semester]
            (task,) = loading.tasks(loading.domain(domain), [semester])
            refs, truth = task.references.theta, task.truth
            runs = {c: contributions.published_ratings(f"truncated-{c}", domain, semester, judge) for c in CONDITIONS[1:]}
            (full,) = [r for r in loading.direct_runs(loading.domain(domain), task) if r.judge == judge and r.condition == "reasoning-shown"]
            runs["full"] = full.table.assign(item=full.table.item.astype(str))
            common = runs["thirds-23"][key]
            for ratings in runs.values():
                common = common.merge(ratings[key].drop_duplicates())
            for condition, ratings in runs.items():
                theta = calibrate(ratings.merge(common), refs).reindex(truth.index)
                errors.append(pd.DataFrame({"protocol": "direct", "domain": domain, "semester": semester, "condition": condition,
                                            "model": truth.index, "err": ((theta - truth) ** 2).to_numpy()}))
            errors.append(pd.DataFrame({"protocol": "direct", "domain": domain, "semester": semester, "condition": "easy",
                                        "model": truth.index, "err": ((refs.mean() - truth) ** 2).to_numpy()}))
            print(domain, semester, judge, "direct:", len(common), "ratings in every condition", flush=True)
    return pd.concat(errors, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redraw", action="store_true", help="draw from the saved errors instead of recomputing")
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS), default=list(PROTOCOLS),
                        help="recompute only these, keeping the others' saved errors")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS,
                        help="for the refitted protocols, recompute only these conditions (and easy), keeping the rest")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(ERRORS) if ERRORS.exists() else pd.DataFrame(columns=["protocol"])
    if "protocol" not in saved:  # errors saved before the other protocols were added: Embedding only
        saved = saved.assign(protocol="embedding")
    if args.redraw:
        errors = saved
    else:
        new = compute(args.protocols, args.conditions)
        replaced = saved.set_index(["protocol", "condition"]).index.isin(new.set_index(["protocol", "condition"]).index)
        errors = pd.concat([saved[~replaced], new], ignore_index=True)
        errors.to_csv(ERRORS, index=False)
    contributions.save_grid(
        errors, PROTOCOLS, PARTS, DOMAIN_COLORS, OUT, "other_protocol_thirds", final="transcript_thirds_other_protocols",
        measures=("only", "without"), shared_x=True,  # the protocols' changes on one scale
        only_against="easy",  # sufficiency: a third kept alone against the baseline; a third removed against the full transcript
        caption="Transcript thirds for the Deterministic features, Embedding and Absolute scoring protocols. Every rendered "
                "transcript (reasoning shown) is cut into three equal character spans and one third is removed, or kept alone. "
                "Deterministic features: the 13 features computed on the cut transcript's text and tool calls, standardised "
                "ridge refitted on the references' cut transcripts. Embedding: vectors of the cut transcript, ridge refitted "
                "on the references' cut transcripts. Absolute scoring: the judge of each cell with the lowest reference leave-one-out error rates the cut "
                "transcripts, calibrated on the references' ratings of the same cut, every condition on the ratings present "
                "in every condition. 2025-II and 2026-I pooled (16 targets per domain), the same 40 train items (10 per difficulty "
                "quartile) as the pairwise removal runs.")
