"""Transcript channels for the Deterministic features and Embedding protocols: the appendix counterpart of the
pairwise channel figures (scripts/analyses/channel_contributions.py), one row of domain panels per protocol.

Both protocols are refitted per condition on the references' channel-filtered transcripts and predict the targets'
θ*; 2025-II and 2026-I pooled (16 targets per domain), the 40 train items of the pairwise removal runs. Conditions, as for the pairwise judge:
each channel (reasoning, messages, tool calls with their outputs) kept alone and removed; Maths and Science have
no tool calls, so reasoning is kept alone and removed and messages removed (= the reasoning alone).
  Embedding: Qwen3-Embedding-8B vectors of the transcript rendered with only / without the channel
    (src/challenges/channels/embed.py, LASR-G5/transcript-embeddings variants channels-<spec>; "answer" is the
    transcript without reasoning for Maths and Science), ridge as in the main analysis.
  Deterministic features: the same 13 features (3 lengths, 9 lexical rates, tool-call count) read from the
    glm_features/ views: full = messages + reasoning ("visible"); without reasoning = "hidden"; reasoning alone =
    src/challenges/channels/features.py (reasoning blocks only); without messages = reasoning alone plus the
    tool-call count; without tools / messages alone = the view with the tool-call count set to 0; tools alone =
    the tool-call count as the only feature.
Statistics and figure: analysis.challenges.contributions (paired change in squared error from the full transcript;
one-sided log likelihood ratio tests).

Run from the repo root (reads data/ and transcript-embeddings/, or SMALL_JUDGE_DATA / SMALL_JUDGE_EMBEDDINGS):
  uv run python scripts/analyses/other_protocol_channels.py [--redraw]
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
from protocols.direct_judge.scoring import calibrate
from capability.task import Cohort
from protocols import embeddings
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS
from protocols.transcript_features.reader import feature_table, rows_table

OUT = Path("analysis/results/figures/channels")
OUT.mkdir(parents=True, exist_ok=True)
# the results' forecasts, for the judge each cell uses (lowest reference leave-one-out error); point it at a rerun's
FORECASTS = Path(os.environ.get("SMALL_JUDGE_FORECASTS", "analysis/results/forecasts.csv"))
REASONING_ONLY = OUT / "features_only_reasoning.csv"  # src/challenges/channels/features.py
DOMAINS, SEMESTERS = ("coding", "cyber", "math", "science"), ("2025-II", "2026-I")
TOOL_DOMAINS = ("coding", "cyber")
PARTS = {"Reasoning": ("only-reasoning", "drop-reasoning"), "Messages": ("only-text", "drop-text"),
         "Tools": ("only-tools", "drop-tools")}
COLOURS = {"Reasoning": "#4a3aa7", "Messages": "#eda100", "Tools": "#1baf7a"}  # the pairwise channel figures' hues
PROTOCOLS = {"glm_regex": "Deterministic features", "embedding": "Embedding", "direct": "Absolute scoring"}
REGEX = FEATURE_SETS["regex"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def conditions(domain: str) -> list[str]:
    if domain not in TOOL_DOMAINS:  # no tool calls: without messages is the reasoning alone
        return ["full", *PARTS["Reasoning"], "drop-text"]
    return ["full", *(run for pair in PARTS.values() for run in pair)]


def embedding_rows(domain: str, condition: str, cohort: Cohort) -> pd.DataFrame:
    variant = {"full": "full", "drop-reasoning": "channels-drop-reasoning" if domain in TOOL_DOMAINS else "answer",
               "drop-text": "channels-drop-text" if domain in TOOL_DOMAINS else "channels-only-reasoning"}
    return embeddings.EmbeddingRegression(variant.get(condition, f"channels-{condition}")).get_features(cohort)


def feature_rows(condition: str, cohort: Cohort, reasoning_only: pd.DataFrame) -> pd.DataFrame:
    visible = lambda: feature_table(cohort, reasoning="visible")[REGEX]  # noqa: E731
    hidden = lambda: feature_table(cohort, reasoning="hidden")[REGEX]  # noqa: E731
    alone = lambda: rows_table(cohort, reasoning_only, REGEX)  # noqa: E731
    if condition == "full":
        return visible()
    if condition == "drop-reasoning":
        return hidden()
    if condition == "only-reasoning":
        return alone()
    if condition == "drop-text":
        return alone().assign(log_tool_calls=visible().log_tool_calls)
    if condition == "drop-tools":
        return visible().assign(log_tool_calls=0.0)
    if condition == "only-text":
        return hidden().assign(log_tool_calls=0.0)
    if condition == "only-tools":
        return visible()[["log_tool_calls"]]
    raise ValueError(condition)


def direct_errors() -> pd.DataFrame:
    """Absolute scoring: each task's leave-one-out judge re-rated the channel-removed transcripts
    (published as logs/direct_channels-<spec>/, collected on the full train set, scored on the 40 removal items like
    the other protocols); every condition and the full transcript recalibrated on the same references. Removal runs only."""
    rows, loo = [], contributions.direct_judges(FORECASTS)
    for domain in DOMAINS:
        for semester in SEMESTERS:
            judge = loo[domain, semester]  # a cell staged for more than one judge keeps the main analysis's
            # the full transcript as the main analysis loads it (loading.direct_runs); every condition on its items
            (task,) = loading.tasks(loading.domain(domain), [semester])
            refs, truth = task.references.theta, task.truth
            (run,) = [r for r in loading.direct_runs(loading.domain(domain), task) if r.judge == judge and r.condition == "reasoning-shown"]
            full = run.table.assign(item=run.table.item.astype(str))
            full = full[full.item.isin(contributions.removal_items(task))]  # the 40 items every other protocol here uses
            for condition in [c for c in conditions(domain) if not c.startswith("only-")]:
                if condition == "full":
                    ratings = full
                else:
                    ratings = contributions.published_ratings(f"channels-{condition}", domain, semester, judge)
                    ratings = ratings.assign(item=ratings.item.astype(str)).merge(full[["model", "bench", "item"]].drop_duplicates())
                theta = calibrate(ratings, refs).reindex(truth.index)
                rows.append(pd.DataFrame({"protocol": "direct", "domain": domain, "semester": semester, "condition": condition,
                                          "model": truth.index, "err": ((theta - truth) ** 2).to_numpy()}))
            rows.append(pd.DataFrame({"protocol": "direct", "domain": domain, "semester": semester, "condition": "easy",
                                      "model": truth.index, "err": ((refs.mean() - truth) ** 2).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every target's squared error and log likelihood ratio per protocol, domain, semester and condition."""
    reasoning_only = pd.read_csv(REASONING_ONLY, dtype={"item": str})
    errors, llrs = [], []
    for domain in DOMAINS:
        for semester in SEMESTERS:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
            keep = contributions.removal_items(task)
            cohorts = [contributions.on_items(c, keep) for c in (task.references, task.test)]
            for protocol in (p for p in PROTOCOLS if p != "direct"):
                for condition in conditions(domain):
                    if protocol == "embedding":
                        x = pd.concat([embedding_rows(domain, condition, c) for c in cohorts])
                        estimator = RidgeCV(alphas=embeddings.ALPHAS)
                    else:
                        x = pd.concat([feature_rows(condition, c, reasoning_only) for c in cohorts])
                        estimator = make_pipeline(StandardScaler(), RidgeCV(alphas=FEATURE_ALPHAS))
                    rule = lr.feature_rule(estimator, x)
                    theta = rule(refs, targets).reindex(truth.index)
                    errors.append(pd.DataFrame({"protocol": protocol, "domain": domain, "semester": semester,
                                                "condition": condition, "model": truth.index, "err": ((theta - truth) ** 2).to_numpy()}))
                    llrs.append(lr.task_llr(rule, refs, truth).pipe(lambda d: d[d["shape"] == lr.SHAPE])
                                .assign(protocol=protocol, domain=domain, semester=semester, condition=condition))
                errors.append(pd.DataFrame({"protocol": protocol, "domain": domain, "semester": semester, "condition": "easy",
                                            "model": truth.index, "err": ((refs.mean() - truth) ** 2).to_numpy()}))
            print(domain, semester, "done", flush=True)
    return pd.concat(errors, ignore_index=True), pd.concat(llrs, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redraw", action="store_true", help="draw from the saved tables instead of recomputing")
    args = parser.parse_args()
    if args.redraw:
        errors = pd.read_csv(OUT / "other_protocol_errors.csv")
        errors = pd.concat([errors[errors.protocol != "direct"], direct_errors()], ignore_index=True)
    else:
        errors, llrs = compute()
        errors = pd.concat([errors, direct_errors()], ignore_index=True)
        errors.to_csv(OUT / "other_protocol_errors.csv", index=False)
        llrs.to_csv(OUT / "other_protocol_llr.csv", index=False)
    contributions.save_grid(
        errors, PROTOCOLS, PARTS, COLOURS, OUT, "other_protocol_channels", final="transcript_channels_other_protocols",
        caption="Transcript channels (reasoning, messages, tool calls with their outputs) for the Deterministic features and "
                "Embedding protocols, refitted on the references' channel-filtered transcripts; 2025-II and 2026-I pooled "
                "(16 targets per domain), the same 40 train items (10 per difficulty quartile) as the pairwise removal runs. Maths and Science have no tool calls: without messages is the "
                "reasoning alone. Embedding: vectors of the transcript rendered with only or without the channel. Deterministic "
                "features: the 13 features computed on the channel's text (the tool-call count is 0 without tools and the "
                "only feature with tools alone). Absolute scoring (channels removed only): each task's leave-one-out judge "
                "re-rated the channel-removed transcripts, on the same 40 train items, recalibrated on the same references.",
        alpha=1, measures=("without",), shared_x=True, ticks=[-2, -1, 0, 1, 2])  # solid bars, as the main channel figure (fig 5)
