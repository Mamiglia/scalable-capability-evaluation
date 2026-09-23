"""Transcript channels for the Deterministic features and Embedding protocols: the appendix counterpart of the
pairwise channel figures (scripts/analyses/channel_contributions.py), one row of domain panels per protocol.

Both protocols are refitted per condition on the references' channel-filtered transcripts and predict the targets'
θ*; 2025-II and 2026-I pooled (16 targets per domain), the full train set. Conditions, as for the pairwise judge:
each channel (reasoning, messages, tool calls with their outputs) kept alone and removed; Maths and Science have
no tool calls, so only reasoning is ablated.
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

Run from the repo root (needs the T7 data and embeddings):
  SMALL_JUDGE_DATA=/Volumes/T7/small-judge-remote/data \\
  SMALL_JUDGE_EMBEDDINGS=/Volumes/T7/small-judge-remote/transcript-embeddings \\
  uv run python scripts/analyses/other_protocol_channels.py [--redraw]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import embeddings, loading
from analysis.challenges import contributions
from analysis.challenges import likelihood_ratio as lr
from capability.task import Cohort
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS
from protocols.transcript_features.reader import feature_table

OUT = Path("analysis/results/figures/channels")
REASONING_ONLY = OUT / "features_only_reasoning.csv"  # src/challenges/channels/features.py
DOMAINS, SEMESTERS = ("coding", "cyber", "math", "science"), ("2025-II", "2026-I")
TOOL_DOMAINS = ("coding", "cyber")
PARTS = {"Reasoning": ("only-reasoning", "drop-reasoning"), "Messages": ("only-text", "drop-text"),
         "Tools": ("only-tools", "drop-tools")}
COLOURS = {"Reasoning": "#4a3aa7", "Messages": "#eda100", "Tools": "#1baf7a"}  # the pairwise channel figures' hues
PROTOCOLS = {"glm_regex": "Deterministic features", "embedding": "Embedding"}
REGEX = FEATURE_SETS["regex"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def conditions(domain: str) -> list[str]:
    parts = list(PARTS.values()) if domain in TOOL_DOMAINS else [PARTS["Reasoning"]]
    return ["full", *(run for pair in parts for run in pair)]


def embedding_rows(domain: str, condition: str, cohort: Cohort) -> pd.DataFrame:
    variant = {"full": "full", "drop-reasoning": "channels-drop-reasoning" if domain in TOOL_DOMAINS else "answer"}
    return embeddings.EmbeddingRegression(variant.get(condition, f"channels-{condition}")).get_features(cohort)


def reasoning_only_table(cohort: Cohort, rows: pd.DataFrame) -> pd.DataFrame:
    """feature_table for the reasoning-alone view: one REGEX row per model, the mean over its correct train transcripts."""
    out = []
    for model in cohort.models.index:
        frames = []
        for bench, ids in cohort.items.groupby("bench").item:
            if isinstance(path := cohort.models.loc[model, f"path_{bench}"], str):
                pick = rows[(rows.path == path) & rows.correct & rows.item.isin(set(map(str, ids)))]
                frames.append(pick[REGEX])
        table = pd.concat(frames) if frames else pd.DataFrame(columns=REGEX)
        out.append(table.mean() if len(table) else pd.Series(np.nan, index=REGEX))
    return pd.DataFrame(out, index=cohort.models.index, columns=REGEX, dtype=float)


def feature_rows(condition: str, cohort: Cohort, reasoning_only: pd.DataFrame) -> pd.DataFrame:
    visible = lambda: feature_table(cohort, reasoning="visible")[REGEX]  # noqa: E731
    hidden = lambda: feature_table(cohort, reasoning="hidden")[REGEX]  # noqa: E731
    alone = lambda: reasoning_only_table(cohort, reasoning_only)  # noqa: E731
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


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every target's squared error and log likelihood ratio per protocol, domain, semester and condition."""
    reasoning_only = pd.read_csv(REASONING_ONLY, dtype={"item": str})
    errors, llrs = [], []
    for domain in DOMAINS:
        for semester in SEMESTERS:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
            for protocol in PROTOCOLS:
                for condition in conditions(domain):
                    if protocol == "embedding":
                        x = pd.concat([embedding_rows(domain, condition, c) for c in (task.references, task.test)])
                        estimator = RidgeCV(alphas=embeddings.ALPHAS)
                    else:
                        x = pd.concat([feature_rows(condition, c, reasoning_only) for c in (task.references, task.test)])
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
    else:
        errors, llrs = compute()
        errors.to_csv(OUT / "other_protocol_errors.csv", index=False)
        llrs.to_csv(OUT / "other_protocol_llr.csv", index=False)
    contributions.save_grid(
        errors, PROTOCOLS, PARTS, COLOURS, OUT, "other_protocol_channels", final="transcript_channels_other_protocols",
        caption="Transcript channels (reasoning, messages, tool calls with their outputs) for the Deterministic features and "
                "Embedding protocols, refitted on the references' channel-filtered transcripts; 2025-II and 2026-I pooled "
                "(16 targets per domain), full train set. Maths and Science have no tool calls, so only reasoning is "
                "ablated. Embedding: vectors of the transcript rendered with only or without the channel. Deterministic "
                "features: the 13 features computed on the channel's text (the tool-call count is 0 without tools and the "
                "only feature with tools alone).")
