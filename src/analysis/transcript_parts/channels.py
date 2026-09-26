"""Transcript channels: each target's squared error when the model's reasoning, its messages or its tool calls with
their outputs are removed from every transcript (conditions "drop-reasoning", "drop-text", "drop-tools"; "full" is
the whole transcript), and each channel's share of the characters the models wrote. Pairwise judge: Figure 5;
Deterministic features, Embedding and Absolute scoring: Figure 24.

Maths and Science have no tool calls: there the transcript without reasoning is the reasoning-hidden transcript, and the
transcript without messages is the reasoning alone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import judges, loading
from analysis.transcript_parts.paired_change import on_items, published_ratings, ridge_forecast, squared_errors
from capability.task import Cohort, Task
from configs.judge import JudgeConfig
from evals.transcripts import read_transcripts
from judge.dataset import response_chars
from protocols.direct_judge.scoring import calibrate
from protocols.embeddings import EmbeddingRegression
from protocols.pairwise.judge import JudgeBT, from_logs
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.reader import feature_table, rows_table

REGEX = FEATURE_SETS["regex"]


def channel_run_dir(judge_config: Path, domain: str, semester: str) -> Path:
    """The log dir a channel judge config (configs/judges/channels/<judge>-<run>.yml) writes the task's verdicts to."""
    return JudgeBT(JudgeConfig.from_cli(judge_config), loading.domain(domain)).log_dir(semester)


def pairwise_errors(domain: str, task: Task, train_runs: dict[str, Path], channel_runs: dict[str, Path],
                    keep: set[str], conversion: str) -> pd.DataFrame:
    """Each target's squared error per pairwise run and of the Baseline ("easy"): domain, semester,
    condition, model, err.

    train_runs: condition -> log dir of a run judged on the full train set, cut here to the `keep` items.
    channel_runs: condition -> log dir of a channel run, judged on the `keep` items only and read whole.
    conversion: verdicts to theta-hat (analysis.judges.pairwise_theta), e.g. "linear".
    """
    refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
    runs = {condition: (v := from_logs(path))[v.item.isin(keep)] for condition, path in train_runs.items()}
    runs |= {condition: from_logs(path) for condition, path in channel_runs.items()}
    cell = dict(domain=domain, semester=task.semester)
    frames = [squared_errors(judges.pairwise_theta(verdicts, refs, targets, conversion, None), truth, **cell,
                             condition=condition) for condition, verdicts in runs.items()]
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)


def embedding_variant(condition: str, tools: bool) -> str:
    """The embedding variant of a condition's transcripts (the vectors of transcript-embeddings/).

    tools: the domain has tool calls; if not, the transcript without reasoning is "answer" and without messages the
        reasoning alone.
    """
    variant = {"full": "full", "drop-reasoning": "channels-drop-reasoning" if tools else "answer",
               "drop-text": "channels-drop-text" if tools else "channels-only-reasoning"}
    return variant.get(condition, f"channels-{condition}")


def feature_rows(condition: str, cohort: Cohort, reasoning_only: pd.DataFrame) -> pd.DataFrame:
    """The 13 deterministic features per model of `cohort` for a condition's transcripts.

    full: messages and reasoning (glm_features "visible"); without reasoning: "hidden"; without messages: the reasoning
    alone (`reasoning_only`, scripts/analysis/features.py) with the full transcript's tool-call count; without tools: the
    full transcript with the tool-call count 0.
    """
    visible = lambda: feature_table(cohort, reasoning="visible")[REGEX]  # noqa: E731
    if condition == "full":
        return visible()
    if condition == "drop-reasoning":
        return feature_table(cohort, reasoning="hidden")[REGEX]
    if condition == "drop-text":
        return rows_table(cohort, reasoning_only, REGEX).assign(log_tool_calls=visible().log_tool_calls)
    if condition == "drop-tools":
        return visible().assign(log_tool_calls=0.0)
    raise ValueError(condition)


def ridge_errors(protocol: str, domain: str, task: Task, keep: set[str], conditions: list[str], tools: bool,
                 reasoning_only: pd.DataFrame) -> pd.DataFrame:
    """Each target's squared error per condition for a ridge protocol refitted on the references' channel-filtered
    transcripts, and of the Baseline ("easy"): protocol, domain, semester, condition, model, err.

    protocol: "embedding" or "glm_regex". tools: the domain has tool calls (embedding_variant).
    reasoning_only: the deterministic features of the reasoning alone (features_reasoning_only.csv).
    """
    refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
    cohorts = [on_items(c, keep) for c in (task.references, task.test)]
    cell = dict(protocol=protocol, domain=domain, semester=task.semester)
    frames = []
    for condition in conditions:
        if protocol == "embedding":
            x = pd.concat([EmbeddingRegression(embedding_variant(condition, tools)).get_features(c) for c in cohorts])
        else:
            x = pd.concat([feature_rows(condition, c, reasoning_only) for c in cohorts])
        frames.append(squared_errors(ridge_forecast(protocol, x, refs, targets), truth, **cell, condition=condition))
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)


def direct_errors(domain: str, task: Task, judge: str, conditions: list[str], keep: set[str], logs: Path) -> pd.DataFrame:
    """Absolute scoring's squared error per condition, of the full transcript and of the Baseline
    ("easy"): protocol, domain, semester, condition, model, err.

    judge: the task's direct judge; its full transcript is its reasoning-shown run as the main analysis loads it, cut to
        the `keep` items. Each condition is its published re-rating (logs/direct_channels-<condition>/) on the full
        transcript's rated (model, bench, item), recalibrated on the references.
    """
    refs, truth = task.references.theta, task.truth
    (run,) = [r for r in loading.direct_runs(loading.domain(domain), task)
              if r.judge == judge and r.condition == "reasoning-shown"]
    full = run.table.assign(item=run.table.item.astype(str))
    full = full[full.item.isin(keep)]
    cell = dict(protocol="direct", domain=domain, semester=task.semester)
    frames = [squared_errors(calibrate(full, refs), truth, **cell, condition="full")]
    for condition in conditions:
        ratings = published_ratings(logs, f"channels-{condition}", domain, task.semester, judge)
        ratings = ratings.assign(item=ratings.item.astype(str)).merge(full[["model", "bench", "item"]].drop_duplicates())
        frames.append(squared_errors(calibrate(ratings, refs), truth, **cell, condition=condition))
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)


def written_chars(task: Task, keep: set[str], channels: dict[str, str], data: Path) -> dict[str, int]:
    """Characters the models themselves wrote (role prefixes removed) in each channel, over the task's references' and
    targets' correct first-attempt transcripts on the `keep` items.

    channels: label -> the channel spec transcripts are read with (evals.transcripts.read_transcripts), e.g.
        "only-reasoning".
    data: the benchmarks checkout the models' log paths are relative to.
    """
    totals = dict.fromkeys(channels, 0)
    models = pd.concat([task.references.models, task.test.models])
    for model in models.index:
        for column in (c for c in models.columns if c.startswith("path_")):
            path = models.loc[model, column]
            if not isinstance(path, str):
                continue
            items = set(task.references.items.loc[task.references.items.bench == column[5:], "item"]) & keep
            for label, spec in channels.items():
                found = read_transcripts(data / path, require_correct=True, include_reasoning=True, items=items, channels=spec)
                totals[label] += sum(response_chars(texts[0].text) for texts in found.values())
    return totals


def shares_figure(shares: pd.DataFrame, colours: dict[str, str], domain_names: dict[str, str]) -> plt.Figure:
    """One pie per domain (rows of `shares`) of each channel's (columns) share of the characters, labelled when >= 10%."""
    fig, axes = plt.subplots(1, len(shares), figsize=(3.8, 1.4))
    for ax, domain in zip(np.atleast_1d(axes), shares.index):
        values = shares.loc[domain]
        ax.pie(values, colors=[colours[c] for c in values.index], startangle=90, counterclock=False,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
               autopct=lambda pct: f"{pct:.0f}%" if pct >= 10 else "", pctdistance=0.6,
               textprops={"fontsize": 7, "color": "white"})
        ax.set_title(domain_names[domain], fontsize=7, color="0.4", pad=2)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in shares.columns], list(shares.columns),
               frameon=False, ncol=len(shares.columns), loc="lower center", bbox_to_anchor=(0.5, -0.08), handlelength=1,
               columnspacing=1)
    return fig
