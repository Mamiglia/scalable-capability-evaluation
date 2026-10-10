"""
Transcript channels: each target's squared error when the model's reasoning, its messages or its tool calls with
their outputs are removed from every transcript (conditions "drop-reasoning", "drop-text", "drop-tools"; "full" is
the whole transcript), for the Pairwise judge, Deterministic features, Embedding and Absolute scoring, and each
channel's share of the characters the models wrote. Maths and Science have no tool calls: there the transcript without
reasoning is the reasoning-hidden transcript, and the transcript without messages is the reasoning alone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import loading
from protocols import judges
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
    """
    The log folder a channel judge config writes the task's verdicts to, as the pairwise harness computes it
    (JudgeBT.log_dir: the config's log folder, suffixed with its condition label if it has one, then domain, semester,
    judge and the condition hash).

    Inputs:
        judge_config: a channel judge config, configs/judges/channels/<judge>-<run>.yml.
        domain: the task's domain key.
        semester: the task's semester.
    Outputs:
        Path of the run folder; it need not exist.
    """
    return JudgeBT(JudgeConfig.from_cli(judge_config), loading.domain(domain)).log_dir(semester)


def pairwise_errors(domain: str, task: Task, train_runs: dict[str, Path], channel_runs: dict[str, Path],
                    keep: set[str], conversion: str) -> pd.DataFrame:
    """
    Each target's squared error under each Pairwise judge run of the channel analysis and under the Baseline ("easy").
    Runs judged on the full train set are cut to the `keep` items, channel runs (judged on those items only) are read
    whole, and each run is scored on all of its own verdicts.

    Inputs:
        domain: the task's domain key.
        task: the task (the references' θ*, the targets and their θ*).
        train_runs: condition -> log folder of a run on the full train set: "full", and for Maths and Science the
            reasoning-hidden run as "drop-reasoning".
        channel_runs: condition -> log folder of a channel run (channel_run_dir).
        keep: the item ids of the channel analysis (removal_items).
        conversion: how verdicts become predicted θ (protocols.judges.pairwise_theta), e.g. "linear".
    Outputs:
        DataFrame with columns domain, semester, condition, model, err; one row per target and condition.
    """
    refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
    runs = {condition: (v := from_logs(path))[v.item.isin(keep)] for condition, path in train_runs.items()}
    runs |= {condition: from_logs(path) for condition, path in channel_runs.items()}
    cell = dict(domain=domain, semester=task.semester)
    frames = [squared_errors(judges.pairwise_theta(verdicts, refs, targets, conversion, None), truth, **cell,
                             condition=condition) for condition, verdicts in runs.items()]
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)


def embedding_variant(condition: str, tools: bool) -> str:
    """
    The name of the embedding variant (a set of vectors in logs/embeddings/) that holds a condition's
    transcripts.

    Inputs:
        condition: "full", "drop-reasoning", "drop-text" or "drop-tools".
        tools: whether the domain has tool calls.
    Outputs:
        str, the variant name: "full" stays "full"; with tool calls every other condition is "channels-<condition>";
        without them "drop-reasoning" is "answer", "drop-text" is "channels-only-reasoning" and any other condition
        is "channels-<condition>".
    """
    variant = {"full": "full", "drop-reasoning": "channels-drop-reasoning" if tools else "answer",
               "drop-text": "channels-drop-text" if tools else "channels-only-reasoning"}
    return variant.get(condition, f"channels-{condition}")


def feature_rows(condition: str, cohort: Cohort, reasoning_only: pd.DataFrame) -> pd.DataFrame:
    """
    The deterministic (regex) features per model of `cohort` on one channel condition's transcripts: "full" is the
    messages with reasoning (glm_features "visible"), "drop-reasoning" the "hidden" view, "drop-text" the reasoning
    alone (`reasoning_only`) with the full transcript's tool-call count, and "drop-tools" the full transcript with
    log_tool_calls set to 0.

    Inputs:
        condition: "full", "drop-reasoning", "drop-text" or "drop-tools"; any other raises ValueError.
        cohort: a task's references or targets, restricted to the channel items (on_items).
        reasoning_only: per-transcript features of the reasoning alone (columns path, item, correct and the regex
            features; features_reasoning_only.csv, written by scripts/analysis/features.py).
    Outputs:
        DataFrame of the regex feature columns, one row per model of the cohort (the mean over its transcripts).
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
    """
    Each target's squared error under a ridge protocol refitted per condition on the references' channel-filtered
    transcripts, and under the Baseline ("easy").

    Inputs:
        protocol: "embedding" (the variant embedding_variant names) or "glm_regex" (the features feature_rows gives).
        domain: the task's domain key.
        task: the task.
        keep: the item ids of the channel analysis (removal_items).
        conditions: the conditions scored, e.g. ["full", "drop-reasoning", "drop-text", "drop-tools"].
        tools: whether the domain has tool calls (embedding_variant).
        reasoning_only: the deterministic features of the reasoning alone (features_reasoning_only.csv), for
            feature_rows.
    Outputs:
        DataFrame with columns protocol, domain, semester, condition, model, err; one row per target and condition.
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
    """
    Each target's squared error under Absolute scoring of the full transcript and of each channel-removed transcript,
    and under the Baseline ("easy"). The full transcript's ratings are cut to the `keep` items; each condition's
    published re-rating (logs/absolute/channels-<condition>/) is cut to the (model, bench, item) rated in the full run,
    and every condition is calibrated on the references' mean ratings.

    Inputs:
        domain: the task's domain key.
        task: the task.
        judge: the task's Absolute scoring judge; its full-transcript ratings are its "reasoning-shown" run from
            loading.direct_runs.
        conditions: the channel conditions scored, e.g. ["drop-reasoning", "drop-text", "drop-tools"].
        keep: the item ids of the channel analysis (removal_items).
        logs: the logs/ checkout (paths.logs).
    Outputs:
        DataFrame with columns protocol ("direct"), domain, semester, condition, model, err; one row per target and
        condition.
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
    """
    Counts the characters the models themselves wrote (the renderer's role prefixes removed) in each channel, over the
    task's references' and targets' correct transcripts of the `keep` items, the first transcript per item.

    Inputs:
        task: the task; uses the references' and targets' path_<bench> log paths and the references' items.
        keep: the item ids of the channel analysis (removal_items).
        channels: label -> the channel spec the transcripts are read with (evals.transcripts.read_transcripts), e.g.
            "only-reasoning".
        data: the benchmarks checkout (paths.data) the log paths are relative to.
    Outputs:
        dict label -> number of characters.
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
    """
    Draws one pie per domain of each channel's share of the characters the models wrote, the percentage printed on
    each slice of at least 10%, with the channels' legend below.

    Inputs:
        shares: DataFrame indexed by domain, one column per channel label, each row summing to 1.
        colours: channel label -> colour.
        domain_names: domain key -> display name.
    Outputs:
        the matplotlib Figure.
    """
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
