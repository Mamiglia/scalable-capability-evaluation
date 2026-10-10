"""
Transcript thirds: each target's squared error when every rendered transcript (reasoning shown) is cut into three
equal character spans and one third is removed or kept alone (conditions "thirds-<kept thirds>", e.g. "thirds-23"
without the first third, "thirds-1" the first third alone), next to the full transcript ("full") and the Baseline
("easy"). The Pairwise judge re-judged the cut transcripts (src/challenges/thirds), and each condition of a task is
scored on the comparisons judged in all of them, since a third too short to judge drops a comparison from that run
only. Deterministic features and Embedding refit their ridge per condition on the references' cut transcripts. For
Absolute scoring the task's judge rated the cut transcripts (logs/absolute/thirds-<kept>/), and each
condition is calibrated on the references' ratings of the transcripts rated in every condition.
"""

from pathlib import Path

import pandas as pd

from analysis import loading
from protocols import judges
from analysis.transcript_parts.paired_change import on_items, published_ratings, ridge_forecast, squared_errors
from capability.task import Task
from protocols.direct_judge.scoring import calibrate
from protocols.embeddings import EmbeddingRegression
from protocols.pairwise.judge import from_logs
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.reader import rows_table

PAIR = ["bench", "item", "model_left", "model_right"]  # one pairwise comparison
RATING = ["model", "bench", "item"]  # one absolute rating


def pairwise_errors(domain: str, task: Task, run_dirs: dict[str, Path], keep: set[str], conversion: str) -> pd.DataFrame:
    """
    Each target's squared error under each Pairwise judge run of the thirds analysis (the full transcript and every
    third removed or kept alone) and under the Baseline ("easy"), every run scored on the same comparisons. Raises
    FileNotFoundError when a run folder holds no .eval logs.

    Inputs:
        domain: the task's domain key.
        task: the task (the references' θ*, the targets and their θ*).
        run_dirs: condition -> the run's log folder (logs/pairwise/<view>/<domain>/<semester>/<judge>/<hash>), "full" first.
            Each run is cut to the `keep` items, a comparison judged twice is kept once (the run's own verdict before
            an imported one), and every run is cut to the comparisons (bench, item, model_left, model_right) judged
            in all of them.
        keep: the item ids judged (removal_items).
        conversion: how verdicts become predicted θ (protocols.judges.pairwise_theta), e.g. "linear".
    Outputs:
        DataFrame with columns domain, semester, condition, model, err; one row per target and condition.
    """
    refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
    runs = {}
    for condition, path in run_dirs.items():
        if not any(path.glob("*.eval")):
            raise FileNotFoundError(f"{domain} {task.semester} {condition}: no logs in {path}")
        verdicts = from_logs(path)
        verdicts = verdicts[verdicts.item.astype(str).isin(keep)]
        runs[condition] = verdicts.sort_values("imported", kind="stable").drop_duplicates(PAIR, keep="first")
    common = runs["full"][PAIR].drop_duplicates()
    for verdicts in runs.values():
        common = common.merge(verdicts[PAIR].drop_duplicates())
    cell = dict(domain=domain, semester=task.semester)
    frames = [squared_errors(judges.pairwise_theta(verdicts.merge(common), refs, targets, conversion, None), truth,
                             **cell, condition=condition) for condition, verdicts in runs.items()]
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)


def ridge_errors(protocol: str, domain: str, task: Task, keep: set[str], conditions: list[str],
                 features: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Each target's squared error under a ridge protocol refitted per condition on the references' cut transcripts, and
    under the Baseline ("easy").

    Inputs:
        protocol: "embedding" (the embedding variant with the condition's name, e.g. "full" or "thirds-23") or
            "glm_regex".
        domain: the task's domain key.
        task: the task.
        keep: the item ids the features are averaged over (removal_items).
        conditions: the conditions scored, e.g. ["full", "thirds-23", "thirds-13", "thirds-12"].
        features: for "glm_regex", the deterministic features of the cut transcripts, one row per transcript and
            condition (columns path, item, correct, condition and the regex features; features_thirds.csv, written by
            scripts/analysis/features.py).
    Outputs:
        DataFrame with columns protocol, domain, semester, condition, model, err; one row per target and condition.
    """
    refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
    cohorts = [on_items(c, keep) for c in (task.references, task.test)]
    cell = dict(protocol=protocol, domain=domain, semester=task.semester)
    frames = []
    for condition in conditions:
        if protocol == "embedding":
            x = pd.concat([EmbeddingRegression(condition).get_features(c) for c in cohorts])
        else:
            rows = features[features.condition == condition]
            x = pd.concat([rows_table(c, rows, FEATURE_SETS["regex"]) for c in cohorts])
        frames.append(squared_errors(ridge_forecast(protocol, x, refs, targets), truth, **cell, condition=condition))
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)


def direct_errors(domain: str, task: Task, judge: str, conditions: list[str], rated_in: list[str],
                  logs: Path) -> pd.DataFrame:
    """
    Each target's squared error under Absolute scoring of the full transcript and of each thirds condition, and under
    the Baseline ("easy"). Every condition is calibrated on the references' mean ratings (protocols.direct_judge.
    scoring.calibrate) over the same transcripts: the (model, bench, item) rated in the full run and in every
    condition of `rated_in`.

    Inputs:
        domain: the task's domain key.
        task: the task.
        judge: the task's Absolute scoring judge; its full-transcript ratings are its "reasoning-shown" run from
            loading.direct_runs.
        conditions: the thirds conditions scored, e.g. ["thirds-23", "thirds-13", "thirds-12"].
        rated_in: the thirds conditions whose published runs (logs/absolute/<condition>/) are read and
            intersected, the kept-alone ones included; must contain `conditions`.
        logs: the logs/ checkout (paths.logs).
    Outputs:
        DataFrame with columns protocol ("direct"), domain, semester, condition, model, err; one row per target and
        condition.
    """
    refs, truth = task.references.theta, task.truth
    runs = {c: published_ratings(logs, c, domain, task.semester, judge) for c in rated_in}
    (full,) = [r for r in loading.direct_runs(loading.domain(domain), task)
               if r.judge == judge and r.condition == "reasoning-shown"]
    runs["full"] = full.table.assign(item=full.table.item.astype(str))
    common = runs[rated_in[0]][RATING]
    for ratings in runs.values():
        common = common.merge(ratings[RATING].drop_duplicates())
    cell = dict(protocol="direct", domain=domain, semester=task.semester)
    frames = [squared_errors(calibrate(runs[c].merge(common), refs), truth, **cell, condition=c)
              for c in ["full", *conditions]]
    return pd.concat([*frames, squared_errors(refs.mean(), truth, **cell, condition="easy")], ignore_index=True)
