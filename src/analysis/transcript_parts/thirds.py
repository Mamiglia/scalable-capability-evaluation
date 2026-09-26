"""Transcript thirds: each target's squared error when every rendered transcript (reasoning shown) is cut into three
equal character spans and one third is removed or kept alone (conditions "thirds-<kept thirds>", e.g. "thirds-23"
without the first third). Pairwise judge: Figure 4 (removed) and Figure 22 (kept alone); Deterministic features,
Embedding and Absolute scoring: Figure 23 (removed).

The pairwise judge re-judged the cut transcripts (the thirds runs of src/challenges/thirds); every condition of a task
is scored on the pairs judged in all of them, since a third too short to judge drops a pair from that run only. The
Deterministic features and Embedding protocols refit their ridge per condition on the references' cut transcripts.
Absolute scoring: the task's direct judge rated the cut transcripts (published as logs/direct_truncated-thirds-<kept>/),
each condition recalibrated on the references' ratings, on the ratings present in every condition.
"""

from pathlib import Path

import pandas as pd

from analysis import judges, loading
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
    """Each target's squared error per pairwise run and of the Baseline ("easy"): domain, semester,
    condition, model, err.

    run_dirs: condition -> the run's log dir, "full" first. Each run is cut to the `keep` items, a verdict judged twice
        kept once (the first, a run's own before an imported one), and every run to the pairs judged in all of them.
    conversion: verdicts to theta-hat (analysis.judges.pairwise_theta), e.g. "linear".
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
    """Each target's squared error per condition for a ridge protocol refitted on the references' cut transcripts, and
    of the Baseline ("easy"): protocol, domain, semester, condition, model, err.

    protocol: "embedding" (the embedding variant named as the condition) or "glm_regex".
    features: for "glm_regex", the deterministic features of the cut transcripts (scripts/analysis/features.py,
        features_thirds.csv; a condition column).
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
    """Absolute scoring's squared error per condition, of the full transcript and of the Baseline
    ("easy"): protocol, domain, semester, condition, model, err.

    judge: the task's direct judge; its full transcript is its reasoning-shown run as the main analysis loads it.
    conditions: the thirds conditions scored.
    rated_in: the thirds conditions whose ratings, with the full transcript's, define the ratings every condition is
        calibrated on (the (model, bench, item) rated in all of them; the kept-alone runs count too).
    """
    refs, truth = task.references.theta, task.truth
    runs = {c: published_ratings(logs, f"truncated-{c}", domain, task.semester, judge) for c in rated_in}
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
