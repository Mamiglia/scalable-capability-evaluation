"""
Squared-error statistics of the predictions in the forecasts table (analysis/forecasts/forecasts.csv): each judge's
core-condition run, each task's judge, each target's squared error, and mean squared errors with bootstrap intervals
over targets, per task and pooled over semesters and domains. Also each prediction set's skill between the Baseline and
the accuracy oracle, and its error minus the Baseline's on the same targets with a paired-bootstrap z.
"""

import numpy as np
import pandas as pd

from configs.logs import logger

KEYS = ["domain", "protocol", "variant", "judge", "condition"]
BASELINE, ORACLE = "easy_items", "accuracy_hard"  # code keys of the Baseline and the accuracy oracle
JUDGED = ("pairwise", "direct")
BEST, MEAN = "best judge on references", "mean over judges"
# every protocol with one prediction per target
PROTOCOLS = (BASELINE, "glm_length", "glm_regex", "embedding", "direct", "pairwise")
METRICS = ("mse", "delta", "z")


def mse_row(g: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """
    Mean squared error of the predictions in g against theta*, with a percentile-bootstrap 95% interval that resamples
    the rows. Each row is taken as one independent prediction of one target; callers pool rows only where that holds.

    Inputs:
        g: prediction rows; uses theta_hat and theta. Rows where either is NaN are dropped.
        seed: seed of the bootstrap's random generator.
        draws: number of bootstrap resamples.
    Outputs:
        dict with n (rows scored), mse, lo and hi (the 2.5 and 97.5 percentiles of the resampled means); mse, lo and hi
        are NaN when no row is scored.
    """
    err = (g.theta_hat - g.theta).dropna().to_numpy() ** 2
    if len(err) == 0:
        return dict(n=0, mse=np.nan, lo=np.nan, hi=np.nan)
    boot = np.random.default_rng(seed).choice(err, size=(draws, len(err))).mean(axis=1)
    return dict(n=len(err), mse=float(err.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def condition_rank(condition: str) -> int:
    """
    Ranks a judge run's condition by how far it is from the core condition, reasoning shown on the full train set; lower
    is closer.

    Inputs:
        condition: the run's condition name: "reasoning-shown", "reasoning-shown-<k>pq" (k train items per difficulty
            quartile), "reasoning-hidden", or a challenge condition.
    Outputs:
        0 for "reasoning-shown"; 1 for any other name starting "reasoning-shown-" (a smaller train set); 2 for
        "reasoning-hidden"; 3 for anything else, which is never a core-condition candidate.
    """
    if condition == "reasoning-shown":
        return 0
    if condition.startswith("reasoning-shown-"):
        return 1
    return 2 if condition == "reasoning-hidden" else 3


def core_condition(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps each judge's run closest to the core condition (condition_rank): its reasoning-shown run on the full train
    set, else its reasoning-shown run on a smaller train set, else its reasoning-hidden run (semesters whose models show
    no reasoning were judged with reasoning hidden only). Every judge stays in, so the choice of judge compares all of
    them; each judge scored in a condition other than reasoning shown on the full train set is logged as a warning.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses domain, semester, protocol, variant,
            judge and condition.
    Outputs:
        DataFrame with forecasts' columns: the rows of every protocol other than pairwise and direct unchanged
        (transcript protocols, the oracle, the pooled ensemble rows), then the pairwise and direct rows of each judge's
        closest run per (domain, semester, protocol, variant). Pairwise and direct rows of challenge conditions are
        dropped.
    """
    judged = forecasts[forecasts.protocol.isin(JUDGED)]
    key = ["domain", "semester", "protocol", "variant", "judge"]
    rank = judged.condition.map(condition_rank).rename("condition_rank")
    candidates = judged[rank < 3].assign(condition_rank=rank[rank < 3])
    core = candidates[candidates.condition_rank == candidates.groupby(key).condition_rank.transform("min")]
    for row in core[core.condition_rank > 0].drop_duplicates(key).itertuples():
        logger.warning("%s %s %s/%s %s: no full-train reasoning-shown run, scoring its %s run",
                       row.domain, row.semester, row.protocol, row.variant, row.judge, row.condition)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], core.drop(columns="condition_rank")], ignore_index=True)


def best_judges(forecasts: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """
    Picks each task's judge: per (domain, semester, judge protocol), the judge whose core-condition run, in the variant
    the protocol is reported in, has the lowest leave-one-model-out squared error on the references. A tie keeps the
    judge that comes first in forecasts.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses domain, semester, protocol, variant,
            judge, condition and judge_loo_mse.
        variants: protocol -> the variant it is reported in, e.g. {"pairwise": "linear", "direct": "raw"}; a judge
            protocol not named is left out.
    Outputs:
        DataFrame with columns domain, semester, protocol, variant, judge, condition, judge_loo_mse; one row per task
        and judge protocol.
    """
    core = core_condition(forecasts)
    core = core[core.protocol.isin(JUDGED) & (core.variant == core.protocol.map(variants))]
    key = ["domain", "semester", "protocol"]
    best = core[core.groupby(key).judge_loo_mse.transform("min") == core.judge_loo_mse]
    return best.drop_duplicates(key)[[*key, "variant", "judge", "condition", "judge_loo_mse"]].reset_index(drop=True)


def judge_summaries(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Builds two summary judges per (domain, semester, judge protocol, variant) from the core-condition runs: the best
    judge on references (lowest leave-one-model-out error on the references, which needs no target's theta*) and the
    mean over judges (every judge's rows together; with the same number of rows per judge, their pooled mean squared
    error is the mean of the judges' mean squared errors).

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses domain, semester, protocol, variant,
            judge, condition, model and judge_loo_mse.
    Outputs:
        DataFrame with forecasts' columns, pairwise and direct rows only, condition "core": the best judge's rows with
        judge BEST ("best judge on references"; one row per target, a tie keeping the first judge's row), then every
        judge's rows with judge MEAN ("mean over judges").
    """
    core = core_condition(forecasts)
    judged = core[core.protocol.isin(JUDGED)]
    key = ["domain", "semester", "protocol", "variant"]
    best = judged.loc[judged.groupby(key).judge_loo_mse.transform("min") == judged.judge_loo_mse]
    best = best.drop_duplicates([*key, "model"])
    return pd.concat([best.assign(judge=BEST), judged.assign(judge=MEAN)], ignore_index=True).assign(condition="core")


def scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Mean squared error of every prediction set in the forecasts table, per task and pooled: one row per (domain,
    semester, protocol, variant, judge, condition), plus semester "all" per domain (every semester of the domain pooled)
    and domain "all" (every task pooled), each with a bootstrap interval over targets (mse_row). The summary judges of
    judge_summaries are included with condition "core"; the mean-over-judges rows get no interval (lo and hi NaN)
    because their rows are several judges' predictions of the same targets, not independent targets.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses domain, semester, protocol, variant,
            judge, condition, model, theta_hat, theta, judge_loo_mse and missing.
    Outputs:
        DataFrame with columns domain, semester, protocol, variant, judge, condition, n, missing, mse, lo, hi; missing
        is the run's count of absent judgments, summed over tasks in the pooled rows.
    """
    rows = []
    for f in (forecasts, judge_summaries(forecasts)):
        for k, g in f.groupby(["semester", *KEYS], sort=False):
            rows.append(dict(zip(["semester", *KEYS], k), missing=int(g.missing.max()), **mse_row(g)))
        for k, g in f.groupby(KEYS, sort=False):
            rows.append(dict(zip(KEYS, k), semester="all", missing=int(g.groupby("semester").missing.max().sum()), **mse_row(g)))
        for k, g in f.groupby(KEYS[1:], sort=False):
            rows.append(dict(zip(KEYS[1:], k), domain="all", semester="all",
                             missing=int(g.groupby(["domain", "semester"]).missing.max().sum()), **mse_row(g)))
    table = pd.DataFrame(rows)[["domain", "semester", *KEYS[1:], "n", "missing", "mse", "lo", "hi"]]
    table.loc[table.judge == MEAN, ["lo", "hi"]] = np.nan
    return table


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Adds skill to the pooled rows of a scores table: skill = (Baseline MSE - MSE) / (Baseline MSE - oracle MSE), 1 at
    the accuracy oracle, 0 at the Baseline and negative when worse than the Baseline. The anchors are the same domain's
    pooled Baseline and oracle rows (for domain "all", the domain-"all" rows).

    Inputs:
        scores: the output of scores(); uses domain, semester, protocol and mse, and needs one pooled Baseline
            (easy_items) and one pooled oracle (accuracy_hard) row per domain.
    Outputs:
        DataFrame: the rows with semester "all", with added columns baseline (the domain's Baseline MSE), oracle (its
        oracle MSE) and skill.
    """
    pooled = scores[scores.semester == "all"].copy()
    anchors = pooled[pooled.protocol.isin([BASELINE, ORACLE])].pivot(index="domain", columns="protocol", values="mse")
    pooled["baseline"] = pooled.domain.map(anchors[BASELINE])
    pooled["oracle"] = pooled.domain.map(anchors[ORACLE])
    pooled["skill"] = (pooled.baseline - pooled.mse) / (pooled.baseline - pooled.oracle)
    return pooled


def one_per_target(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps one prediction per target for each protocol, variant and condition: the rows of protocols other than pairwise
    and direct as they are, and for pairwise and direct the rows of the best judge on references (judge_summaries,
    condition "core"). The mean over judges is left out because it has several predictions per target.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses the columns core_condition and
            judge_summaries use.
    Outputs:
        DataFrame with forecasts' columns; the pairwise and direct rows have judge BEST ("best judge on references") and
        condition "core".
    """
    best = judge_summaries(forecasts)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], best[best.judge == BEST]], ignore_index=True)


def paired_metrics(errors: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """
    Compares one prediction set's squared errors with the Baseline's on the same targets: the mean difference, and its
    z, the difference divided by the standard deviation of a paired bootstrap that resamples targets.

    Inputs:
        errors: one row per target; uses err_condition (the prediction's squared error) and err_baseline (the
            Baseline's).
        seed: seed of the bootstrap's random generator.
        draws: number of bootstrap resamples.
    Outputs:
        dict with n, mse (mean of err_condition), baseline_mse (mean of err_baseline), delta (mse - baseline_mse) and z
        (delta over the bootstrap standard deviation; NaN when the paired differences do not vary). With no rows, n is 0
        and the rest NaN.
    """
    c, b = errors.err_condition.to_numpy(float), errors.err_baseline.to_numpy(float)
    if len(c) == 0:
        return dict(n=0, mse=np.nan, baseline_mse=np.nan, delta=np.nan, z=np.nan)
    idx = np.random.default_rng(seed).integers(0, len(c), size=(draws, len(c)))
    sd = float((c[idx] - b[idx]).mean(axis=1).std(ddof=1))
    delta = float((c - b).mean())
    return dict(n=len(c), mse=float(c.mean()), baseline_mse=float(b.mean()), delta=delta, z=delta / sd if sd > 0 else np.nan)


def task_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    The Baseline-paired metrics of every prediction set, per task and pooled: for each (domain, semester, protocol,
    variant, judge, condition) of one_per_target, the mean squared error, the Baseline's on the same targets, their
    difference and its paired-bootstrap z (paired_metrics); plus semester "all" per domain and domain "all" over every
    task. A target that either side did not predict is dropped from both.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses domain, semester, model, protocol,
            variant, judge, condition, theta_hat and theta, and the columns one_per_target uses.
    Outputs:
        DataFrame with columns domain, semester, protocol, variant, judge, condition, n, mse, baseline_mse, delta, z.
    """
    frame = one_per_target(forecasts)
    frame = frame.assign(err_condition=(frame.theta_hat - frame.theta) ** 2)
    base = frame[frame.protocol == BASELINE][["domain", "semester", "model", "err_condition"]].rename(columns={"err_condition": "err_baseline"})
    paired = frame.merge(base, on=["domain", "semester", "model"]).dropna(subset=["err_condition", "err_baseline"])
    rows = []
    for k, g in paired.groupby(["semester", *KEYS], sort=False):
        rows.append(dict(zip(["semester", *KEYS], k), **paired_metrics(g)))
    for k, g in paired.groupby(KEYS, sort=False):
        rows.append(dict(zip(KEYS, k), semester="all", **paired_metrics(g)))
    for k, g in paired.groupby(KEYS[1:], sort=False):
        rows.append(dict(zip(KEYS[1:], k), domain="all", semester="all", **paired_metrics(g)))
    return pd.DataFrame(rows)[["domain", "semester", *KEYS[1:], "n", "mse", "baseline_mse", "delta", "z"]]


def in_variants(table: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """
    Keeps each protocol's rows in the variant it is reported in.

    Inputs:
        table: any table with protocol and variant columns.
        variants: protocol -> reported variant; a protocol not named keeps only its rows with variant "-".
    Outputs:
        the rows of table whose variant is variants[protocol] ("-" for protocols not in variants).
    """
    return table[[variants.get(p, "-") == v for p, v in zip(table.protocol, table.variant)]]


def target_errors(forecasts: pd.DataFrame, variants: dict[str, str], semesters: tuple[str, ...] | None = None,
                  protocols: tuple[str, ...] = PROTOCOLS) -> pd.DataFrame:
    """
    Each target's squared error under each protocol: one prediction per target and protocol, pairwise and direct from
    each task's best judge on references, every protocol in the variant it is reported in; targets without a prediction
    or without theta* are dropped.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv).
        variants: protocol -> reported variant, e.g. {"pairwise": "linear", "direct": "raw", "embedding": "full"}.
        semesters: keep only these semesters (None: all).
        protocols: keep only these protocols (default PROTOCOLS: the Baseline, Length, Deterministic features,
            Embedding, Absolute scoring and Pairwise judge).
    Outputs:
        DataFrame with columns domain, semester, model, protocol, theta_hat, theta, err (the squared error); one row per
        target and protocol.
    """
    f = in_variants(one_per_target(forecasts), variants)
    f = f[f.protocol.isin(protocols)]
    if semesters is not None:
        f = f[f.semester.isin(semesters)]
    f = f.dropna(subset=["theta_hat", "theta"])
    return f.assign(err=(f.theta_hat - f.theta) ** 2)[["domain", "semester", "model", "protocol", "theta_hat", "theta", "err"]]


def with_baseline(errors: pd.DataFrame) -> pd.DataFrame:
    """
    Adds to each row of a target_errors table the Baseline's squared error on the same target.

    Inputs:
        errors: target_errors rows; uses domain, semester, model, protocol and err.
    Outputs:
        errors with an added column base (NaN where the Baseline has no row for the target).
    """
    base = errors[errors.protocol == BASELINE].set_index(["domain", "semester", "model"]).err
    return errors.assign(base=errors.set_index(["domain", "semester", "model"]).index.map(base))


