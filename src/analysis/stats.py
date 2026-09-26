"""Stats: MSE of theta-hat against theta* with a bootstrap interval over targets, pooled over
semesters and domains; judge summaries (best judge on references, mean over judges); skill
relative to easy-item prediction and the oracle; and the per-task metrics the figures and
tables use: raw MSE, delta from easy-item prediction, and the paired-bootstrap z of that delta.
"""

import numpy as np
import pandas as pd

from configs.logs import logger

KEYS = ["domain", "protocol", "variant", "judge", "condition"]
BASELINE, ORACLE = "easy_items", "accuracy_hard"
JUDGED = ("pairwise", "direct")
BEST, MEAN = "best judge on references", "mean over judges"
PROTOCOLS = (BASELINE, "glm_length", "glm_regex", "embedding", "direct", "pairwise")  # every protocol with one forecast per target
METRICS = ("mse", "delta", "z")


def mse_row(g: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """n, mse and a percentile-bootstrap 95% interval (lo, hi) of the mean squared error of
    g.theta_hat against g.theta, NaN forecasts dropped. The replication unit is the row
    (one target forecast); callers pool rows only where that is the right unit."""
    err = (g.theta_hat - g.theta).dropna().to_numpy() ** 2
    if len(err) == 0:
        return dict(n=0, mse=np.nan, lo=np.nan, hi=np.nan)
    boot = np.random.default_rng(seed).choice(err, size=(draws, len(err))).mean(axis=1)
    return dict(n=len(err), mse=float(err.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def condition_rank(condition: str) -> int:
    """How far a judge run is from the core condition: 0 reasoning-shown on the full train set, 1 reasoning-shown
    on a smaller train set ("-<k>pq"), 2 reasoning-hidden, 3 not a core candidate (a challenge condition)."""
    if condition == "reasoning-shown":
        return 0
    if condition.startswith("reasoning-shown-"):
        return 1
    return 2 if condition == "reasoning-hidden" else 3


def core_condition(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The judge rows of the core condition, chosen per judge within a cell (domain, semester,
    protocol, variant): each judge's closest run to reasoning shown on the full train set (see
    `condition_rank`) — else its reasoning-shown run on a smaller train set, else its
    reasoning-hidden run (semesters whose subjects have no reasoning were judged hidden only).
    Every judge stays in the cell, so "best judge on references" compares all of them; a judge
    scored below full-train shown is logged. Transcript-protocol rows pass through unchanged."""
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
    """The judge each task uses: per (domain, semester, protocol), the core-condition run with the lowest reference
    leave-one-model-out error; a tie keeps the judge that comes first in `forecasts`.

    forecasts: the forecasts table (analysis/forecasts/forecasts.csv).
    variants: protocol -> the variant it is reported in, e.g. {"pairwise": "linear", "direct": "raw"}; judge protocols
        not named are left out.
    Returns one row per (domain, semester, protocol): variant, judge, condition, judge_loo_mse.
    """
    core = core_condition(forecasts)
    core = core[core.protocol.isin(JUDGED) & (core.variant == core.protocol.map(variants))]
    key = ["domain", "semester", "protocol"]
    best = core[core.groupby(key).judge_loo_mse.transform("min") == core.judge_loo_mse]
    return best.drop_duplicates(key)[[*key, "variant", "judge", "condition", "judge_loo_mse"]].reset_index(drop=True)


def judge_summaries(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Two derived judge rows per (domain, semester, judge protocol, variant), on the core
    condition, condition set to "core": judge BEST = the rows of the judge with the lowest
    reference leave-one-out MSE that semester (known without target labels; a tie keeps one
    judge); judge MEAN = the rows of every judge (pooling equal-sized per-judge row sets
    makes the MSE the mean of the per-judge MSEs)."""
    core = core_condition(forecasts)
    judged = core[core.protocol.isin(JUDGED)]
    key = ["domain", "semester", "protocol", "variant"]
    best = judged.loc[judged.groupby(key).judge_loo_mse.transform("min") == judged.judge_loo_mse]
    best = best.drop_duplicates([*key, "model"])
    return pd.concat([best.assign(judge=BEST), judged.assign(judge=MEAN)], ignore_index=True).assign(condition="core")


def scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per domain x semester x protocol x variant x judge x condition (n, missing,
    mse, lo, hi), plus semester "all" per domain and domain "all" per protocol x variant x
    judge x condition (pooled over every semester of every domain). The judge summary rows
    are included with condition "core"; the mean-over-judges rows' interval is NaN because
    their pooled rows are not independent targets (the unit there is the judge), unlike the
    best-judge rows, which are one forecast per target like any other protocol."""
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
    """Pooled (semester "all") rows with the domain's easy-item-prediction and oracle MSE
    alongside and skill = (baseline - mse) / (baseline - oracle): 1 at the oracle, 0 at
    easy-item prediction, negative below it. For domain "all" the anchors are the
    domain-"all" rows."""
    pooled = scores[scores.semester == "all"].copy()
    anchors = pooled[pooled.protocol.isin([BASELINE, ORACLE])].pivot(index="domain", columns="protocol", values="mse")
    pooled["baseline"] = pooled.domain.map(anchors[BASELINE])
    pooled["oracle"] = pooled.domain.map(anchors[ORACLE])
    pooled["skill"] = (pooled.baseline - pooled.mse) / (pooled.baseline - pooled.oracle)
    return pooled


def one_per_target(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The forecast rows with exactly one row per target and condition: transcript protocols
    as they are, judge protocols as the best judge on references (condition "core"). The
    mean over judges is excluded because its rows are not one forecast per target."""
    best = judge_summaries(forecasts)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], best[best.judge == BEST]], ignore_index=True)


def paired_metrics(errors: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """From aligned squared errors (columns `err_condition`, `err_baseline`, one row per target):
    n, mse, baseline_mse, delta = mean(condition - baseline), and z = delta / sd of the
    paired bootstrap of delta (targets resampled with replacement; NaN when the paired
    differences do not vary)."""
    c, b = errors.err_condition.to_numpy(float), errors.err_baseline.to_numpy(float)
    if len(c) == 0:
        return dict(n=0, mse=np.nan, baseline_mse=np.nan, delta=np.nan, z=np.nan)
    idx = np.random.default_rng(seed).integers(0, len(c), size=(draws, len(c)))
    sd = float((c[idx] - b[idx]).mean(axis=1).std(ddof=1))
    delta = float((c - b).mean())
    return dict(n=len(c), mse=float(c.mean()), baseline_mse=float(b.mean()), delta=delta, z=delta / sd if sd > 0 else np.nan)


def task_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per (domain, semester) x protocol x variant x judge x condition with the three
    metrics paired against easy-item prediction on the same targets (one_per_target rows only), plus
    semester "all" per domain and domain "all" per condition over the pooled targets.
    Columns: domain, semester, protocol, variant, judge, condition, n, mse, baseline_mse,
    delta, z. A target the condition did not forecast is dropped from both sides."""
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
    """The rows of `table` in the variant `variants` names for their protocol ("-" for protocols it does not name)."""
    return table[[variants.get(p, "-") == v for p, v in zip(table.protocol, table.variant)]]


def target_errors(forecasts: pd.DataFrame, variants: dict[str, str], semesters: tuple[str, ...] | None = None,
                  protocols: tuple[str, ...] = PROTOCOLS) -> pd.DataFrame:
    """One forecast per target and protocol, with its squared error: judge protocols as each task's best judge on
    references, every protocol in its reported variant; targets without a forecast dropped.

    variants: protocol -> reported variant, e.g. {"pairwise": "linear", "direct": "raw", "embedding": "full"}.
    semesters: keep only these semesters (None: all).
    Returns columns domain, semester, model, protocol, theta_hat, theta, err.
    """
    f = in_variants(one_per_target(forecasts), variants)
    f = f[f.protocol.isin(protocols)]
    if semesters is not None:
        f = f[f.semester.isin(semesters)]
    f = f.dropna(subset=["theta_hat", "theta"])
    return f.assign(err=(f.theta_hat - f.theta) ** 2)[["domain", "semester", "model", "protocol", "theta_hat", "theta", "err"]]


def with_baseline(errors: pd.DataFrame) -> pd.DataFrame:
    """`errors` (target_errors) with a `base` column: the baseline's squared error on the same target."""
    base = errors[errors.protocol == BASELINE].set_index(["domain", "semester", "model"]).err
    return errors.assign(base=errors.set_index(["domain", "semester", "model"]).index.map(base))


def prediction_score(errors: np.ndarray | pd.Series, baseline: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """-log2(error / baseline error): the number of halvings of the error, 0 at the baseline, 1 at half its error."""
    return -np.log2(errors / baseline)


def family(model: pd.Series) -> pd.Series:
    """The developer prefix of each model name: glm-4.7 -> glm, kimi-k2.5 -> kimi, qwen3-235b -> qwen."""
    return model.str.extract(r"^([a-z]+)", expand=False)


def best_other_family(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Like one_per_target, but each target's judge is the best on references among the core-condition judges from
    another developer than the target, so no model is judged by its own family; targets with no such judge dropped."""
    core = core_condition(forecasts)
    judged = core[core.protocol.isin(JUDGED) & (family(core.judge) != family(core.model))]
    key = ["domain", "semester", "protocol", "variant", "model"]
    best = judged.loc[judged.groupby(key).judge_loo_mse.transform("min") == judged.judge_loo_mse].drop_duplicates(key)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], best.assign(judge="best-other-family", condition="core")],
                     ignore_index=True)
