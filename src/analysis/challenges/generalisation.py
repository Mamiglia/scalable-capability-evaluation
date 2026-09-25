"""Research question 3: when does the capability signal generalise?

Three analyses that only reshape `analysis/results/forecasts.csv`: the prediction score over
target semesters, the forecast error against how far a target sits from its references, and the
reasoning-model targets of 2025-I, whose references are all non-reasoning models. Transcripts
from one domain calibrated on another domain's theta* live in cross_domain.py.

Judge protocols are the best judge on references, one variant per protocol as in figure one.
The prediction score is signal_glm's, -log2(error / the saturated-item prediction's error): the
number of halvings of the error, 0 at the baseline. Squared errors are pooled before the score is
taken, because a target the saturated-item prediction happens to nail has a near-zero baseline
error and would otherwise dominate.

Design: docs/superpowers/specs/2026-09-17-generalisation-design.md. Figures: plots.generalisation.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.challenges.signal_glm import DOMAINS, PROTOCOLS, score
from analysis.stats import DOMAIN_LABELS, chosen
from analysis.stats import BASELINE, one_per_target

DOMAIN_LABELS = {**DOMAIN_LABELS, "math": "Maths"}  # the manuscript spells the domain with an s

# 2025-I targets whose transcripts carry a reasoning trace: deepseek-r1 and minimax-m1 always
# reason, the three Qwen3 models were collected with `reasoning: {enabled: true}` (configs/models).
REASONING_2025_I = ("deepseek-r1-0528", "minimax-m1", "qwen3-235b-a22b", "qwen3-30b-a3b", "qwen3-32b")


def rows(forecasts: pd.DataFrame, variants: dict[str, str], semesters: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One row per target x protocol in PROTOCOLS: domain, semester, model, protocol, theta_hat,
    theta and err (the squared error). Judge protocols are the best judge on references; every
    protocol is taken in the variant `variants` names for it. Rows without a forecast are dropped."""
    f = chosen(one_per_target(forecasts), variants)
    f = f[f.protocol.isin(PROTOCOLS)]
    if semesters is not None:
        f = f[f.semester.isin(semesters)]
    f = f.dropna(subset=["theta_hat", "theta"])
    return f.assign(err=(f.theta_hat - f.theta) ** 2)[["domain", "semester", "model", "protocol", "theta_hat", "theta", "err"]]


def with_baseline(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with a `base` column: the saturated-item prediction's squared error on the same
    (domain, semester, model). Baseline rows are kept."""
    base = table[table.protocol == BASELINE].set_index(["domain", "semester", "model"]).err
    return table.assign(base=table.set_index(["domain", "semester", "model"]).index.map(base))


def task_scores(table: pd.DataFrame, draws: int = 2000, seed: int = 0, metric: str = "score") -> pd.DataFrame:
    """Per (domain, semester, protocol), from the task's targets: n, mse, baseline_mse, score =
    -log2(mse / baseline_mse), and `value`, `lo`, `hi`, `p` in the chosen metric, from `draws`
    bootstrap resamples of the task's targets (each target's pair of squared errors moves together).
    metric "score": value = the score, lo/hi its 2.5/97.5 percentiles, p the two-sided test of the
    score against 0. metric "error": value = the mean squared error, lo/hi its own 2.5/97.5
    percentiles, p the two-sided test of the paired difference from the saturated-item
    prediction's error. The figures replace p by the log likelihood ratio's
    (likelihood_ratio.task_tests)."""
    assert metric in ("score", "error"), metric
    rng = np.random.default_rng(seed)
    out = []
    for (domain, semester, protocol), g in with_baseline(table).dropna(subset=["base"]).groupby(["domain", "semester", "protocol"]):
        err, base = g.err.to_numpy(float), g.base.to_numpy(float)
        idx = rng.integers(0, len(err), size=(draws, len(err)))
        boot_err, boot_base = err[idx].mean(axis=1), base[idx].mean(axis=1)
        boot = score(boot_err, boot_base) if metric == "score" else boot_err
        gap = boot if metric == "score" else boot_err - boot_base  # what the test holds against 0
        keep = np.isfinite(boot) & np.isfinite(gap)
        lo, hi = np.percentile(boot[keep], [2.5, 97.5]) if len(g) >= 3 else (np.nan, np.nan)
        tail = min((gap[keep] <= 0).mean(), (gap[keep] >= 0).mean()) if len(g) >= 3 else np.nan
        value = float(score(err.mean(), base.mean())) if metric == "score" else float(err.mean())
        out.append(dict(domain=domain, semester=semester, protocol=protocol, n=len(g), mse=float(err.mean()),
                        baseline_mse=float(base.mean()), score=float(score(err.mean(), base.mean())), value=value,
                        p=np.nan if protocol == BASELINE else float(min(1.0, 2 * tail)),
                        lo=float(lo), hi=float(hi)))
    return pd.DataFrame(out).sort_values(["domain", "semester", "protocol"], ignore_index=True)


BASELINE_LABEL = "Saturated-item prediction"


def reference_theta(domain: str, semester: str) -> pd.Series:
    """The references' theta* for one task, from the frozen model table."""
    from analysis import loading
    from reference import tables
    models, _ = tables.load(loading.domain(domain))
    return models.loc[models[f"{semester}_role"] == "reference", f"{semester}_theta"].dropna()


def distances(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with the signed error (theta_hat - theta) and two distances per target: `above_top`,
    its theta* minus the strongest reference's, and `above_mean`, its theta* minus the references'
    mean. The saturated-item prediction's signed error is exactly -above_mean."""
    refs = {key: reference_theta(*key) for key in table.groupby(["domain", "semester"]).groups}
    key = table.set_index(["domain", "semester"]).index
    return table.assign(signed=table.theta_hat - table.theta,
                        above_top=table.theta.to_numpy() - np.array([refs[k].max() for k in key]),
                        above_mean=table.theta.to_numpy() - np.array([refs[k].mean() for k in key]))


def distance_slopes(table: pd.DataFrame, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per protocol and distance, the OLS slope of the signed error on the distance with a 95%
    percentile-bootstrap interval over targets. A slope of 0 means the error does not grow as
    targets move away from the references; -1 means the protocol tracks the saturated-item
    prediction, whose signed error is -above_mean by construction. Targets within a task share
    references, so the interval is optimistic."""
    rng = np.random.default_rng(seed)
    out = []
    for distance in ("above_top", "above_mean"):
        for p in PROTOCOLS:
            g = table[table.protocol == p].dropna(subset=[distance, "signed"])
            x, y = g[distance].to_numpy(float), g.signed.to_numpy(float)
            if len(x) < 3:
                continue
            idx = rng.integers(0, len(x), size=(draws, len(x)))
            boot = [np.polyfit(x[i], y[i], 1)[0] for i in idx]
            out.append(dict(distance=distance, protocol=p, n=len(x), slope=float(np.polyfit(x, y, 1)[0]),
                            lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(out)


def with_bins(table: pd.DataFrame, distance: str = "above_top", bins: int = 3,
              within: tuple[float, float] = (-1.0, 1.0)) -> pd.DataFrame:
    """`table` with `bin`, 0 to bins - 1: the target's equal-count bin of `distance` within its
    domain, over the targets whose distance lies in `within`; the rest get no bin."""
    x = table[distance].where(table[distance].between(*within))
    return table.assign(bin=x.groupby(table.domain).transform(lambda v: pd.qcut(v, bins, labels=False)))


def binned(x: np.ndarray, y: np.ndarray, bins: int = 4) -> tuple[list[float], list[float]]:
    """The means of x and y in each of `bins` equal-count bins of x."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    full = [b for b in range(bins) if (which == b).any()]
    return [x[which == b].mean() for b in full], [y[which == b].mean() for b in full]


DISTANCE_LABELS = {"above_top": "θ* above the strongest reference", "above_mean": "θ* above the mean reference"}


def sliding_curves(table: pd.DataFrame, distance: str = "above_top", k: int = 5, draws: int = 2000,
                   seed: int = 0) -> pd.DataFrame:
    """Per domain and protocol (the saturated-item prediction and the paper protocols), the mean
    squared error over each run of `k` targets adjacent in `distance` (the k nearest to one another,
    stepping one target at a time), placed at the run's mean distance, with the 95%
    percentile-bootstrap interval over the run's targets. The window is narrow where targets are
    dense and wide where they are sparse, and every point rests on k targets. Columns: domain,
    protocol, centre, first, last (the run's distances), n, mean, lo, hi."""
    protocols = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    rng = np.random.default_rng(seed)
    rows = []
    for domain in DOMAINS:
        for p in protocols:
            d = table[(table.domain == domain) & (table.protocol == p)].dropna(subset=["err"]).sort_values(distance)
            x, e = d[distance].to_numpy(float), d.err.to_numpy(float)
            for i in range(len(x) - k + 1):
                v = e[i:i + k]
                boot = v[rng.integers(0, k, (draws, k))].mean(axis=1)
                rows.append(dict(domain=domain, protocol=p, centre=float(x[i:i + k].mean()), first=float(x[i]),
                                 last=float(x[i + k - 1]), n=k, mean=float(v.mean()),
                                 lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def degradation_tests(table: pd.DataFrame, distance: str = "above_top") -> pd.DataFrame:
    """Per domain and protocol (the saturated-item prediction and the paper protocols), the
    least-squares slope of squared error on `distance` over the domain's targets and the two-sided
    t-test (n - 2 degrees of freedom) that it is 0; a positive slope: the protocol's error grows as
    targets sit further above the references. Targets share their task's references, so p is optimistic.
    Columns: domain, protocol, n, slope, se, p."""
    protocols = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    rows = []
    for domain in DOMAINS:
        for p in protocols:
            d = table[(table.domain == domain) & (table.protocol == p)].dropna(subset=["err", distance])
            fit = scipy_stats.linregress(d[distance], d.err)
            rows.append(dict(domain=domain, protocol=p, n=len(d), slope=fit.slope, se=fit.stderr, p=fit.pvalue))
    return pd.DataFrame(rows)


def bin_boxes(table: pd.DataFrame, distance: str = "above_top", draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per domain, distance bin (with_bins) and series (the saturated-item prediction and the paper
    protocols): the bin's mean distance, its targets, the mean squared error over them and its 95%
    percentile-bootstrap interval. Columns: domain, bin, protocol, x, n, mean, lo, hi."""
    series = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    rng = np.random.default_rng(seed)
    rows = []
    for (domain, b, p), g in table[table.protocol.isin(series)].dropna(subset=["bin"]).groupby(["domain", "bin", "protocol"]):
        v = g.err.to_numpy(float)
        boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(axis=1)
        rows.append(dict(domain=domain, bin=b, protocol=p, x=float(g[distance].mean()), n=len(v), mean=float(v.mean()),
                         lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def fit_band(x: np.ndarray, y: np.ndarray, xs: np.ndarray, degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The least-squares polynomial of `degree` through (x, y) at `xs`, and its 95% confidence half-width."""
    X, Xs = np.vander(x, degree + 1), np.vander(xs, degree + 1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid, dof = y - X @ beta, len(x) - degree - 1
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    return Xs @ beta, scipy_stats.t.ppf(0.975, dof) * np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))


def reasoning_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The 2025-I math and science targets with a `group` column, "reasoning" or "non-reasoning"
    (REASONING_2025_I), and the signed error. Their references, the 2024-II roster, are all
    non-reasoning models."""
    t = table[(table.semester == "2025-I") & table.domain.isin(["math", "science"])]
    group = np.where(t.model.isin(REASONING_2025_I), "reasoning", "non-reasoning")
    return t.assign(group=group, signed=t.theta_hat - t.theta)


def reasoning_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Per domain, protocol and group: n, mse, baseline_mse and the prediction score from the
    group's pooled squared errors. Five reasoning and three non-reasoning targets, so no test."""
    out = []
    for (domain, protocol, group), g in with_baseline(reasoning_rows(table)).groupby(["domain", "protocol", "group"]):
        mse, base = float(g.err.mean()), float(g.base.mean())
        out.append(dict(domain=domain, protocol=protocol, group=group, n=len(g), mse=mse,
                        baseline_mse=base, score=float(score(mse, base))))
    return pd.DataFrame(out)


def reasoning_model_scores(table: pd.DataFrame) -> pd.DataFrame:
    """One prediction score per 2025-I target and protocol, its squared errors pooled over math
    and science as figure one pools over the four domains: model, group, protocol, score."""
    t = with_baseline(reasoning_rows(table)).dropna(subset=["base"])
    pooled = t.groupby(["model", "group", "protocol"])[["err", "base"]].mean().reset_index()
    return pooled.assign(score=score(pooled.err, pooled.base))


def release_dates() -> dict[str, pd.Timestamp]:
    """Each model's release date from configs/models, keyed by its data-dir shortname (the config
    name's last segment, lowercased) and by the config's file stem; a date range keeps its start."""
    import yaml
    from pathlib import Path
    out = {}
    for path in Path("configs/models").glob("*/*.yml"):
        config = yaml.safe_load(path.read_text()) or {}
        if config.get("release_date"):
            date = pd.Timestamp(str(config["release_date"]).split("/")[0])
            out[path.stem] = date
            if "name" in config:
                out[config["name"].split("/")[-1].lower()] = date
    return out


def prediction_example(forecasts: pd.DataFrame, variants: dict[str, str], domain: str, semester: str, model: str,
                       protocols: list[str]) -> tuple[dict[str, tuple[float, float]], pd.Series, pd.DataFrame]:
    """One target's forecast under `protocols` and the saturated-item prediction, as the Student t
    distributions of θ* the log likelihood ratio compares (likelihood_ratio.log_p). Returns per rule its
    (forecast, spread); the task's references' θ*; and per rule its forecast, spread, squared error, log LR
    and judge. Judge protocols use the task's best judge on references (forecast and spread from
    `forecasts`); feature protocols are refitted on the references, so SMALL_JUDGE_DATA and
    SMALL_JUDGE_EMBEDDINGS must be set."""
    from analysis import loading
    from analysis.challenges import likelihood_ratio as lr
    from protocols.embeddings import EmbeddingRegression
    from analysis.stats import core_condition
    from protocols import PROTOCOLS
    task = loading.tasks(loading.domain(domain), [semester])[0]
    refs, truth, df = task.references.theta, float(task.truth[model]), len(task.references.theta) - 1
    judged = core_condition(forecasts).merge(lr.chosen_judges(forecasts, variants),
                                             on=["domain", "semester", "protocol", "model", "judge"])
    spreads = {BASELINE: (float(refs.mean()), float(np.sqrt(lr.loo_mse(lr.saturated, refs))))}
    judges = {BASELINE: "-"}
    for p in protocols:
        if p in ("pairwise", "direct"):
            row = judged[(judged.domain == domain) & (judged.semester == semester) & (judged.protocol == p)
                         & (judged.variant == variants[p]) & (judged.model == model)].iloc[0]
            spreads[p], judges[p] = (float(row.theta_hat), float(np.sqrt(row.judge_loo_mse))), row.judge
        else:
            feature = PROTOCOLS[p] if p != "embedding" else EmbeddingRegression(variants["embedding"])
            rule = lr.feature_rule(feature.estimator, pd.concat([feature.get_features(task.references),
                                                                 feature.get_features(task.test)]))
            spreads[p], judges[p] = (float(rule(refs, [model]).iloc[0]), float(np.sqrt(lr.loo_mse(rule, refs)))), "-"
    log_p = {p: float(scipy_stats.t.logpdf(truth, df, c, s)) for p, (c, s) in spreads.items()}
    values = pd.DataFrame([dict(protocol=p, judge=judges[p], theta=truth, forecast=c, spread=s, squared_error=(c - truth) ** 2,
                                log_lr=log_p[p] - log_p[BASELINE], references=len(refs)) for p, (c, s) in spreads.items()])
    print(values.round(3).to_string(index=False))
    return spreads, refs, values
