"""Log likelihood ratio of each protocol's forecast against the references' own distribution.

For one target, log LR = log p(theta* | protocol) - log p(theta* | references). Each term is a
distribution centred on a forecast fitted on the references and scaled by the square root of
that forecast rule's leave-one-model-out squared error on the references, in one of two SHAPES:
"normal", or "t", a Student t with n - 1 degrees of freedom (n references) that allows for the
scale being estimated from few models. The references' rule is their mean theta* (the saturated-item
prediction), which ignores transcripts. Both rules see only the references, so no target label
enters either term.

A unit's statistic (a distance bin, a task, a model pooled over domains) is the mean log LR over
its targets. Its uncertainty has two parts: the targets, whose spread enters as in a t-test, and
the reference set, from dropping each reference in turn and recomputing everything
(leave-one-model-out over the references, per task; the tasks' parts add).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import clone

from analysis import judges, stats

SHAPES = ("normal", "t")
SHAPE = "t"  # the one the figures use (user's choice, 2026-09-21); "normal" is kept alongside for comparison
Rule = Callable[[pd.Series, list[str]], pd.Series]  # (references' theta*, models) -> the models' theta-hat


TEST_TEXT = ("For each target the log likelihood ratio is log p(θ* | protocol) − log p(θ* | references): each term "
             "is a Student t distribution with n − 1 degrees of freedom (n = 7 or 8 references per task), centred "
             "on a forecast fitted on the task's references (for the references' term, their mean θ*) and scaled "
             "by that forecast's root-mean-square leave-one-model-out error on the references. The mean log "
             "likelihood ratio over units is tested against 0 with a two-sided t-test. "
             "Uncorrected for multiple comparisons; * p < .05, ** p < .01, *** p < .001.")


def p_text(p: float) -> str:
    """A p-value as the captions print it: "p < .001", or three decimals without the leading zero."""
    if not np.isfinite(p):
        return "p not computed"
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    """A number as the captions print it, with a true minus sign (and a plus sign when `sign`)."""
    return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−")


def saturated(refs: pd.Series, models: list[str]) -> pd.Series:
    """The saturated-item prediction: the references' mean theta* for every model."""
    return pd.Series(float(refs.mean()), index=models)


def judge_rule(table: pd.DataFrame, protocol: str, variant: str) -> Rule:
    """A judge run's conversion (analysis.judges) as a rule; verdicts or ratings of a model not
    in `refs` never calibrate it."""
    if protocol == "pairwise":
        return lambda refs, models: judges.pairwise_theta(table, refs, models, variant)
    return lambda refs, models: judges.direct_theta(table, refs, models, variant)


def feature_rule(estimator, x: pd.DataFrame) -> Rule:
    """A fresh clone of `estimator` fitted on the references' rows of `x` (feature rows of
    references and targets), as TabularRegression.fit and predict do: rows with a nonfinite
    value are left out of the fit and predicted NaN."""
    def rule(refs: pd.Series, models: list[str]) -> pd.Series:
        out = pd.Series(np.nan, index=models)
        xr = x.reindex(refs.index)
        known = np.isfinite(xr).all(axis=1) & np.isfinite(refs)
        if known.sum() < 2:
            return out
        fitted = clone(estimator).fit(xr[known], refs[known])
        xm = x.reindex(models)
        has = np.isfinite(xm).all(axis=1).to_numpy()
        out[has] = fitted.predict(xm[has])
        return out
    return rule


def loo_mse(rule: Rule, refs: pd.Series) -> float:
    """The rule's mean squared miss on each reference, forecast from the other references."""
    return float(np.nanmean([(rule(refs.drop(r), [r]).iloc[0] - refs[r]) ** 2 for r in refs.index]))


def log_p(rule: Rule, refs: pd.Series, truth: pd.Series) -> dict[str, np.ndarray]:
    """Per target, log p(theta* | rule): centred on the rule's forecast from `refs`, scaled by its
    leave-one-model-out error on them, one array per shape in SHAPES."""
    centre = rule(refs, list(truth.index)).reindex(truth.index).to_numpy(float)
    x, scale = truth.to_numpy(float), np.sqrt(loo_mse(rule, refs))
    return {"normal": scipy_stats.norm.logpdf(x, centre, scale), "t": scipy_stats.t.logpdf(x, len(refs) - 1, centre, scale)}


def log_lr(rule: Rule, refs: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """Per target (truth: theta* indexed by target), log p(theta* | rule) - log p(theta* | references),
    one column per shape in SHAPES."""
    protocol, references = log_p(rule, refs, truth), log_p(saturated, refs, truth)
    return pd.DataFrame({k: protocol[k] - references[k] for k in SHAPES}, index=truth.index)


def task_log_p(rule: Rule, refs: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """log p(theta* | rule) in SHAPE, targets as columns, one row per reference set: "-" for the full
    set, else the reference dropped, with everything refitted on the rest."""
    return pd.DataFrame({dropped: log_p(rule, subset, truth)[SHAPE]
                         for dropped, subset in [("-", refs), *[(r, refs.drop(r)) for r in refs.index]]},
                        index=truth.index).T


def difference_test(a: pd.DataFrame, b: pd.DataFrame, paired: bool = False) -> dict:
    """Mean over `a`'s targets minus mean over `b`'s (task_log_p tables of one task), and the one-sided
    p that it is above 0. The squared standard error is the targets' part (paired: the variance of the
    per-target differences over n, n - 1 df; else Welch's, with its df) plus the leave-one-model-out
    variance of the difference over the references."""
    diff = a.mean(axis=1) - b.mean(axis=1)  # per reference set
    shifts = diff.drop("-").to_numpy()
    k = len(shifts)
    var_refs = (k - 1) / k * float(((shifts - shifts.mean()) ** 2).sum())
    x, y = a.loc["-"].to_numpy(float), b.loc["-"].to_numpy(float)
    if paired:
        v, df = np.var(x - y, ddof=1) / len(x), len(x) - 1
    else:
        vx, vy = np.var(x, ddof=1) / len(x), np.var(y, ddof=1) / len(y)
        v, df = vx + vy, (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    se, d = float(np.sqrt(v + var_refs)), float(diff["-"])
    p = float(scipy_stats.t.sf(d / se, df)) if se > 0 else (0.0 if d > 0 else 1.0)
    return dict(diff=d, se=se, df=float(df), p=p)


def task_llr(rule: Rule, refs: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """Rows (model, dropped, shape, llr): `dropped` "-" for the full reference set, else the
    reference left out of it, with everything (centres and spreads) refitted on the rest."""
    frames = []
    for dropped, subset in [("-", refs), *[(r, refs.drop(r)) for r in refs.index]]:
        wide = log_lr(rule, subset, truth).rename_axis("model").reset_index()
        frames.append(wide.melt(id_vars="model", var_name="shape", value_name="llr").assign(dropped=dropped))
    return pd.concat(frames, ignore_index=True)


TARGET = ("domain", "semester", "model")  # one target of one task


def unit_test(rows: pd.DataFrame, unit: tuple[str, ...] = TARGET) -> dict:
    """One protocol's llr rows (domain, semester, model, dropped, llr) for the targets of one unit
    of analysis: n (replication units), the mean over units of each unit's mean log LR, its
    standard error and the one-sided p that it is above 0 (t, n - 1 df). `unit` names the
    replication unit: a target by default, ("semester", "model") for a target model pooled over
    its domains as in figure one. The squared standard error is the units' variance over n plus,
    per task, the leave-one-model-out variance of the statistic over that task's references."""
    full = rows[rows.dropped == "-"].dropna(subset=["llr"])
    per_unit = full.groupby(list(unit)).llr.mean()
    n = len(per_unit)
    if n < 2:
        return dict(n=n, llr=float(per_unit.mean()) if n else np.nan, se=np.nan, p=np.nan)
    # each target's weight in the statistic: 1 / (units x targets in its unit)
    weight = (1 / (n * full.groupby(list(unit)).llr.transform("size"))).to_numpy()
    index = pd.MultiIndex.from_frame(full[list(TARGET)])
    base, weight = pd.Series(full.llr.to_numpy(), index=index).sort_index(), pd.Series(weight, index=index).sort_index()
    var_refs = 0.0
    for (domain, semester), g in rows[rows.dropped != "-"].groupby(["domain", "semester"]):
        if (domain, semester) not in base.index.droplevel("model"):
            continue
        wide = g.pivot_table(index="dropped", columns="model", values="llr")
        own = base.loc[(domain, semester)].reindex(wide.columns)
        shift = ((wide - own) * weight.loc[(domain, semester)].reindex(wide.columns)).sum(axis=1)  # the statistic's move
        k = len(shift)
        var_refs += (k - 1) / k * float(((shift - shift.mean()) ** 2).sum())
    mean = float(per_unit.mean())
    se = float(np.sqrt(per_unit.var(ddof=1) / n + var_refs))
    if se == 0:  # every log LR identical, e.g. all 0 for a protocol that forecasts the references' mean
        return dict(n=n, llr=mean, se=se, p=0.0 if mean > 0 else 1.0)
    return dict(n=n, llr=mean, se=se, p=float(scipy_stats.t.sf(mean / se, n - 1)))


def group_tests(llr: pd.DataFrame, groups: pd.DataFrame, group: str) -> pd.DataFrame:
    """unit_test per (domain, `group`, protocol), the units given by `groups` (domain, semester,
    model, `group`); targets without a group are left out."""
    keyed = llr.merge(groups[["domain", "semester", "model", group]].drop_duplicates(), on=["domain", "semester", "model"])
    return pd.DataFrame([dict(domain=d, **{group: g_}, protocol=p, **unit_test(rows))
                         for (d, g_, p), rows in keyed.dropna(subset=[group]).groupby(["domain", group, "protocol"])])


def task_tests(llr: pd.DataFrame) -> pd.DataFrame:
    """unit_test per (domain, semester, protocol): each task's targets."""
    return pd.DataFrame([dict(domain=d, semester=s, protocol=p, **unit_test(rows))
                         for (d, s, p), rows in llr.groupby(["domain", "semester", "protocol"])])


def pooled_tests(llr: pd.DataFrame, unit: tuple[str, ...] = TARGET) -> pd.DataFrame:
    """unit_test per protocol over every row of `llr` (filter it first)."""
    return pd.DataFrame([dict(protocol=p, **unit_test(rows, unit)) for p, rows in llr.groupby("protocol")])


def slope_test(rows: pd.DataFrame, x: pd.Series) -> dict:
    """One protocol's llr rows for one domain against `x` (a distance indexed by domain, semester,
    model): n, the OLS slope of the log LR on x, its standard error and the one-sided p that it is
    above 0 (the protocol gains on the references' distribution as targets move away; t, n - 2 df).
    The squared standard error is the regression's plus, per task, the leave-one-model-out variance
    of the slope over that task's references."""
    full = rows[rows.dropped == "-"].dropna(subset=["llr"])
    key = pd.MultiIndex.from_frame(full[list(TARGET)])
    xs, base = x.reindex(key).to_numpy(float), pd.Series(full.llr.to_numpy(float), index=key)
    if len(base) < 3:
        return dict(n=len(base), slope=np.nan, se=np.nan, p=np.nan)
    fit = scipy_stats.linregress(xs, base.to_numpy())
    var_refs = 0.0
    for (domain, semester), g in rows[rows.dropped != "-"].groupby(["domain", "semester"]):
        slopes = []
        for _, h in g.groupby("dropped"):
            moved = base.copy()
            index = pd.MultiIndex.from_frame(h[list(TARGET)])
            keep = index.isin(moved.index)
            moved.loc[index[keep]] = h.llr.to_numpy(float)[keep]
            slopes.append(scipy_stats.linregress(xs, moved.to_numpy()).slope)
        slopes = np.array(slopes)
        var_refs += (len(slopes) - 1) / len(slopes) * float(((slopes - slopes.mean()) ** 2).sum())
    se = float(np.sqrt(fit.stderr ** 2 + var_refs))
    p = float(scipy_stats.t.sf(fit.slope / se, len(base) - 2)) if se > 0 else (0.0 if fit.slope > 0 else 1.0)
    return dict(n=len(base), slope=float(fit.slope), se=se, p=p)


def slope_tests(llr: pd.DataFrame, table: pd.DataFrame, distance: str) -> pd.DataFrame:
    """slope_test per (domain, protocol), `distance` read from `table` (domain, semester, model, distance)."""
    x = table.drop_duplicates(list(TARGET)).set_index(list(TARGET))[distance]
    keyed = llr.merge(table[list(TARGET)].drop_duplicates(), on=list(TARGET))
    return pd.DataFrame([dict(domain=d, protocol=p, **slope_test(rows, x)) for (d, p), rows in keyed.groupby(["domain", "protocol"])])


def chosen_judges(forecasts: pd.DataFrame, variants: dict[str, str], other_family: bool = False) -> pd.DataFrame:
    """(domain, semester, protocol, model, judge): the judge each target's forecast comes from, in the
    core condition. By default the task's best judge on references (the results' choice); with
    `other_family`, the best among judges from another developer than the target
    (signal_glm.best_other_family), so no model is judged by its own lineage."""
    from analysis.challenges.signal_glm import family
    core = stats.core_condition(forecasts)
    judged = [p for p in variants if p in ("pairwise", "direct")]
    core = core[core.protocol.isin(judged) & np.array([variants.get(p) == v for p, v in zip(core.protocol, core.variant)])]
    key = ["domain", "semester", "protocol"]
    if other_family:
        core, key = core[family(core.judge) != family(core.model)], key + ["model"]
    best = core[core.groupby(key).judge_loo_mse.transform("min") == core.judge_loo_mse]
    return best[["domain", "semester", "protocol", "model", "judge"]].drop_duplicates(["domain", "semester", "protocol", "model"])


def select(llr: pd.DataFrame, judges_: pd.DataFrame) -> pd.DataFrame:
    """The transcript protocols' rows and, for judge protocols, each target's chosen judge's rows."""
    chosen = llr.merge(judges_, on=["domain", "semester", "protocol", "model", "judge"])
    return pd.concat([llr[llr.judge == "-"], chosen], ignore_index=True)


def demo() -> None:
    """Self-check on a toy task: a rule that knows the truth beats the references' distribution,
    and the references' own rule scores exactly 0."""
    refs = pd.Series(np.linspace(-1, 1, 8), index=[f"r{i}" for i in range(8)])
    truth = pd.Series([0.2, 1.5, 2.0], index=["a", "b", "c"])
    every = pd.concat([refs, truth])
    oracle = lambda r, models: every.reindex(models) + 0.05  # noqa: E731
    assert (log_lr(oracle, refs, truth) > 0).all().all()
    assert np.allclose(log_lr(saturated, refs, truth), 0)
    rows = task_llr(oracle, refs, truth).assign(domain="d", semester="s").query("shape == 'normal'")
    assert unit_test(rows)["p"] < 0.05
    assert np.isclose(unit_test(rows, ("semester",))["llr"], rows[rows.dropped == "-"].llr.mean())  # one unit: its mean
    x = pd.Series([0.2, 1.5, 2.0], index=pd.MultiIndex.from_tuples([("d", "s", m) for m in truth.index], names=list(TARGET)))
    assert slope_test(rows, x)["slope"] > 0  # the null's likelihood collapses further out, the oracle's does not
    close, far = task_log_p(oracle, refs, truth), task_log_p(saturated, refs, truth)
    assert difference_test(close, far, paired=True)["p"] < 0.05  # the oracle explains theta* better than the mean
    assert difference_test(close, close, paired=True)["diff"] == 0


if __name__ == "__main__":
    demo()
    print("ok")
