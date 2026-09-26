"""
Each target's log likelihood ratio against the Baseline, log p(theta* | protocol) - log p(theta* | Baseline), and the
one-sided t-tests on it. Each term is a distribution centred on a rule's prediction fitted on the references alone and
scaled by the square root of that rule's leave-one-model-out squared error on the references, either normal or a Student
t with n - 1 degrees of freedom for n references (SHAPES; the results use SHAPE); the Baseline's rule is the references'
mean theta*, so no target's theta* enters either term. A test's statistic is the mean over its units (targets, tasks, or
target models pooled over their domains) of each unit's mean log likelihood ratio, and its squared standard error is the
variance over units divided by their number, as in a t-test, plus the variance due to the reference set: per task, the
jackknife variance of the statistic when each reference is dropped in turn and everything refitted, summed over tasks.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import clone

from analysis import judges, stats

SHAPES = ("normal", "t")
SHAPE = "t"  # the shape the results use (user's choice, 2026-09-21); "normal" is kept alongside for comparison
Rule = Callable[[pd.Series, list[str]], pd.Series]  # (references' theta*, models) -> the models' theta-hat


TEST_TEXT = ("For each target the log likelihood ratio is log p(θ* | protocol) − log p(θ* | references): each term "
             "is a Student t distribution with n − 1 degrees of freedom (n = 7 or 8 references per task), centred "
             "on a forecast fitted on the task's references (for the references' term, their mean θ*) and scaled "
             "by that forecast's root-mean-square leave-one-model-out error on the references. The mean log "
             "likelihood ratio over units is tested against 0 with a two-sided t-test. "
             "Uncorrected for multiple comparisons; * p < .05, ** p < .01, *** p < .001.")


def p_text(p: float) -> str:
    """
    Formats a p-value for a caption: "p < .001" below 0.001, else three decimals without the leading zero ("p = .012").

    Inputs:
        p: the p-value.
    Outputs:
        the text; "p not computed" when p is NaN or infinite.
    """
    if not np.isfinite(p):
        return "p not computed"
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    """
    Formats a number for a caption, with a true minus sign (U+2212) in place of the hyphen.

    Inputs:
        x: the number.
        digits: decimals shown.
        sign: if True, a number that is not negative gets a plus sign.
    Outputs:
        the text.
    """
    return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−")


def saturated(refs: pd.Series, models: list[str]) -> pd.Series:
    """
    The Baseline's prediction rule: every model is predicted the references' mean theta*.

    Inputs:
        refs: the references' theta*, indexed by model.
        models: the models to predict.
    Outputs:
        Series indexed by models, every value the mean of refs.
    """
    return pd.Series(float(refs.mean()), index=models)


def judge_rule(table: pd.DataFrame, protocol: str, variant: str) -> Rule:
    """
    Turns one judge run's verdicts or ratings into a prediction rule: given the references' theta*, the rule converts
    the table to theta-hat of the requested models (analysis.judges.pairwise_theta or direct_theta), so only the
    references passed in calibrate it, and verdicts or ratings of any other model never do.

    Inputs:
        table: the run's verdicts (pairwise: bench, item, model_left, model_right, pick_left, ...) or ratings (bench,
            model, item, score).
        protocol: "pairwise" for verdicts; any other value reads the table as ratings.
        variant: the conversion passed to analysis.judges, e.g. "linear" for verdicts or "raw" for ratings; no frontier
            prior is passed.
    Outputs:
        a Rule: (references' theta*, models) -> Series of theta-hat indexed by the models.
    """
    if protocol == "pairwise":
        return lambda refs, models: judges.pairwise_theta(table, refs, models, variant)
    return lambda refs, models: judges.direct_theta(table, refs, models, variant)


def feature_rule(estimator, x: pd.DataFrame) -> Rule:
    """
    Turns a regression on per-model features into a prediction rule, as TabularRegression.fit and predict do: given the
    references' theta*, a fresh clone of the estimator is fitted on the references' feature rows and predicts the
    requested models' rows. References with a nonfinite feature or theta* are left out of the fit, and requested models
    with a nonfinite feature are predicted NaN.

    Inputs:
        estimator: an unfitted scikit-learn regressor, e.g. a ridge pipeline; cloned on every call.
        x: feature rows of the references and targets, indexed by model.
    Outputs:
        a Rule: (references' theta*, models) -> Series of theta-hat indexed by the models; all NaN when fewer than two
        references have finite features and theta*.
    """
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
    """
    The rule's leave-one-model-out mean squared error on the references: each reference predicted by the rule from the
    other references, the squared misses averaged over the references (NaN misses skipped).

    Inputs:
        rule: a prediction rule (judge_rule, feature_rule or saturated).
        refs: the references' theta*, indexed by model.
    Outputs:
        the mean squared error (NaN when every miss is NaN).
    """
    return float(np.nanmean([(rule(refs.drop(r), [r]).iloc[0] - refs[r]) ** 2 for r in refs.index]))


def log_p(rule: Rule, refs: pd.Series, truth: pd.Series) -> dict[str, np.ndarray]:
    """
    Each target's log density of its theta* under the rule's predictive distribution: centred on the rule's prediction
    from refs and scaled by the square root of the rule's leave-one-model-out error on refs, in both shapes.

    Inputs:
        rule: a prediction rule.
        refs: the references' theta*, indexed by model.
        truth: the targets' theta*, indexed by target.
    Outputs:
        dict shape -> array of log densities in the order of truth: "normal" (a normal distribution) and "t" (a Student
        t with len(refs) - 1 degrees of freedom).
    """
    centre = rule(refs, list(truth.index)).reindex(truth.index).to_numpy(float)
    x, scale = truth.to_numpy(float), np.sqrt(loo_mse(rule, refs))
    return {"normal": scipy_stats.norm.logpdf(x, centre, scale), "t": scipy_stats.t.logpdf(x, len(refs) - 1, centre, scale)}


def log_lr(rule: Rule, refs: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """
    Each target's log likelihood ratio, log p(theta* | rule) - log p(theta* | Baseline), with both distributions fitted
    on refs alone.

    Inputs:
        rule: the protocol's prediction rule.
        refs: the references' theta*, indexed by model.
        truth: the targets' theta*, indexed by target.
    Outputs:
        DataFrame indexed by target with one column per shape ("normal", "t").
    """
    protocol, references = log_p(rule, refs, truth), log_p(saturated, refs, truth)
    return pd.DataFrame({k: protocol[k] - references[k] for k in SHAPES}, index=truth.index)


def task_log_p(rule: Rule, refs: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """
    log p(theta* | rule) in the shape SHAPE for every target of one task, on the full reference set and with each
    reference dropped in turn and the rule refitted on the rest.

    Inputs:
        rule: a prediction rule.
        refs: the references' theta*, indexed by model.
        truth: the targets' theta*, indexed by target.
    Outputs:
        DataFrame with the targets as columns and one row per reference set, indexed by the reference dropped ("-" for
        the full set).
    """
    return pd.DataFrame({dropped: log_p(rule, subset, truth)[SHAPE]
                         for dropped, subset in [("-", refs), *[(r, refs.drop(r)) for r in refs.index]]},
                        index=truth.index).T


def difference_test(a: pd.DataFrame, b: pd.DataFrame, paired: bool = False) -> dict:
    """
    Tests whether one rule's mean log density over its targets exceeds another's, within one task: the difference of the
    two means on the full reference set, with a squared standard error that adds the targets' part to the jackknife
    variance of the difference over the dropped references.

    Inputs:
        a: task_log_p table of the first rule (rows: "-" and each dropped reference; columns: targets).
        b: task_log_p table of the second rule, for the same task.
        paired: True when a and b have the same targets in the same order: the targets' part is then the variance of the
            per-target differences over their number, with n - 1 degrees of freedom; otherwise Welch's variance and
            degrees of freedom.
    Outputs:
        dict with diff (mean of a minus mean of b), se, df and p (the one-sided upper-tail p of diff / se under a t
        distribution with df degrees of freedom; degenerate_p(diff) when se is 0).
    """
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
    p = float(scipy_stats.t.sf(d / se, df)) if se > 0 else degenerate_p(d)
    return dict(diff=d, se=se, df=float(df), p=p)


def task_llr(rule: Rule, refs: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """
    Each target's log likelihood ratio on the full reference set and with each reference dropped in turn, everything
    (the predictions and the scales of both distributions) refitted on the remaining references.

    Inputs:
        rule: the protocol's prediction rule.
        refs: the references' theta*, indexed by model.
        truth: the targets' theta*, indexed by target.
    Outputs:
        DataFrame with columns model, shape, llr, dropped; one row per target, shape and reference set, dropped being
        "-" for the full set and otherwise the reference left out.
    """
    frames = []
    for dropped, subset in [("-", refs), *[(r, refs.drop(r)) for r in refs.index]]:
        wide = log_lr(rule, subset, truth).rename_axis("model").reset_index()
        frames.append(wide.melt(id_vars="model", var_name="shape", value_name="llr").assign(dropped=dropped))
    return pd.concat(frames, ignore_index=True)


TARGET = ("domain", "semester", "model")  # one target of one task


def degenerate_p(effect: float) -> float:
    """
    The upper-tail p of an effect whose standard error is 0: 0 above zero, 1 below, 0.5 at exactly zero (so the
    two-sided p, 2 min(p, 1 - p), is 1 for no effect rather than 0).

    Inputs:
        effect: the effect (a mean, a difference or a slope).
    Outputs:
        the p: 0.0, 0.5 or 1.0.
    """
    return 0.0 if effect > 0 else 1.0 if effect < 0 else 0.5


def unit_test(rows: pd.DataFrame, unit: tuple[str, ...] = TARGET) -> dict:
    """
    Tests whether a protocol's mean log likelihood ratio is above 0. The statistic is the mean over units of each unit's
    mean log likelihood ratio on the full reference set; its squared standard error is the variance over units divided
    by their number, plus, per task, the jackknife variance of the statistic over that task's dropped references.

    Inputs:
        rows: one protocol's log likelihood ratio rows (one shape and one judge per target) for every reference set;
            uses domain, semester, model, dropped and llr.
        unit: the columns that name a replication unit: a target by default (TARGET), ("semester", "model") for a target
            model pooled over its domains, ("domain", "semester") for a task.
    Outputs:
        dict with n (units), llr (the statistic), se and p (the one-sided upper-tail p, t with n - 1 degrees of
        freedom). se and p are NaN with fewer than two units; when se is 0, p is degenerate_p(llr).
    """
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
    if se == 0:  # every log LR identical, e.g. all 0 for a protocol that predicts the references' mean
        return dict(n=n, llr=mean, se=se, p=degenerate_p(mean))
    return dict(n=n, llr=mean, se=se, p=float(scipy_stats.t.sf(mean / se, n - 1)))


def group_tests(llr: pd.DataFrame, groups: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    unit_test for each (domain, group, protocol), with a table assigning targets to groups; targets without a group are
    left out.

    Inputs:
        llr: log likelihood ratio rows (one shape and judge per target); uses domain, semester, model, protocol, dropped
            and llr.
        groups: one row per target: domain, semester, model and the group column.
        group: the name of the group column.
    Outputs:
        DataFrame with columns domain, the group column, protocol, n, llr, se, p (one-sided upper-tail p); one row per
        domain, group and protocol.
    """
    keyed = llr.merge(groups[["domain", "semester", "model", group]].drop_duplicates(), on=["domain", "semester", "model"])
    return pd.DataFrame([dict(domain=d, **{group: g_}, protocol=p, **unit_test(rows))
                         for (d, g_, p), rows in keyed.dropna(subset=[group]).groupby(["domain", group, "protocol"])])


def task_tests(llr: pd.DataFrame) -> pd.DataFrame:
    """
    unit_test for each (domain, semester, protocol), with the task's targets as the units.

    Inputs:
        llr: log likelihood ratio rows (one shape and judge per target); uses domain, semester, model, protocol, dropped
            and llr.
    Outputs:
        DataFrame with columns domain, semester, protocol, n, llr, se, p (one-sided upper-tail p); one row per task and
        protocol.
    """
    return pd.DataFrame([dict(domain=d, semester=s, protocol=p, **unit_test(rows))
                         for (d, s, p), rows in llr.groupby(["domain", "semester", "protocol"])])


def pooled_tests(llr: pd.DataFrame, unit: tuple[str, ...] = TARGET) -> pd.DataFrame:
    """
    unit_test for each protocol over every row given; the caller filters the rows first.

    Inputs:
        llr: log likelihood ratio rows (one shape and judge per target); uses protocol and the columns unit_test uses.
        unit: the replication unit, as in unit_test.
    Outputs:
        DataFrame with columns protocol, n, llr, se, p (one-sided upper-tail p); one row per protocol.
    """
    return pd.DataFrame([dict(protocol=p, **unit_test(rows, unit)) for p, rows in llr.groupby("protocol")])


def slope_test(rows: pd.DataFrame, x: pd.Series) -> dict:
    """
    Tests whether a protocol's log likelihood ratio rises with a target's distance, that is whether the protocol gains
    on the Baseline's distribution as targets move away from the references. The statistic is the least-squares slope of
    the log likelihood ratio on the distance over the targets; its squared standard error is the regression's plus, per
    task, the jackknife variance of the slope over that task's dropped references.

    Inputs:
        rows: one protocol's log likelihood ratio rows for one domain (one shape and judge per target), for every
            reference set; uses domain, semester, model, dropped and llr.
        x: the distance, indexed by (domain, semester, model).
    Outputs:
        dict with n (targets), slope, se and p (the one-sided upper-tail p, t with n - 2 degrees of freedom); slope, se
        and p are NaN with fewer than three targets.
    """
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
    p = float(scipy_stats.t.sf(fit.slope / se, len(base) - 2)) if se > 0 else degenerate_p(fit.slope)
    return dict(n=len(base), slope=float(fit.slope), se=se, p=p)


def slope_tests(llr: pd.DataFrame, table: pd.DataFrame, distance: str) -> pd.DataFrame:
    """
    slope_test for each (domain, protocol), with the distance read from a table of targets; only targets in the table
    are kept.

    Inputs:
        llr: log likelihood ratio rows (one shape and judge per target); uses domain, semester, model, protocol, dropped
            and llr.
        table: one row per target: domain, semester, model and the distance column.
        distance: the name of the distance column.
    Outputs:
        DataFrame with columns domain, protocol, n, slope, se, p (one-sided upper-tail p); one row per domain and
        protocol.
    """
    x = table.drop_duplicates(list(TARGET)).set_index(list(TARGET))[distance]
    keyed = llr.merge(table[list(TARGET)].drop_duplicates(), on=list(TARGET))
    return pd.DataFrame([dict(domain=d, protocol=p, **slope_test(rows, x)) for (d, p), rows in keyed.groupby(["domain", "protocol"])])


def chosen_judges(forecasts: pd.DataFrame, variants: dict[str, str], other_family: bool = False) -> pd.DataFrame:
    """
    The judge each target's pairwise and direct predictions come from, among the core-condition runs in the reported
    variant: by default the task's best judge on references (the lowest leave-one-model-out error on the references, as
    stats.best_judges picks); with other_family, per target the best among the judges of another developer family than
    the target (as stats.best_other_family), so no model is judged by its own family.

    Inputs:
        forecasts: the forecasts table (analysis/forecasts/forecasts.csv); uses domain, semester, protocol, variant,
            judge, condition, model and judge_loo_mse.
        variants: protocol -> reported variant; only the judge protocols named here (pairwise, direct) are kept.
        other_family: if True, pick per target among the judges of another family.
    Outputs:
        DataFrame with columns domain, semester, protocol, model, judge; one row per target and judge protocol, a tie
        keeping the judge that comes first in forecasts. With other_family, a target with no such judge has no row.
    """
    family = stats.family
    core = stats.core_condition(forecasts)
    judged = [p for p in variants if p in ("pairwise", "direct")]
    core = core[core.protocol.isin(judged) & np.array([variants.get(p) == v for p, v in zip(core.protocol, core.variant)])]
    key = ["domain", "semester", "protocol"]
    if other_family:
        core, key = core[family(core.judge) != family(core.model)], key + ["model"]
    best = core[core.groupby(key).judge_loo_mse.transform("min") == core.judge_loo_mse]
    return best[["domain", "semester", "protocol", "model", "judge"]].drop_duplicates(["domain", "semester", "protocol", "model"])


def select(llr: pd.DataFrame, judges_: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the rows the tests use: every row of the protocols without a judge (judge "-"), and for the judge protocols
    the rows of each target's chosen judge.

    Inputs:
        llr: rows with domain, semester, protocol, model and judge columns, e.g.
            analysis/forecasts/likelihood_ratio.csv.
        judges_: chosen_judges' output (domain, semester, protocol, model, judge).
    Outputs:
        DataFrame with the columns of llr: the judge "-" rows, then the chosen judges' rows.
    """
    chosen = llr.merge(judges_, on=["domain", "semester", "protocol", "model", "judge"])
    return pd.concat([llr[llr.judge == "-"], chosen], ignore_index=True)


def demo() -> None:
    """
    Self-check on a toy task of eight references and three targets: a rule that knows the truth has positive log
    likelihood ratios, a significant unit_test, a positive slope_test and a significant difference_test against the
    Baseline; the Baseline's own rule scores exactly 0; a unit_test with one unit returns that unit's mean; and a
    difference_test of a rule against itself gives a difference of 0. Runs when the module is executed.

    Inputs:
        none.
    Outputs:
        none; raises AssertionError if a check fails.
    """
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
