"""
Reasoning targets predicted from non-reasoning references: the Pairwise judge on the Maths 2025-I targets, whose
references (the 2024-II roster) are all non-reasoning models. It compares the squared error of the non-reasoning
targets, the reasoning targets, and the reasoning targets judged with their reasoning removed, with Welch's and paired
t-tests between the groups.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.plots import ci95, stars, unclip_markers

REMOVED_CONDITION = "reasoning-hidden"  # the judge's run on transcripts with the reasoning removed
# non-reasoning targets grey circles, reasoning targets diamonds in the Pairwise judge's blues: its dark shade with
# the reasoning shown, its light shade with it removed
STYLE = {"non-reasoning": dict(marker="o", face="#9a9a9a", edge="#333333", mean="#222222"),
         "reasoning": dict(marker="D", face="#90afd5", edge="#2260ab", mean="#2260ab"),
         "removed": dict(marker="D", face="#b4d0f0", edge="#6aa1e2", mean="#6aa1e2")}
GROUPS = {"non-reasoning": "Non-reasoning targets", "reasoning": "Reasoning targets"}


def shown(errors: pd.DataFrame, domain: str, semester: str, protocol: str, reasoning_models: list[str]) -> pd.DataFrame:
    """
    One protocol's per-target errors on one task, each target labelled "reasoning" if it is in `reasoning_models`
    and "non-reasoning" otherwise.

    Inputs:
        errors: per-target errors from stats.target_errors (domain, semester, model, protocol, theta_hat, theta, err).
        domain: the task's domain key.
        semester: the task's semester.
        protocol: the protocol, e.g. "pairwise".
        reasoning_models: the targets whose transcripts carry reasoning (config.yaml generalisation.reasoning_targets).
    Outputs:
        the task's and protocol's rows of `errors` with a group column; one row per target.
    """
    t = errors[(errors.domain == domain) & (errors.semester == semester) & (errors.protocol == protocol)]
    return t.assign(group=np.where(t.model.isin(reasoning_models), "reasoning", "non-reasoning"))


def removed(forecasts: pd.DataFrame, domain: str, semester: str, protocol: str, variant: str, judge: str,
            reasoning_models: list[str]) -> pd.DataFrame:
    """
    The same judge's predictions of the reasoning targets from transcripts with the reasoning removed (its
    "reasoning-hidden" run), with their squared error.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); uses domain,
            semester, protocol, variant, condition, judge, model, theta_hat and theta.
        domain: the task's domain key.
        semester: the task's semester.
        protocol: the judge protocol, e.g. "pairwise".
        variant: the protocol's reported variant, e.g. "linear".
        judge: the judge used with the reasoning shown (the task's best judge on the references).
        reasoning_models: the reasoning targets.
    Outputs:
        the matching rows of `forecasts` with an err column; one row per reasoning target.
    """
    r = forecasts[(forecasts.domain == domain) & (forecasts.semester == semester) & (forecasts.protocol == protocol)
                  & (forecasts.variant == variant) & (forecasts.condition == REMOVED_CONDITION) & (forecasts.judge == judge)]
    r = r[r.model.isin(reasoning_models)]
    return r.assign(err=(r.theta_hat - r.theta) ** 2)


def group_tests(shown: pd.DataFrame, removed: pd.DataFrame) -> dict[str, dict]:
    """
    Two-sided t-tests of squared error: the reasoning targets, and the reasoning targets with their reasoning removed,
    each against the non-reasoning targets (Welch's, unequal variances); and the reasoning targets with the reasoning
    removed against the same targets with it shown (paired).

    Inputs:
        shown: the rows shown returns (model, group, err).
        removed: the rows removed returns (model, err).
    Outputs:
        dict test name -> {diff: the mean difference (first group minus second), t, df, p}; the names are "Reasoning
        minus non-reasoning targets", "Reasoning targets with reasoning removed minus non-reasoning targets" and
        "Reasoning targets, reasoning removed minus shown".
    """
    err = shown.set_index("model").err
    reasoning, other = err[(shown.group == "reasoning").to_numpy()], err[(shown.group == "non-reasoning").to_numpy()]
    cut = removed.set_index("model").err.reindex(reasoning.index)

    def welch(a: pd.Series, b: pd.Series) -> dict:
        t = scipy_stats.ttest_ind(a, b, equal_var=False)
        return dict(diff=float(a.mean() - b.mean()), t=float(t.statistic), df=float(t.df), p=float(t.pvalue))

    paired = scipy_stats.ttest_rel(cut, reasoning)
    return {"Reasoning minus non-reasoning targets": welch(reasoning, other),
            "Reasoning targets with reasoning removed minus non-reasoning targets": welch(cut, other),
            "Reasoning targets, reasoning removed minus shown": dict(
                diff=float((cut - reasoning).mean()), t=float(paired.statistic), df=float(paired.df), p=float(paired.pvalue))}


def reasoning_targets_figure(shown: pd.DataFrame, removed: pd.DataFrame, tests: dict[str, dict]) -> plt.Figure:
    """
    Draws two panels. On the left, predicted against true capability with the diagonal for a perfect prediction, each
    group labelled beside its points and the reasoning targets with their reasoning removed hollow; on the right, one
    slot per group (non-reasoning, reasoning, reasoning removed) with a small marker per target, a large one at the
    mean, a box over the mean's 95% t-interval floored at 0, and brackets marking the two Welch tests against the
    non-reasoning targets, the wider one higher.

    Inputs:
        shown: the rows shown returns (group, theta, theta_hat, err).
        removed: the rows removed returns (theta, theta_hat, err).
        tests: group_tests' dict; the brackets use the two Welch tests' p.
    Outputs:
        the matplotlib Figure.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.8, 1.5), layout="constrained", width_ratios=[1, 1.6])
    reach = max(2.0, float(np.ceil(np.abs(shown[["theta", "theta_hat"]].to_numpy()).max() * 2) / 2)) + 0.2
    left.plot([-reach, reach], [-reach, reach], color="grey", lw=0.8, zorder=1)
    rng = np.random.default_rng(0)
    for i, group in enumerate(GROUPS):
        g, st = shown[shown.group == group], STYLE[group]
        left.scatter(g.theta, g.theta_hat, s=22, marker=st["marker"], facecolor=st["face"], edgecolor=st["edge"],
                     linewidths=0.7, zorder=3, clip_on=False)
        name = GROUPS[group].replace(" ", "\n")
        if group == "non-reasoning":  # label to the left of the cloud, reasoning to the right
            left.text(g.theta.min() - 0.25, g.theta_hat.min(), name, ha="right", va="center", fontsize=5.5, color=st["edge"])
        else:
            left.text(g.theta.max() + 0.25, g.theta_hat.mean(), name, ha="left", va="center", fontsize=5.5, color=st["edge"])
        v = g.err.to_numpy(float)
        half = ci95(v)
        if np.isfinite(half):  # a box over the mean's 95% t-interval, floored at 0
            lo = max(v.mean() - half, 0.0)
            right.add_patch(plt.Rectangle((i - 0.25, lo), 0.5, v.mean() + half - lo, facecolor=st["face"],
                                          edgecolor="none", alpha=0.45, zorder=2))
        right.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, marker=st["marker"], facecolor=st["face"],
                      edgecolor=st["edge"], linewidths=0.5, zorder=3, clip_on=False)
        right.scatter([i], [v.mean()], s=34, marker=st["marker"], color=st["mean"], linewidths=0, zorder=5)
    st = STYLE["removed"]  # hollow, in its own colour
    left.scatter(removed.theta, removed.theta_hat, s=22, marker="D", facecolor="white", edgecolor=st["edge"],
                 linewidths=0.7, zorder=3, clip_on=False)
    left.text(removed.theta.max() + 0.25, removed.theta_hat.max(), "Reasoning\nremoved", ha="left", va="center",
              fontsize=5.5, color=st["edge"])
    v = removed.err.to_numpy(float)
    mean, half = v.mean(), ci95(v)
    lo = max(mean - half, 0.0)
    right.add_patch(plt.Rectangle((2 - 0.25, lo), 0.5, mean + half - lo, facecolor=st["face"], edgecolor="none",
                                  alpha=0.45, zorder=2))
    right.scatter(2 + rng.uniform(-0.12, 0.12, len(v)), v, s=9, marker="D", facecolor="white", edgecolor=st["edge"],
                  linewidths=0.5, zorder=3, clip_on=False)
    right.scatter([2], [mean], s=34, marker="D", facecolor="white", edgecolor=st["mean"], linewidths=1, zorder=5)
    left.set_xlim(-reach, reach)
    left.set_ylim(-reach, reach)
    left.set_aspect("equal")
    left.set_xticks([-1.5, 0, 1.5], ["−1.5", "0", "1.5"])
    left.set_yticks([-2, 0, 2])
    # mathtext for the hat: Arial's combining circumflex renders as a prime
    left.set_xlabel(r"True capability ($\theta^*$)")
    left.set_ylabel(r"Predicted capability ($\hat{\theta}$)")
    slots = ["Non-\nreasoning\ntargets", "Reasoning\ntargets", "Reasoning\nremoved"]
    right.set_xticks(range(len(slots)), slots)
    right.set_xlim(-0.6, len(slots) - 0.4)
    right.set_ylabel("Squared error")
    right.set_ylim(bottom=0)
    right.autoscale_view()  # the boxes are patches, which the y-axis does not follow on its own
    right.set_ylim(0, max(right.get_ylim()[1], *(p.get_y() + p.get_height() for p in right.patches)))
    brackets = [(0, 1, tests["Reasoning minus non-reasoning targets"]["p"]),
                (0, 2, tests["Reasoning targets with reasoning removed minus non-reasoning targets"]["p"])]
    top = right.get_ylim()[1]  # brackets above the data; the axis line stops at its last tick below them
    ticks = [t for t in right.get_yticks() if 0 <= t <= top]
    right.set_yticks(ticks)
    right.spines["left"].set_bounds(0, ticks[-1])
    for k, (a, b, p) in enumerate(sorted(brackets, key=lambda t: abs(t[1] - t[0]))):
        y = top * (1.04 + 0.14 * k)
        right.plot([a, a, b, b], [y - 0.03 * top, y, y, y - 0.03 * top], color="black", lw=0.6, clip_on=False)
        right.text((a + b) / 2, y, stars(p, "n.s."), ha="center", va="bottom", fontsize=7)
    right.set_ylim(0, top * (1.04 + 0.14 * len(brackets)))
    return unclip_markers(fig)


def target_table(shown: pd.DataFrame, removed: pd.DataFrame, judge: str) -> pd.DataFrame:
    """
    Every target's θ*, predicted θ and squared error, one row per point drawn: the non-reasoning and reasoning targets
    with the reasoning shown, and the reasoning targets with it removed.

    Inputs:
        shown: the rows shown returns (model, group, theta, theta_hat, err).
        removed: the rows removed returns (model, theta, theta_hat, err).
        judge: the Pairwise judge used in both conditions.
    Outputs:
        DataFrame with columns group ("non-reasoning", "reasoning" or "reasoning removed"), model, judge, theta,
        theta_hat, err; sorted by group and theta.
    """
    rows = pd.concat([shown, removed.assign(group="reasoning removed")], ignore_index=True)
    rows = rows.assign(judge=judge)[["group", "model", "judge", "theta", "theta_hat", "err"]]
    return rows.sort_values(["group", "theta"], ignore_index=True)  # the groups' names sort in the figure's order


def group_errors(targets: pd.DataFrame) -> pd.DataFrame:
    """
    Per group, the mean squared error over its targets and the mean's 95% t-interval with its lower end floored at 0,
    as the figure's boxes draw it.

    Inputs:
        targets: target_table rows (group, err).
    Outputs:
        DataFrame with columns group, n, mean, lo, hi; one row per group, in the figure's order.
    """
    rows = []
    for group, g in targets.groupby("group", sort=False):
        v = g.err.to_numpy(float)
        rows.append(dict(group=group, n=len(v), mean=v.mean(), lo=max(v.mean() - ci95(v), 0.0), hi=v.mean() + ci95(v)))
    return pd.DataFrame(rows)
