"""
Judge choice: each task has three candidate judges (its three strongest references), and this module measures how much
the prediction error of the Pairwise judge and of Absolute scoring depends on which of them judges. It also lists each
task's candidates ordered by their leave-one-model-out error on the references, the error that selects the judge.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from analysis import loading, stats
from analysis.plots import unclip_markers
from reference import tables

# the figure's look as the exploratory scripts' house style drew it (scripts/analyses/plots.py)
INK2, MUTED, GRID = "#33322f", "#4a4844", "#d3d2c9"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Arimo", "Liberation Sans"], "font.size": 8,
         "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
         "lines.linewidth": 1.6, "figure.dpi": 400, "savefig.dpi": 400}


def judge_errors(forecasts: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """
    Every candidate judge's prediction of every target, in the core condition and in each judge protocol's reported
    variant, with its squared error, the Baseline's squared error on the same target, and the judge's rank by theta*.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); uses domain,
            semester, protocol, variant, judge, condition, model, theta_hat and theta.
        variants: protocol -> the variant it is reported in; only its judge protocols (pairwise, direct) are kept.
    Outputs:
        DataFrame, one row per task, judge protocol, judge and target: forecasts.csv's columns plus e (squared error),
        e_base (the Baseline's squared error: the references' mean theta* from data/meta/<domain>/models.csv),
        judge_theta (the judge's theta* as a reference of the task), task ("<domain> <semester>") and rank (the
        judge's dense rank by that theta* among the task's candidates, 1 = strongest). Rows without e or e_base are
        dropped.
    """
    judged = [p for p in variants if p in stats.JUDGED]
    f = stats.core_condition(forecasts[forecasts.protocol.isin(judged)])
    f = f[f.variant == f.protocol.map(variants)].copy()
    base = {}
    for (d, s), _ in f.groupby(["domain", "semester"]):
        models = tables.load(loading.domain(d))[0]
        refs = models.loc[models[f"{s}_role"] == "reference", f"{s}_theta"]
        base[(d, s)] = (refs.mean(), refs)
    f["e"] = (f.theta_hat - f.theta) ** 2
    f["e_base"] = [(base[(d, s)][0] - t) ** 2 for d, s, t in zip(f.domain, f.semester, f.theta)]
    f["judge_theta"] = [base[(d, s)][1].get(j, np.nan) for d, s, j in zip(f.domain, f.semester, f.judge)]
    f["task"] = f.domain + " " + f.semester
    f["rank"] = f.groupby(["task", "protocol"]).judge_theta.rank(ascending=False, method="dense").astype(int)
    return f.dropna(subset=["e", "e_base"])


def medians(f: pd.DataFrame, semesters: list[str]) -> pd.Series:
    """
    Summarises the judge spread over the tasks of `semesters`: the median over tasks of the range (largest minus
    smallest) of the Pairwise judge's MSE across its candidate judges, and of the gaps from the Pairwise judge's MSE
    (averaged over its candidates) to Absolute scoring's (likewise averaged) and to the Baseline's.

    Inputs:
        f: judge_errors' output; uses semester, task, protocol, judge, e and e_base.
        semesters: the semesters whose tasks are summarised.
    Outputs:
        Series with entries "pairwise judge range", "absolute scoring - pairwise" and "baseline - pairwise", each a
        median over tasks of an MSE range or MSE difference.
    """
    f = f[f.semester.isin(semesters)]
    t = f.groupby(["task", "protocol", "judge"]).agg(mse=("e", "mean"), base=("e_base", "mean")).reset_index()
    spread = t.groupby(["task", "protocol"]).mse.agg(lambda x: x.max() - x.min()).unstack()
    mse, base = t.groupby(["task", "protocol"]).mse.mean().unstack(), t.groupby("task").base.first()
    return pd.Series({"pairwise judge range": spread.pairwise.median(), "absolute scoring - pairwise": (mse.direct - mse.pairwise).median(),
                      "baseline - pairwise": (base - mse.pairwise).median()})


def by_rank(f: pd.DataFrame, semesters: list[str]) -> pd.DataFrame:
    """
    Each target's squared error with the candidate judge of each rank, and the Baseline's, averaged over the domains in
    which the model is a target.

    Inputs:
        f: judge_errors' output; uses semester, task, domain, protocol, rank, model, judge, e and e_base.
        semesters: the semesters whose targets are kept.
    Outputs:
        DataFrame with columns protocol ("easy_items" for the Baseline), judge_rank (the judge's rank as a string, "1"
        to "3"; "-" for the Baseline), model, e (the squared error averaged over the model's domains) and n_domains
        (their number); one row per protocol, rank and target, the Baseline's first.
    """
    g = f[f.semester.isin(semesters)]
    base = (g.drop_duplicates(["domain", "semester", "model"]).groupby("model", as_index=False)
            .agg(e=("e_base", "mean"), n_domains=("domain", "nunique")).assign(protocol=stats.BASELINE, judge_rank="-"))
    ranked = g.assign(judge_rank=g["rank"].astype(str)).drop_duplicates(["task", "protocol", "judge_rank", "model", "judge"])
    judged = ranked.groupby(["protocol", "judge_rank", "model"], as_index=False).agg(e=("e", "mean"), n_domains=("domain", "nunique"))
    return pd.concat([base, judged], ignore_index=True)[["protocol", "judge_rank", "model", "e", "n_domains"]]


def figure(f: pd.DataFrame, semesters: list[str], names: dict[str, str], colours: dict[str, str], path: str) -> pd.DataFrame:
    """
    Draws each target's squared error (averaged over its domains) in one column for the Baseline, then one column per
    judge rank within each judge protocol: a small point per target, the mean as a large circle and its 95% t-interval
    as a box, on a log y-axis.

    Inputs:
        f: judge_errors' output.
        semesters: the semesters whose targets are drawn.
        names: protocol -> legend label, the Baseline's ("easy_items") included.
        colours: protocol -> colour, the Baseline's included.
        path: the output path without extension.
    Outputs:
        writes path.pdf, path.png and path.svg; returns the plotted numbers as a DataFrame with columns protocol,
        judge_rank ("-" for the Baseline), n (targets), mean, lo and hi (the 95% t-interval) and half (its
        half-width), one row per drawn column.
    """
    t = by_rank(f, semesters)
    judged = [p for p in ("pairwise", "direct") if p in set(f.protocol)]
    columns = [(p, k, label, t[(t.protocol == p) & (t.judge_rank == k)].set_index("model").e)
               for p, k, label in [(stats.BASELINE, "-", "Baseline")]
               + [(p, str(k), label) for p in judged for k, label in ((1, "Strongest ref."), (2, "2nd strongest"), (3, "3rd strongest"))]]
    rows = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(2.6, 2.3))
        ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
            ax.spines[sp].set_position(("outward", 4))
        ax.tick_params(colors=MUTED, labelcolor=INK2)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 3)
        ax.set_yticks([0.01, 0.1, 1, 3])
        ax.set_yticklabels(["0.01", "0.1", "1", "3"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        rng = np.random.default_rng(0)  # jitter of the points
        xs, labels, x = [], [], 0
        for p, col, label, e in columns:
            if e.empty:
                continue
            if col == "1":
                x += 0.6
            m, h = e.mean(), st.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / np.sqrt(len(e))
            c = colours[p]
            ax.add_patch(plt.Rectangle((x - 0.3, m - h), 0.6, 2 * h, color=c, alpha=0.3, lw=0, zorder=2))
            ax.plot(x + rng.uniform(-0.2, 0.2, len(e)), e, "o", ms=2, mfc=c, mec="none", alpha=0.7, zorder=3)
            ax.plot(x, m, "o", ms=5, mfc=c, mec=c, zorder=4)
            xs.append(x), labels.append(label)
            rows.append(dict(protocol=p, judge_rank=col, n=len(e), mean=m, lo=m - h, hi=m + h, half=h))
            x += 1
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
        ax.set_ylabel("Squared error (θ²)")
        handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=colours[p], mec=colours[p], label=names[p]) for p in (stats.BASELINE, *judged)]
        leg = ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, handles=handles, columnspacing=0.8,
                        fontsize=7, handletextpad=0.4, labelspacing=0.3)
        for text in leg.get_texts():
            text.set_color(INK2)
        fig.tight_layout()
        unclip_markers(fig)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(f"{path}.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    return pd.DataFrame(rows)


def candidate_judges(forecasts: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """
    The three candidate judges of each task and judge protocol, ordered from the lowest to the highest
    leave-one-model-out error on the references (core condition, reported variant); barring ties, the first is the
    judge that stats.best_judges selects.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); uses domain,
            semester, protocol, variant, judge, condition and judge_loo_mse.
        variants: protocol -> the variant it is reported in.
    Outputs:
        DataFrame with columns semester, domain, protocol, Best, 2nd best, 3rd best (judge names); one row per task
        and judge protocol.
    """
    core = stats.core_condition(forecasts)
    core = core[core.protocol.isin(stats.JUDGED) & (core.variant == core.protocol.map(variants))]
    loo = core.drop_duplicates(["domain", "semester", "protocol", "judge"]).sort_values("judge_loo_mse")
    return (loo.groupby(["semester", "domain", "protocol"]).judge.agg(lambda j: list(j)[:3]).apply(pd.Series)
            .rename(columns={0: "Best", 1: "2nd best", 2: "3rd best"}).reset_index())
