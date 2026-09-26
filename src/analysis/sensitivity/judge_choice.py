"""Judge choice: how much the prediction error depends on which of the three candidate judges (the three strongest
references of a task) judges it (Figure 15, the Appendix D.3 medians), and the candidates ordered by reference
leave-one-model-out error (Table 8).
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
    """Every candidate judge's forecast of every target, core condition, in each judge protocol's reported variant,
    with its squared error `e`, the baseline's `e_base` (the references' mean theta*) and the judge's `rank` by theta*
    among the task's candidates (1 = strongest). Targets whose error ratio to the baseline is undefined are dropped.

    variants: protocol -> reported variant; its judge protocols (pairwise, direct) are the ones kept.
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
    """Appendix D.3: the median over the tasks of `semesters` of the range of pairwise MSE over the three judges,
    and of the gap from pairwise judging (mean MSE over judges) to absolute scoring and to the baseline.

    f: judge_errors' output.
    """
    f = f[f.semester.isin(semesters)]
    t = f.groupby(["task", "protocol", "judge"]).agg(mse=("e", "mean"), base=("e_base", "mean")).reset_index()
    spread = t.groupby(["task", "protocol"]).mse.agg(lambda x: x.max() - x.min()).unstack()
    mse, base = t.groupby(["task", "protocol"]).mse.mean().unstack(), t.groupby("task").base.first()
    return pd.Series({"pairwise judge range": spread.pairwise.median(), "absolute scoring - pairwise": (mse.direct - mse.pairwise).median(),
                      "baseline - pairwise": (base - mse.pairwise).median()})


def by_rank(f: pd.DataFrame, semesters: list[str]) -> pd.DataFrame:
    """One row per protocol x judge rank (column "1", "2", "3") x target: the target's squared error averaged over the
    domains it has (e), and their number (n_domains).

    f: judge_errors' output.
    """
    g = f[f.semester.isin(semesters)]
    ranked = g.assign(column=g["rank"].astype(str)).drop_duplicates(["task", "protocol", "column", "model", "judge"])
    return ranked.groupby(["protocol", "column", "model"], as_index=False).agg(e=("e", "mean"), n_domains=("domain", "nunique"))


def figure(f: pd.DataFrame, semesters: list[str], names: dict[str, str], colours: dict[str, str], path: str) -> pd.DataFrame:
    """Figure 15: squared error per target (averaged over its domains), the baseline then one column per judge rank
    within each judge protocol: small points, the mean as a large circle and its 95% t-interval as a box; log y-axis.
    Writes path.{pdf,png,svg}; returns the plotted numbers, one row per column: protocol, column, n, mean, half
    (the t-interval's half-width).

    names, colours: protocol -> label and colour, the baseline's ("easy_items") included.
    """
    t = by_rank(f, semesters)
    base = f[f.semester.isin(semesters)].drop_duplicates(["domain", "semester", "model"]).groupby("model").e_base.mean()
    judged = [p for p in ("pairwise", "direct") if p in set(f.protocol)]
    columns = [(stats.BASELINE, "-", "Baseline", base)] + [(p, str(k), label, t[(t.protocol == p) & (t.column == str(k))].set_index("model").e)
                                                           for p in judged for k, label in ((1, "Strongest ref."), (2, "2nd strongest"), (3, "3rd strongest"))]
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
            rows.append(dict(protocol=p, column=col, n=len(e), mean=m, half=h))
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
    """Table 8: per task and judge protocol, the three candidate judges ordered by reference leave-one-model-out error
    (core condition, reported variant). Columns semester, domain, protocol, Best, 2nd best, 3rd best."""
    core = stats.core_condition(forecasts)
    core = core[core.protocol.isin(stats.JUDGED) & (core.variant == core.protocol.map(variants))]
    loo = core.drop_duplicates(["domain", "semester", "protocol", "judge"]).sort_values("judge_loo_mse")
    return (loo.groupby(["semester", "domain", "protocol"]).judge.agg(lambda j: list(j)[:3]).apply(pd.Series)
            .rename(columns={0: "Best", 1: "2nd best", 2: "3rd best"}).reset_index())
