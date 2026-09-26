"""How much does the choice of judge model change the outcome of pairwise and direct judging?

    PYTHONPATH=src uv run python scripts/analyses/judge_choice.py

Reads analysis/results/forecasts.csv (scripts/analyses/results.py): one theta-hat per target x protocol x judge,
judges = the top-3 references of each semester task, core condition, pairwise "linear" and direct "raw". Per target
the prediction score S = -log2(e / e_base) against the reference-mean constant (e = squared error in theta); a
task's score is the mean over its targets. Writes analysis/judge/judge_choice.csv (one row per task x protocol x
judge) and judge_choice.{png,svg}, and prints:

  spread      per task: the score range across the three judges vs the pairwise-direct gap of the same judge
  variance    nested sums of squares of the per-target score: task | judge within task | target within judge
  forecast    per target: sd of theta-hat across judges vs the rmse of the mean forecast (theta units)
  selection   score when the judge is the best on reference leave-one-out (label-free), the oracle best,
              a random pick (mean over judges), the worst, and the strongest reference (top-1)
  capability  does the judge's own theta* predict its score within a task (slope, Spearman)
  agreement   correlation of the judges' forecasts and of their errors over the 8 targets of a task
  gap         relative strength: per target, the signed error (theta-hat - theta*) and the score against
              gap = theta*_target - theta*_judge (> 0: the target is stronger than its judge); judge_choice_gap.{png,svg}
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).parent))
from plots import INK2, MUTED, legend, save, style  # noqa: E402  (sets the Agg backend)

import matplotlib.pyplot as plt  # noqa: E402

from analysis.stats import core_condition  # noqa: E402

FORECASTS = Path("analysis/results/forecasts.csv")
OUT = Path("analysis/judge/judge_choice")
VARIANT = {"pairwise": "linear", "direct": "raw"}
RANK_COLOR = {1: "#1f63b6", 2: "#7f9cc4", 3: "#b8b5ab"}  # judge rank by theta* among the references


def references(domain: str, semester: str) -> pd.Series:
    """theta* of the semester's references, indexed by model shortname."""
    m = pd.read_csv(f"data/meta/{domain}/models.csv", index_col="model")
    return m.loc[m[f"{semester}_role"] == "reference", f"{semester}_theta"]


def load() -> pd.DataFrame:
    f = pd.read_csv(FORECASTS)
    f = core_condition(f[f.protocol.isin(VARIANT)])
    f = f[f.variant == f.protocol.map(VARIANT)].copy()
    base = {}
    for (d, s), _ in f.groupby(["domain", "semester"]):
        refs = references(d, s)
        base[(d, s)] = (refs.mean(), refs)
    f["e"] = (f.theta_hat - f.theta) ** 2
    f["e_base"] = [(base[(d, s)][0] - t) ** 2 for d, s, t in zip(f.domain, f.semester, f.theta)]
    f["score"] = -np.log2(f.e / f.e_base)
    f["judge_theta"] = [base[(d, s)][1].get(j, np.nan) for d, s, j in zip(f.domain, f.semester, f.judge)]
    f["task"] = f.domain + " " + f.semester
    # a judge's rank by theta* among the judges of the same task and protocol (1 = strongest)
    f["rank"] = f.groupby(["task", "protocol"]).judge_theta.rank(ascending=False, method="dense").astype(int)
    return f.dropna(subset=["score"])


def per_task(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby(["domain", "semester", "task", "protocol", "judge"], sort=False)
    t = g.agg(n=("score", "size"), score=("score", "mean"), mse=("e", "mean"), judge_theta=("judge_theta", "first"),
              rank=("rank", "first"), loo=("judge_loo_mse", "first")).reset_index()
    return t


def ci(x: pd.Series, d: int = 1) -> str:
    x = x.dropna()
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return f"{x.mean():.{d}f} ± {h:.{d}f} (n={len(x)})"


def nested_ss(f: pd.DataFrame) -> pd.Series:
    """Fractions of the total sum of squares of the per-target score: between tasks, between judges within a
    task, and between targets within a judge (residual)."""
    y = f.score
    task_mean = f.groupby("task").score.transform("mean")
    judge_mean = f.groupby(["task", "judge"]).score.transform("mean")
    ss = pd.Series({"task": ((task_mean - y.mean()) ** 2).sum(), "judge | task": ((judge_mean - task_mean) ** 2).sum(),
                    "target | judge": ((y - judge_mean) ** 2).sum()})
    return ss / ss.sum()


def selection(t: pd.DataFrame) -> pd.DataFrame:
    """Per task x protocol: score under every way of picking one judge."""
    rows = []
    for (task, p), g in t.groupby(["task", "protocol"], sort=False):
        rows.append(dict(task=task, protocol=p, n_judges=len(g), loo_pick=g.loc[g.loo.idxmin(), "score"],
                         oracle=g.score.max(), random=g.score.mean(), worst=g.score.min(),
                         top1=g.loc[g["rank"].idxmin(), "score"], range=g.score.max() - g.score.min()))
    return pd.DataFrame(rows)


def agreement(f: pd.DataFrame) -> pd.DataFrame:
    """Per task x protocol x judge pair: Pearson r of theta-hat and of the error over the targets."""
    rows = []
    for (task, p), g in f.groupby(["task", "protocol"], sort=False):
        wide = g.pivot(index="model", columns="judge", values="theta_hat")
        err = wide.sub(g.groupby("model").theta.first(), axis=0)
        js = list(wide.columns)
        for i in range(len(js)):
            for j in range(i + 1, len(js)):
                rows.append(dict(task=task, protocol=p, a=js[i], b=js[j], r_forecast=wide[js[i]].corr(wide[js[j]]),
                                 r_error=err[js[i]].corr(err[js[j]])))
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, out: Path):
    """Two rows (pairwise, direct): per task the score of each judge, colour = its theta* rank, ring = the judge
    the reference leave-one-out error would pick; a thin line spans the judges' range."""
    tasks = list(dict.fromkeys(t.task))
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True, sharey=True)
    for ax, p in zip(axes, VARIANT):
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 4))
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for x, task in enumerate(tasks):
            g = t[(t.task == task) & (t.protocol == p)]
            if g.empty:
                continue
            ax.plot([x, x], [g.score.min(), g.score.max()], color=MUTED, lw=0.8, zorder=2)
            pick = g.loo.idxmin()
            for i, r in g.iterrows():
                ax.plot(x, r.score, "o", ms=4, mfc=RANK_COLOR[r["rank"]], mec=INK2 if i == pick else RANK_COLOR[r["rank"]],
                        mew=0.9 if i == pick else 0.4, zorder=3)
        ax.text(0.0, 1.0, p, transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="bottom")
    axes[0].set_ylim(-2, 10)
    axes[0].set_yticks([-2, 0, 5, 10])
    fig.supylabel("Prediction score", fontsize=8, x=0.02)
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels([s.replace("science", "sci").replace("coding", "code") for s in tasks], rotation=90, ha="center")
    axes[1].set_xlim(-0.6, len(tasks) - 0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=c, mec=c, label=f"Judge θ* rank {k}") for k, c in RANK_COLOR.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=INK2, mew=0.9, label="Picked by reference LOO"))
    legend(axes[0], loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, handles=handles, columnspacing=1.0, fontsize=7)
    save(fig, out)


def gap_figure(f: pd.DataFrame, out: Path):
    """Per protocol: signed error of every target forecast against the target-minus-judge theta* gap, with the
    mean signed error in gap bins of 0.5 (bins with < 5 forecasts skipped)."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6), sharex=True, sharey=True)
    for ax, p in zip(axes, VARIANT):
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 4))
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        g = f[f.protocol == p]
        gap, err = g.theta - g.judge_theta, g.theta_hat - g.theta
        ax.plot(gap, err, "o", ms=2, mfc="#b8b5ab", mec="none", zorder=2)
        bins = np.arange(np.floor(gap.min() * 2) / 2, gap.max() + 0.5, 0.5)
        b = pd.cut(gap, bins)
        m = err.groupby(b, observed=True).agg(["mean", "size"])
        m = m[m["size"] >= 5]
        ax.plot([iv.mid for iv in m.index], m["mean"], "-o", color="#1f63b6", lw=1, ms=3, zorder=3)
        ax.text(0.0, 1.0, p, transform=ax.transAxes, fontsize=7, color=MUTED, ha="left", va="bottom")
        ax.set_xlabel("Target − judge θ*")
    axes[0].set_ylabel("θ̂ − θ*")
    axes[0].set_xlim(-2, 3)
    axes[0].set_xticks([-2, -1, 0, 1, 2, 3])
    axes[0].set_ylim(-4, 2)
    axes[0].set_yticks([-4, -2, 0, 2])
    handles = [plt.Line2D([], [], marker="o", ls="", ms=2, mfc="#b8b5ab", mec="none", label="One target forecast"),
               plt.Line2D([], [], marker="o", ls="-", ms=3, lw=1, color="#1f63b6", label="Mean per 0.5 θ bin")]
    fig.tight_layout()
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7,
                     columnspacing=1.0, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out)


def gap_stats(f: pd.DataFrame, p: str) -> str:
    g = f[f.protocol == p]
    gap, err = g.theta - g.judge_theta, g.theta_hat - g.theta
    r_err, r_score = st.spearmanr(gap, err)[0], st.spearmanr(gap, g.score)[0]
    slope = np.polyfit(gap, err, 1)[0]
    # replication unit for the split: judge x task cell, its mean over the targets on each side of the judge
    side = (g.assign(side=np.where(gap > 0, "stronger", "weaker"), err=err)
            .groupby(["task", "judge", "side"]).agg(err=("err", "mean"), score=("score", "mean")))
    s, w = side.xs("stronger", level="side"), side.xs("weaker", level="side")
    # the pooled gap mixes two things: which target (a strong target is underestimated by every judge: cohort shift)
    # and which judge. Within a target only the judge varies: centre error and judge theta* per (task, target).
    key = ["task", "model"]
    within = g.assign(err=err).assign(de=lambda x: x.err - x.groupby(key).err.transform("mean"),
                                       dj=lambda x: x.judge_theta - x.groupby(key).judge_theta.transform("mean"),
                                       ds=lambda x: x.score - x.groupby(key).score.transform("mean"))
    b_err, b_score = np.polyfit(within.dj, within.de, 1)[0], np.polyfit(within.dj, within.ds, 1)[0]
    return ("gap: n=%d forecasts, %d judge-task cells; Spearman(gap, signed error) %.2f, slope %.2f theta/theta; Spearman(gap, score) %.2f\n"
            "     targets stronger than the judge: signed error %s, score %s\n     targets weaker than the judge:   signed error %s, score %s\n"
            "     within a target (judge is the only thing that varies): a judge 1 theta stronger changes the signed error by %+.2f theta, the score by %+.1f"
            % (len(g), g.groupby(["task", "judge"]).ngroups, r_err, slope, r_score, ci(s.err, 2), ci(s.score), ci(w.err, 2), ci(w.score),
               b_err, b_score))


BOX_SEMESTERS = ["2025-II", "2026-I"]  # the 16 targets every domain has
BOX_COLOR = {"baseline": "#9a9890", "pairwise": "#2a78d6", "direct": "#1baf7a"}
BOX_COLUMNS = [("rank", 1, "Strongest ref."), ("rank", 2, "2nd strongest"), ("rank", 3, "3rd strongest")]


def box_table(f: pd.DataFrame) -> pd.DataFrame:
    """One row per protocol x column x target: the target's squared error averaged over the domains it has under
    that judge choice. Columns: the judge of theta* rank 3/2/1 in each task, or the judge the reference LOO picks."""
    g = f[f.semester.isin(BOX_SEMESTERS)]
    pick = g.loc[g.groupby(["task", "protocol"]).judge_loo_mse.transform("min") == g.judge_loo_mse].assign(column="loo")
    ranked = g.assign(column=g["rank"].astype(str))
    both = pd.concat([ranked, pick]).drop_duplicates(["task", "protocol", "column", "model", "judge"])
    return both.groupby(["protocol", "column", "model"], as_index=False).agg(e=("e", "mean"), n_domains=("domain", "nunique"))


def box_figure(f: pd.DataFrame, out: Path):
    """Squared error per target model (N=16, 2025-II + 2026-I, averaged over its domains), one column per judge
    choice within each protocol (the judge of theta* rank 1, 2, 3 in each task):
    small points, the mean as a large circle, its 95% t-interval as a box; first column = the reference-mean constant
    forecast (the no-information baseline). Log y-axis."""
    t = box_table(f)
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    style(ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_position(("outward", 4))
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3)
    ax.set_yticks([0.01, 0.1, 1, 3])
    ax.set_yticklabels(["0.01", "0.1", "1", "3"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    rng = np.random.default_rng(0)
    xs, labels, x = [], [], 0
    base = f[f.semester.isin(BOX_SEMESTERS)].drop_duplicates(["domain", "semester", "model"]).groupby("model").e_base.mean()
    columns = [("baseline", "-", "Baseline", base)] + [(p, "loo" if kind == "pick" else str(key), label, None)
                                                          for p in VARIANT for kind, key, label in BOX_COLUMNS]
    for p, col, label, e in columns:
        if e is None:
            e = t[(t.protocol == p) & (t.column == col)].set_index("model").e
        if e.empty:
            continue
        if col == "1":
            x += 0.6
        m, h = e.mean(), st.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / np.sqrt(len(e))
        c = BOX_COLOR[p]
        ax.add_patch(plt.Rectangle((x - 0.3, m - h), 0.6, 2 * h, color=c, alpha=0.3, lw=0, zorder=2))
        ax.plot(x + rng.uniform(-0.2, 0.2, len(e)), e, "o", ms=2, mfc=c, mec="none", alpha=0.7, zorder=3)
        ax.plot(x, m, "o", ms=5, mfc=c, mec=c, zorder=4)
        xs.append(x), labels.append(label)
        x += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.set_ylabel("Squared error (θ²)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=c, mec=c, label={"baseline": "Baseline", "pairwise": "Pairwise judge", "direct": "Absolute scoring"}[p])
               for p, c in BOX_COLOR.items()]
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, handles=handles, columnspacing=0.8, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.03)
    save(fig, out)
    return t


def main():
    f = load()
    t = per_task(f)
    t.to_csv(OUT.with_suffix(".csv"), index=False, float_format="%.3f")
    sel = selection(t)
    agree = agreement(f)
    print("per task x judge\n", t.round(2).to_string(index=False))
    for p in VARIANT:
        s, g = sel[sel.protocol == p], f[f.protocol == p]
        print(f"\n== {p} ({len(s)} tasks)")
        print("spread: score range across judges, median %.1f, IQR %.1f-%.1f" % tuple(s.range.quantile([0.5, 0.25, 0.75])))
        print("variance of per-target score:", nested_ss(g).round(2).to_dict())
        wide = g.pivot_table(index=["task", "model"], columns="judge", values="theta_hat")
        sd_judge = wide.std(axis=1, ddof=1)
        rmse = np.sqrt(((wide.mean(axis=1) - g.groupby(["task", "model"]).theta.first()) ** 2))
        print("forecast: median sd of theta-hat across judges %.2f vs median |error| of the judges' mean forecast %.2f"
              % (sd_judge.median(), rmse.median()))
        print("selection (mean ± t95 CI over tasks):", {k: ci(s[k]) for k in ("loo_pick", "oracle", "random", "worst", "top1")})
        print("loo picks the oracle judge in %d/%d tasks; top-1 reference is the oracle in %d/%d"
              % ((s.loo_pick == s.oracle).sum(), len(s), (s.top1 == s.oracle).sum(), len(s)))
        tt = t[t.protocol == p]
        c = tt.assign(ds=tt.score - tt.groupby("task").score.transform("mean"), dth=tt.judge_theta - tt.groupby("task").judge_theta.transform("mean"))
        slope = np.polyfit(c.dth, c.ds, 1)[0]
        print("capability: within-task slope of score on judge theta* %.1f per theta unit, Spearman %.2f (p=%.2f)"
              % (slope, *st.spearmanr(c.dth, c.ds)))
        a = agree[agree.protocol == p]
        print("agreement: median r of forecasts %.2f, of errors %.2f" % (a.r_forecast.median(), a.r_error.median()))
        print(gap_stats(f, p))
    both = sel.pivot(index="task", columns="protocol", values="random")
    print("\nprotocol gap (pairwise - direct, same task, judge-averaged): median %.1f, IQR %.1f-%.1f"
          % tuple((both.pairwise - both.direct).quantile([0.5, 0.25, 0.75])))
    figure(t, OUT)
    gap_figure(f, OUT.with_name(OUT.name + "_gap"))
    box = box_figure(f, OUT.with_name(OUT.name + "_box"))
    print("\nsquared error per target (2025-II + 2026-I, mean over domains), mean ± t95 CI over targets:")
    print(box.groupby(["protocol", "column"]).e.apply(ci, 2).to_string())


if __name__ == "__main__":
    main()
