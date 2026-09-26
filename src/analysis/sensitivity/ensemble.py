"""
Judge ensemble: compares the mean of the three candidate judges' Pairwise judge predictions with the prediction of the
single judge the main analysis selects, both computed from the train items all three judged. No judging is run.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import loading
from analysis.plots import unclip_markers
from analysis.sensitivity.item_budget import candidates, pairwise_verdicts
from capability.task import Task
from protocols.pairwise.linear import weighted_theta


def ensemble_forecasts(runs: dict[tuple[str, str, str], Path], domains: list[str]) -> pd.DataFrame:
    """
    For every task, predicts every target with each candidate judge alone (linear conversion, weighted_theta; method
    "single"), with the mean of the candidates' predictions (method "mean"), both from the items every candidate
    judged, and with the Baseline (method "constant", estimator "mean_ref": the references' mean theta*).

    Inputs:
        runs: item_budget.core_run_dirs' output; a candidate without a run is left out with a printed "missing".
        domains: the domains, in order; their tasks are the semesters of data/meta/<domain>/fit.yml `top`.
    Outputs:
        DataFrame with columns domain, semester, judges (the judges pooled, "|"-joined; empty for the Baseline), k
        (their number), method, estimator ("linear" or "mean_ref"), model, forecast (the predicted theta*) and truth
        (the target's theta*); one row per task, method, judge set and target.
    """
    rows = []
    for name in domains:
        dom = loading.domain(name)
        for sem, judges in candidates(name).items():
            task = Task.from_semester(sem, dom)
            refs, targets = task.references.theta, list(task.test.models.index)
            per_judge = {}
            for judge in judges:
                if (name, sem, judge) not in runs:
                    print("missing", name, sem, judge, flush=True)
                    continue
                per_judge[judge] = pairwise_verdicts(runs[(name, sem, judge)]).merge(task.references.items[["bench", "item"]]).assign(judge=judge)
            common = set.intersection(*(set(zip(v.bench, v.item)) for v in per_judge.values()))
            per_judge = {j: v[[k in common for k in zip(v.bench, v.item)]] for j, v in per_judge.items()}
            print(name, sem, list(per_judge), len(common), "items", flush=True)

            def add(subset, method, estimator, forecasts):
                """Appends one row per target of `forecasts` (model -> predicted theta*) to rows."""
                rows.extend(dict(domain=name, semester=sem, judges="|".join(subset), k=len(subset), method=method, estimator=estimator,
                                 model=m, forecast=f, truth=task.truth.get(m)) for m, f in forecasts.items())

            single = {j: weighted_theta(v, refs, targets) for j, v in per_judge.items()}
            for j, forecasts in single.items():
                add([j], "single", "linear", forecasts)
            add(list(per_judge), "mean", "linear", pd.concat(single.values(), axis=1).mean(axis=1))
            add([], "constant", "mean_ref", pd.Series(refs.mean(), index=targets))
    return pd.DataFrame(rows)


def task_errors(forecasts: pd.DataFrame, judges: pd.DataFrame, n_candidates: int, domains: list[str]) -> pd.DataFrame:
    """
    The MSE over each task's targets of the Baseline, of the selected pairwise judge alone and of the mean of the
    candidate judges, the latter two from the items every candidate judged.

    Inputs:
        forecasts: ensemble_forecasts' output as read back from its csv (the Baseline's empty judges read as NaN).
        judges: the selected pairwise judge of each task (stats.best_judges); uses domain, semester, judge.
        n_candidates: the number of judges the ensemble must pool; a task with fewer is left out.
        domains: the domains, in row order.
    Outputs:
        DataFrame with columns domain, semester, baseline, judge, single and ensemble (MSEs); one row per task,
        ordered by `domains`, then semester.
    """
    cells = ["domain", "semester"]
    f = forecasts.dropna(subset=["truth"])
    mse = f.assign(se=(f.forecast - f.truth) ** 2).groupby([*cells, "judges", "k", "method", "estimator"], dropna=False).se.mean().reset_index()
    lin = mse[mse.estimator == "linear"].merge(mse[mse.estimator == "mean_ref"][[*cells, "se"]].rename(columns={"se": "baseline"}))
    ens = lin[(lin.method == "mean") & (lin.k == n_candidates)]
    single = lin[lin.method == "single"].merge(judges[[*cells, "judge"]].rename(columns={"judge": "judges"}))
    both = ens[[*cells, "baseline"]].merge(single[[*cells, "judges", "se"]]).merge(ens[[*cells, "se"]], on=cells, suffixes=("_single", "_ens"))
    both = both.assign(order=both.domain.map({d: i for i, d in enumerate(domains)})).sort_values(["order", "semester"])
    return both.drop(columns="order").reset_index(drop=True).rename(columns={"judges": "judge", "se_single": "single", "se_ens": "ensemble"})


def figure(t: pd.DataFrame, domains: list[str], colours: dict[str, str], markers: dict[str, str], names: dict[str, str],
           path: str) -> None:
    """
    Draws, in the left panel, each task's ensemble MSE (domain marker) joined by a line to the selected judge's MSE
    (grey dot), with the Baseline's MSE as a grey tick, tasks in `t`'s order with a gap between domains, on a log
    axis; in the right panel, each task's difference, ensemble minus single judge.

    Inputs:
        t: task_errors' output.
        domains: the domains, in x-axis order.
        colours, markers, names: domain -> colour, marker, legend label.
        path: the output path without extension.
    Outputs:
        writes path.png, path.svg and path.pdf.
    """
    t = t.assign(x=np.arange(len(t)) + t.domain.map({d: i for i, d in enumerate(domains)}) * 0.6)
    style = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
             "font.family": "sans-serif", "font.sans-serif": ["Arial", "Arimo", "Liberation Sans"], "lines.linewidth": 1.6}
    with plt.rc_context(style):
        fig, (ax, dx) = plt.subplots(1, 2, figsize=(3.6, 1.5), gridspec_kw={"width_ratios": [6, 1], "wspace": 0.45})
        ax.plot(t.x, t.baseline, "_", ms=6, color="#9a9a9a", mew=1, ls="", clip_on=False)
        ax.vlines(t.x, t.single, t.ensemble, color="#c8c8c8", lw=0.8)
        ax.plot(t.x, t.single, "o", ms=3, color="#6e6e6e", mew=0, clip_on=False)
        for name, marker in markers.items():
            e = t[t.domain == name]
            ax.plot(e.x, e.ensemble, marker, ms=4, color=colours[name], mfc="white", mew=1, ls="", clip_on=False, label=names[name])
            dx.plot(np.linspace(-0.25, 0.25, len(e)), e.ensemble - e.single, marker, ms=3, color=colours[name], mfc="white", mew=0.7, ls="",
                    clip_on=False)
        ax.set_xticks(t.x, t.semester), ax.tick_params(axis="x", labelrotation=90)
        ax.set_yscale("log"), ax.set_ylim(0.05, 4), ax.set_yticks([0.1, 1], ["0.1", "1"]), ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel("Squared error (θ²)")
        dx.axhline(0, color="#9a9a9a", lw=0.8, ls="--")
        dx.set_xlim(-0.5, 0.5), dx.set_xticks([]), dx.spines["bottom"].set_visible(False)
        dx.set_ylim(-1.3, 0.4), dx.set_yticks([-1, -0.5, 0]), dx.set_ylabel("Δ vs single judge (θ²)", labelpad=1)
        for a in (ax, dx):
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_position(("outward", 5))
        handles = [plt.Line2D([], [], marker="_", ms=6, color="#9a9a9a", mew=1, ls="", label="Baseline"),
                   plt.Line2D([], [], marker="o", ms=3, color="#6e6e6e", mew=0, ls="", label="Single judge")] + ax.get_legend_handles_labels()[0]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.92), handletextpad=0.1, columnspacing=0.6)
        unclip_markers(fig)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
