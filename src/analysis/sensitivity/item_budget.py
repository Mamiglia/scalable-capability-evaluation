"""Item and reference budget: the prediction error when a judge's verdicts on only some of the judged train items are
used (Figure 19; per candidate judge in the breakdowns), and when only some of the references are kept (Figure 20).
No judging is run: every subset is cut from the verdicts and ratings already on disk.
"""
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis import loading, stats
from analysis.plots import unclip_markers
from capability.evaluate import score
from capability.task import Task
from configs.judge import DirectJudgeConfig
from protocols import PROTOCOLS
from protocols.direct_judge import JudgeDirect, from_logs as direct_logs
from protocols.direct_judge.scoring import calibrate
from protocols.pairwise import from_logs
from protocols.pairwise.linear import weighted_theta
from reference import tables

KEY = ["bench", "item", "model_left", "model_right"]


def candidates(domain: str) -> dict[str, list[str]]:
    """semester -> the task's three candidate judges, strongest reference first (data/meta/<domain>/fit.yml `top`)."""
    top = yaml.safe_load((tables.frozen_dir(loading.domain(domain)) / "fit.yml").read_text())["top"]
    return {sem: [j.lower() for j in judges] for sem, judges in top.items()}


def core_run_dirs(forecasts: pd.DataFrame, variant: str, logs: Path) -> dict[tuple[str, str, str], Path]:
    """(domain, semester, judge) -> the directory of each pairwise judge's core-condition run (stats.core_condition),
    the run the main analysis scores (for the 2024 tasks, whose subjects show no reasoning, the reasoning-hidden run).

    forecasts: analysis/forecasts/forecasts.csv. variant: the pairwise variant reported.
    logs: the judge-logs checkout (logs/judge/<domain>/<semester>/<judge>/<condition hash>).
    """
    configs = loading.pairwise_configs()
    core = stats.core_condition(forecasts)
    core = core[(core.protocol == "pairwise") & (core.variant == variant)].drop_duplicates(["domain", "semester", "judge"])
    return {(d, s, j): next(run for run in sorted((logs / "judge" / d / s / j).glob("*/")) if configs.get((j, run.name), ("", ""))[1] == c)
            for d, s, j, c in core[["domain", "semester", "judge", "condition"]].itertuples(index=False)}


def pairwise_verdicts(run_dir: Path) -> pd.DataFrame:
    """A pairwise run's verdicts, one per (bench, item, left, right), the run's own verdict over an imported one."""
    return from_logs(run_dir).sort_values("imported", kind="stable").drop_duplicates(KEY)


def baseline_mse(task: Task) -> float:
    """The baseline's MSE on the task's targets (the references' mean theta*)."""
    return score(PROTOCOLS["mean_ref"](task), task.truth)["mse"]


def item_subsets(items: pd.DataFrame, sizes: list[int], draws: int, rng: np.random.Generator) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """(n, draw, items kept): `draws` random subsets of n items for every size below the number of items, then all.

    items: the judged items, sorted, so the draws do not depend on the log order.
    """
    for n in [s for s in sizes if s < len(items)] + [len(items)]:
        for draw in range(1 if n == len(items) else draws):
            yield n, draw, items.iloc[rng.choice(len(items), n, replace=False)]


def pairwise_item_draws(runs: dict[tuple[str, str, str], Path], domains: list[str], sizes: list[int], draws: int,
                        seed: int) -> pd.DataFrame:
    """Per task, candidate judge and item subset, the MSE of the targets placed from that subset's pairwise verdicts
    (JudgeLinear) and the baseline's MSE. One random stream over every task, in domain and semester order.

    runs: core_run_dirs' output; a candidate without a run is skipped.
    Returns columns domain, semester, judge, judge_rank, n_judged, n, draw, baseline, linear.
    """
    rows, rng = [], np.random.default_rng(seed)
    for name in domains:
        dom = loading.domain(name)
        for sem, judges in candidates(name).items():
            task = Task.from_semester(sem, dom)
            baseline = baseline_mse(task)
            refs, targets = task.references.theta, list(task.test.models.index)
            for judge_rank, judge in enumerate(judges, 1):
                if (name, sem, judge) not in runs:
                    print("missing", name, sem, judge, flush=True)
                    continue
                v = pairwise_verdicts(runs[(name, sem, judge)]).merge(task.references.items[["bench", "item"]])
                items = v[["bench", "item"]].drop_duplicates().sort_values(["bench", "item"]).reset_index(drop=True)
                for n, draw, keep in item_subsets(items, sizes, draws, rng):
                    mse = score(weighted_theta(v.merge(keep), refs, targets), task.truth)["mse"]
                    rows.append(dict(domain=name, semester=sem, judge=judge, judge_rank=judge_rank, n_judged=len(items),
                                     n=n, draw=draw, baseline=baseline, linear=mse))
                print(name, sem, judge_rank, judge, len(items), "items", flush=True)
    return pd.DataFrame(rows)


def direct_item_draws(domains: list[str], sizes: list[int], draws: int, seed: int) -> pd.DataFrame:
    """As pairwise_item_draws for absolute scoring: each candidate judge's 1-10 ratings (reasoning shown, every bench
    set's run directory), mean rating -> theta* calibrated on the references.

    Returns columns domain, semester, judge, judge_rank, n_judged, n, draw, baseline, direct.
    """
    rows, rng = [], np.random.default_rng(seed)
    for name in domains:
        dom = loading.domain(name)
        for sem, judges in candidates(name).items():
            task = Task.from_semester(sem, dom)
            for judge_rank, judge in enumerate(judges, 1):
                config = DirectJudgeConfig.from_cli(loading.JUDGES / f"{judge}.yml")
                runs = JudgeDirect(config, dom, include_reasoning=True).log_dirs(sem)
                if not runs:
                    print("missing", name, sem, judge, flush=True)
                    continue
                refs, targets = task.references.theta, task.truth.index
                r = pd.concat([direct_logs(d) for d in runs], ignore_index=True).merge(task.references.items[["bench", "item"]])
                r = r[(r.semester == sem) & r.model.isin(refs.index.union(targets))]
                items = r[["bench", "item"]].drop_duplicates().sort_values(["bench", "item"]).reset_index(drop=True)
                baseline = baseline_mse(task)
                for n, draw, keep in item_subsets(items, sizes, draws, rng):
                    theta_hat = calibrate(r.merge(keep), anchors=refs).reindex(targets)
                    rows.append(dict(domain=name, semester=sem, judge=judge, judge_rank=judge_rank, n_judged=len(items), n=n,
                                     draw=draw, baseline=baseline, direct=score(theta_hat, task.truth)["mse"]))
                print(name, sem, judge, len(items), "items", flush=True)
    return pd.DataFrame(rows)


def reference_subsets(runs: dict[tuple[str, str, str], Path], domains: list[str], judges: pd.DataFrame, sizes: list[int]) -> pd.DataFrame:
    """Per task, the MSE of the targets placed (JudgeLinear) with every k-subset of the references as the whole
    reference set, for k in `sizes` below the number of references, then all of them. Only the subset's theta* and
    its verdicts are kept.

    runs: core_run_dirs' output. judges: the judge of each task, columns domain, semester, judge (stats.best_judges, pairwise).
    Returns columns domain, semester, judge, judge_rank, baseline, n_refs, k, subset ("|"-joined), linear.
    """
    judge_of = judges.set_index(["domain", "semester"]).judge
    rows = []
    for name in domains:
        dom = loading.domain(name)
        for sem, ranked in candidates(name).items():
            judge, task = judge_of[(name, sem)], Task.from_semester(sem, dom)
            v = pairwise_verdicts(runs[(name, sem, judge)]).merge(task.references.items[["bench", "item"]])
            print(name, sem, judge, flush=True)
            cell = dict(domain=name, semester=sem, judge=judge, judge_rank=ranked.index(judge) + 1, baseline=baseline_mse(task))
            refs, targets = task.references.theta, list(task.test.models.index)
            for k in [s for s in sizes if s < len(refs)] + [len(refs)]:
                for subset in combinations(refs.index, k):
                    kept = refs[list(subset)]
                    among = v[v.model_left.isin([*kept.index, *targets]) & v.model_right.isin([*kept.index, *targets])]
                    rows.append(dict(**cell, n_refs=len(refs), k=k, subset="|".join(subset),
                                     linear=score(weighted_theta(among, kept, targets), task.truth)["mse"]))
    return pd.DataFrame(rows)


def keep_stronger_or_weaker(subsets: pd.DataFrame) -> pd.DataFrame:
    """The reference subsets of each task and k split by the mean theta* of the references they keep: the top third
    (keep "stronger") and the bottom third (keep "weaker"), each with the all-reference set (where the two meet).

    subsets: reference_subsets' output.
    Returns its rows with a `keep` column.
    """
    out = []
    for (name, sem), cell in subsets.groupby(["domain", "semester"]):
        models = tables.load(loading.domain(name))[0]
        theta = models.loc[models[f"{sem}_role"] == "reference", f"{sem}_theta"].sort_values()
        out.append(cell.assign(theta_kept=cell.subset.map(lambda kept: theta[kept.split("|")].mean())))
    out = pd.concat(out)
    third = out.groupby(["domain", "semester", "k"]).theta_kept.rank(pct=True)
    df = out.assign(keep=np.where(third > 2 / 3, "stronger", np.where(third <= 1 / 3, "weaker", "")))
    full = df[df.k == df.n_refs]
    return pd.concat([full.assign(keep="stronger"), df[df.keep == "stronger"], full.assign(keep="weaker"), df[df.keep == "weaker"]])


def curves_figure(df: pd.DataFrame, key: str, x: str, xlabel: str, xticks: tuple, colours: dict[str, str],
                  markers: dict[str, str], names: dict[str, str], path: str) -> pd.DataFrame:
    """Figures 19 and 20 (and the per-judge breakdowns): rows = df.row (ordered by judge_rank), one panel per semester,
    one line per domain: the task's median squared error over the subsets, band 2.5-97.5 percentile, the task's
    baseline MSE as a dashed line in the domain's colour; log y-axis. The x-axis starts at its first tick.
    Writes path.{pdf,png,svg}; returns the plotted numbers: row, semester, domain, x, median, lo, hi, baseline.

    key: the column holding each subset's MSE. colours, markers, names: domain -> colour, marker, label.
    """
    df = df[df[x] >= xticks[0]]
    semesters, ranks = sorted(df.semester.unique()), list(dict.fromkeys(df.sort_values("judge_rank", kind="stable").row))
    draws = df[key].groupby([df.row, df.semester, df.domain, df[x]])
    floor = (1e-9, np.inf)
    median, lo, hi = draws.median(), draws.quantile(0.025), draws.quantile(0.975)
    base = df.groupby(["row", "semester", "domain"]).baseline.first()
    style = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(len(ranks), len(semesters), figsize=(0.3 + 1.2 * len(semesters), 0.3 + 1.1 * len(ranks)),
                                 sharex=True, sharey=True, squeeze=False, gridspec_kw={"wspace": 0.4, "hspace": 0.35})
        for row, rank in zip(axes, ranks):
            for ax, sem in zip(row, semesters):
                for name, marker in markers.items():
                    cell = (rank, sem, name)
                    if cell in median.index.droplevel(x):
                        y = median[cell].clip(*floor)
                        ax.fill_between(y.index, lo[cell].clip(*floor), hi[cell].clip(*floor), color=colours[name], alpha=0.15, lw=0)
                        ax.plot(xticks[::len(xticks) - 1], [base[cell]] * 2, color=colours[name], lw=0.7, ls="--", alpha=0.7)
                        ax.plot(y.index, y, color=colours[name], lw=1, marker=marker, ms=2.5, mfc="white", mew=0.7, clip_on=False)
                if rank == ranks[0]:
                    ax.text(0, 1.04, sem, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_position(("outward", 4))
            row[0].set_ylabel(rank)
        fig.supylabel("Squared error (θ²)", fontsize=8, x=0.125 - 0.7 / fig.get_figwidth())
        ax = axes[0, 0]
        ax.set_xlim(xticks[0], xticks[-1]), ax.set_xticks(xticks)
        ax.set_yscale("log"), ax.set_ylim(0.02, 6), ax.set_yticks([0.1, 1]), ax.set_yticklabels(["0.1", "1"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        fig.supxlabel(xlabel, fontsize=8, y=0.11 - 0.45 / fig.get_figheight())
        handles = [plt.Line2D([], [], color=colours[name], lw=1, marker=marker, ms=2.5, mfc="white", mew=0.7, label=names[name])
                   for name, marker in markers.items() if name in set(df.domain)]
        handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.7, ls="--", label="Baseline"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.9 + 0.1 / len(ranks)),
                   handletextpad=0.3, columnspacing=0.8, handlelength=1.4)
        unclip_markers(fig)
        fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{path}.svg", bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)
    table = pd.DataFrame({"median": median, "lo": lo, "hi": hi}).reset_index()
    return table.assign(baseline=[base[c] for c in zip(table.row, table.semester, table.domain)])
