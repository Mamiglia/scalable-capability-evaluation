"""
Item and reference budget: the prediction error when a judge's verdicts or ratings on only a random subset of the
judged train items are used, for every candidate judge, and when the Pairwise judge places the targets against only a
subset of the references. No judging is run: every subset is cut from the verdicts and ratings already on disk.
"""
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    """
    The three candidate judges of every frozen task of a domain, as the freeze recorded them: the task's three
    references with the highest global theta, strongest first.

    Inputs:
        domain: the domain's short name ("science", "math", "coding", "cyber"); reads data/meta/<domain>/fit.yml,
            key `top`.
    Outputs:
        dict semester -> list of the three judge names in lower case.
    """
    top = tables.fit_settings(domain)["top"]
    return {sem: [j.lower() for j in judges] for sem, judges in top.items()}


def core_run_dirs(forecasts: pd.DataFrame, variant: str, logs: Path) -> dict[tuple[str, str, str], Path]:
    """
    Finds the log folder of the run the main analysis scores for each pairwise judge: its core-condition run
    (stats.core_condition), reasoning shown on the full train set where that run exists, else reasoning shown on a
    smaller train set, else reasoning hidden.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); uses domain,
            semester, protocol, variant, judge and condition.
        variant: the pairwise variant reported (e.g. "linear").
        logs: the protocol-outputs checkout; the run folders are logs/pairwise/<view>/<domain>/<semester>/<judge>/<condition hash>.
    Outputs:
        dict (domain, semester, judge) -> the run folder whose condition hash maps (loading.pairwise_configs) to the
        judge's core condition; the first such folder in sorted order.
    """
    configs = loading.pairwise_configs()
    core = stats.core_condition(forecasts)
    core = core[(core.protocol == "pairwise") & (core.variant == variant)].drop_duplicates(["domain", "semester", "judge"])
    return {(d, s, j): next(run for run in sorted(loading.pairwise_dirs(logs, d, s, j), key=lambda r: r.name) if configs.get((j, run.name), ("", ""))[1] == c)
            for d, s, j, c in core[["domain", "semester", "judge", "condition"]].itertuples(index=False)}


def pairwise_verdicts(run_dir: Path) -> pd.DataFrame:
    """
    Reads a pairwise run's verdicts, one per bench, item and ordered pair of models; where the run's own verdict and
    one imported from another run overlap, the run's own is kept.

    Inputs:
        run_dir: a pairwise run folder (logs/pairwise/<view>/<domain>/<semester>/<judge>/<condition hash>).
    Outputs:
        DataFrame with protocols.pairwise.from_logs' columns (bench, judge, item, model_left, model_right, pick_left,
        semester, imported, ...); one row per bench, item, model_left and model_right.
    """
    return from_logs(run_dir).sort_values("imported", kind="stable").drop_duplicates(KEY)


def baseline_mse(task: Task) -> float:
    """
    The Baseline's MSE on the task's targets: every target predicted as the references' mean theta*.

    Inputs:
        task: the semester task (references with theta*, targets, and the targets' theta* in task.truth).
    Outputs:
        float, the MSE over the targets.
    """
    return score(PROTOCOLS["mean_ref"](task), task.truth)["mse"]


def item_subsets(items: pd.DataFrame, sizes: list[int], draws: int, rng: np.random.Generator) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """
    Yields random item subsets: `draws` subsets of n items, drawn without replacement, for every n in `sizes` below the
    number of items, then one subset holding every item.

    Inputs:
        items: the judged items (columns bench, item), sorted so that the draws do not depend on the log order.
        sizes: the subset sizes.
        draws: the number of random subsets per size.
        rng: the random generator, shared across calls.
    Outputs:
        an iterator of (n, draw, the rows of `items` kept).
    """
    for n in [s for s in sizes if s < len(items)] + [len(items)]:
        for draw in range(1 if n == len(items) else draws):
            yield n, draw, items.iloc[rng.choice(len(items), n, replace=False)]


def pairwise_item_draws(runs: dict[tuple[str, str, str], Path], domains: list[str], sizes: list[int], draws: int,
                        seed: int) -> pd.DataFrame:
    """
    For every task, candidate judge and random subset of its judged train items, places the targets from the Pairwise
    judge's verdicts on those items alone (the linear conversion, weighted_theta) and records their MSE. One random
    stream runs over every task, in domain and semester order.

    Inputs:
        runs: core_run_dirs' output; a candidate without a run is skipped with a printed "missing".
        domains: the domains, in order; their tasks are the semesters of data/meta/<domain>/fit.yml `top`.
        sizes: the subset sizes (numbers of items).
        draws: the number of random subsets per size.
        seed: the seed of the random stream.
    Outputs:
        DataFrame with columns domain, semester, judge, judge_rank (1 = strongest candidate), n_judged (items the
        judge judged), n (items kept), draw, baseline (the Baseline's MSE on the task) and linear (the MSE from the
        kept items); one row per task, judge, n and draw.
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
    """
    As pairwise_item_draws for Absolute scoring: for every task, candidate judge and random subset of its rated train
    items, each target's mean 1-10 rating on those items is mapped to theta* by the line fitted on the references
    (calibrate), and the MSE over the targets recorded. The ratings are those of the judge's reasoning-shown runs, read
    from every run folder of its bench sets; the ratings of models other than the task's references and targets are
    dropped.

    Inputs:
        domains: the domains, in order; their tasks are the semesters of data/meta/<domain>/fit.yml `top`.
        sizes: the subset sizes (numbers of items).
        draws: the number of random subsets per size.
        seed: the seed of the random stream, which runs over every task in domain and semester order.
    Outputs:
        DataFrame with columns domain, semester, judge, judge_rank, n_judged, n, draw, baseline and direct (the MSE
        from the kept items); one row per task, judge, n and draw. A candidate without a run is skipped.
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
    """
    For every task, places the targets with the selected Pairwise judge (linear conversion, weighted_theta) using each
    subset of k references in place of all of them, for every k in `sizes` below the number of references and then
    all of them, and records the MSE. Only the kept references' theta* and the verdicts among kept references and
    targets are used.

    Inputs:
        runs: core_run_dirs' output.
        domains: the domains, in order; their tasks are the semesters of data/meta/<domain>/fit.yml `top`.
        judges: the selected pairwise judge of each task (stats.best_judges); uses domain, semester, judge.
        sizes: the numbers of references kept.
    Outputs:
        DataFrame with columns domain, semester, judge, judge_rank, baseline (the Baseline's MSE with every
        reference), n_refs (the task's references), k, subset (the kept references, "|"-joined) and linear (the
        MSE); one row per task and reference subset.
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
    """
    Splits the reference subsets of each task and k by the mean theta* of the references they keep: the top third keep
    the stronger references, the bottom third the weaker; the set of all references is added to both, where the two
    curves meet.

    Inputs:
        subsets: reference_subsets' output; uses domain, semester, k, n_refs and subset. The references' theta* come
            from data/meta/tasks.csv.
    Outputs:
        DataFrame with subsets' columns plus theta_kept (the kept references' mean theta*) and keep ("stronger" or
        "weaker"); the middle third is left out.
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
    """
    Draws the MSE against the subset size: one row of panels per value of df.row (ordered by judge_rank), one panel
    per semester, one line per domain showing the task's median MSE over the subsets with a 2.5-97.5 percentile band,
    and the task's Baseline MSE as a dashed line in the domain's colour; log y-axis. Sizes below the first tick are
    left out.

    Inputs:
        df: one row per task and subset; uses row, judge_rank, semester, domain, baseline, the column `x` and the
            column `key`.
        key: the column holding each subset's MSE.
        x: the column holding the subset size.
        xlabel: the x-axis label.
        xticks: the x-axis ticks; the first and last are its limits.
        colours, markers, names: domain -> line colour, marker, legend label.
        path: the output path without extension.
    Outputs:
        writes path.png, path.svg and path.pdf; returns the plotted numbers as a DataFrame with columns row, semester,
        domain, the x column, median, lo, hi and baseline, one row per line point.
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
