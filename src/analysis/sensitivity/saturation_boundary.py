"""
Saturation boundary: an item counts as saturated (EASY) when the 3 strongest references all solve it; this module
tightens that to the k = 4 to 8 strongest references, or to every reference and target ("all"). At each level theta*
is refitted on the enlarged set of unsaturated (HARD) items, every protocol is refitted on the train items still
saturated, and each protocol's MSE is compared with the Baseline's.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import calibration, loading, matching, stats
from analysis.baselines import EasyItemPrediction
from analysis.plots import unclip_markers
from capability.task import Cohort, Task
from configs.domain import DomainConfig
from protocols import PROTOCOLS
from protocols.embeddings import EmbeddingRegression
from reference import tables
from reference.freeze import easy_hard, local_fit

ORDER = ("pairwise", "direct", "glm_regex", "embedding")  # the figure's drawing and legend order


def saturating_models(task: Task, models: pd.DataFrame, level: int | str) -> list[str]:
    """
    The models that must all solve an item for it to count as saturated at `level`.

    Inputs:
        task: the semester task; uses its references and targets.
        models: the frozen models table (data/meta/<domain>/models.csv); uses global_theta.
        level: a number of references, or "all".
    Outputs:
        list of model names: the `level` references with the highest global theta, or every reference and target
        for "all".
    """
    if level == "all":
        return [*task.references.models.index, *task.truth.index]
    return models.global_theta.loc[task.references.models.index].nlargest(level).index.tolist()


def level_split(domain: DomainConfig, task: Task, level: int | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the domain's items at `level` from the repeat-0 answers of the models that must solve them: HARD items
    (every such model answered, some wrongly), and observed items (the task's frozen train items that every such model
    solved).

    Inputs:
        domain: the domain config; reads data/meta/<domain>/models.csv, items.csv and responses.csv.
        task: the semester task.
        level: a number of references, or "all" (see saturating_models).
    Outputs:
        (hard, observed): two DataFrames of rows of the frozen items table (bench, item, guess, a, b, ...).
    """
    models, items = tables.load(domain)
    status = easy_hard(tables.load_responses(domain), items, saturating_models(task, models, level))
    observed = items[items[f"{task.semester}_train"].astype(bool) & (status == "easy")]
    return items[status == "hard"], observed


def refit_theta(domain: DomainConfig, task: Task, hard: pd.DataFrame, seed: int = 0) -> pd.Series:
    """
    Refits theta* on the items `hard` as reference.freeze.semester_task does: the references and targets in one fit,
    then, in a second fit, every model of the semesters before the references' that answered some of those items. The
    model set is the freeze's, so level 3 reproduces the frozen theta*.

    Inputs:
        domain: the domain config; reads data/meta/<domain>/responses.csv and models.csv, and configs/semesters.yml.
        task: the semester task.
        hard: the HARD rows of the items table (level_split's first output).
        seed: the sampler's seed.
    Outputs:
        Series of theta* indexed by model.
    """
    responses = tables.load_responses(domain)
    theta = local_fit(responses, hard, [*task.references.models.index, *task.truth.index], seed=seed).theta
    rosters, models = tables.load_semesters(), tables.load(domain)[0]
    earlier = list(rosters)[:list(rosters).index(task.semester) - 1]
    answered = set(responses.merge(hard[["bench", "item"]]).model)
    historical = [m for past in earlier for m in rosters[past] if m in models.index and m in answered]
    if historical:
        theta = pd.concat([theta, local_fit(responses, hard, historical, seed=seed).theta])
    return theta


def level_task(task: Task, theta: pd.Series, observed: pd.DataFrame) -> Task:
    """
    The task at one saturation level: every theta* replaced by the refitted one and every cohort's items cut to the
    observed items. A reference or historical model without a refitted theta* (no HARD response at this level) is
    dropped.

    Inputs:
        task: the semester task at the frozen boundary.
        theta: the refitted theta* indexed by model (refit_theta's output).
        observed: the items still saturated (level_split's second output); uses bench, item.
    Outputs:
        a Task with the same cohorts cut to `observed`, their theta* and truth taken from `theta`; transcripts already
        read are carried over, cut to the kept items.
    """
    items = task.references.items.merge(observed[["bench", "item"]])

    def cohort(c: Cohort, labelled: bool) -> Cohort:
        """The cohort cut to the kept items, with the refitted theta* if it is labelled."""
        models = c.models.assign(theta=theta.reindex(c.models.index)) if labelled else c.models
        if labelled:
            models = models[np.isfinite(models.theta)]
        out = Cohort(models, items, c.semester)
        if "transcripts" in c.__dict__:  # reuse the transcripts already read (cached_property), cut to the items
            keep = set(zip(items.bench, items.item))
            out.__dict__["transcripts"] = {m: {k: t for k, t in ts.items() if k in keep} for m, ts in c.transcripts.items()}
        return out

    return Task(task.semester, cohort(task.references, True), cohort(task.historical, True),
                cohort(task.test, False), theta.reindex(task.truth.index))


def cut(run: loading.JudgeRun, items: pd.DataFrame) -> loading.JudgeRun:
    """
    A shallow copy of a matched judge run whose verdicts or ratings are cut to the given items.

    Inputs:
        run: a judge run matched to the task (matching.match).
        items: the items kept; uses bench, item.
    Outputs:
        a JudgeRun whose table holds only the rows on `items`.
    """
    out = copy.copy(run)
    out.table = run.table.merge(items[["bench", "item"]])
    return out


def level_forecasts(domains: list[str], semesters: list[str], levels: list, cached_theta: pd.DataFrame | None,
                    embedding: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every task and saturation level, refits theta* on the level's HARD items and computes every protocol's
    prediction of every target on the level's observed items (calibration.forecasts): the Baseline, Deterministic
    features, Embedding, and every judge run on disk. A level above the task's number of references is skipped.

    Inputs:
        domains: the domains, in order.
        semesters: the semesters whose tasks are used.
        levels: the saturation levels, e.g. [3, 4, 5, 6, 7, 8, "all"].
        cached_theta: a previous call's theta table, used instead of refitting theta* (the slow step); None refits.
        embedding: the Embedding variant reported.
    Outputs:
        (theta, forecasts). theta: DataFrame with columns domain, semester, level (a string), model, theta, n_hard
        and n_observed (the level's numbers of HARD and observed items); one row per task, level and model.
        forecasts: calibration.forecasts' rows (domain, semester, protocol, variant, judge, condition, judge_loo_mse,
        model, theta_hat, theta, missing) with a level column; one row per task, level, protocol, variant, judge,
        condition and target.
    """
    thetas, frames = [], []
    for name in domains:
        dom = loading.domain(name)
        protocols = [(EasyItemPrediction(), "-"), (PROTOCOLS["glm_regex"], "-"), (EmbeddingRegression(embedding), embedding)]
        for task in loading.tasks(dom, semesters):
            runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
            for level in levels:
                if level != "all" and level > len(task.references.models):  # fewer references than the level: no task
                    continue
                hard, observed = level_split(dom, task, level)
                if cached_theta is not None:
                    theta = cached_theta[(cached_theta.domain == name) & (cached_theta.semester == task.semester)
                                         & (cached_theta.level == str(level))].set_index("model").theta
                else:
                    theta = refit_theta(dom, task, hard)
                thetas.append(pd.DataFrame({"domain": name, "semester": task.semester, "level": str(level), "model": theta.index,
                                            "theta": theta.to_numpy(), "n_hard": len(hard), "n_observed": len(observed)}))
                lt = level_task(task, theta, observed)
                frames.append(calibration.forecasts(dom, lt, protocols, [cut(r, observed) for r in runs]).assign(level=str(level)))
                print(name, task.semester, level, len(hard), "hard", len(observed), "observed", flush=True)
    return pd.concat(thetas, ignore_index=True), pd.concat(frames, ignore_index=True)


def ratio(g: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """
    The MSE of a protocol over the Baseline's MSE on the same targets, with a percentile-bootstrap 95% interval that
    resamples targets.

    Inputs:
        g: one row per target; uses err (the protocol's squared error) and err_baseline (the Baseline's).
        seed: the bootstrap's seed.
        draws: the number of bootstrap draws.
    Outputs:
        dict with n (targets), mse, baseline_mse, ratio (mse / baseline_mse), lo and hi (the interval's bounds).
    """
    c, b = g.err.to_numpy(float), g.err_baseline.to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(c), size=(draws, len(c)))
    boot = c[idx].mean(1) / b[idx].mean(1)
    return dict(n=len(c), mse=c.mean(), baseline_mse=b.mean(), ratio=c.mean() / b.mean(),
                lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))


def ratios(forecasts: pd.DataFrame, variants: dict[str, str], seed: int, draws: int) -> pd.DataFrame:
    """
    Each protocol's MSE over the Baseline's MSE on the same targets, per saturation level and domain and pooled over
    every domain, with a bootstrap interval over targets. A judge protocol is scored with the judge that has the
    lowest leave-one-model-out error on the references at that level (stats.one_per_target).

    Inputs:
        forecasts: level_forecasts' second output; uses level, domain, semester, protocol, variant, judge, model,
            theta_hat and theta.
        variants: protocol -> the variant it is reported in; uses embedding, direct and pairwise.
        seed: the bootstrap's seed.
        draws: the number of bootstrap draws.
    Outputs:
        DataFrame with columns level, domain ("all" for every target pooled), protocol, judge ("best judge on
        references" for a judge protocol, empty otherwise), n, mse, baseline_mse, ratio, lo and hi; one row per level,
        domain and protocol.
    """
    shown = {stats.BASELINE: "-", "glm_regex": "-", **{p: variants[p] for p in ("embedding", "direct", "pairwise")}}
    rows = []
    for level, f in forecasts.groupby("level", sort=False):
        f = stats.one_per_target(f.drop(columns="level"))
        f = f[f.protocol.map(shown).eq(f.variant)].assign(err=lambda d: (d.theta_hat - d.theta) ** 2)
        base = f[f.protocol == stats.BASELINE][["domain", "semester", "model", "err"]].rename(columns={"err": "err_baseline"})
        f = f.merge(base).dropna(subset=["err", "err_baseline"])
        for (domain, protocol), g in [*f.groupby(["domain", "protocol"]), *((("all", p), g) for p, g in f.groupby("protocol"))]:
            rows.append(dict(level=level, domain=domain, protocol=protocol, judge=",".join(sorted(set(g.judge) - {"-"})),
                             **ratio(g, seed, draws)))
    return pd.DataFrame(rows)


def pairwise_table(table: pd.DataFrame, theta: pd.DataFrame, domains: list[str], levels: list, names: dict[str, str]) -> pd.DataFrame:
    """
    The Pairwise judge's MSE ratio to the Baseline per domain and level, with the number of observed items at each
    level summed over the domain's semesters.

    Inputs:
        table: ratios' output.
        theta: level_forecasts' first output; uses domain, semester, level and n_observed.
        domains: the domains, in row order.
        levels: the saturation levels, in column order.
        names: domain -> display name.
    Outputs:
        DataFrame with columns domain (display name, then "Pooled" for every domain pooled), one ratio column per
        level and one n_<level> column per level (empty on the pooled row); one row per domain.
    """
    levels = [str(l) for l in levels]
    ratio = table[table.protocol == "pairwise"].pivot(index="domain", columns="level", values="ratio")[levels]
    n = theta.drop_duplicates(["domain", "semester", "level"]).groupby(["domain", "level"]).n_observed.sum().unstack()[levels]
    out = ratio.join(n.add_prefix("n_").astype("Int64")).reindex([*domains, "all"])
    return out.rename(index={**names, "all": "Pooled"}).rename_axis("domain").reset_index()


def figure(table: pd.DataFrame, domains: list[str], levels: list, protocols: list[str], names: dict[str, str],
           colours: dict[str, str], domain_names: dict[str, str], path: str) -> None:
    """
    Draws one panel per domain (2 x 2), each protocol's MSE ratio to the Baseline against the saturation level with its
    bootstrap band, on a log y-axis with a dashed line at 1.

    Inputs:
        table: ratios' output.
        domains: the domains, one panel each.
        levels: the saturation levels, in x-axis order.
        protocols: the protocols drawn (drawn in ORDER).
        names, colours: protocol -> legend label, colour.
        domain_names: domain -> panel title.
        path: the output path without extension.
    Outputs:
        writes path.svg, path.pdf and path.png.
    """
    style = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial"], "axes.linewidth": 0.8,
             "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False}
    levels = [str(l) for l in levels]
    x = range(len(levels))
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(4.2, 3.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.15})
        for ax, domain in zip(axes.flat, domains):
            ax.axhline(1, color="#9a9a9a", lw=0.8, ls="--")
            for protocol in [p for p in ORDER if p in protocols]:
                t = table[(table.domain == domain) & (table.protocol == protocol)].set_index("level").reindex(levels)
                ax.fill_between(x, t.lo, t.hi, color=colours[protocol], alpha=0.15, lw=0)
                ax.plot(x, t.ratio, color=colours[protocol], lw=1, marker="o", ms=3, clip_on=False, label=names[protocol])
            ax.set_title(domain_names[domain], loc="left", fontsize=8, color="#7a7a7a")
            ax.set_yscale("log"), ax.set_ylim(0.03, 10)
            ax.set_yticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03×", "0.1×", "0.3×", "1×", "3×", "10×"])
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.set_xticks(list(x), levels), ax.set_xlim(-0.3, len(levels) - 0.7)
            ax.spines["bottom"].set_bounds(0, len(levels) - 1)
        fig.supylabel("Squared error vs baseline", fontsize=8, x=0.02)
        fig.supxlabel("Models that must solve an item", fontsize=8, y=0.0)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
        unclip_markers(fig)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(f"{path}.{ext}", bbox_inches="tight", **({"dpi": 300} if ext == "png" else {}))
        plt.close(fig)
