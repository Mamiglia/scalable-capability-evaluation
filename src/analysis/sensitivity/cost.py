"""
Cost against error per domain: the selected judge's Pairwise judge and Absolute scoring runs are cut to fractions of
their judged train items and priced by the tokens their logs record, giving a curve of MSE against dollars;
Deterministic features and Embedding are single points. No judging is run and no API is called.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from analysis import loading, matching
from protocols import judges
from analysis.plots import unclip_markers
from capability.evaluate import forecast
from protocols import PROTOCOLS

FREE = 1e-3  # where $0 sits on the log cost axis
YMAX = 3  # the top of the error axis; a point above it is drawn as a triangle at the edge with its value
ORDER = ("pairwise", "direct", "glm_regex", "embedding")  # the figure's drawing and legend order


def run_cost(run_dir: Path, prices: dict[str, list[float]], proxies: dict[str, str]) -> tuple[float, int]:
    """
    Prices a judge run from the token usage in its log headers: uncached input tokens at the input price, cache reads
    at the cached-input price, output tokens at the output price.

    Inputs:
        run_dir: a run folder; every .eval log in it is read.
        prices: OpenRouter model id -> dollars per million tokens (input, cached input, output).
        proxies: logged model -> the OpenRouter model priced in its place (judges without a list price).
    Outputs:
        (dollars, input tokens) summed over the run's logs.
    """
    dollars, tokens = 0.0, 0
    for f in Path(run_dir).glob("*.eval"):
        for model, u in (read_eval_log(str(f), header_only=True).stats.model_usage or {}).items():
            inp, cached, out = prices[proxies.get(model, model.removeprefix("openrouter/"))]
            cache = u.input_tokens_cache_read or 0
            dollars += ((u.input_tokens - cache) * inp + cache * cached + u.output_tokens * out) / 1e6
            tokens += u.input_tokens
    return dollars, tokens


def cost_curves(forecasts: pd.DataFrame, judges_used: pd.DataFrame, domains: list[str], variants: dict[str, str],
                fractions: list[float], draws: int, seed: int, prices: dict[str, list[float]], proxies: dict[str, str],
                embedding_price: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the error and cost of every protocol over every frozen task of each domain. For the Pairwise judge and
    Absolute scoring, the selected run is cut to each fraction of its judged items (at least 3 items, `draws` random
    subsets, one subset at fraction 1) and each target's squared error averaged over the subsets; the cost is the
    run's full cost times the fraction. Deterministic features cost $0; Embedding is priced as embedding the Absolute
    scoring run's input tokens ($0 for a task without a selected Absolute scoring run). One random stream runs over
    every task.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); uses domain,
            semester, protocol, variant, model, theta_hat and theta for Deterministic features and Embedding.
        judges_used: the selected judge of each task and judge protocol (stats.best_judges); uses domain, semester,
            protocol, judge and condition.
        domains: the domains, in order.
        variants: protocol -> the variant it is reported in.
        fractions: the fractions of the judged items kept.
        draws: the number of random subsets per fraction below 1.
        seed: the seed of the random stream.
        prices, proxies: as for run_cost.
        embedding_price: dollars per million embedded tokens.
    Outputs:
        (curves, points). curves: DataFrame with columns model, se (squared error), baseline_se (the Baseline's),
        domain, semester, protocol, judge, condition, fraction, n_items and cost; one row per task, judge protocol,
        fraction and target. points: DataFrame with columns domain, semester, protocol, model, se and cost; one row
        per task, protocol (glm_regex, embedding) and target.
    """
    rng = np.random.default_rng(seed)
    pick = judges_used.set_index(["domain", "semester", "protocol"])
    curves, points = [], []
    for name in domains:
        dom = loading.domain(name)
        for task in loading.tasks(dom):
            sem, refs, targets = task.semester, task.references.theta, list(task.truth.index)
            runs = loading.judge_runs(dom, task)
            base = forecast(task, PROTOCOLS["mean_ref"]).set_index("model").theta_hat.reindex(targets)
            embed_tokens = 0
            for protocol in ("pairwise", "direct"):
                if (name, sem, protocol) not in pick.index:
                    print("no", protocol, "run", name, sem, flush=True)
                    continue
                judge, condition = pick.loc[(name, sem, protocol), ["judge", "condition"]]
                run = matching.match(next(r for r in runs if (r.protocol, r.judge, r.condition) == (protocol, judge, condition)), task)
                dirs = run.harness.log_dirs(sem) if protocol == "direct" else [run.harness.log_dir(sem)]
                dollars, tokens = map(sum, zip(*(run_cost(d, prices, proxies) for d in dirs)))
                embed_tokens = tokens if protocol == "direct" else embed_tokens
                items = run.table[["bench", "item"]].drop_duplicates().sort_values(["bench", "item"]).reset_index(drop=True)
                for frac in fractions:
                    n = max(3, round(frac * len(items)))
                    err = []
                    for _ in range(1 if frac == 1 else draws):
                        sub = run.table.merge(items.iloc[rng.choice(len(items), n, replace=False)])
                        theta_hat = (judges.pairwise_theta(sub, refs, targets, variants[protocol]) if protocol == "pairwise"
                                     else judges.direct_theta(sub, refs, targets, variants[protocol]))
                        err.append((theta_hat.reindex(targets) - task.truth) ** 2)
                    curves.append(pd.DataFrame({"model": targets, "se": pd.concat(err, axis=1).mean(axis=1).to_numpy(),
                                                "baseline_se": ((base - task.truth) ** 2).to_numpy()})
                                  .assign(domain=name, semester=sem, protocol=protocol, judge=run.judge, condition=run.condition,
                                          fraction=frac, n_items=n, cost=frac * dollars))
            for protocol in ("glm_regex", "embedding"):
                g = forecasts[(forecasts.domain == name) & (forecasts.semester == sem) & (forecasts.protocol == protocol)
                              & (forecasts.variant == variants.get(protocol, "-"))]
                points.append(g.assign(se=(g.theta_hat - g.theta) ** 2, cost=0.0 if protocol == "glm_regex" else embed_tokens * embedding_price / 1e6)
                              [["domain", "semester", "protocol", "model", "se", "cost"]])
            print(name, sem, flush=True)
    return pd.concat(curves, ignore_index=True), pd.concat(points, ignore_index=True)


def boot(se: pd.Series, seed: int, draws: int) -> tuple[float, float, float]:
    """
    The mean of the squared errors with a percentile-bootstrap 95% interval that resamples targets.

    Inputs:
        se: one squared error per target; missing values are dropped.
        seed: the bootstrap's seed.
        draws: the number of bootstrap draws.
    Outputs:
        (mse, lo, hi).
    """
    err = se.dropna().to_numpy()
    b = np.random.default_rng(seed).choice(err, size=(draws, len(err))).mean(axis=1)
    return err.mean(), *np.percentile(b, [2.5, 97.5])


def summarise(curves: pd.DataFrame, points: pd.DataFrame, seed: int, draws: int) -> pd.DataFrame:
    """
    Summarises cost_curves per domain: the mean cost per semester and the MSE over every target of the domain with its
    bootstrap interval, for each judge protocol at each fraction, for Deterministic features and Embedding, and for
    the Baseline (protocol "mean_ref", cost $0, from the fraction-1 rows of the curves).

    Inputs:
        curves, points: cost_curves' outputs as read back from their csv.
        seed: the bootstrap's seed.
        draws: the number of bootstrap draws.
    Outputs:
        DataFrame with columns domain, protocol, fraction (empty for protocols without a judge), cost, mse, lo and hi;
        one row per domain, protocol and fraction.
    """
    rows = []
    for (name, protocol, frac), g in curves.groupby(["domain", "protocol", "fraction"]):
        cost = g.drop_duplicates("semester").cost.mean()
        rows.append(dict(domain=name, protocol=protocol, fraction=frac, cost=cost, **dict(zip(["mse", "lo", "hi"], boot(g.se, seed, draws)))))
    for (name, protocol), g in points.groupby(["domain", "protocol"]):
        rows.append(dict(domain=name, protocol=protocol, fraction=np.nan, cost=g.drop_duplicates("semester").cost.mean(),
                         **dict(zip(["mse", "lo", "hi"], boot(g.se, seed, draws)))))
    for name, g in curves[curves.fraction == 1].drop_duplicates(["domain", "semester", "model"]).groupby("domain"):
        rows.append(dict(domain=name, protocol="mean_ref", fraction=np.nan, cost=0.0,
                         **dict(zip(["mse", "lo", "hi"], boot(g.baseline_se, seed, draws)))))
    return pd.DataFrame(rows)


def figure(s: pd.DataFrame, domains: list[str], protocols: list[str], names: dict[str, str], colours: dict[str, str],
           domain_names: dict[str, str], path: str) -> None:
    """
    Draws one panel per domain (2 x 2) of MSE against cost per semester on log axes: the judge protocols as curves over
    the item fractions with their bootstrap band, Deterministic features and Embedding as points with their interval
    ($0 drawn at FREE; a point above YMAX as a triangle at the edge labelled with its value), and the Baseline as a
    dashed line.

    Inputs:
        s: summarise's output.
        domains: the domains, one panel each.
        protocols: the protocols drawn (drawn in ORDER).
        names, colours: protocol -> legend label, colour.
        domain_names: domain -> panel title.
        path: the output path without extension.
    Outputs:
        writes path.png, path.svg and path.pdf.
    """
    judged, drawn = ("pairwise", "direct"), [p for p in ORDER if p in protocols]
    style = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25, "hspace": 0.45})
        for ax, name in zip(axes.flat, domains):
            d = s[s.domain == name]
            ax.axhline(d[d.protocol == "mean_ref"].mse.iloc[0], color="#9a9a9a", lw=0.8, ls="--", zorder=0)
            for key in drawn:
                color = colours[key]
                g = d[d.protocol == key].sort_values("fraction")
                x = g.cost.clip(lower=FREE)
                if key in judged:
                    ax.fill_between(x, g.lo, g.hi, color=color, alpha=0.18, lw=0)
                    ax.plot(x, g.mse, color=color, lw=1, marker="o", ms=2.5, mfc="white", mew=0.7)
                elif (g.mse > YMAX).any():  # off the chart: a triangle at the edge and its value
                    ax.plot(x, [YMAX] * len(g), "^", color=color, ms=3.5, clip_on=False)
                    for xi, m in zip(x, g.mse):
                        ax.text(xi * 1.6, YMAX, f"{m:.0f}", fontsize=5, color=color, va="center")
                else:
                    ax.errorbar(x, g.mse, yerr=[g.mse - g.lo, g.hi - g.mse], fmt="o", color=color, ms=3.5, lw=1, capsize=0)
            ax.text(0, 1.04, domain_names[name], transform=ax.transAxes, fontsize=7, color="#9a9a9a")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_position(("outward", 4))
        ax = axes[0, 0]
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(FREE / 1.5, 1000)
        ax.set_xticks([FREE, 0.01, 0.1, 1, 10, 100, 1000], ["0", ".01", ".1", "1", "10", "100", "1k"])
        ax.set_ylim(0.03, YMAX)
        ax.set_yticks([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        for a in axes[1]:
            a.tick_params(axis="x", labelrotation=90)
            a.set_xlabel("Cost per semester ($)")
        for a in axes[:, 0]:
            a.set_ylabel("Squared error (θ²)")
        handles = [plt.Line2D([], [], color=colours[k], lw=1, marker="o", ms=2.5, mfc="white", mew=0.7, label=names[k]) for k in drawn if k in judged]
        handles += [plt.Line2D([], [], color=colours[k], ls="", marker="o", ms=3.5, label=names[k]) for k in drawn if k not in judged]
        handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Baseline"))
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97),
                   handletextpad=0.3, columnspacing=0.8, handlelength=1.4)
        unclip_markers(fig)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
