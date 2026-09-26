"""One target's forecasts drawn as the predictive distributions the log likelihood ratio compares (Figure 10): the
references' θ* at their release dates, the target's true θ*, and from the target's release date each protocol's
Student t distribution of θ* as a one-sided violin.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

from analysis import likelihood_ratio as lr
from analysis import loading
from analysis.plots import unclip_markers
from analysis.stats import BASELINE, best_judges
from protocols import PROTOCOLS
from protocols.embeddings import EmbeddingRegression


def release_dates(model_configs: Path) -> dict[str, pd.Timestamp]:
    """Each model's release date, keyed by its data-dir shortname (the config name's last segment, lowercased) and by
    the config's file stem; a date range keeps its start.

    model_configs: the model configs' folder (configs/models), one subfolder per provider.
    """
    out = {}
    for path in Path(model_configs).glob("*/*.yml"):
        config = yaml.safe_load(path.read_text()) or {}
        if config.get("release_date"):
            date = pd.Timestamp(str(config["release_date"]).split("/")[0])
            out[path.stem] = date
            if "name" in config:
                out[config["name"].split("/")[-1].lower()] = date
    return out


def example_distributions(forecasts: pd.DataFrame, variants: dict[str, str], domain: str, semester: str, target: str,
                          protocols: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """The references' θ* and, per rule (the Baseline, then `protocols`), the target's predictive distribution: one row
    each with judge, theta (the target's θ*), forecast, spread (the rule's root-mean-square leave-one-model-out error
    on the references), squared_error, log_lr and references (their number).

    Judge protocols use the task's best judge on references, forecast and spread from `forecasts`; transcript protocols
    are refitted on the references (reads the task's features and embeddings).
    """
    task = loading.tasks(loading.domain(domain), [semester])[0]
    refs, truth, df = task.references.theta, float(task.truth[target]), len(task.references.theta) - 1
    best = best_judges(forecasts, variants).set_index(["domain", "semester", "protocol"])
    spreads = {BASELINE: (float(refs.mean()), float(np.sqrt(lr.loo_mse(lr.saturated, refs))))}
    judges = {BASELINE: "-"}
    for p in protocols:
        if p in ("pairwise", "direct"):
            run = best.loc[(domain, semester, p)]
            row = forecasts[(forecasts.domain == domain) & (forecasts.semester == semester) & (forecasts.protocol == p)
                            & (forecasts.variant == run.variant) & (forecasts.judge == run.judge)
                            & (forecasts.condition == run.condition) & (forecasts.model == target)].iloc[0]
            spreads[p], judges[p] = (float(row.theta_hat), float(np.sqrt(row.judge_loo_mse))), run.judge
        else:
            feature = PROTOCOLS[p] if p != "embedding" else EmbeddingRegression(variants["embedding"])
            rule = lr.feature_rule(feature.estimator, pd.concat([feature.get_features(task.references),
                                                                 feature.get_features(task.test)]))
            spreads[p], judges[p] = (float(rule(refs, [target]).iloc[0]), float(np.sqrt(lr.loo_mse(rule, refs)))), "-"
    log_p = {p: float(scipy_stats.t.logpdf(truth, df, c, s)) for p, (c, s) in spreads.items()}
    values = pd.DataFrame([dict(protocol=p, judge=judges[p], theta=truth, forecast=c, spread=s, squared_error=(c - truth) ** 2,
                                log_lr=log_p[p] - log_p[BASELINE], references=len(refs)) for p, (c, s) in spreads.items()])
    return refs, values


def prediction_example_figure(values: pd.DataFrame, refs: pd.Series, dates: dict[str, pd.Timestamp], target: str,
                              marker: str, display: dict, width_days: float = 90) -> plt.Figure:
    """The references' θ* at their release dates (grey), the target's true θ* (black) and, from the target's release
    date, each rule's Student t distribution of θ* as a one-sided violin, cut 3 spreads either side of its forecast or
    at the axis, with a line across it at the forecast. The black line at θ* reaches the widest violin, so its length
    inside each violin is that distribution's density at the truth. Every violin shares one width per unit of density,
    the widest `width_days` long.

    values, refs: example_distributions' output. dates: release_dates. marker: the domain's marker.
    display: config.yaml's display section (protocol_names, protocol_colours with the Baseline's).
    """
    names, colours = display["protocol_names"], display["protocol_colours"]
    spreads = {r.protocol: (r.forecast, r.spread) for r in values.itertuples()}
    log_lr = dict(zip(values.protocol, values.log_lr))
    protocols = [p for p in spreads if p != BASELINE]
    truth, df = float(values.theta.iloc[0]), len(refs) - 1
    fig, ax = plt.subplots(figsize=(2.8, 2.5), layout="constrained")
    ax.scatter([dates[m] for m in refs.index], refs.to_numpy(), s=14, marker=marker, color=colours[BASELINE],
               linewidths=0, zorder=3)
    base_c, base_s = spreads[BASELINE]
    extent = [*refs, truth, base_c - 3 * base_s, base_c + 3 * base_s, *(c for c, _ in spreads.values())]
    lo, hi = np.floor(min(extent) * 2) / 2, np.ceil(max(extent) * 2) / 2
    x0 = mdates.date2num(dates[target])
    scale = width_days / max(scipy_stats.t.pdf(0, df, 0, s) for _, s in spreads.values())  # days per unit of density
    for p, (centre, spread) in spreads.items():
        ys = np.linspace(max(centre - 3 * spread, lo), min(centre + 3 * spread, hi), 400)  # the violin's cut
        edge = x0 + scale * scipy_stats.t.pdf(ys, df, centre, spread)
        ax.fill_betweenx(ys, x0, edge, color=colours[p], alpha=0.25, lw=0, zorder=1)
        ax.plot(edge, ys, color=colours[p], lw=1, zorder=2)
        ax.plot([x0, x0 + scale * scipy_stats.t.pdf(centre, df, centre, spread)], [centre] * 2, color=colours[p], lw=1, zorder=2)
    reach = max(scipy_stats.t.pdf(truth, df, c, s) for c, s in spreads.values())
    ax.plot([x0, x0 + scale * reach], [truth] * 2, color="black", lw=0.8, zorder=4)
    ax.scatter([x0], [truth], s=18, marker=marker, color="black", linewidths=0, zorder=5)
    for p in protocols:  # each protocol's squared error, left of the violins at its forecast
        centre = spreads[p][0]
        ax.text(x0 - 5, centre, f"squared error {(centre - truth) ** 2:.2f}", color=colours[p], ha="right", va="center", fontsize=7)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(np.ceil(lo), np.floor(hi) + 1))
    ax.spines["left"].set_bounds(np.ceil(lo), np.floor(hi))
    ax.set_ylabel("Model Capability (θ)")
    ax.set_xlim(mdates.date2num(min(dates[m] for m in refs.index).to_period("Q").to_timestamp()), x0 + 1.05 * width_days)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
    ax.set_xlabel("Release date")
    point = lambda colour: plt.Line2D([], [], color=colour, marker=marker, ls="none", ms=4, markeredgewidth=0)  # noqa: E731
    patch = lambda colour: plt.matplotlib.patches.Patch(facecolor=colour, alpha=0.5, edgecolor="none")  # noqa: E731
    handles = [point(colours[BASELINE]), point("black"), *(patch(colours[p]) for p in spreads)]
    labels = ["References", "Target", names[BASELINE],
              *(f"{names[p]}, log LR {log_lr[p]:.2f}".replace("-", "−") for p in protocols)]
    fig.legend(handles=handles, labels=labels, loc="outside lower center", ncol=2, frameon=False)
    return unclip_markers(fig)


def example_caption(values: pd.DataFrame, domain_name: str, semester: str, target: str, names: dict[str, str]) -> str:
    """The .txt of the prediction example: what the figure draws, then every rule's forecast, spread, squared error
    and log likelihood ratio."""
    by = values.set_index("protocol")
    n = int(values.references.iloc[0])
    lines = [f"{names[p]}: forecast {lr.num(r.forecast)}, spread {lr.num(r.spread)}, squared error {lr.num(r.squared_error)}"
             + ("" if p == BASELINE else f", log likelihood ratio {lr.num(r.log_lr)}") + "." for p, r in by.iterrows()]
    return (f"What one target's forecast looks like: {target} in {domain_name} {semester}, true θ* {lr.num(by.theta.iloc[0])}. "
            f"Grey markers are the task's {n} references at their release dates; the black marker is the target's true θ* "
            "at its release date. From that date each forecast is drawn as the distribution the log likelihood ratio uses: "
            f"a Student t with {n - 1} degrees of freedom, centred on the forecast (the line across the violin) and scaled "
            "by the forecast rule's root-mean-square leave-one-model-out error on the references (its spread). Grey: the "
            "Baseline, the references' mean θ*; each protocol in its colour (judge protocols: the task's best judge on "
            f"references, {', '.join(f'{names[p]} {j}' for p, j in by.judge.items() if j != '-')}). Every violin has the "
            "same width per unit of density, so a narrower distribution is wider at its peak; each is cut 3 spreads from "
            "its forecast or at the axis. The black line at θ* runs to the widest violin: where it crosses each violin's "
            "edge is that distribution's density at the truth, and a protocol's log likelihood ratio is the log of its "
            "density there over the Baseline's.\n\n" + "\n".join(lines) + "\n")
