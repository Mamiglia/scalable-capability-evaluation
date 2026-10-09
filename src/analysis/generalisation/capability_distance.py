"""
Prediction error against how far a target's capability sits above the strongest reference's. Each domain's targets
are put in equal-count bins of that distance, and every protocol's mean squared error per bin is given with a
bootstrap interval; the Baseline's least-squares line of squared error on distance, and per protocol the slope of
squared error on distance with its t-test, show the trend over the targets.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis import loading
from analysis.plots import unclip_markers
from analysis.stats import BASELINE
from reference import tables

DISTANCE_LABEL = "θ* above the strongest reference"


def distances(errors: pd.DataFrame) -> pd.DataFrame:
    """
    Adds each target's distance above its task's strongest reference: the target's θ* minus the highest θ* among the
    task's references, read from the frozen model table (the <semester>_role and <semester>_theta columns).

    Inputs:
        errors: per-target errors from stats.target_errors (domain, semester, model, protocol, theta, err, ...).
    Outputs:
        `errors` with an above_top column.
    """
    top = {}
    for domain, semester in errors.groupby(["domain", "semester"]).groups:
        models, _ = tables.load(loading.domain(domain))
        top[domain, semester] = models.loc[models[f"{semester}_role"] == "reference", f"{semester}_theta"].dropna().max()
    key = errors.set_index(["domain", "semester"]).index
    return errors.assign(above_top=errors.theta.to_numpy() - np.array([top[k] for k in key]))


def with_bins(table: pd.DataFrame, bins: int) -> pd.DataFrame:
    """
    Adds each row's equal-count bin of distance (above_top) within its domain, cut at the domain's quantiles.

    Inputs:
        table: rows with domain and above_top (the output of distances).
        bins: the number of bins per domain (config.yaml generalisation.distance_bins).
    Outputs:
        `table` with a bin column, 0 to bins - 1.
    """
    return table.assign(bin=table.above_top.groupby(table.domain).transform(lambda v: pd.qcut(v, bins, labels=False)))


def bin_errors(table: pd.DataFrame, protocols: list[str], draws: int, seed: int) -> pd.DataFrame:
    """
    Per domain, bin and series (the Baseline and each protocol), the bin's mean distance, its number of targets, the
    mean squared error over them and the 2.5 and 97.5 percentiles of that mean over `draws` resamples of the bin's
    targets.

    Inputs:
        table: with_bins rows (domain, bin, protocol, above_top, err).
        protocols: the protocols besides the Baseline.
        draws: bootstrap resamples per bin and series.
        seed: the random generator's seed.
    Outputs:
        DataFrame with columns domain, bin, protocol, x (the mean distance), n, mean, lo, hi; one row per domain, bin
        and series.
    """
    rng = np.random.default_rng(seed)
    rows = []
    binned = table[table.protocol.isin([BASELINE, *protocols])].dropna(subset=["bin"])
    for (domain, b, p), g in binned.groupby(["domain", "bin", "protocol"]):
        v = g.err.to_numpy(float)
        boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(axis=1)
        rows.append(dict(domain=domain, bin=b, protocol=p, x=float(g.above_top.mean()), n=len(v), mean=float(v.mean()),
                         lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def slope_tests(table: pd.DataFrame, domains: list[str], protocols: list[str]) -> pd.DataFrame:
    """
    Per domain and series (the Baseline and each protocol), the least-squares line of squared error on distance over
    the domain's targets: its slope, the slope's standard error and two-sided t-test that it is 0 (n - 2 degrees of
    freedom), and its intercept (the Baseline's line is the figure's dashed line). Targets share their task's
    references, so p is optimistic.

    Inputs:
        table: rows with domain, protocol, above_top and err (the output of distances or with_bins).
        domains: domain keys, in order.
        protocols: the protocols besides the Baseline.
    Outputs:
        DataFrame with columns domain, protocol, n, slope, se, p, intercept; one row per domain and series.
    """
    rows = []
    for domain in domains:
        for p in [BASELINE, *protocols]:
            d = table[(table.domain == domain) & (table.protocol == p)].dropna(subset=["err", "above_top"])
            fit = scipy_stats.linregress(d.above_top, d.err)
            rows.append(dict(domain=domain, protocol=p, n=len(d), slope=fit.slope, se=fit.stderr, p=fit.pvalue,
                             intercept=fit.intercept))
    return pd.DataFrame(rows)


def fit_band(x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The least-squares line of y on x evaluated at `xs`, and the half-width of its 95% confidence band there.

    Inputs:
        x: the points' x values (distance).
        y: the points' y values (squared error).
        xs: where to evaluate the line.
    Outputs:
        (the line's values at xs, the band's half-widths at xs), both arrays.
    """
    X, Xs = np.vander(x, 2), np.vander(xs, 2)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid, dof = y - X @ beta, len(x) - 2
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    return Xs @ beta, scipy_stats.t.ppf(0.975, dof) * np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))


def capability_distance_figure(boxes: pd.DataFrame, table: pd.DataFrame, domains: list[str], protocols: list[str],
                               display: dict) -> plt.Figure:
    """
    Draws one panel per domain with its own distance axis: per bin, each protocol's mean squared error as a dot over a
    translucent box of its interval, side by side around the bin's mean distance, and the Baseline as its
    least-squares line over the domain's targets, dashed black with its 95% confidence band. The shared squared-error
    axis holds every protocol's box.

    Inputs:
        boxes: bin_errors rows.
        table: the per-target rows with_bins returns (domain, protocol, above_top, err).
        domains: domain keys, in panel order.
        protocols: the protocols drawn, in order.
        display: config.yaml's display section; uses protocol_names, protocol_colours and domain_names.
    Outputs:
        the matplotlib Figure.
    """
    names, colours = display["protocol_names"], display["protocol_colours"]
    fig, axes = plt.subplots(1, len(domains), figsize=(4.4, 1.6), layout="constrained", sharey=True)
    for ax, domain in zip(axes, domains):
        panel = boxes[boxes.domain == domain]
        xs = panel.drop_duplicates("bin").x
        gap = float(np.diff(np.sort(xs.to_numpy(float))).min())  # the series share a bin's slot: fit the closest pair
        step = min(0.06 * float(xs.max() - xs.min()), 0.85 * gap / len(protocols))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
        base = table[(table.domain == domain) & (table.protocol == BASELINE)]
        line_x = np.linspace(float(xs.min()) - 0.2, float(xs.max()) + 0.2, 50)
        fit, band = fit_band(base.above_top.to_numpy(float), base.err.to_numpy(float), line_x)
        ax.plot(line_x, fit, color="black", lw=0.8, ls="--", zorder=4)
        ax.fill_between(line_x, np.maximum(fit - band, 0), fit + band, color="black", alpha=0.12, linewidth=0, zorder=1)
        for k, p in enumerate(protocols):
            d = panel[panel.protocol == p]
            x = d.x.to_numpy(float) + step * (k - (len(protocols) - 1) / 2)
            for xi, lo, hi in zip(x, d.lo, d.hi):
                ax.add_patch(plt.Rectangle((xi - 0.4 * step, lo), 0.8 * step, hi - lo, facecolor=colours[p],
                                           edgecolor="none", alpha=0.35, zorder=2))
            ax.scatter(x, d["mean"], s=10, color=colours[p], linewidths=0, edgecolors="none", zorder=3)
        margin = max(step * len(protocols) / 2 + 0.05 * float(xs.max() - xs.min()), 0.2)  # the line runs 0.2 past the bins
        ax.set_xlim(float(xs.min()) - margin, float(xs.max()) + margin)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.text(0, 1.03, display["domain_names"][domain], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    for ax in axes:
        ax.set_ylim(0, float(boxes[boxes.protocol.isin(protocols)].hi.max()) * 1.05)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.supxlabel(DISTANCE_LABEL, fontsize=8)
    fig.supylabel("Squared error", fontsize=8)
    handles = [plt.Line2D([], [], color="black", lw=0.8, ls="--", label=names[BASELINE]),
               *[plt.Line2D([], [], color=colours[p], marker="o", ls="none", ms=3.5, label=names[p]) for p in protocols]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7)
    return unclip_markers(fig)

