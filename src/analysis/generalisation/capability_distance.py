"""Prediction error against how far a target's capability sits above the strongest reference's (Figure 8): each
domain's targets in equal-count bins of that distance, every protocol's mean squared error per bin with a bootstrap
interval, the baseline's least-squares line, and per protocol the slope of squared error on distance with its t-test
(the slopes the text quotes)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis import loading
from analysis.likelihood_ratio import num, p_text
from analysis.plots import unclip_markers
from analysis.stats import BASELINE
from reference import tables

DISTANCE_LABEL = "θ* above the strongest reference"


def distances(errors: pd.DataFrame) -> pd.DataFrame:
    """`errors` (stats.target_errors) with `above_top`: each target's θ* minus its task's strongest reference's."""
    top = {}
    for domain, semester in errors.groupby(["domain", "semester"]).groups:
        models, _ = tables.load(loading.domain(domain))
        top[domain, semester] = models.loc[models[f"{semester}_role"] == "reference", f"{semester}_theta"].dropna().max()
    key = errors.set_index(["domain", "semester"]).index
    return errors.assign(above_top=errors.theta.to_numpy() - np.array([top[k] for k in key]))


def with_bins(table: pd.DataFrame, bins: int) -> pd.DataFrame:
    """`table` with `bin`, 0 to bins - 1: the target's equal-count bin of `above_top` within its domain."""
    return table.assign(bin=table.above_top.groupby(table.domain).transform(lambda v: pd.qcut(v, bins, labels=False)))


def bin_errors(table: pd.DataFrame, protocols: list[str], draws: int, seed: int) -> pd.DataFrame:
    """Per domain, bin and series (the baseline and `protocols`): x, the bin's mean distance, n, its mean squared error
    and lo, hi, the 2.5 and 97.5 percentiles over `draws` resamples of the bin's targets."""
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
    """Per domain and series (the baseline and `protocols`): the least-squares slope of squared error on distance over
    the domain's targets, its standard error and the two-sided t-test that it is 0 (n - 2 df). Targets share their
    task's references, so p is optimistic. Columns: domain, protocol, n, slope, se, p."""
    rows = []
    for domain in domains:
        for p in [BASELINE, *protocols]:
            d = table[(table.domain == domain) & (table.protocol == p)].dropna(subset=["err", "above_top"])
            fit = scipy_stats.linregress(d.above_top, d.err)
            rows.append(dict(domain=domain, protocol=p, n=len(d), slope=fit.slope, se=fit.stderr, p=fit.pvalue))
    return pd.DataFrame(rows)


def fit_band(x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The least-squares line through (x, y) at `xs`, and its 95% confidence half-width."""
    X, Xs = np.vander(x, 2), np.vander(xs, 2)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid, dof = y - X @ beta, len(x) - 2
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    return Xs @ beta, scipy_stats.t.ppf(0.975, dof) * np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))


def capability_distance_figure(boxes: pd.DataFrame, table: pd.DataFrame, domains: list[str], protocols: list[str],
                               display: dict) -> plt.Figure:
    """One panel per domain, each with its own distance axis: per bin, each protocol's mean squared error as a dot over
    a translucent box of its interval, side by side around the bin's mean distance; the baseline as its least-squares
    line over the domain's targets, dashed black with its 95% confidence band. The shared squared-error axis holds
    every box.

    boxes: bin_errors rows. table: the per-target rows with_bins returns.
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


def caption(boxes: pd.DataFrame, table: pd.DataFrame, slopes: pd.DataFrame, domains: list[str], protocols: list[str],
            display: dict, draws: int) -> str:
    """The figure's .txt: what it draws, every bin's mean squared error and interval, the baseline's line, and each
    series' slope test."""
    names = display["protocol_names"]
    sizes = table[table.protocol == BASELINE].groupby(["domain", "bin"]).size()
    lines = []
    for domain in domains:
        base = table[(table.domain == domain) & (table.protocol == BASELINE)]
        b1, b0 = np.polyfit(base.above_top, base.err, 1)
        lines.append(f"{display['domain_names'][domain]}: the baseline's line, squared error = {num(b0)} "
                     f"{'+' if b1 >= 0 else '−'} {num(abs(b1))} × distance.")
        for _, g in boxes[boxes.domain == domain].groupby("bin"):
            row = g.set_index("protocol")
            lines.append(f"  bin at {num(row.x.iloc[0], sign=True)} ({int(row.n.iloc[0])} targets): " + "; ".join(
                f"{names[p]} {num(row.loc[p, 'mean'])} ({num(row.loc[p, 'lo'])} to {num(row.loc[p, 'hi'])})"
                for p in protocols if p in row.index) + ".")
        for r in slopes[slopes.domain == domain].itertuples():
            lines.append(f"  slope of squared error on distance, {names[r.protocol]}: {num(r.slope)} (SE {num(r.se)}), "
                         f"{p_text(r.p)}.")
    n = int(sizes.groupby("domain").sum().iloc[0])
    per_bin = f"{sizes.min()}" if sizes.min() == sizes.max() else f"{sizes.min()} to {sizes.max()}"
    return ("Squared error against how far each target's θ* sits above the strongest reference's, for 2025-II and "
            f"2026-I, one panel per domain with its own distance axis. Each domain's {n} targets are split into "
            f"{sizes.index.get_level_values('bin').nunique()} equal-count bins of distance ({per_bin} targets each). In "
            "each bin, each dot is a protocol's mean squared error over the bin's "
            "targets, placed side by side around the bin's mean distance, and the box is its 95% percentile-bootstrap "
            f"interval over those targets ({draws:,} resamples); with 4 targets the intervals are rough. The dashed line "
            "is the baseline's least-squares line of squared error on distance over the domain's targets, drawn 0.2 "
            "beyond the outermost bins, with its 95% confidence band. Judge protocols use each task's best judge on "
            "references. No test is drawn. Listed, not drawn: each series' least-squares slope of squared error on "
            f"distance over the domain's {n} targets, two-sided t-test on {n - 2} degrees of freedom, uncorrected; "
            "targets share their task's references, so p is optimistic.\n\nPer domain and bin (mean distance above the "
            "strongest reference): mean squared error (95% interval):\n" + "\n".join(lines) + "\n")
