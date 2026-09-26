"""Signal in saturated transcripts: one GLM of forecast error over every protocol and domain.

    squared_error ~ protocol + domain + (protocol | domain) + (1 | model:domain) [+ (1 | protocol:judge)],   Gamma, log link

on the balanced semesters (2025-II, 2026-I: every protocol in every domain x semester). Rows
are one forecast per target x protocol (best judge on references); the saturated-item
prediction is the reference level of `protocol`, so every coefficient is a log error ratio to it. The hierarchical fit runs through hibayes
(scripts/analyses/signal_glm/); this module builds its rows, reads its posterior into a ratio
table, fits the flat statsmodels cross-check and draws the figure.

Design: docs/superpowers/specs/2026-09-16-signal-glm-design.md
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices

from scipy import stats as scipy_stats

from analysis.plots import CONDITIONS, DOMAIN_LABELS, break_edges, break_strip, chosen, ci95, compress, fix_axes_height
from analysis.plots import LABELS as plots_labels
from analysis.stats import BASELINE, JUDGED, core_condition, one_per_target

LABELS = {p: plots_labels[p] for p in CONDITIONS}  # the paper's protocol names, one source

# The project's protocol palette (CLAUDE.md). Domains never share a hue with a protocol.
COLORS = {"glm_length": "#f1a48a", "glm_regex": "#eb6834", "embedding": "#4a3aa7", "direct": "#1baf7a", "pairwise": "#2a78d6"}
DOMAIN_COLORS = {"science": "#cc79a7", "math": "#d55e00", "coding": "#6a3d9a", "cyber": "#c9a800"}
DOMAIN_MARKERS = {"science": "o", "math": "s", "coding": "^", "cyber": "D"}

PAPER_PROTOCOLS = [p for p in CONDITIONS if p != "glm_length"]  # Length alone is a diagnostic, not a paper protocol
PROTOCOLS = [BASELINE, *PAPER_PROTOCOLS]  # the GLM's levels (Appendix D.4), the baseline first: the reference level everywhere
DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = ("2025-II", "2026-I")  # the semesters every domain has: a balanced design needs no semester term
POOLED = "all"
FLOOR = 1e-6  # a Gamma outcome must be positive; an exact hit is clipped here


def score(errors, baseline):
    """Prediction score per model, -log2(error / the model's own saturated-item error): the number
    of halvings of the error. 0 is the baseline, 1 half its error, 2 a quarter, -1 twice it.
    Symmetric in the ratio (twice as good and twice as bad are equidistant from 0), so a model
    the baseline happens to predict almost exactly cannot dominate the mean, unlike
    100 * (1 - error / baseline) (gpt-oss-20b: -114%); the same scale as the hierarchical GLM."""
    return -np.log2(errors / baseline)


def rows(forecasts: pd.DataFrame, variants: dict[str, str], semesters: tuple[str, ...] = SEMESTERS,
         judges: str = "best") -> pd.DataFrame:
    """The model's table from the forecasts table: one row per target x protocol in PROTOCOLS,
    each protocol in the variant `variants` names for it (pairwise conversion, direct
    normalisation, embedding variant; "-" when the protocol is not named), semesters in
    `semesters` only. judges "best": judge protocols as the best judge on references, one row
    per target; "best_other_family": the best judge on references among judges from another
    developer than the target (no self-judging: glm-4.7 never judges glm-5.x); "all": one row
    per target x judge on the core condition. Columns: y (squared
    error, >= FLOOR), protocol, domain, semester, task ("<domain> <semester>"), model,
    model_domain ("<model> <domain>", the cell every protocol forecasts once), judge ("-" for
    transcript protocols and under "best"), protocol_judge ("<protocol> <judge>", "-" likewise).
    Rows without a forecast are dropped."""
    assert judges in ("best", "best_other_family", "all"), judges
    f = one_per_target(forecasts) if judges == "best" else best_other_family(forecasts) if judges == "best_other_family" else core_condition(forecasts)
    f = chosen(f, variants)
    f = f[f.protocol.isin(PROTOCOLS) & f.semester.isin(semesters)].dropna(subset=["theta_hat", "theta"])
    judge = f.judge.where(f.protocol.isin(JUDGED) & (judges == "all"), "-")
    return pd.DataFrame({
        "y": ((f.theta_hat - f.theta) ** 2).clip(lower=FLOOR).to_numpy(),
        "protocol": f.protocol.to_numpy(), "domain": f.domain.to_numpy(), "semester": f.semester.to_numpy(),
        "task": (f.domain + " " + f.semester).to_numpy(), "model": f.model.to_numpy(),
        "model_domain": (f.model + " " + f.domain).to_numpy(),
        "judge": judge.to_numpy(), "protocol_judge": (f.protocol + " " + judge).where(judge != "-", "-").to_numpy(),
    })


def family(model: pd.Series) -> pd.Series:
    """The developer prefix of a model name: glm-4.7 -> glm, kimi-k2.5 -> kimi, qwen3-235b -> qwen."""
    return model.str.extract(r"^([a-z]+)", expand=False)


def best_other_family(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Like one_per_target, but each target's judge is the best (lowest reference leave-one-out
    MSE) among the core-condition judges from another developer family than the target, so no
    model is judged by its own lineage. Targets with no such judge are dropped."""
    core = core_condition(forecasts)
    judged = core[core.protocol.isin(JUDGED) & (family(core.judge) != family(core.model))]
    key = ["domain", "semester", "protocol", "variant", "model"]
    best = judged.loc[judged.groupby(key).judge_loo_mse.transform("min") == judged.judge_loo_mse].drop_duplicates(key)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], best.assign(judge="best-other-family", condition="core")], ignore_index=True)


def flat_glm(table: pd.DataFrame) -> pd.DataFrame:
    """The frequentist cross-check: Gamma GLM with a log link of y on protocol * domain +
    model_domain (a fixed intercept per (target model, domain) cell, the paired analogue of the
    hierarchical fit's random one; semester and domain main effects are implied by the cell).
    With every judge's rows the judges are pooled: no judge term, the effect is the mean over judges.
    Returns one row per non-baseline protocol x domain plus a pooled row (domain = "all", the
    mean of the four domain log ratios) with flat_ratio, flat_lo, flat_hi (95% Wald interval),
    the error ratio to the saturated-item prediction in that domain. A domain without a
    protocol's forecasts gets NaN."""
    formula = f"y ~ C(protocol, Treatment('{BASELINE}')) * C(domain, Treatment('science')) + C(model_domain)"
    # Newton, not IRLS: the 64 cell dummies and the floored exact hits make IRLS overflow
    fit = smf.glm(formula, table, family=sm.families.Gamma(sm.families.links.Log())).fit(method="newton", maxiter=5000, disp=0)
    design = fit.model.data.model_spec  # patsy DesignInfo
    first_cell = table.model_domain.min()  # any cell: it cancels in every contrast

    def contrast(protocol: str, domain: str) -> np.ndarray | None:
        if not ((table.protocol == protocol) & (table.domain == domain)).any():
            return None
        frame = pd.DataFrame({"protocol": [protocol, BASELINE], "domain": domain, "model_domain": first_cell})
        (x,) = build_design_matrices([design], frame)
        return np.asarray(x)[0] - np.asarray(x)[1]  # log mu(protocol) - log mu(baseline), same cell

    out = []
    for protocol in PROTOCOLS[1:]:
        cells = {d: contrast(protocol, d) for d in DOMAINS}
        present = [c for c in cells.values() if c is not None]
        cells[POOLED] = np.mean(present, axis=0) if present else None
        for domain, c in cells.items():
            if c is None:
                out.append(dict(protocol=protocol, domain=domain, flat_ratio=np.nan, flat_lo=np.nan, flat_hi=np.nan))
                continue
            t = fit.t_test(c)
            lo, hi = np.asarray(t.conf_int()).ravel()
            out.append(dict(protocol=protocol, domain=domain, flat_ratio=float(np.exp(np.ravel(t.effect)[0])), flat_lo=float(np.exp(lo)), flat_hi=float(np.exp(hi))))
    return pd.DataFrame(out)


def ratio_table(posterior, coords: dict) -> pd.DataFrame:
    """The hierarchical fit's error ratios from the posterior (an xarray Dataset or dict of
    arrays with leading chain/draw dims): protocol_effects [.., P], interaction [.., P, D],
    sigma_protocol [.., P]; coords["protocol"] and coords["domain"] in the model's order, the
    baseline first. One row per non-baseline protocol x domain (ratio = exp(alpha + delta))
    plus a pooled row per protocol (domain = "all", ratio = exp(alpha), sigma_protocol with
    sigma_lo, sigma_hi = the protocol's across-domain SD on the log scale). ratio, lo, hi and
    the sigma triple = posterior median and 2.5 / 97.5 percentiles."""
    alpha = np.asarray(posterior["protocol_effects"])
    delta = np.asarray(posterior["interaction"])
    sigma = np.asarray(posterior["sigma_protocol"])
    alpha, delta, sigma = (a.reshape(-1, *a.shape[-k:]) for a, k in ((alpha, 1), (delta, 2), (sigma, 1)))

    def summary(draws: np.ndarray) -> dict:
        lo, mid, hi = np.percentile(np.exp(draws), [2.5, 50, 97.5])
        return dict(ratio=mid, lo=lo, hi=hi)

    out = []
    for p, protocol in enumerate(coords["protocol"]):
        if protocol == BASELINE:
            continue
        for d, domain in enumerate(coords["domain"]):
            out.append(dict(protocol=protocol, domain=domain, **summary(alpha[:, p] + delta[:, p, d]),
                            sigma_protocol=np.nan, sigma_lo=np.nan, sigma_hi=np.nan))
        s_lo, s_mid, s_hi = np.percentile(sigma[:, p], [2.5, 50, 97.5])
        out.append(dict(protocol=protocol, domain=POOLED, **summary(alpha[:, p]), sigma_protocol=s_mid, sigma_lo=s_lo, sigma_hi=s_hi))
    return pd.DataFrame(out)


def error_table(posterior, coords: dict) -> pd.DataFrame:
    """The hierarchical fit's expected squared error per protocol x domain for a typical target
    (cell intercept 0): exp(intercept + alpha + gamma + delta), the baseline included, plus a
    pooled row per protocol (domain = "all", exp(intercept + alpha): the domain effects sum to
    zero, so this is the mean over domains on the log scale). mse, lo, hi = posterior median
    and 2.5 / 97.5 percentiles."""
    intercept = np.asarray(posterior["intercept"]).reshape(-1)
    alpha = np.asarray(posterior["protocol_effects"])
    gamma = np.asarray(posterior["domain_effects"])
    delta = np.asarray(posterior["interaction"])
    alpha, gamma, delta = (a.reshape(-1, *a.shape[-k:]) for a, k in ((alpha, 1), (gamma, 1), (delta, 2)))

    def summary(draws: np.ndarray) -> dict:
        lo, mid, hi = np.percentile(np.exp(draws), [2.5, 50, 97.5])
        return dict(mse=mid, lo=lo, hi=hi)

    out = []
    for p, protocol in enumerate(coords["protocol"]):
        for d, domain in enumerate(coords["domain"]):
            out.append(dict(protocol=protocol, domain=domain, **summary(intercept + alpha[:, p] + gamma[:, d] + delta[:, p, d])))
        out.append(dict(protocol=protocol, domain=POOLED, **summary(intercept + alpha[:, p])))
    return pd.DataFrame(out)


def wide_table(table: pd.DataFrame) -> pd.DataFrame:
    """ratio_table rows as protocol x domain: index protocol (the baseline first, ratio 1 by
    definition, no interval), columns <domain>, <domain>_lo, <domain>_hi for each domain and
    "all" (the pooled effect), then sigma, sigma_lo, sigma_hi (the across-domain SD on the log
    scale, NaN for the baseline)."""
    cols = [d for d in [*DOMAINS, POOLED] if (table.domain == d).any()]
    wide = table.pivot(index="protocol", columns="domain", values=["ratio", "lo", "hi"])
    out = pd.DataFrame(index=[BASELINE, *[p for p in PROTOCOLS[1:] if p in wide.index]])
    for d in cols:
        out[d] = wide["ratio"][d].reindex(out.index).fillna({BASELINE: 1.0})
        out[f"{d}_lo"], out[f"{d}_hi"] = wide["lo"][d].reindex(out.index), wide["hi"][d].reindex(out.index)
    pooled = table[table.domain == POOLED].set_index("protocol").reindex(out.index)
    out["sigma"], out["sigma_lo"], out["sigma_hi"] = pooled.sigma_protocol, pooled.sigma_lo, pooled.sigma_hi
    out.index.name = "protocol"
    return out


def table_figure(table: pd.DataFrame) -> plt.Figure:
    """The hierarchical ratio table as an image: one row per protocol, columns pooled and per
    domain as "median [lo, hi]", plus the across-domain spread; caption below."""
    protocols = [p for p in PROTOCOLS[1:] if (table.protocol == p).any()]
    cols = [d for d in [POOLED, *DOMAINS] if (table.domain == d).any()]
    head = ["Protocol", *["Pooled" if d == POOLED else DOMAIN_LABELS[d] for d in cols], "SD across domains"]
    rows = []
    for p in protocols:
        sub = table[table.protocol == p].set_index("domain")
        rows.append([LABELS[p], *[f"{sub.loc[d].ratio:.2f}  [{sub.loc[d].lo:.2f}, {sub.loc[d].hi:.2f}]" for d in cols],
                     f"{sub.loc[POOLED].sigma_protocol:.2f}  [{sub.loc[POOLED].sigma_lo:.2f}, {sub.loc[POOLED].sigma_hi:.2f}]"])
    fig = plt.figure(figsize=(9.5, 1.9))
    ax = fig.add_axes([0.02, 0.16, 0.96, 0.82]); ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center",
                   colWidths=[0.15, *[0.14] * len(cols), 0.15])
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.6)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.8); cell.set_edgecolor("black")
        cell.visible_edges = {0: "TB", len(rows): "B"}.get(r, "")
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.set_text_props(ha="left"); cell.PAD = 0.03; cell._loc = "left"
    fig.text(0.02, 0.02, "MSE ratio to the saturated-item prediction, posterior median [95% CI]; SD across domains on the log scale.",
             fontsize=7, va="top")
    return fig


def ratio_figure(table: pd.DataFrame, scale: str = "ratio") -> plt.Figure:
    """One panel per domain plus the pooled panel, one point per protocol with its 95% interval
    (filled = hierarchical, open = flat cross-check when the table has flat_ratio). scale
    "ratio": error ratio on a log axis, dashed line at 1, below 1 beats the saturated-item
    prediction. scale "score": prediction score -log2(ratio), halvings of the error as `score`,
    linear axis, line at 0, above 0 beats it."""
    assert scale in ("ratio", "score"), scale
    tf = (lambda v: -np.log2(v)) if scale == "score" else (lambda v: v)
    panels = [d for d in [*DOMAINS, POOLED] if (table.domain == d).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.1 * len(panels) + 0.6, 2.0), sharey=True)
    protocols = [p for p in PAPER_PROTOCOLS if (table.protocol == p).any()]
    x = np.arange(len(protocols))
    for ax, domain in zip(np.atleast_1d(axes), panels):
        sub = table[table.domain == domain].set_index("protocol").reindex(protocols)
        for i, p in enumerate(protocols):
            r = sub.loc[p]
            mid, lo, hi = sorted([tf(r.ratio), tf(r.lo), tf(r.hi)])[1], min(tf(r.lo), tf(r.hi)), max(tf(r.lo), tf(r.hi))
            ax.errorbar(i - 0.12, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", ms=3.5, lw=0.8, color=COLORS[p], capsize=0)
            if "flat_ratio" in sub and np.isfinite(r.flat_ratio):
                fmid, flo, fhi = tf(r.flat_ratio), min(tf(r.flat_lo), tf(r.flat_hi)), max(tf(r.flat_lo), tf(r.flat_hi))
                ax.errorbar(i + 0.12, fmid, yerr=[[fmid - flo], [fhi - fmid]], fmt="o", ms=3.5, lw=0.8, mfc="white", color=COLORS[p], capsize=0)
        ax.axhline(tf(1), ls="--", lw=0.6, color="0.4")
        if scale == "ratio":
            ax.set_yscale("log")
        ax.set_xticks(x, [LABELS[p] for p in protocols], rotation=60, ha="right")
        ax.set_xlim(-0.6, len(protocols) - 0.4)
        ax.set_title(DOMAIN_LABELS.get(domain, "Pooled"), fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("MSE ratio to prediction\nfrom saturated items" if scale == "ratio" else "Prediction score\n(halvings of error)")
    if "flat_ratio" in table:
        handles = [plt.Line2D([], [], marker="o", ms=3.5, lw=0.8, color="0.3", label="Hierarchical fit (partial pooling), 95% posterior interval"),
                   plt.Line2D([], [], marker="o", ms=3.5, lw=0.8, color="0.3", mfc="white", label="Maximum-likelihood fit (no pooling), 95% Wald interval")]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, handlelength=1.5, fontsize=6)
    fig.tight_layout()
    return fig


def posterior_figure(posterior, coords: dict, scale: str = "ratio") -> plt.Figure:
    """Posterior densities of the protocol effects. scale "ratio": error ratio exp(alpha), log x,
    dashed line at 1, below 1 beats the baseline. scale "score": prediction score
    -log2(exp(alpha)), halvings of the error as `score`, linear x, line at 0, above 0 beats the baseline. Top left: the pooled
    protocol effects, one curve per protocol. Top right: the domain effects exp(gamma), the
    baseline's error relative to the grand mean over domains (always a ratio), one curve per domain. Bottom: the
    per-domain protocol effects exp(alpha + delta), one small panel per protocol with one curve
    per domain, so "below 1 in every curve" = beats the baseline in every domain and the spread
    of the curves = the (protocol | domain) term. Protocol hue is used only in the top-left
    panel, domain hue in the other two."""
    from scipy.stats import gaussian_kde

    alpha = np.asarray(posterior["protocol_effects"]); alpha = alpha.reshape(-1, alpha.shape[-1])
    gamma = np.asarray(posterior["domain_effects"]); gamma = gamma.reshape(-1, gamma.shape[-1])
    delta = np.asarray(posterior["interaction"]); delta = delta.reshape(-1, *delta.shape[-2:])
    assert scale in ("ratio", "score"), scale
    protocols = [p for p in PAPER_PROTOCOLS if p in coords["protocol"]]
    domains = list(coords["domain"])
    to_score = lambda x: -x / np.log(2)  # the log effect to the prediction score, halvings of the error

    bars: dict = {}  # axis -> [(lo, mid, hi, color, row)], drawn above the curves once the panel's height is known

    def density(ax, draws: np.ndarray, color: str, label: str | None = None, row: int = 0, tf=np.exp) -> None:
        """One proper density (unit area on the plotted axis: in log x for the ratio scale, in
        score for the score scale), so a tighter posterior is a taller curve; its 95% interval
        and median are queued for a bar above the curves."""
        if tf is np.exp:  # log axis: the density of the log effect, drawn at exp(grid)
            grid = np.linspace(draws.min(), draws.max(), 200)
            x, y = np.exp(grid), gaussian_kde(draws)(grid)
        else:
            values = tf(draws)
            x = np.linspace(values.min(), values.max(), 200)
            y = gaussian_kde(values)(x)
        ax.fill_between(x, y, color=color, alpha=0.2, lw=0)
        ax.plot(x, y, color=color, lw=1, label=label)
        lo, mid, hi = sorted(tf(np.percentile(draws, [2.5, 50, 97.5])))
        bars.setdefault(ax, []).append((lo, mid, hi, color, row))

    fig = plt.figure(figsize=(5.5, 3.6))
    gs = fig.add_gridspec(2, len(protocols), height_ratios=[1.3, 1], hspace=0.9, wspace=0.35)
    top = gs[0, :].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=0.3)  # the protocol panel holds more curves than the domain panel
    top_l = fig.add_subplot(top[0])
    top_r = fig.add_subplot(top[1])
    tf = to_score if scale == "score" else np.exp
    for i, p in enumerate(protocols):
        density(top_l, alpha[:, coords["protocol"].index(p)], COLORS[p], LABELS[p], row=i, tf=tf)
    for i, d in enumerate(domains):  # sum-to-zero: every domain is a departure from the grand mean
        density(top_r, gamma[:, domains.index(d)], DOMAIN_COLORS[d], DOMAIN_LABELS[d], row=i)
    top_l.set_xlabel("Prediction score (halvings of error)" if scale == "score" else "MSE ratio to saturated-item prediction")
    top_r.set_xlabel("Baseline MSE ratio to grand mean")
    top_l.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, columnspacing=0.8, handlelength=1.2)
    top_r.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, columnspacing=0.8, handlelength=1.2)
    bottom = []
    for i, p in enumerate(protocols):
        ax = fig.add_subplot(gs[1, i], sharey=bottom[0] if bottom else None)
        bottom.append(ax)
        for j, d in enumerate(domains):
            k = coords["protocol"].index(p)
            density(ax, alpha[:, k] + delta[:, k, domains.index(d)], DOMAIN_COLORS[d], row=j, tf=tf)
        ax.text(0.02, 1.02, LABELS[p], transform=ax.transAxes, fontsize=7, color="0.35", va="bottom")
        ax.set_xlabel(("Prediction score (halvings of error), per domain" if scale == "score" else "MSE ratio to saturated-item prediction, per domain") if i == len(protocols) // 2 else "")
        if scale == "ratio":
            ax.set_xlim(0.02, 5)
        if i:
            ax.tick_params(labelleft=False)
    from matplotlib.ticker import NullLocator
    score_axes = [top_l, *bottom] if scale == "score" else []
    for ax, ticks in ((top_l, None if scale == "score" else [0.1, 0.3, 1, 3]), (top_r, [0.5, 1, 2]), *((ax, None if scale == "score" else [0.1, 1, 3]) for ax in bottom)):
        if ax not in score_axes:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())  # log minor ticks clutter panels this small
        if ticks is None:  # halvings: whole numbers over the drawn range
            lo, hi = ax.get_xlim()
            ticks = list(np.arange(np.ceil(lo), np.floor(hi) + 1))
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
        ax.axvline(0 if ax in score_axes else 1, ls="--", lw=0.8, color="0.5")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax, items in bars.items():
        top = max(line.get_ydata().max() for line in ax.get_lines() if len(line.get_xdata()) > 2)
        for lo, mid, hi, color, row in items:
            level = top * (1.12 + 0.1 * row)
            ax.plot([lo, hi], [level, level], color=color, lw=1, solid_capstyle="butt")
            ax.plot(mid, level, "o", ms=2.5, color=color)
    top_l.set_ylabel("Posterior density", labelpad=2)
    bottom[0].set_ylabel("Posterior density", labelpad=2)
    return fig


def per_model(table: pd.DataFrame) -> pd.DataFrame:
    """Squared error pooled over domains: one row per target model, one column per protocol in
    PROTOCOLS, the mean of y over the model's domains. A model missing a protocol in any domain
    gets NaN there."""
    wide = table.pivot_table(index="model", columns=["protocol", "domain"], values="y")
    complete = wide.T.groupby(level="protocol").count().T.eq(len(DOMAINS))
    return wide.T.groupby(level="protocol").mean().T.where(complete).reindex(columns=PROTOCOLS)


def model_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Prediction score per target model: `score` of the model's squared error pooled over its
    domains (`per_model`) against the saturated-item prediction's pooled the same way. Pooling
    first keeps one lucky near-exact cell from dominating a model (scored per domain, an error of
    0.0000 against a baseline of 2 is 17 halvings). One row per model, one column per protocol
    in PROTOCOLS (the baseline's column is 0); NaN where a model lacks the protocol in any domain."""
    pooled = per_model(table)
    return pd.concat({p: score(pooled[p], pooled[BASELINE]) for p in pooled.columns}, axis=1).reindex(columns=PROTOCOLS)


def score_axis(values: np.ndarray) -> tuple[float, float]:
    """Axis limits for prediction scores: the data's range rounded out to whole halvings, always including 0."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    return float(min(0, np.floor(lo))), float(max(0, np.ceil(hi)))


def hodges_lehmann_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """The exact (1 - alpha) Hodges-Lehmann interval of one sample: order statistics of the
    Walsh averages (v_i + v_j) / 2 at the critical value of the signed-rank statistic's exact
    null distribution, so 0 lies outside it exactly when the two-sided Wilcoxon signed-rank
    test of `values` against 0 rejects at `alpha`."""
    v = np.asarray(values, float)
    n = len(v)
    walsh = np.sort([(v[i] + v[j]) / 2 for i in range(n) for j in range(i, n)])
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1
    for r in range(1, n + 1):  # each rank r is in T+ with probability 1/2
        dist = (dist + np.roll(dist, r)) / 2
    c = int(np.max(np.where(np.cumsum(dist) <= alpha / 2)[0]))
    return float(walsh[c]), float(walsh[len(walsh) - c - 1])


def two_lines(label: str) -> str:
    """A tick label broken at its last space, for figure one's 30-degree two-line labels."""
    return "\n".join(label.rsplit(" ", 1))


def stars(pval: float) -> str:
    return "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""


def model_bar_figure(pooled: pd.DataFrame, tests: pd.DataFrame, metric: str = "mse") -> plt.Figure:
    """One bar per protocol over the target models pooled over domains, one small filled circle
    per model (jittered), stars from `tests`. metric "mse": bar at the mean squared error with a
    capless 95% t-interval over models, the saturated-item prediction's mean as a dashed line,
    far-high values compressed into a band above the axis (compress; the break glyph is the
    user's). metric "delta": each model's error minus its saturated-item error, same drawing
    with the line at 0. metric "ratio": each model's error divided by its saturated-item error,
    log axis, bars from 1 to the geometric mean with the t-interval of the log ratios. metric
    "score": prediction score per model (`score`, halvings of the error), line at 0, axis over
    the data in whole halvings."""
    assert metric in ("mse", "delta", "ratio", "score"), metric
    protocols = [p for p in PAPER_PROTOCOLS if pooled[p].notna().any()]
    values = {p: pooled[p].dropna() for p in protocols}
    if metric == "ratio":
        values = {p: np.log(v / pooled.loc[v.index, BASELINE]) for p, v in values.items()}
    elif metric == "score":
        values = {p: score(v, pooled.loc[v.index, BASELINE]) for p, v in values.items()}
    elif metric == "delta":
        values = {p: v - pooled.loc[v.index, BASELINE] for p, v in values.items()}
    means = {p: float(v.mean()) for p, v in values.items()}
    halves = {p: ci95(v.to_numpy()) for p, v in values.items()}
    baseline = float(pooled[BASELINE].mean()) if metric == "mse" else 0.0
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    if metric != "ratio":
        raw = np.array(list(means.values()) + [v for s in values.values() for v in s] + [baseline])
        if metric == "score":
            positions, mapping, extent = raw.copy(), {}, None
        else:
            _, first = compress(raw)
            extent = np.array([means[p] + halves[p] for p in means if means[p] not in first])
            positions, mapping = compress(raw, extent)
        place = dict(zip(raw.tolist(), positions.tolist()))
    else:
        mapping, extent, raw = {}, None, np.array([])
        place = None
    for i, p in enumerate(protocols):
        mean = means[p]
        if metric != "ratio":
            ax.bar(i, place[mean], color=COLORS[p], width=0.72, zorder=1)
            if mean not in mapping:
                cap = break_edges(raw, mapping, extent)[0]
                lo, hi = mean - halves[p], mean + halves[p]
                ax.plot([i, i], [lo, min(hi, cap)], color="black", lw=0.8, zorder=4)
            ys = [place[float(v)] for v in values[p]]
        else:
            ax.bar(i, np.exp(mean) - 1, bottom=1, color=COLORS[p], width=0.72, zorder=1)
            ax.plot([i, i], np.exp([mean - halves[p], mean + halves[p]]), color="black", lw=0.8, zorder=4)
            ys = np.exp(values[p]).tolist()
        xs = i + rng.uniform(-0.22, 0.22, len(ys))
        ax.scatter(xs, ys, s=5, color="#333333", linewidth=0, zorder=5, clip_on=False)  # markers on the -1 edge stay whole
        star = stars(float(tests.set_index("protocol").loc[p, "p"]))
        if star:
            ax.text(i, max(ys), star, ha="center", va="bottom", fontsize=7, zorder=10)  # above the break strip
    if metric != "ratio":
        ax.axhline(place[baseline], color="grey", lw=0.8, ls="--", zorder=0)
        if mapping:
            inside = [t for t in ax.get_yticks() if t <= max(v for v in raw if v not in mapping)]
            ax.set_yticks(inside + list(mapping.values()), [f"{t:g}" for t in inside] + [f"{v:.0f}" for v in mapping])
            break_strip(ax, raw, mapping, extent)
        ax.set_ylabel({"mse": "Prediction error\n(squared error, θ)", "delta": "Δ prediction error vs\nsaturated-item prediction",
                       "score": "Prediction score (%)"}[metric])
        if metric == "score":
            lo, hi = score_axis(raw)
            ax.set_ylim(lo, hi)
            ax.set_yticks(np.arange(lo, hi + 1, 1))
            ax.spines["left"].set_bounds(lo, hi)
    else:
        ax.axhline(1, color="grey", lw=0.8, ls="--", zorder=0)
        ax.set_yscale("log")
        ax.set_ylabel("Error ratio to prediction\nfrom saturated items")
    ax.set_xticks(range(len(protocols)), [two_lines(LABELS[p]) for p in protocols])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", multialignment="right")  # figure one
    fix_axes_height(fig, ax)
    return fig


def model_violin_figure(pooled: pd.DataFrame, tests: pd.DataFrame) -> plt.Figure:
    """Prediction score per model (`score`) as one violin per protocol
    in its colour, the same points, line at 0, stars and axis (-50 to 100, lower values clipped to
    -50) as model_bar_figure("score"); a black tick at the median."""
    protocols = [p for p in PAPER_PROTOCOLS if pooled[p].notna().any()]
    values = {p: score(pooled[p].dropna(), pooled.loc[pooled[p].notna(), BASELINE]).clip(lower=-50) for p in protocols}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    parts = ax.violinplot([values[p].to_numpy() for p in protocols], positions=range(len(protocols)), widths=0.8,
                          showextrema=False, showmedians=False)
    for body, p in zip(parts["bodies"], protocols):
        body.set_facecolor(COLORS[p]); body.set_edgecolor("none"); body.set_alpha(0.6); body.set_zorder(1)
    for i, p in enumerate(protocols):
        ys = values[p].to_numpy()
        ax.plot([i - 0.2, i + 0.2], [np.median(ys)] * 2, color="black", lw=1, zorder=4)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(ys)), ys, s=5, color="#333333", linewidth=0, zorder=5, clip_on=False)
        star = stars(float(tests.set_index("protocol").loc[p, "p"]))
        if star:
            ax.text(i, 100, star, ha="center", va="bottom", fontsize=7, zorder=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=0)
    ax.set_ylim(-50, 100)
    ax.set_yticks([-50, 0, 50, 100])
    ax.spines["left"].set_bounds(-50, 100)
    ax.set_ylabel("Prediction score (%)")
    ax.set_xticks(range(len(protocols)), [two_lines(LABELS[p]) for p in protocols])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", multialignment="right")  # figure one
    fix_axes_height(fig, ax)
    return fig


def model_box_figure(scores: pd.DataFrame, tests: pd.DataFrame, statistic: str = "mean") -> plt.Figure:
    """Prediction score per protocol as a centre-and-interval box. statistic "mean": a large
    filled circle at the mean over models, a translucent box spanning the 95% t-interval of the
    mean; "median": circle at the median, box the Hodges-Lehmann 95% interval. The per-model
    points in the protocol's colour (translucent, jittered); line at 0, stars and axis (-50 to
    100, points beyond it drawn on its edges) as model_violin_figure. The centre and interval
    use the unclipped scores."""
    protocols = [p for p in PAPER_PROTOCOLS if scores[p].notna().any()]
    values = {p: scores[p].dropna() for p in protocols}
    rng = np.random.default_rng(0)
    bottom, top = score_axis(np.concatenate([v.to_numpy() for v in values.values()]))
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    for i, p in enumerate(protocols):
        v = values[p].to_numpy()
        if statistic == "mean":
            centre, half = float(v.mean()), ci95(v)
            lo, hi = centre - half, centre + half
        else:
            centre, (lo, hi) = float(np.median(v)), hodges_lehmann_interval(v)
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, facecolor=COLORS[p], edgecolor="none", alpha=0.35, zorder=1))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, color=COLORS[p], alpha=0.5, linewidth=0, zorder=3)
        ax.scatter([i], [centre], s=28, color=COLORS[p], linewidth=0, zorder=5)
        star = stars(float(tests.set_index("protocol").loc[p, "p"]))
        if star:
            ax.text(i, top, star, ha="center", va="bottom", fontsize=7, zorder=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-0.6, len(protocols) - 0.4)
    ax.set_ylim(bottom, top)
    ax.set_yticks(np.arange(bottom, top + 1, 1))
    ax.spines["left"].set_bounds(bottom, top)
    ax.set_ylabel("Prediction score\n(halvings of error)")
    ax.set_xticks(range(len(protocols)), [two_lines(LABELS[p]) for p in protocols])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", multialignment="right")  # figure one
    fix_axes_height(fig, ax)
    return fig


def contrast_table(posterior, coords: dict, against: str = "pairwise") -> pd.DataFrame:
    """Error ratios of `against` to every other non-baseline protocol, from the same posterior
    draws: one row per protocol x domain (exp of the difference of the two protocols' alpha +
    delta in that domain) plus a pooled row (domain "all", exp(alpha_against - alpha_p)).
    ratio, lo, hi = posterior median and 2.5 / 97.5 percentiles; p_below_1 = posterior mass
    with `against` the smaller error. Shared cell, domain and judge terms cancel in the
    difference, so these intervals are tighter than the two marginal ones suggest."""
    alpha = np.asarray(posterior["protocol_effects"]); alpha = alpha.reshape(-1, alpha.shape[-1])
    delta = np.asarray(posterior["interaction"]); delta = delta.reshape(-1, *delta.shape[-2:])
    a = coords["protocol"].index(against)
    out = []
    for p, protocol in enumerate(coords["protocol"]):
        if protocol in (BASELINE, against):
            continue
        cells = {d: (alpha[:, a] + delta[:, a, i]) - (alpha[:, p] + delta[:, p, i]) for i, d in enumerate(coords["domain"])}
        cells[POOLED] = alpha[:, a] - alpha[:, p]
        for domain, draws in cells.items():
            lo, mid, hi = np.percentile(np.exp(draws), [2.5, 50, 97.5])
            out.append(dict(against=against, protocol=protocol, domain=domain, ratio=mid, lo=lo, hi=hi, p_below_1=float((draws < 0).mean())))
    return pd.DataFrame(out)


def contrast_figure(table: pd.DataFrame, scale: str = "ratio") -> plt.Figure:
    """One row per protocol: the pooled contrast of `against` to it (black, filled, thick
    interval) and the four per-domain contrasts (domain colours, open, thin). scale "ratio":
    error ratio, log x, line at 1, below 1 = `against` has the smaller error. scale "score":
    -log2(ratio), halvings of the error as `score`, linear x, line at 0, above 0 = `against` has
    the smaller error."""
    assert scale in ("ratio", "score"), scale
    tf = (lambda x: -np.log2(x)) if scale == "score" else (lambda x: x)
    protocols = [p for p in PAPER_PROTOCOLS if (table.protocol == p).any()]
    fig, ax = plt.subplots(figsize=(2.6, 0.35 * len(protocols) + 0.9), layout="constrained")
    for i, p in enumerate(protocols):
        sub = table[table.protocol == p].set_index("domain")
        for j, d in enumerate([d for d in DOMAINS if d in sub.index]):
            r = sub.loc[d]; y = i + 0.3 - 0.15 * j
            ax.plot([tf(r.lo), tf(r.hi)], [y, y], color=DOMAIN_COLORS[d], lw=0.6, zorder=2)
            ax.plot(tf(r.ratio), y, "o", ms=2.5, mfc="white", mec=DOMAIN_COLORS[d], mew=0.6, zorder=3)
        r = sub.loc[POOLED]
        ax.plot([tf(r.lo), tf(r.hi)], [i - 0.35, i - 0.35], color="black", lw=1.0, zorder=4)
        ax.plot(tf(r.ratio), i - 0.35, "o", ms=3.5, color="black", zorder=5)
    ax.axvline(tf(1), ls="--", lw=0.8, color="grey", zorder=0)
    if scale == "ratio":
        ax.set_xscale("log")
    ax.set_yticks(range(len(protocols)), [LABELS[p] for p in protocols])
    ax.set_ylim(len(protocols) - 0.4, -0.7)
    against = table.against.iloc[0]
    ax.set_xlabel(f"Error ratio, {LABELS[against]} vs protocol" if scale == "ratio" else f"Halvings of error, {LABELS[against]} vs protocol")
    handles = [plt.Line2D([], [], color="black", marker="o", ms=3.5, lw=1.0, label="Pooled")] + \
              [plt.Line2D([], [], color=DOMAIN_COLORS[d], marker="o", mfc="white", ms=2.5, lw=0.6, label=DOMAIN_LABELS[d]) for d in DOMAINS]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, handlelength=1.5)
    return fig


def _domain_axis(ax, labels: list[str]) -> None:
    """Categorical x axis: pooled then the four domains, spines trimmed to the ticks."""
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.spines["bottom"].set_bounds(0, len(labels) - 1)


def _ratio_axis(ax, ticks: list[float], label: str | None) -> None:
    """Log MSE-ratio y axis with the equal-error line, spine trimmed to the ticks."""
    ax.axhline(1, color="0.5", ls="--", lw=0.8, zorder=0)
    ax.set(yscale="log", ylim=(ticks[0], ticks[-1]))
    ax.set_yticks(ticks, [f"{t:g}".replace("0.", ".") for t in ticks])
    ax.minorticks_off()
    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
    if label:
        ax.set_ylabel(label)


def _point(ax, x: float, row, color: str, domain: str) -> None:
    """One estimate: a 95% interval as a vertical line and a marker, filled when pooled, open for a domain."""
    pooled = domain == POOLED
    ax.plot([x, x], [row.lo, row.hi], color=color, lw=1 if pooled else 0.8, solid_capstyle="butt")
    ax.plot(x, row.ratio, marker="o" if pooled else DOMAIN_MARKERS[domain], ms=4 if pooled else 3.5,
            mfc=color if pooled else "white", mec=color, mew=0.9, ls="none", clip_on=False)


def appendix_effect_figure(table: pd.DataFrame) -> plt.Figure:
    """Posterior MSE ratios to the baseline: one panel per paper protocol, pooled then each domain
    along x, so a protocol's domain dependence reads as the spread of its own points."""
    protocols = [p for p in PAPER_PROTOCOLS if (table.protocol == p).any()]
    domains = [POOLED, *DOMAINS]
    labels = ["Pooled", *(DOMAIN_LABELS[d] for d in DOMAINS)]
    ticks = [0.05, 0.25, 1, 4]
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.5, 1.9), layout="constrained", sharey=True)
    for ax, protocol in zip(axes, protocols):
        rows = table[table.protocol == protocol].set_index("domain")
        for x, domain in enumerate(domains):
            if domain in rows.index:
                _point(ax, x, rows.loc[domain], COLORS[protocol], domain)
        _domain_axis(ax, labels)
        ax.tick_params(axis="x", rotation=90)
        _ratio_axis(ax, ticks, "MSE ratio to baseline" if ax is axes[0] else None)
        ax.set_title(LABELS[protocol], loc="left", fontsize=7, fontweight="bold", pad=3)
    return fig


def appendix_error_figure(table: pd.DataFrame) -> plt.Figure:
    """Expected squared error of the baseline and each paper protocol (error_table), one panel
    for the pooled estimate and one per domain; the baseline's median as a dashed line."""
    protocols = [BASELINE, *[p for p in PAPER_PROTOCOLS if (table.protocol == p).any()]]
    domains = [POOLED, *DOMAINS]
    ticks = [0.05, 0.25, 1, 4]
    fig, axes = plt.subplots(1, len(domains), figsize=(5.5, 1.7), layout="constrained", sharey=True)
    for ax, domain in zip(axes, domains):
        rows = table[table.domain == domain].set_index("protocol")
        marker = "o" if domain == POOLED else DOMAIN_MARKERS[domain]
        ax.axhline(rows.loc[BASELINE, "mse"], color="0.5", ls="--", lw=0.8, zorder=0)
        for x, protocol in enumerate(protocols):
            row = rows.loc[protocol]
            color = "0.45" if protocol == BASELINE else COLORS[protocol]
            ax.plot([x, x], [row.lo, row.hi], color=color, lw=1, solid_capstyle="butt")
            ax.plot(x, row.mse, marker=marker, ms=4, color=color, ls="none", clip_on=False)
        ax.set_xticks([])
        ax.set_xlim(-0.5, len(protocols) - 0.5)
        ax.spines["bottom"].set_visible(False)
        ax.set(yscale="log", ylim=(ticks[0], ticks[-1]))
        ax.set_yticks(ticks, [f"{t:g}".replace("0.", ".") for t in ticks])
        ax.minorticks_off()
        ax.spines["left"].set_bounds(ticks[0], ticks[-1])
        if ax is axes[0]:
            ax.set_ylabel("Squared error")
        ax.set_title("Pooled" if domain == POOLED else DOMAIN_LABELS[domain], loc="left", fontsize=7, fontweight="bold", pad=3)
    handles = [plt.Line2D([], [], color="0.45" if p == BASELINE else COLORS[p], marker="o", ms=4, lw=0,
                          label="Baseline" if p == BASELINE else LABELS[p]) for p in protocols]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False, handletextpad=0.3, columnspacing=0.8)
    return fig


def appendix_contrast_figure(contrasts: pd.DataFrame, effects: pd.DataFrame) -> plt.Figure:
    """a: Pairwise judge's MSE against each comparator, pooled then per domain, colour the comparator.
    b: each protocol's between-domain standard deviation of log MSE."""
    comparators = [p for p in PAPER_PROTOCOLS if p != "pairwise" and (contrasts.protocol == p).any()]
    domains = [POOLED, *DOMAINS]
    labels = ["Pooled", *(DOMAIN_LABELS[d] for d in DOMAINS)]
    ticks = [0.05, 0.25, 1, 4]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), layout="constrained", width_ratios=[1.6, 1])
    ax = axes[0]
    offsets = np.linspace(-0.22, 0.22, len(comparators))
    for offset, protocol in zip(offsets, comparators):
        rows = contrasts[contrasts.protocol == protocol].set_index("domain")
        for x, domain in enumerate(domains):
            if domain in rows.index:
                _point(ax, x + offset, rows.loc[domain], COLORS[protocol], domain)
    _domain_axis(ax, labels)
    ax.tick_params(axis="x", rotation=90)
    _ratio_axis(ax, ticks, "Pairwise / comparator MSE")
    ax.text(-0.3, 1.02, "a", transform=ax.transAxes, fontweight="bold")
    handles = [plt.Line2D([], [], color=COLORS[p], marker="o", ms=4, lw=0, label=LABELS[p]) for p in comparators]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)

    ax = axes[1]
    sigma = effects[effects.domain == POOLED].set_index("protocol")
    protocols = [p for p in PAPER_PROTOCOLS if p in sigma.index]
    for x, protocol in enumerate(protocols):
        row = sigma.loc[protocol]
        ax.plot([x, x], [row.sigma_lo, row.sigma_hi], color=COLORS[protocol], lw=1, solid_capstyle="butt")
        ax.plot(x, row.sigma_protocol, "o", color=COLORS[protocol], ms=4, clip_on=False)
    sigma_ticks = [0, 0.5, 1, 1.5]
    ax.set_xticks(range(len(protocols)), [two_lines(LABELS[p]) for p in protocols], rotation=90)
    ax.set_xlim(-0.5, len(protocols) - 0.5)
    ax.set(ylim=(sigma_ticks[0], sigma_ticks[-1]), ylabel="Between-domain SD")
    ax.set_yticks(sigma_ticks)
    ax.spines["bottom"].set_bounds(0, len(protocols) - 1)
    ax.spines["left"].set_bounds(sigma_ticks[0], sigma_ticks[-1])
    ax.text(-0.5, 1.02, "b", transform=ax.transAxes, fontweight="bold")
    return fig


def appendix_fit_figure(idata, features: dict, table: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Observed versus posterior-predicted squared error, coverage and log-error R²."""
    obs = np.asarray(idata.observed_data["obs"])
    mu = np.exp(np.median(log_mu_draws(idata.posterior, features), axis=0))
    pp = np.asarray(idata.posterior_predictive["obs"]).reshape(-1, obs.size)
    lo, hi = np.percentile(pp, [2.5, 97.5], axis=0)
    protocols = [p for p in [BASELINE, *PAPER_PROTOCOLS] if (table.protocol == p).any()]
    fig = plt.figure(figsize=(4.3, 3.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1])
    axes = np.atleast_1d(top.subplots(1, len(protocols), sharex=True, sharey=True))
    limits = (1e-6, 30)
    ticks = [1e-6, 1e-3, 1, 30]
    coverage = []
    for ax, protocol in zip(axes, protocols):
        rows = (table.protocol == protocol).to_numpy()
        color = COLORS.get(protocol, "black")
        ax.vlines(mu[rows], lo[rows], hi[rows], color="0.8", lw=0.4, zorder=1)
        ax.plot(mu[rows], obs[rows], "o", ms=1.8, color=color, mec="none", zorder=2)
        ax.plot(limits, limits, color="0.5", ls="--", lw=0.8, zorder=0)
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        ax.set_xticks(ticks, [r"$10^{-6}$", r"$10^{-3}$", "1", "30"])
        ax.set_yticks(ticks, [r"$10^{-6}$", r"$10^{-3}$", "1", "30"])
        ax.spines["bottom"].set_bounds(*limits); ax.spines["left"].set_bounds(*limits)
        ax.text(0.03, 0.97, two_lines(LABELS.get(protocol, protocol)), transform=ax.transAxes,
                va="top", color="0.3", fontsize=6)
        inside = ((obs[rows] >= lo[rows]) & (obs[rows] <= hi[rows])).mean()
        r2 = np.corrcoef(np.log(mu[rows]), np.log(obs[rows]))[0, 1] ** 2
        coverage.append(dict(protocol=protocol, n=int(rows.sum()), coverage95=inside, r2_log=r2))
    axes[0].set_ylabel("Observed squared error")
    axes[len(axes) // 2].set_xlabel("Expected squared error")
    axes[0].text(-0.35, 1.02, "a", transform=axes[0].transAxes, fontweight="bold", color="0.3")

    fit = pd.DataFrame(coverage)
    bottom_axes = bottom.subplots(1, 2)
    y = np.arange(len(protocols))[::-1]
    for ax, column, xlabel, reference in (
        (bottom_axes[0], "coverage95", "95% predictive coverage", 0.95),
        (bottom_axes[1], "r2_log", "Log-error R²", None),
    ):
        values = fit.set_index("protocol").loc[protocols, column]
        for yi, protocol, value in zip(y, protocols, values):
            ax.plot(value, yi, "o", ms=4, color=COLORS.get(protocol, "black"))
        if reference is not None:
            ax.axvline(reference, color="0.5", ls="--", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(-0.45, len(protocols) - 0.55), xlabel=xlabel)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks(y, [two_lines(LABELS.get(p, p)) for p in protocols])
        ax.spines["bottom"].set_bounds(0, 1); ax.spines["left"].set_bounds(y.min(), y.max())
    for ax, letter in zip(bottom_axes, "bc"):
        ax.text(-0.25, 1.02, letter, transform=ax.transAxes, fontweight="bold", color="0.3")
    return fig, fit


def log_mu_draws(posterior, features: dict) -> np.ndarray:
    """log mu per draw and row (draws x rows) from the fitted effects and the row indices in
    `features` (protocol_index, domain_index, model_domain_index, protocol_judge_index)."""
    get = lambda k: np.asarray(posterior[k]).reshape(-1, *np.asarray(posterior[k]).shape[2:])
    p, d, c, j = (np.asarray(features[k]) for k in ("protocol_index", "domain_index", "model_domain_index", "protocol_judge_index"))
    return (get("intercept")[:, None] + get("protocol_effects")[:, p] + get("domain_effects")[:, d]
            + get("interaction")[:, p, d] + get("cell_effects")[:, c] + get("judge_effects")[:, j])


def diagnostics(idata, features: dict) -> dict:
    """Fit and convergence summary of one model: divergences, max r_hat and min ESS over the
    free parameters, LOO (elpd, se, p_loo) with Pareto-k counts, and the Bayesian R^2 on the
    log scale (posterior median of var(log mu) / (var(log mu) + var(log y - log mu)))."""
    summ = az.summary(idata, var_names=["~cell_z", "~judge_z", "~interaction_z"], filter_vars="like", round_to="none")
    loo = az.loo(idata, pointwise=True)
    k = np.asarray(loo.pareto_k)
    y = np.log(np.asarray(idata.observed_data["obs"]))
    mu = log_mu_draws(idata.posterior, features)
    r2 = np.var(mu, axis=1) / (np.var(mu, axis=1) + np.var(y[None, :] - mu, axis=1))
    return dict(divergences=int(idata.sample_stats.diverging.sum()), r_hat_max=float(summ.r_hat.max()),
                ess_bulk_min=float(summ.ess_bulk.min()), ess_tail_min=float(summ.ess_tail.min()),
                elpd_loo=float(loo.elpd_loo), elpd_se=float(loo.se), p_loo=float(loo.p_loo),
                pareto_k_bad=int(((k > 0.7) & (k <= 1)).sum()), pareto_k_very_bad=int((k > 1).sum()), n=int(k.size),
                r2_log=float(np.median(r2)), r2_log_lo=float(np.percentile(r2, 2.5)), r2_log_hi=float(np.percentile(r2, 97.5)))


def convergence_figure(idata, coords: dict, table: pd.DataFrame, fig=None) -> plt.Figure:
    """Sampler and LOO diagnostics of one fit in three panels. Left: trace of each free protocol
    effect (log scale), the four chains overlaid, protocol colours. Middle: r_hat (x) and bulk ESS
    (marker size) of every free parameter summarised by az.summary, one strip per parameter
    family, dashed line at r_hat 1.01. Right: Pareto-k per observation from PSIS-LOO, coloured by
    protocol (from `table`, the rows in observation order), dashed lines at 0.7 (unreliable) and
    1 (very unreliable)."""
    protocols = coords["protocol"]
    post = idata.posterior
    fig = fig or plt.figure(figsize=(6.5, 1.8), layout="constrained")
    axes = fig.subplots(1, 3, width_ratios=[1.4, 1, 1.2])
    alpha = np.asarray(post["protocol_effects"])  # chains x draws x protocols
    for k, p in enumerate(protocols):
        if p == BASELINE:
            continue
        for chain in range(alpha.shape[0]):
            axes[0].plot(alpha[chain, :, k], color=COLORS[p], lw=0.3, alpha=0.6)
    axes[0].legend(handles=[plt.Line2D([], [], color=COLORS[p], lw=1.5, label=LABELS[p]) for p in protocols if p != BASELINE],
                   frameon=False, fontsize=5, loc="lower left", handlelength=1, handletextpad=0.4, ncol=2, columnspacing=0.6)
    axes[0].set_xlabel("draw"); axes[0].set_ylabel("protocol effect (log)")
    axes[0].set_xticks([0, alpha.shape[1]])
    summ = az.summary(idata, var_names=["~cell_z", "~judge_z", "~interaction_z"], filter_vars="like", round_to="none").dropna(subset=["r_hat"])
    family = summ.index.str.replace(r"\[.*", "", regex=True)
    families = [f for f in dict.fromkeys(family) if not f.endswith("_free")]  # the deterministic versions carry the same draws
    rng = np.random.default_rng(0)
    for i, fam in enumerate(families):
        rows = summ[family == fam]
        axes[1].scatter(rows.r_hat, i + rng.uniform(-0.2, 0.2, len(rows)), s=np.clip(rows.ess_bulk / 100, 2, 30), color="0.3", alpha=0.6, linewidth=0)
    axes[1].axvline(1.01, ls="--", lw=0.8, color="0.5")
    axes[1].set_yticks(range(len(families)), families, fontsize=6)
    axes[1].set_xlabel("r_hat (marker size: bulk ESS)")
    axes[1].set_xlim(0.995, max(1.015, float(summ.r_hat.max()) + 0.003))
    k = np.asarray(az.loo(idata, pointwise=True).pareto_k)
    assert len(table) == k.size, (len(table), k.size)
    for p in protocols:
        rows = np.flatnonzero((table.protocol == p).to_numpy())
        axes[2].scatter(rows, k[rows], s=3, color=COLORS.get(p, "black"), linewidth=0, label=LABELS.get(p, "Saturated items"))
    for level in (0.7, 1.0):
        axes[2].axhline(level, ls="--", lw=0.8, color="0.5")
    axes[2].set_xlabel("observation"); axes[2].set_ylabel("Pareto-k")
    axes[2].set_ylim(top=max(1.3, float(k.max()) + 0.1) + 0.5)  # room for the legend above the lines
    axes[2].legend(frameon=False, fontsize=5, ncol=2, loc="upper left", handletextpad=0.2, columnspacing=0.6, markerscale=1.5)
    return fig


def fit_quality_figure(idata, features: dict, coords: dict, table: pd.DataFrame) -> plt.Figure:
    """The fit and convergence evidence in one figure, three rows: posterior predictive densities
    per protocol (predictive_figure), observed against predicted with 95% predictive intervals
    (fit_figure), and the sampler diagnostics (convergence_figure), with the scalar summary
    (divergences, max r_hat, min ESS, Bayesian R^2, Pareto-k counts) from `diagnostics` as the
    title line. Panel letters a to c."""
    d = diagnostics(idata, features)
    fig = plt.figure(figsize=(6.5, 5.6), layout="constrained")
    rows = fig.subfigures(3, 1, height_ratios=[1, 1.1, 1.3], hspace=0.06)
    predictive_figure(idata, table, fig=rows[0])
    fit_figure(idata, features, table, fig=rows[1])
    convergence_figure(idata, coords, table, fig=rows[2])
    for row, letter in zip(rows, "abc"):
        row.text(0.0, 1.0, letter, fontsize=9, fontweight="bold", va="top", ha="left")
    fig.suptitle(f"divergences {d['divergences']}, max r_hat {d['r_hat_max']:.3f}, min bulk ESS {d['ess_bulk_min']:.0f}, "
                 f"min tail ESS {d['ess_tail_min']:.0f}, Bayesian R² (log) {d['r2_log']:.2f} [{d['r2_log_lo']:.2f}, {d['r2_log_hi']:.2f}], "
                 f"Pareto-k > 0.7: {d['pareto_k_bad'] + d['pareto_k_very_bad']} of {d['n']} (> 1: {d['pareto_k_very_bad']})",
                 fontsize=6.5, color="0.3")
    return fig


def loo_compare(idatas: dict) -> pd.DataFrame:
    """arviz.compare over the named fits (elpd_loo, elpd_diff, weight, se, dse, warning)."""
    return az.compare(idatas, ic="loo", scale="log")


def predictive_figure(idata, table: pd.DataFrame, draws: int = 40, seed: int = 0, fig=None) -> plt.Figure:
    """Posterior predictive check on the log scale, one small panel per protocol: the observed
    log squared errors' density (black) over `draws` posterior predictive densities (thin grey)."""
    from scipy.stats import gaussian_kde

    obs = np.log(np.asarray(idata.observed_data["obs"]))
    pp = np.log(np.asarray(idata.posterior_predictive["obs"]).reshape(-1, obs.size))
    pick = np.random.default_rng(seed).choice(pp.shape[0], draws, replace=False)
    protocols = [p for p in PROTOCOLS if (table.protocol == p).any()]
    fig = fig or plt.figure(figsize=(1.0 * len(protocols) + 0.5, 1.5), layout="constrained")
    axes = fig.subplots(1, len(protocols), sharex=True, sharey=True)
    grid = np.linspace(-10, 6, 200)
    for ax, p in zip(axes, protocols):
        rows = (table.protocol == p).to_numpy()
        for i in pick:
            ax.plot(grid, gaussian_kde(pp[i, rows])(grid), color="0.75", lw=0.4, zorder=1)
        ax.plot(grid, gaussian_kde(obs[rows])(grid), color="black", lw=1.0, zorder=2)
        ax.set_title(LABELS.get(p, "Saturated items"), fontsize=7, color="0.3", loc="left")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[0].set_xlabel("log squared error (lower = better forecast)")
    return fig


def fit_figure(idata, features: dict, table: pd.DataFrame, fig=None) -> tuple[plt.Figure, pd.DataFrame]:
    """Observed against predicted per row, on the log scale, one panel per protocol: the
    posterior median of log mu on x, the observed log squared error on y, a thin vertical line
    per row for the 95% posterior predictive interval, and the identity line. Returns the figure
    and the per-protocol coverage of that interval with the log-scale R^2."""
    obs = np.log(np.asarray(idata.observed_data["obs"]))
    mu = np.median(log_mu_draws(idata.posterior, features), axis=0)
    pp = np.log(np.asarray(idata.posterior_predictive["obs"]).reshape(-1, obs.size))
    lo, hi = np.percentile(pp, [2.5, 97.5], axis=0)
    protocols = [p for p in PROTOCOLS if (table.protocol == p).any()]
    fig = fig or plt.figure(figsize=(1.0 * len(protocols) + 0.5, 1.6), layout="constrained")
    axes = fig.subplots(1, len(protocols), sharex=True, sharey=True)
    lim = (-14, 5)
    coverage = []
    for ax, p in zip(axes, protocols):
        r = (table.protocol == p).to_numpy()
        ax.vlines(mu[r], lo[r], hi[r], color="0.8", lw=0.4, zorder=1)
        ax.plot(mu[r], obs[r], "o", ms=1.8, color=COLORS.get(p, "black"), mec="none", zorder=2)
        ax.plot([-10, 5], [-10, 5], ls="--", lw=0.8, color="0.5", zorder=0)
        ax.set_title(LABELS.get(p, "Saturated items"), fontsize=7, color="0.3", loc="left")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xticks([-10, -5, 0, 5]); ax.set_yticks([-10, -5, 0, 5])
        ax.spines["left"].set_bounds(-10, 5); ax.spines["bottom"].set_bounds(-10, 5)
        inside = ((obs[r] >= lo[r]) & (obs[r] <= hi[r])).mean()
        coverage.append(dict(protocol=p, n=int(r.sum()), coverage95=inside, r2_log=np.corrcoef(mu[r], obs[r])[0, 1] ** 2))
    axes[0].set_xlabel("model-expected log squared error"); axes[0].set_ylabel("actual log squared error")
    return fig, pd.DataFrame(coverage)
