"""
The paired change in squared error that every transcript-parts figure draws, the items the cut-transcript conditions
are judged on (10 per difficulty quartile), and the figure and caption text shared by the thirds and channel analyses.

Each analysis predicts the same targets from transcripts with one part removed or kept alone, next to the full
transcript (condition "full") and the Baseline ("easy"). A row's change is the mean over a domain's targets of the
target's squared error under the row's condition minus under the comparison condition (the full transcript, or the
Baseline for a part kept alone), with its 95% t-interval and the two-sided paired t-test of it against 0 (the same
test: an interval excluding 0 is p < .05), the targets of every semester pooled.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import likelihood_ratio as lr
from analysis.plots import stars, unclip_markers
from capability.task import Cohort, Task
from protocols import embeddings
from protocols.pairwise.judge import stratified_items
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS

# comparison condition -> (its name in captions, on the x label)
AGAINST = {"full": ("the full transcript", "full transcript"), "easy": ("the Baseline", "baseline")}


def removal_items(task: Task, per_quartile: int, seed: int) -> set[str]:
    """
    The task's items every cut-transcript condition is judged on: a seeded draw of `per_quartile` train items from
    each quartile of item difficulty (b), or every train item when `per_quartile` is 0.

    Inputs:
        task: the task; uses task.references.items (the frozen train items: bench, item, b).
        per_quartile: items drawn per difficulty quartile (config.yaml removal_items.per_quartile, 10).
        seed: the draw's random seed (removal_items.seed).
    Outputs:
        set of item ids, as strings.
    """
    return set(map(str, stratified_items(task.references.items, per_quartile, seed)))


def on_items(cohort: Cohort, keep: set[str]) -> Cohort:
    """
    A copy of `cohort` with its items restricted to those in `keep`; its models are unchanged.

    Inputs:
        cohort: a task's references or targets (capability.task.Cohort); uses cohort.items.item.
        keep: the item ids (strings) to keep, e.g. from removal_items.
    Outputs:
        the restricted Cohort.
    """
    return replace(cohort, items=cohort.items[cohort.items.item.astype(str).isin(keep)])


def published_ratings(logs: Path, condition: str, domain: str, semester: str, judge: str) -> pd.DataFrame:
    """
    Reads the audited ratings of a published Absolute scoring run on cut transcripts from its ratings.csv.

    Inputs:
        logs: the logs/ checkout (paths.logs).
        condition: the run's condition, e.g. "channels-drop-reasoning" or "truncated-thirds-23"; the file read is
            <logs>/direct_<condition>/<domain>/<semester>/<judge>/<hash>/ratings.csv, of which exactly one must exist.
        domain: the task's domain key.
        semester: the task's semester.
        judge: the judge's shortname.
    Outputs:
        DataFrame of the file's rows with item read as a string; the callers use model, bench, item and score.
    """
    (path,) = Path(logs, f"direct_{condition}", domain, semester, judge).glob("*/ratings.csv")
    return pd.read_csv(path, dtype={"item": str})


def squared_errors(theta: pd.Series | float, truth: pd.Series, **columns: str) -> pd.DataFrame:
    """
    Each target's squared error of a prediction, labelled with constant columns.

    Inputs:
        theta: the predicted θ per model (a Series, aligned to the targets), or one number for every target (the
            Baseline: the references' mean θ*).
        truth: the targets' θ*, indexed by model.
        columns: label columns with one value each, e.g. domain, semester, condition.
    Outputs:
        DataFrame with the `columns`, model and err (the squared error, NaN for a target without a prediction); one
        row per target of `truth`.
    """
    if isinstance(theta, pd.Series):
        theta = theta.reindex(truth.index)
    return pd.DataFrame({**columns, "model": truth.index, "err": ((theta - truth) ** 2).to_numpy()})


def ridge_forecast(protocol: str, x: pd.DataFrame, refs: pd.Series, targets: list[str]) -> pd.Series:
    """
    Predicts the targets' θ with a ridge regression fitted on the references' rows of `x`, its penalty chosen by
    cross-validation (RidgeCV). A reference with a non-finite feature is left out of the fit and such a target is
    predicted NaN.

    Inputs:
        protocol: "embedding" (RidgeCV on the embedding vectors, with protocols.embeddings.ALPHAS); any other value
            is taken as "glm_regex" (RidgeCV on the standardised deterministic features, with the feature protocol's
            ALPHAS).
        x: one row of features per model, references and targets, indexed by model.
        refs: the references' θ*, indexed by model.
        targets: the target models.
    Outputs:
        Series of predicted θ, indexed by target.
    """
    if protocol == "embedding":
        estimator = RidgeCV(alphas=embeddings.ALPHAS)
    else:
        estimator = make_pipeline(StandardScaler(), RidgeCV(alphas=FEATURE_ALPHAS))
    return lr.feature_rule(estimator, x)(refs, targets)


def paired_t(diff: np.ndarray) -> dict[str, float]:
    """
    The mean of the per-target differences, its 95% t-interval and the two-sided one-sample t-test that it is 0.
    NaN differences are dropped; p is 1 when the differences have no spread.

    Inputs:
        diff: per-target differences in squared error (under a condition minus under the comparison condition).
    Outputs:
        dict with estimate (the mean), lo and hi (the interval's ends) and p.
    """
    diff = diff[np.isfinite(diff)]
    mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
    half = float(scipy_stats.t.ppf(0.975, len(diff) - 1)) * se
    p = float(scipy_stats.ttest_1samp(diff, 0).pvalue) if se > 0 else 1.0
    return dict(estimate=mean, lo=mean - half, hi=mean + half, p=p)


def change_table(errors: pd.DataFrame, rows: dict[str, str], against: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    For one domain, each row's mean change in squared error from the comparison condition, the targets paired within
    each semester and every semester pooled, with its interval and test (paired_t); and the domain's mean squared
    error with the full transcript and with the Baseline.

    Inputs:
        errors: one domain's per-target errors, columns semester, condition, model, err; conditions "full", "easy"
            and those `rows` names.
        rows: row label -> condition; a condition missing in any semester is left out.
        against: the comparison condition, "full" or "easy".
    Outputs:
        (DataFrame with columns row, estimate, lo, hi, p, one row per row label kept; dict {"full": ..., "easy": ...},
        the mean squared error of the full transcript and of the Baseline over every semester's targets).
    """
    wide = {s: g.pivot_table(index="model", columns="condition", values="err") for s, g in errors.groupby("semester")}
    present = set.intersection(*(set(w.columns) for w in wide.values()))
    table = pd.DataFrame([dict(row=label, **paired_t(np.concatenate([(wide[s][c] - wide[s][against]).to_numpy(float)
                                                                     for s in wide])))
                          for label, c in rows.items() if c in present])
    mean = lambda c: float(np.concatenate([wide[s][c].to_numpy() for s in wide]).mean())  # noqa: E731
    return table, dict(full=mean("full"), easy=mean("easy"))


def domain_changes(errors: pd.DataFrame, rows: dict[str, str], against: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs change_table for every domain in `errors`.

    Inputs:
        errors: per-target errors, columns domain, semester, condition, model, err.
        rows: row label -> condition, as in change_table.
        against: the comparison condition, "full" or "easy".
    Outputs:
        (DataFrame with columns domain, row, estimate, lo, hi, p, one row per domain and row label; DataFrame indexed
        by domain with columns full and easy, the domain's mean squared error of each).
    """
    tables, means = [], {}
    for domain, part in errors.groupby("domain"):
        table, means[domain] = change_table(part, rows, against)
        tables.append(table.assign(domain=domain))
    table = pd.concat(tables, ignore_index=True)
    return table[["domain", *table.columns.drop("domain")]], pd.DataFrame(means).T


def baseline_change(errors: pd.DataFrame) -> pd.DataFrame:
    """
    Per domain, the Baseline's mean change in squared error from the full transcript over the domain's targets, every
    semester pooled; it is drawn as a grey point for scale and not tested.

    Inputs:
        errors: per-target errors, columns domain, semester, condition, model, err, with conditions "easy" and "full".
    Outputs:
        DataFrame with columns domain, estimate, lo and hi (both equal to estimate); one row per domain.
    """
    rows = []
    for domain, part in errors.groupby("domain"):
        wide = {s: g.pivot_table(index="model", columns="condition", values="err") for s, g in part.groupby("semester")}
        diff = np.concatenate([(w["easy"] - w["full"]).to_numpy(float) for w in wide.values()])
        mean = float(diff[np.isfinite(diff)].mean())
        rows.append(dict(domain=domain, estimate=mean, lo=mean, hi=mean))
    return pd.DataFrame(rows)


def method_text(against: str, removed: bool) -> str:
    """
    The caption sentences saying what a point, a box and a star show.

    Inputs:
        against: the comparison condition, "full" or "easy" (named through AGAINST).
        removed: True when the rows remove a part, False when they keep a part alone.
    Outputs:
        str, the sentences.
    """
    name = AGAINST[against][0]
    tested = "without that part" if removed else "with only that part"
    return (f"Each point is the mean over a domain's targets of the change in squared error from {name} (the target's "
            f"squared error under the condition minus under {name}; below 0 is better than {name}); the box is its 95% "
            f"t-interval. The dashed line at 0 is {name}. Stars: two-sided paired t-test over the targets of the squared "
            f"error {tested} against {name}'s (the interval's own test: a box excluding 0 is p < .05). Uncorrected for "
            "multiple comparisons; * p < .05, ** p < .01, *** p < .001; n.s., not significant.")


def number_lines(table: pd.DataFrame, means: pd.DataFrame, targets: pd.Series, against: str,
                 domain_names: dict[str, str]) -> list[str]:
    """
    The caption's list of plotted numbers: per domain, the mean squared error of the Baseline and of the full
    transcript, then each row's change, interval and p.

    Inputs:
        table: domain_changes' first table (domain, row, estimate, lo, hi, p).
        means: domain_changes' second table (columns full and easy, indexed by domain).
        targets: the number of targets per domain.
        against: the comparison condition, "full" or "easy".
        domain_names: domain key -> display name; also the order of the domains.
    Outputs:
        list of str: a heading line, then one line per domain.
    """
    lines = [f"Change in squared error from {AGAINST[against][0]} (95% interval), p:"]
    for domain in (d for d in domain_names if d in means.index):
        cells = [(f"mean squared error: Baseline {means.loc[domain, 'easy']:.2f}, full transcript "
                  f"{means.loc[domain, 'full']:.2f}")]
        cells += [f"{r.row} {r.estimate:+.2f} ({r.lo:+.2f} to {r.hi:+.2f}), {lr.p_text(r.p)}"
                  for r in table[table.domain == domain].itertuples()]
        lines.append(f"{domain_names[domain]} ({targets[domain]} targets): " + "; ".join(cells) + ".")
    return lines


def draw_row(axes, table: pd.DataFrame, rows: list[str], removed: bool, colours: dict[str, str],
             domain_names: dict[str, str], titles: bool = True, alpha: float | None = None,
             baselines: pd.DataFrame | None = None, span: tuple[float, float] | None = None,
             ticks: list[float] | None = None) -> None:
    """
    Draws one row of panels, one per domain, into `axes`: a line per row label with a dot at the mean change, a box
    over its 95% t-interval and the significance mark just past the panel's widest box, and a dashed line at 0 for the
    comparison condition. Every panel gets the same x ticks and limits.

    Inputs:
        axes: matplotlib axes, one per domain present in `table`, in domain_names order.
        table: domain_changes' first table (domain, row, estimate, lo, hi, p).
        rows: the row labels, top to bottom.
        removed: True for parts removed (hollow dots, darker boxes), False for parts kept alone.
        colours: row label -> colour, or domain -> colour when every row of a panel shares one.
        domain_names: domain key -> display name, in drawing order.
        titles: write each domain's name above its panel.
        alpha: the boxes' opacity; by default 0.6 for parts removed and 0.35 for parts kept alone.
        baselines: baseline_change's table, drawn as a grey point on an extra last line, "Baseline prediction".
        span: (low, high) the axis must cover, so several rows of panels can share one scale.
        ticks: the x ticks; by default the finest step of 1, 2 or 5 times a power of ten that gives at most five ticks
            covering every interval and 0.
    Outputs:
        none; draws into `axes`.
    """
    domains = [d for d in domain_names if d in set(table.domain)]
    labels = list(rows)
    width = 0.5
    picked = table
    if baselines is not None:
        labels.append("Baseline prediction")
        picked = pd.concat([picked, baselines])
    low, high = min(0.0, float(picked.lo.min())), max(0.0, float(picked.hi.max()))  # one axis for every panel
    if span is not None:
        low, high = min(low, span[0]), max(high, span[1])
    if ticks is None:  # else the caller's ticks; an interval may run a little past the last one
        for step in (m * 10.0 ** e for e in range(-2, 2) for m in (1, 2, 5)):  # the finest nice step with at most 5 ticks
            if np.ceil(high / step) - np.floor(low / step) <= 4:
                break
        ticks = list(np.arange(np.floor(low / step), np.ceil(high / step) + 1) * step)  # through 0, covering every interval
    extent = ticks[-1] - ticks[0]
    gap, text = 0.04 * extent, 0.2 * extent  # marks, in axis units
    reach = ticks[-1] + text / 2  # how far right any panel's marks go: one x limit for every panel, past the last label
    for ax, domain in zip(axes, domains):
        part = picked[picked.domain == domain]
        ax.vlines(0, -0.5, len(labels) - 0.4, colors="0.5", lw=0.8, linestyles="--", zorder=1.5)  # the comparison
        drawn = []
        for y, label in enumerate(rows):
            pick = part[part.row == label]
            if not pick.empty:
                drawn.append((y, colours.get(label, colours.get(domain, "0.55")), pick.iloc[0]))
        for y, colour, row in drawn:
            ax.add_patch(plt.Rectangle((row.lo, y - width / 2), row.hi - row.lo, width, color=colour,
                                       alpha=alpha or (0.6 if removed else 0.35), linewidth=0, zorder=1))
            ax.plot(row.estimate, y, "o", ms=3.5, mfc="white" if removed else colour, mec="black", mew=0.4, zorder=2)
        edge = max(r.hi for _, _, r in drawn) + gap  # the marks start just past the panel's widest box
        for y, _, row in drawn:
            ax.text(edge, y, stars(row.p, "n.s."), ha="left", va="center", fontsize=6)
        if baselines is not None and not (row := baselines[baselines.domain == domain]).empty:
            ax.plot(row.iloc[0].estimate, len(rows), "o", ms=3.5, mfc="0.6", mec="black", mew=0.4, zorder=2)
        reach = max(reach, edge + text)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(labels) - 0.4, -0.6)  # the first row on top
        if titles:
            ax.set_title(domain_names[domain], loc="left", fontsize=7, fontweight="bold")
    for ax in axes:
        ax.set_xlim(ticks[0], reach)
    axes[0].set_yticks(range(len(labels)), labels)


def figure(table: pd.DataFrame, rows: list[str], removed: bool, colours: dict[str, str], domain_names: dict[str, str],
           against: str, baselines: pd.DataFrame | None = None) -> plt.Figure:
    """
    Draws draw_row's row of domain panels on a figure of its own, 5.2 inches wide and taller with more rows, with the
    x label "Δ Squared error vs <comparison>".

    Inputs:
        table: domain_changes' first table.
        rows, removed, colours, domain_names, baselines: as in draw_row.
        against: the comparison condition, "full" or "easy".
    Outputs:
        the matplotlib Figure.
    """
    domains = [d for d in domain_names if d in set(table.domain)]
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2, 1.3 + 0.3 * (len(rows) - 3 + (baselines is not None))),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    draw_row(np.atleast_1d(axes), table, rows, removed, colours, domain_names, baselines=baselines)
    fig.supxlabel(f"Δ Squared error vs {AGAINST[against][1]}", fontsize=8, y=-0.12)
    return unclip_markers(fig)


def grid(tables: dict[str, pd.DataFrame], rows: list[str], removed: bool, colours: dict[str, str],
         domain_names: dict[str, str], against: str, alpha: float | None = None, shared_x: bool = False,
         ticks: list[float] | None = None) -> plt.Figure:
    """
    Stacks one draw_row per protocol in a 5.2-inch-wide figure, domain titles on the top row only and each protocol's
    name to the left of its row.

    Inputs:
        tables: protocol display name -> its domain_changes first table, top to bottom.
        rows, removed, colours, domain_names, alpha, ticks: as in draw_row.
        against: the comparison condition, "full" or "easy".
        shared_x: one x range for every protocol's row, the zeros aligned.
    Outputs:
        the matplotlib Figure.
    """
    domains = [d for d in domain_names if d in set(pd.concat(tables.values()).domain)]
    fig, axes = plt.subplots(len(tables), len(domains), figsize=(5.2, 1.3 * len(tables)), sharey="row", squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.55})
    every = pd.concat(tables.values())
    span = (float(every.lo.min()), float(every.hi.max())) if shared_x else None
    for i, (label, table) in enumerate(tables.items()):
        draw_row(axes[i], table, rows, removed, colours, domain_names, titles=i == 0, alpha=alpha, span=span, ticks=ticks)
        axes[i, 0].annotate(label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-92, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
    if shared_x:  # each row's right limit makes room for its own marks: take the widest, so every zero lines up
        right = max(ax.get_xlim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_xlim(ax.get_xlim()[0], right)
    fig.supxlabel(f"Δ Squared error vs {AGAINST[against][1]}", fontsize=8, y=-0.12 / len(tables))
    return unclip_markers(fig)
