"""Which part of a transcript carries the capability signal: the statistics and the figure shared by
the channel analysis (scripts/analyses/channel_contributions.py) and the thirds analysis
(scripts/analyses/position_contributions.py).

Each analysis judges the same targets with one part of the transcript kept alone ("only") and with
that part removed ("without"), next to the full transcript and the saturated-item prediction
("easy"). Figures plot each condition's mean change in squared error from the full transcript over
the targets (paired).

Each interval is the 95% t-interval of the mean paired change and each p the two-sided paired t-test
of it against 0 (the same test: an interval excluding 0 is p < .05), the targets of every semester
pooled.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.challenges import likelihood_ratio as lr
from analysis.plots import unclip_markers

PER_QUARTILE = 10  # every removal analysis uses the 40-item subset the pairwise removal runs judged


def removal_items(task) -> set[str]:
    """The task's 40 removal-analysis train items: stratified_items(seed 0), 10 per difficulty quartile."""
    from protocols.pairwise.judge import stratified_items
    return set(map(str, stratified_items(task.references.items, PER_QUARTILE)))


def direct_judges(forecasts: Path) -> pd.Series:
    """(domain, semester) -> the main analysis's absolute-scoring judge: the core condition's raw
    direct judge with the lowest reference leave-one-out error in `forecasts` (analysis/results/forecasts.csv)."""
    from analysis import stats
    f = stats.core_condition(pd.read_csv(forecasts))
    f = f[(f.protocol == "direct") & (f.variant == "raw")]
    return f.loc[f.groupby(["domain", "semester"]).judge_loo_mse.idxmin()].set_index(["domain", "semester"]).judge


def on_items(cohort, keep: set[str]):
    """`cohort` restricted to the train items in `keep`."""
    return replace(cohort, items=cohort.items[cohort.items.item.astype(str).isin(keep)])


DOMAIN_TITLES = {"science": "Science", "math": "Maths", "coding": "Coding", "cyber": "Cyber"}  # the paper's domain order
FINAL = Path.home() / "Documents/AIS/LASR/manuscript/final_figs"  # the manuscript's final figures
# the final figures' names and the caption sentences that differ between the two
FINAL_NAMES = {"only": "kept_alone", "without": "removed"}
# what a condition is compared with -> (in captions, on the x label): the full transcript or easy-item prediction
AGAINST = {"full": ("the full transcript", "full transcript"), "easy": ("the saturated-item prediction", "baseline")}
CAPTION_TESTS = {"only": "with only that part", "without": "without that part"}
# the full transcript as a row of its own next to the parts kept alone (tested like them, against `only_against`)
FULL_ROW = {"Full transcript": ("full", None)}


def row_label(measure: str, part: str) -> str:
    """A figure/caption row: "Only first third", "Without reasoning", ...; the full-transcript row as it is."""
    return part if part in FULL_ROW else f"{measure.capitalize()} {part[0].lower()}{part[1:]}"


def llr_p(llr: pd.DataFrame, run: str, against: str | None = None) -> float:
    """One-sided p over the targets of `llr` (rows: domain, semester, model, condition, dropped, llr):
    `run` against the references' distribution or, with `against`, `run`'s log likelihood against
    `against`'s on the same targets (the references' term cancels)."""
    rows = llr[llr.condition == run]
    if against is not None:
        rows = rows.merge(llr[llr.condition == against], on=["domain", "semester", "model", "dropped"], suffixes=("", "_against"))
        rows = rows.assign(llr=rows.llr - rows.llr_against)
    return float(lr.unit_test(rows)["p"])


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."




def paired_t(diff: np.ndarray) -> dict[str, float]:
    """Mean of the per-target differences `diff`, its 95% t-interval and the two-sided p that it is 0."""
    diff = diff[np.isfinite(diff)]
    mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
    half = float(scipy_stats.t.ppf(0.975, len(diff) - 1)) * se
    p = float(scipy_stats.ttest_1samp(diff, 0).pvalue) if se > 0 else 1.0
    return dict(estimate=mean, lo=mean - half, hi=mean + half, p=p)


def part_stats(errors: pd.DataFrame, parts: dict[str, tuple[str, str]], only_against: str = "full") -> tuple[pd.DataFrame, dict]:
    """Change in squared error from the full transcript per part kept alone ("only") and per part
    removed ("without") for one domain: the mean over targets of the target's squared error under
    the run minus under the full transcript, with its paired t-interval and two-sided paired t-test.

    `errors`: one row per (semester, condition, target) with the target's squared error `err`;
    conditions include "full", "easy" and every run named in `parts` (label -> (only, without)).
    Parts whose runs are missing are skipped. Returns (rows: part, measure, estimate, lo, hi, p;
    full: the full transcript's mean squared error `estimate` and the saturated-item prediction's `easy`)."""
    wide = {s: g.pivot_table(index="model", columns="condition", values="err") for s, g in errors.groupby("semester")}
    conditions = set.intersection(*(set(w.columns) for w in wide.values()))
    rows = []
    for label, (only, without) in parts.items():
        for measure, run in (("only", only), ("without", without)):
            if run in conditions:
                base = only_against if measure == "only" else "full"  # a part kept alone: against the full transcript or "easy"
                diff = np.concatenate([(wide[s][run] - wide[s][base]).to_numpy(float) for s in wide])
                rows.append(dict(part=label, measure=measure, **paired_t(diff)))
    mean = lambda c: float(np.concatenate([wide[s][c].to_numpy() for s in wide]).mean())  # noqa: E731
    return pd.DataFrame(rows), dict(estimate=mean("full"), easy=mean("easy"))


def draw_row(axes, stats: pd.DataFrame, parts: list[str], measure: str, colours: dict[str, str],
             titles: bool = True, alpha: float | None = None, baselines: pd.DataFrame | None = None,
             span: tuple[float, float] | None = None, ticks: list[float] | None = None) -> None:
    """One panel per domain of `stats` into `axes` (one per domain, in DOMAIN_TITLES order), one row
    per part kept alone (`measure` "only") or removed ("without"). Dot = mean change in squared error
    from the full transcript, box = 95% t-interval of the paired change; the dashed line at 0
    is the full transcript. Each mark sits just past the panel's widest box. No grid, no line at the
    saturated-item prediction, which the marks do not test. `stats`: domain, part, measure, estimate,
    lo, hi, p. `colours`: part -> colour, or domain -> colour when every part of a domain shares one.
    Every panel shares one axis, from the first tick under the lowest interval to the first tick past
    the widest. `baselines` (baseline_stats) adds a last row, easy-item prediction's change from the full
    transcript as a grey point, for scale; no interval, no test mark. `span` (low, high) widens the axis to
    cover it, so several rows can share one scale."""
    domains = [d for d in DOMAIN_TITLES if d in set(stats.domain)]
    rows = [row_label(measure, p) for p in parts]
    width = 0.5
    picked = stats[stats.measure == measure]
    if baselines is not None:
        rows.append("Baseline prediction")
        picked = pd.concat([picked, baselines])
    low, high = min(0.0, float(picked.lo.min())), max(0.0, float(picked.hi.max()))  # one axis for every panel
    if span is not None:
        low, high = min(low, span[0]), max(high, span[1])
    if ticks is None:  # else the caller's ticks; an interval may run a little past the last one
        for step in (m * 10.0 ** e for e in range(-2, 2) for m in (1, 2, 5)):  # the finest nice step with at most 5 ticks
            if np.ceil(high / step) - np.floor(low / step) <= 4:
                break
        ticks = list(np.arange(np.floor(low / step), np.ceil(high / step) + 1) * step)  # through 0, covering every interval
    span = ticks[-1] - ticks[0]
    gap, text = 0.04 * span, 0.2 * span  # marks, in axis units
    reach = ticks[-1] + text / 2  # how far right any panel's marks go: one x limit for every panel, past the last label
    for ax, domain in zip(axes, domains):
        part = picked[picked.domain == domain]
        ax.vlines(0, -0.5, len(rows) - 0.4, colors="0.5", lw=0.8, linestyles="--", zorder=1.5)  # the full transcript
        drawn = []
        for y, label in enumerate(parts):
            pick = part[part.part == label]
            if not pick.empty:
                drawn.append((y, colours.get(label, colours.get(domain, "0.55")), pick.iloc[0]))
        for y, colour, row in drawn:
            ax.add_patch(plt.Rectangle((row.lo, y - width / 2), row.hi - row.lo, width, color=colour,
                                       alpha=alpha or (0.6 if measure == "without" else 0.35), linewidth=0, zorder=1))
            ax.plot(row.estimate, y, "o", ms=3.5, mfc="white" if measure == "without" else colour, mec="black", mew=0.4, zorder=2)
        edge = max(r.hi for _, _, r in drawn) + gap  # the marks start just past the panel's widest box
        for y, _, row in drawn:
            ax.text(edge, y, stars(row.p), ha="left", va="center", fontsize=6)
        if baselines is not None and not (row := baselines[baselines.domain == domain]).empty:
            row, y = row.iloc[0], len(parts)
            ax.plot(row.estimate, y, "o", ms=3.5, mfc="0.6", mec="black", mew=0.4, zorder=2)
        reach = max(reach, edge + text)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows) - 0.4, -0.6)  # the first part on top
        if titles:
            ax.set_title(DOMAIN_TITLES[domain], loc="left", fontsize=7, fontweight="bold")
    for ax in axes:
        ax.set_xlim(ticks[0], reach)
    axes[0].set_yticks(range(len(rows)), rows)


def figure(stats: pd.DataFrame, parts: list[str], measure: str, colours: dict[str, str],
           baselines: pd.DataFrame | None = None, against: str = "full") -> plt.Figure:
    """draw_row on its own figure, 5.2 in wide, with the shared x label."""
    domains = [d for d in DOMAIN_TITLES if d in set(stats.domain)]
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2, 1.3 + 0.3 * (len(parts) - 3 + (baselines is not None))), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    draw_row(np.atleast_1d(axes), stats, parts, measure, colours, baselines=baselines)
    fig.supxlabel(f"Δ Squared error vs {AGAINST[against][1]}", fontsize=8, y=-0.12)
    return unclip_markers(fig)


def grid(stats: dict[str, pd.DataFrame], parts: list[str], measure: str, colours: dict[str, str],
         alpha: float | None = None, shared_x: bool = False, against: str = "full",
         baselines: dict[str, pd.DataFrame] | None = None, ticks: list[float] | None = None) -> plt.Figure:
    """One draw_row per protocol (`stats`: label -> its stats table), stacked, domain titles on the
    top row only, each row's protocol label to the left of its condition labels, its own x axis.
    `baselines`: label -> baseline_stats, a grey baseline point per protocol row as in `figure`."""
    domains = [d for d in DOMAIN_TITLES if d in set(pd.concat(stats.values()).domain)]
    fig, axes = plt.subplots(len(stats), len(domains), figsize=(5.2, (1.3 + 0.3 * (baselines is not None)) * len(stats)),
                             sharey="row", squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.55})
    every = pd.concat(stats.values())
    every = pd.concat([every[every.measure == measure], *(baselines or {}).values()])
    span = (float(every.lo.min()), float(every.hi.max())) if shared_x else None  # one x scale for every row
    for i, (label, table) in enumerate(stats.items()):
        draw_row(axes[i], table, parts, measure, colours, titles=i == 0, alpha=alpha, span=span,
                 baselines=baselines and baselines[label], ticks=ticks)
        axes[i, 0].annotate(label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-92, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
    if shared_x:  # each row's right limit makes room for its own marks: take the widest, so every zero lines up
        right = max(ax.get_xlim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_xlim(ax.get_xlim()[0], right)
    fig.supxlabel(f"Δ Squared error vs {AGAINST[against][1]}", fontsize=8, y=-0.12 / len(stats))
    return unclip_markers(fig)


def baseline_stats(errors: pd.DataFrame) -> pd.DataFrame:
    """Per domain, easy-item prediction's mean change in squared error from the full transcript, semesters pooled:
    domain, estimate (lo = hi = estimate: a point only, since the paper tests the baseline with the LLR, not this)."""
    rows = []
    for domain, part in errors.groupby("domain"):
        wide = {s: g.pivot_table(index="model", columns="condition", values="err") for s, g in part.groupby("semester")}
        diff = np.concatenate([(w["easy"] - w["full"]).to_numpy(float) for w in wide.values()])
        mean = float(diff[np.isfinite(diff)].mean())
        rows.append(dict(domain=domain, estimate=mean, lo=mean, hi=mean))
    return pd.DataFrame(rows)


def domain_stats(errors: pd.DataFrame, parts: dict[str, tuple[str, str]], only_against: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """part_stats per domain of `errors`: (stats with a domain column, fulls indexed by domain)."""
    rows, fulls = [], {}
    for domain, part in errors.groupby("domain"):
        table, fulls[domain] = part_stats(part, parts, only_against)
        rows.append(table.assign(domain=domain))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fulls).T


def caption_lines(stats: pd.DataFrame, fulls: pd.DataFrame, targets: pd.Series, measure: str, caption: str,
                  against: str = "full") -> list[str]:
    """The .txt caption: `caption` (what was judged, by which judges), the shared method, every plotted number."""
    name = AGAINST[against][0]
    lines = [f"{caption} Each point is the mean over a domain's targets of the change in squared error from {name} "
             f"(the target's squared error under the condition minus under {name}; below 0 is better than {name}); "
             f"the box is its 95% t-interval. The dashed line at 0 is {name}. Stars: two-sided paired t-test over the "
             "targets of the squared error " + CAPTION_TESTS[measure] + f" against {name}'s (the interval's own test: "
             "a box excluding 0 is p < .05). Uncorrected for multiple comparisons; * p < .05, ** p < .01, *** p < .001; "
             "n.s., not significant.",
             "", f"Change in squared error from {name} (95% interval), p:"]
    for domain in (d for d in DOMAIN_TITLES if d in fulls.index):
        full = fulls.loc[domain]
        cells = [f"mean squared error: saturated-item prediction {full.easy:.2f}, full transcript {full.estimate:.2f}"]
        for r in stats[(stats.domain == domain) & (stats.measure == measure)].itertuples():
            cells.append(f"{row_label(measure, r.part)} {r.estimate:+.2f} "
                         f"({r.lo:+.2f} to {r.hi:+.2f}), {lr.p_text(r.p)}")
        lines.append(f"{DOMAIN_TITLES[domain]} ({targets[domain]} targets): " + "; ".join(cells) + ".")
    return lines


def save(fig: plt.Figure, out: Path, name: str, final: str | None) -> None:
    """The figure as .png and .svg under `out`, and under FINAL when `final` names it."""
    for folder, stem in [(out, name)] + ([(FINAL, final)] if final else []):
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(folder / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
        print(folder / f"{stem}.png")
    plt.close(fig)


def save_figures(errors: pd.DataFrame, parts: dict[str, tuple[str, str]], colours: dict[str, str],
                 out: Path, name: str, final: str | None = None, caption: str = "",
                 measures: tuple[str, ...] = ("only", "without"), baseline: bool = False,
                 only_against: str = "full") -> None:
    """part_stats for every domain of `errors`, written to <name>_delta_squared_error.csv, and
    the "only" and "without" figures to <name>_only and <name>_without (.png, .svg) under `out`. With
    `final`, also into FINAL as <final>_kept_alone and <final>_removed, each with a .txt caption.
    `measures` limits the figures, e.g. to ("without",) when no part was run alone; `baseline` adds the
    easy-item prediction point (baseline_stats) for scale; `only_against` "easy" tests a part kept alone
    against the baseline instead of the full transcript."""
    stats, fulls = domain_stats(errors, parts, only_against)
    stats.to_csv(out / f"{name}_delta_squared_error.csv", index=False)
    targets = errors[errors.condition == "full"].groupby("domain").size()
    for measure in measures:
        against = only_against if measure == "only" else "full"
        save(figure(stats, list(parts), measure, colours, baseline_stats(errors) if baseline else None, against), out, f"{name}_{measure}", final and f"{final}_{FINAL_NAMES[measure]}")
        if final:
            (FINAL / f"{final}_{FINAL_NAMES[measure]}.txt").write_text("\n".join(caption_lines(stats, fulls, targets, measure, caption, against)) + "\n")


def save_grid(errors: pd.DataFrame, protocols: dict[str, str], parts: dict[str, tuple[str, str]],
              colours: dict[str, str], out: Path, name: str, final: str | None = None, caption: str = "",
              alpha: float | None = None, measures: tuple[str, ...] = ("only", "without"), shared_x: bool = False,
              only_against: str = "full", baseline: bool = False, ticks: list[float] | None = None) -> None:
    """save_figures for several protocols at once (`errors` carries a protocol column; `protocols`:
    key -> row label), one grid figure per measure with a row of panels per protocol. `baseline` adds
    each protocol's baseline point to the kept-alone figure, as save_figures does."""
    stats, fulls, targets = {}, {}, {}
    for key, label in protocols.items():
        e = errors[errors.protocol == key]
        stats[label], fulls[label] = domain_stats(e, parts, only_against)
        targets[label] = e[e.condition == "full"].groupby("domain").size()
    pd.concat(stats, names=["protocol"]).reset_index(level=0).to_csv(out / f"{name}_delta_squared_error.csv", index=False)
    for measure in measures:
        against = only_against if measure == "only" else "full"
        baselines = ({label: baseline_stats(errors[errors.protocol == key]) for key, label in protocols.items()}
                     if baseline and measure == "only" else None)
        save(grid(stats, list(parts), measure, colours, alpha, shared_x, against, baselines, ticks), out, f"{name}_{measure}", final and f"{final}_{FINAL_NAMES[measure]}")
        if final:
            lines = [caption]
            for label in protocols.values():
                lines += ["", f"== {label}"] + caption_lines(stats[label], fulls[label], targets[label], measure, "", against)[1:]
            (FINAL / f"{final}_{FINAL_NAMES[measure]}.txt").write_text("\n".join(lines) + "\n")
