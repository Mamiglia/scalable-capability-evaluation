"""Which part of a transcript carries the capability signal: the statistics and captions shared by
the channel analysis (scripts/analyses/channel_contributions.py) and the thirds analysis
(scripts/analyses/position_contributions.py). Their figures: plots.contributions.

Each analysis judges the same targets with one part of the transcript kept alone ("only") and with
that part removed ("without"), next to the full transcript and the saturated-item prediction
("easy"). Figures plot each condition's mean change in squared error from the full transcript over
the targets (paired).

Each interval is the 95% t-interval of the mean paired change and each p the two-sided paired t-test
of it against 0 (the same test: an interval excluding 0 is p < .05), the targets of every semester
pooled.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.challenges import likelihood_ratio as lr

PER_QUARTILE = 10  # every removal analysis uses the 40-item subset the pairwise removal runs judged


def removal_items(task) -> set[str]:
    """The task's 40 removal-analysis train items: stratified_items(seed 0), 10 per difficulty quartile."""
    from protocols.pairwise.judge import stratified_items
    return set(map(str, stratified_items(task.references.items, PER_QUARTILE)))


def on_items(cohort, keep: set[str]):
    """`cohort` restricted to the train items in `keep`."""
    return replace(cohort, items=cohort.items[cohort.items.item.astype(str).isin(keep)])


DOMAIN_TITLES = {"coding": "Coding", "cyber": "Cyber", "math": "Maths", "science": "Science"}
CAPTION_TESTS = {"only": "with only that part", "without": "without that part"}


def llr_p(llr: pd.DataFrame, run: str, against: str | None = None) -> float:
    """One-sided p over the targets of `llr` (rows: domain, semester, model, condition, dropped, llr):
    `run` against the references' distribution or, with `against`, `run`'s log likelihood against
    `against`'s on the same targets (the references' term cancels)."""
    rows = llr[llr.condition == run]
    if against is not None:
        rows = rows.merge(llr[llr.condition == against], on=["domain", "semester", "model", "dropped"], suffixes=("", "_against"))
        rows = rows.assign(llr=rows.llr - rows.llr_against)
    return float(lr.unit_test(rows)["p"])


def paired_t(diff: np.ndarray) -> dict[str, float]:
    """Mean of the per-target differences `diff`, its 95% t-interval and the two-sided p that it is 0."""
    diff = diff[np.isfinite(diff)]
    mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
    half = float(scipy_stats.t.ppf(0.975, len(diff) - 1)) * se
    p = float(scipy_stats.ttest_1samp(diff, 0).pvalue) if se > 0 else 1.0
    return dict(estimate=mean, lo=mean - half, hi=mean + half, p=p)


def part_stats(errors: pd.DataFrame, parts: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, dict]:
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
                diff = np.concatenate([(wide[s][run] - wide[s]["full"]).to_numpy(float) for s in wide])
                rows.append(dict(part=label, measure=measure, **paired_t(diff)))
    mean = lambda c: float(np.concatenate([wide[s][c].to_numpy() for s in wide]).mean())  # noqa: E731
    return pd.DataFrame(rows), dict(estimate=mean("full"), easy=mean("easy"))


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


def domain_stats(errors: pd.DataFrame, parts: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """part_stats per domain of `errors`: (stats with a domain column, fulls indexed by domain)."""
    rows, fulls = [], {}
    for domain, part in errors.groupby("domain"):
        table, fulls[domain] = part_stats(part, parts)
        rows.append(table.assign(domain=domain))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fulls).T


def caption_lines(stats: pd.DataFrame, fulls: pd.DataFrame, targets: pd.Series, measure: str, caption: str) -> list[str]:
    """The .txt caption: `caption` (what was judged, by which judges), the shared method, every plotted number."""
    lines = [f"{caption} Each point is the mean over a domain's targets of the change in squared error from the "
             "full transcript (the target's squared error under the condition minus under the full transcript; "
             "below 0 is better than the full transcript); the box is its 95% t-interval. The dashed line at 0 is "
             "the full transcript. Stars: two-sided paired t-test over the targets of the squared error "
             + CAPTION_TESTS[measure] + " against the full transcript's (the interval's own test: a box excluding 0 "
             "is p < .05). Uncorrected for multiple comparisons; * p < .05, ** p < .01, *** p < .001; n.s., not "
             "significant.",
             "", "Change in squared error from the full transcript (95% interval), p:"]
    for domain in (d for d in DOMAIN_TITLES if d in fulls.index):
        full = fulls.loc[domain]
        cells = [f"mean squared error: saturated-item prediction {full.easy:.2f}, full transcript {full.estimate:.2f}"]
        for r in stats[(stats.domain == domain) & (stats.measure == measure)].itertuples():
            cells.append(f"{measure.capitalize()} {r.part[0].lower()}{r.part[1:]} {r.estimate:+.2f} "
                         f"({r.lo:+.2f} to {r.hi:+.2f}), {lr.p_text(r.p)}")
        lines.append(f"{DOMAIN_TITLES[domain]} ({targets[domain]} targets): " + "; ".join(cells) + ".")
    return lines
