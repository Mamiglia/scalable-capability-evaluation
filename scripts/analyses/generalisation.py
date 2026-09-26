"""Research question 3 figures: prediction score over time, error against distance from the
references, and the reasoning-model targets of 2025-I.

Cell script: run top to bottom as `PYTHONPATH=src uv run python scripts/analyses/generalisation.py`,
or cell by cell (`# %%`). Reads `analysis/results/forecasts.csv` written by results.py; the frozen
model tables supply the references' theta*. Launches nothing.

Design: docs/superpowers/specs/2026-09-17-generalisation-design.md
"""

# %% settings
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis.challenges import generalisation as gen  # noqa: E402  (after the path is set)
from analysis.challenges import likelihood_ratio as lr  # noqa: E402

VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}  # as figure one
OUT = Path("analysis/results/generalisation")
FIG = Path("analysis/results/figures/generalisation")
FINAL = Path("analysis/results/figures/paper")  # the paper figures and their .txt captions
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)

forecasts = pd.read_csv("analysis/results/forecasts.csv")
rows = gen.rows(forecasts, VARIANTS)
# every p-value: the log likelihood ratio against the references' distribution, best judges (scripts/analyses/likelihood_ratio.py)
llr = lr.select(pd.read_csv("analysis/results/likelihood_ratio.csv"), lr.chosen_judges(forecasts, VARIANTS))
llr = llr[llr["shape"] == lr.SHAPE]

# %% prediction score over target semesters
scores = gen.task_scores(rows)
scores = scores.drop(columns="p").merge(lr.task_tests(llr)[["domain", "semester", "protocol", "p"]], how="left")
scores.to_csv(OUT / "score_over_time.csv", index=False, float_format="%.4f")
# check: the pooled MSE per task matches the results runner's, which computes it by another path
metrics = pd.read_csv("analysis/results/metrics.csv").query("semester != 'all' and domain != 'all'")
metrics = metrics[[VARIANTS.get(p, "-") == v for p, v in zip(metrics.protocol, metrics.variant)]]
merged = scores.merge(metrics[["domain", "semester", "protocol", "mse"]], on=["domain", "semester", "protocol"],
                      suffixes=("", "_results"))
assert len(merged), "no task matched metrics.csv"
# metrics.csv is written to four decimals, so the comparison is to its rounding, not exact. It is a
# cross-check against another script's output, so a disagreement is reported, not fatal: metrics.csv
# is stale whenever results.py has not been re-run since the judge selection or the logs changed.
off = merged[(merged.mse - merged.mse_results).abs() > 5e-4 * merged.mse.abs().clip(lower=1)]
if len(off):
    print("task MSE disagrees with metrics.csv (stale results.py output?):")
    print(off[["domain", "semester", "protocol", "mse", "mse_results"]].round(3).to_string(index=False))
for ext in ("png", "svg"):
    gen.time_figure(scores).savefig(FIG / f"score_over_time.{ext}", bbox_inches="tight")
    gen.time_figure(scores, by="domain").savefig(FIG / f"score_over_time_by_domain.{ext}", bbox_inches="tight")
    gen.time_figure(scores, by="domain", stack=True).savefig(FIG / f"score_over_time_stacked.{ext}", bbox_inches="tight")
    gen.time_figure(scores, by="domain", box=True).savefig(FIG / f"score_over_time_box.{ext}", bbox_inches="tight")
print(scores.pivot_table(index=["domain", "semester"], columns="protocol", values="score").round(0).to_string())

# %% error against distance from the references
# the two semesters every domain has, as in figure one
BINS = 4  # equal-count bins of every target, per domain
distance = gen.with_bins(gen.distances(gen.rows(forecasts, VARIANTS, ("2025-II", "2026-I"))), bins=BINS,
                         within=(-np.inf, np.inf))
bin_sizes = distance[distance.protocol == "easy_items"].groupby(["domain", "bin"]).size()
assert (bin_sizes >= 3).all(), f"a bin with fewer than 3 targets: {bin_sizes.to_dict()}"
slopes = gen.distance_slopes(distance)
distance.to_csv(OUT / "distance.csv", index=False, float_format="%.4f")
slopes.to_csv(OUT / "distance_slopes.csv", index=False, float_format="%.4f")
# per domain, each protocol's slope of the log likelihood ratio on the distance
slope_tests = {which: lr.slope_tests(llr, distance, which) for which in gen.DISTANCE_LABELS}
pd.concat(slope_tests, names=["distance"]).reset_index(0).to_csv(OUT / "llr_slope_tests.csv", index=False,
                                                                 float_format="%.4g")
for ext in ("png", "svg"):
    for which in gen.DISTANCE_LABELS:
        for metric in ("signed", "score"):
            for points in (True, False):
                name = f"distance_{which}" + ("" if metric == "signed" else "_score") + ("" if points else "_binned")
                gen.distance_figure(distance, which, metric, points, slope_tests[which] if metric == "signed" else None).savefig(
                    FIG / f"{name}.{ext}", bbox_inches="tight")
print(slopes.round(2).to_string(index=False))
print(slope_tests["above_top"].round(3).to_string(index=False))
# equal-count bins of every target, one dot each, filled where its log likelihood ratio is above 0; OLS lines per protocol
bin_errors = lr.group_tests(llr, distance, "bin")
bin_errors.to_csv(OUT / "distance_bin_tests.csv", index=False, float_format="%.4g")
for ext in ("png", "svg"):
    for metric in ("signed", "squared"):
        gen.bin_distance_figure(distance, slope_tests["above_top"], bin_errors, metric=metric).savefig(
            FIG / f"distance_above_top_bins{'' if metric == 'signed' else '_squared'}.{ext}", bbox_inches="tight")
    gen.bin_distance_figure(distance, slope_tests["above_top"], bin_errors, metric="squared", lines=False,
                            dot_slopes=True).savefig(FIG / f"distance_above_top_bins_squared_baseline_line.{ext}",
                                                     bbox_inches="tight")
    gen.bin_distance_figure(distance, slope_tests["above_top"], None, metric="squared", lines=False, log=True).savefig(
        FIG / f"distance_above_top_bins_squared_log.{ext}", bbox_inches="tight")
    # the final figure: linear axis, every dot the same, no per-bin test
    fig = gen.bin_distance_figure(distance, slope_tests["above_top"], None, metric="squared", lines=False, fits=True)
    for folder, name in ((FIG, "distance_above_top_bins_squared_untested"), (FINAL, "capability_generalisation")):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    gen.bin_distance_figure(distance, slope_tests["above_top"], bin_errors, metric="squared", lines=False,
                                 log=True, bars=True).savefig(FIG / f"distance_above_top_bins_squared_log_bars.{ext}",
                                                              bbox_inches="tight")


# the final figure's caption, every value from the tables the figure draws
from analysis.stats import BASELINE  # noqa: E402

binned_points = distance.dropna(subset=["bin"]).groupby(["domain", "bin", "protocol"]).agg(
    x=("above_top", "mean"), err=("err", "mean"), n=("err", "size")).reset_index()
names = {p: gen.LABELS[p] for p in ("glm_regex", "embedding", "direct", "pairwise")}
lines = []
for domain in gen.DOMAINS:
    d = distance[(distance.domain == domain) & (distance.protocol == BASELINE)]
    slope, intercept = np.polyfit(d.above_top, d.err, 1)
    fitted = {p: np.polyfit(g.above_top, g.err, 1) for p, g in distance[distance.domain == domain].groupby("protocol")
              if p in names or p == BASELINE}
    line = lambda b, a: f"{lr.num(a)} {'+' if b >= 0 else '−'} {lr.num(abs(b))} × d"  # noqa: E731
    lines.append(f"{gen.DOMAIN_LABELS[domain]} ({len(d)} targets): lines of squared error on "
                 f"distance d, saturated-item prediction {line(*fitted[BASELINE])}; "
                 + "; ".join(f"{names[p]} {line(*fitted[p])}" for p in names
                             if p in fitted) + ".")
    for b_, g in binned_points[binned_points.domain == domain].groupby("bin"):
        err = g.set_index("protocol").err
        parts = [f"{names[p]} {lr.num(err[p])}" for p in names if p in err.index]  # paper order
        lines.append(f"  bin at {lr.num(g.x.iloc[0], sign=True)} ({int(g.n.iloc[0])} targets): saturated-item prediction "
                     f"{lr.num(err[BASELINE])}; " + "; ".join(parts) + ".")
caption = ("Squared error against how far each target's θ* sits above the strongest reference's, for 2025-II and 2026-I, "
           "one panel per domain, each with its own distance axis. Each domain's targets are split into "
           f"{BINS} equal-count bins of distance ("
           + (f"{bin_sizes.min()}" if bin_sizes.min() == bin_sizes.max() else f"{bin_sizes.min()} to {bin_sizes.max()}")
           + " targets each); "
           "each dot is a bin's mean squared "
           "error at the bin's mean distance. Each line is a protocol's least-squares line of squared error on distance over "
           "all the domain's targets, drawn 0.2 beyond the outermost dots, with its 95% confidence band; the "
           "saturated-item prediction's is dashed black. Judge protocols use each task's best judge on references. No test is drawn."
           "\n\nPer domain and bin (mean distance above the strongest reference): mean squared error:\n"
           + "\n".join(lines) + "\n")
(FINAL / "capability_generalisation.txt").write_text(caption)
print("wrote", FINAL / "capability_generalisation.txt")

# the same bins as dots with boxes over their 95% intervals, no lines
boxes = gen.bin_boxes(distance)
boxes.to_csv(OUT / "distance_bin_boxes.csv", index=False, float_format="%.4f")
fig = gen.bin_box_figure(boxes, distance)
for ext in ("png", "svg"):
    for folder, name in ((FIG, "distance_above_top_bins_boxes"), (FINAL, "capability_generalisation_boxes")):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
series = ["glm_regex", "embedding", "direct", "pairwise"]
lines = []
for domain in gen.DOMAINS:
    base = distance[(distance.domain == domain) & (distance.protocol == BASELINE)]
    b1, b0 = np.polyfit(base.above_top, base.err, 1)
    lines.append(f"{gen.DOMAIN_LABELS[domain]}: saturated-item prediction's line, squared error = {lr.num(b0)} "
                 f"{'+' if b1 >= 0 else '−'} {lr.num(abs(b1))} × distance.")
    for b_, g in boxes[boxes.domain == domain].groupby("bin"):
        row = g.set_index("protocol")
        lines.append(f"  bin at {lr.num(row.x.iloc[0], sign=True)} ({int(row.n.iloc[0])} targets): " + "; ".join(
            f"{gen.BASELINE_LABEL if p == BASELINE else gen.LABELS[p]} {lr.num(row.loc[p, 'mean'])} "
            f"({lr.num(row.loc[p, 'lo'])} to {lr.num(row.loc[p, 'hi'])})" for p in series if p in row.index) + ".")
(FINAL / "capability_generalisation_boxes.txt").write_text(
    "Squared error against how far each target's θ* sits above the strongest reference's, for 2025-II and 2026-I, one "
    "panel per domain with its own distance axis. Each domain's 16 targets are split into 4 equal-count bins of distance "
    "(4 targets each). In each bin, each dot is a series' mean squared error over the bin's targets, placed side by side "
    "around the bin's mean distance, and the box is its 95% percentile-bootstrap interval over those 4 targets (2,000 "
    "resamples); with 4 targets the intervals are rough. The dashed line is the saturated-item prediction's "
    "least-squares line of squared error on distance over the domain's targets, drawn 0.2 beyond the outermost bins, "
    "with its 95% confidence band. Judge protocols use each "
    "task's best judge on references. No test is drawn.\n\nPer domain and bin (mean distance above the strongest "
    "reference): mean squared error (95% interval):\n" + "\n".join(lines) + "\n")
print("wrote", FINAL / "capability_generalisation_boxes.txt")

# the same squared errors as sliding means over 5 targets adjacent in distance, with bootstrap bands
curves = gen.sliding_curves(distance)
curves.to_csv(OUT / "distance_sliding.csv", index=False, float_format="%.4f")
degrading = gen.degradation_tests(distance)  # does each protocol's squared error rise above the references?
degrading.to_csv(OUT / "distance_degradation_tests.csv", index=False, float_format="%.4g")
fig = gen.sliding_distance_figure(curves, tests=degrading)
for ext in ("png", "svg"):
    for folder, name in ((FIG, "distance_above_top_sliding"), (FINAL, "capability_generalisation_sliding")):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
lines = []
for domain in gen.DOMAINS:
    lines.append(f"{gen.DOMAIN_LABELS[domain]}:")
    for p in (BASELINE, "glm_regex", "embedding", "direct", "pairwise"):
        c = curves[(curves.domain == domain) & (curves.protocol == p)]
        test = degrading.set_index(["domain", "protocol"]).loc[(domain, p)]
        lines.append(f"  {gen.BASELINE_LABEL if p == BASELINE else gen.LABELS[p]}: slope {lr.num(test.slope)} (SE "
                     f"{lr.num(test.se)}), {lr.p_text(test.p)}; windows " + "; ".join(
            f"{lr.num(r.centre, sign=True)}: {lr.num(r.mean)} ({lr.num(r.lo)} to {lr.num(r.hi)})" for r in c.itertuples()) + ".")
(FINAL / "capability_generalisation_sliding.txt").write_text(
    "Squared error against how far each target's θ* sits above the strongest reference's, for 2025-II and 2026-I, one "
    "panel per domain with its own distance axis. Each point of a line is a protocol's mean squared error over 5 targets "
    "adjacent in distance (the 5 nearest to one another), placed at their mean distance; the window steps one target at "
    "a time, 12 windows over a domain's 16 targets, so it is narrow where targets are dense and wide where they are "
    "sparse. The shading is the 95% percentile-bootstrap interval over the window's 5 targets (2,000 resamples). The "
    "saturated-item prediction is dashed black. Judge protocols use each task's best judge on references. Stars mark "
    "a protocol whose squared error rises with distance above the strongest reference: the least-squares slope of its "
    "squared error on distance over the domain's 16 targets (not the windows), two-sided t-test on 14 degrees of freedom, "
    "uncorrected; targets share their task's references, so p is optimistic. * p < .05, ** p < .01, *** p < .001."
    "\n\nPer domain and line: slope of squared error on distance (SE), p; then window mean squared error (95% "
    "interval) at each window's mean distance:\n"
    + "\n".join(lines) + "\n")
print("wrote", FINAL / "capability_generalisation_sliding.txt")


# %% reasoning-model targets of 2025-I
reasoning = gen.reasoning_scores(rows)
reasoning.to_csv(OUT / "reasoning_2025-I.csv", index=False, float_format="%.4f")
gen.reasoning_model_scores(rows).to_csv(OUT / "reasoning_2025-I_models.csv", index=False, float_format="%.4f")
for ext in ("png", "svg"):
    gen.reasoning_figure(rows).savefig(FIG / f"reasoning_2025-I.{ext}", bbox_inches="tight")
    gen.reasoning_pairwise_figure(rows).savefig(FIG / f"reasoning_2025-I_pairwise_math.{ext}", bbox_inches="tight")
    gen.reasoning_pairwise_figure(rows, metric="error").savefig(FIG / f"reasoning_2025-I_pairwise_math_error.{ext}",
                                                              bbox_inches="tight")
    gen.reasoning_score_figure(rows).savefig(FIG / f"reasoning_2025-I_score.{ext}", bbox_inches="tight")
    gen.reasoning_scatter_figure(rows, "math").savefig(FIG / f"reasoning_2025-I_scatter_math.{ext}", bbox_inches="tight")
print(reasoning.pivot_table(index=["domain", "protocol"], columns="group", values="score").round(0).to_string())
