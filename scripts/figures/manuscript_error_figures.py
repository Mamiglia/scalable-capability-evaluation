"""The manuscript figures with squared error in place of the prediction score, into draft3_se, and the
split total/unique channel and thirds figures (prediction score) next to the draft2 figures.

Squared error on the axes; figure one's and the time figure's stars are the log likelihood ratio
against the references' distribution (src/analysis/challenges/likelihood_ratio.py), boxes and bars
each protocol's own 95% interval. Figure one and the time figure also go to final_figs. The
channel/thirds figures still test the error with a part alone against the baseline's error. Figures whose
y-axis is not a score (the distance and reasoning scatters, in theta) are copied unchanged.

Reads analysis/results/forecasts.csv and the figures the channel and thirds scripts write
(scripts/analyses/channel_contributions.py, position_contributions.py): run those first.
Run: PYTHONPATH=src uv run python scripts/figures/manuscript_error_figures.py
"""

# %% settings
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as scipy_stats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis.challenges import generalisation as gen  # noqa: E402
from analysis.challenges import likelihood_ratio as lr  # noqa: E402
from analysis.challenges.signal_glm import LABELS  # noqa: E402
from plots import generalisation as gen_plots  # noqa: E402

MANUSCRIPT = Path.home() / "Documents/AIS/LASR/manuscript"
SCORE_DIR, ERROR_DIR, FINAL = MANUSCRIPT / "draft2_score", MANUSCRIPT / "draft3_se", MANUSCRIPT / "final_figs"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}  # as figure one
RESULTS = Path("analysis/results/figures")

forecasts = pd.read_csv("analysis/results/forecasts.csv")
rows = gen.rows(forecasts, VARIANTS)
llr = lr.select(pd.read_csv("analysis/results/likelihood_ratio.csv"), lr.chosen_judges(forecasts, VARIANTS))
llr = llr[llr["shape"] == lr.SHAPE]


def save(fig, name: str, folder: Path = ERROR_DIR, final: str | None = None) -> None:
    """Into `folder` as `name`, and into final_figs as `final` when given."""
    for ext in ("svg", "png"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        if final:
            fig.savefig(FINAL / f"{final}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", folder / f"{name}.svg", *([FINAL / f"{final}.svg"] if final else []))


def copy(source: Path, name: str, folder: Path = ERROR_DIR) -> None:
    for ext in ("svg", "png"):
        shutil.copyfile(source.with_suffix(f".{ext}"), folder / f"{name}.{ext}")
    print("copied", folder / f"{name}.svg")


# %% squared error: figure one, time, simplified reasoning
# a plain one-sample t-test: the full reference set only, no refits with a reference left out
tests = lr.pooled_tests(llr[llr.semester.isin(["2025-II", "2026-I"]) & (llr.dropped == "-")], unit=("semester", "model"))  # a model over its domains
tests["p"] = 2 * tests.p.clip(upper=1 - tests.p)  # two-sided: pooled_tests gives the upper tail
tests.to_csv(ERROR_DIR / "fig2_signal_error_tests.csv", index=False, float_format="%.4g")
print(tests.round(4).to_string(index=False))
save(gen_plots.model_error_figure(forecasts, VARIANTS, tests), "fig2_signal_error_box", final="signal_by_protocol")

errors = gen.task_scores(rows, metric="error")
errors = errors.drop(columns="p").merge(lr.task_tests(llr)[["domain", "semester", "protocol", "p"]], how="left")
errors.to_csv(ERROR_DIR / "time_generalisation_error.csv", index=False, float_format="%.4f")
# no per-semester stars: 8 targets and 7 or 8 references per task leave the test little power; figure one tests
save(gen_plots.time_figure(errors.assign(p=float("nan")), by="domain", box=True, metric="error", log=True), "time_generalisation_error",
     final="time_generalisation")
save(gen_plots.time_figure(errors.assign(p=float("nan")), by="domain", metric="error"), "across_time_error")

# captions for the two final figures, every value from the tables the figures draw
from analysis.challenges import signal_glm as sg  # noqa: E402
from analysis.stats import ci95  # noqa: E402
from analysis.stats import BASELINE  # noqa: E402


def label(p: str) -> str:
    return gen.BASELINE_LABEL if p == BASELINE else LABELS[p]


pooled = sg.per_model(sg.rows(forecasts, VARIANTS))
by = tests.set_index("protocol")
lines = []
for p in [BASELINE, *sg.PAPER_PROTOCOLS]:
    v = pooled[p].dropna().to_numpy(float)
    entry = (f"{label(p)}: mean squared error {lr.num(v.mean())} (95% CI {lr.num(v.mean() - ci95(v))} to "
             f"{lr.num(v.mean() + ci95(v))}, {len(v)} models)")
    if p != BASELINE:
        t = by.loc[p]
        entry += f"; mean log likelihood ratio {lr.num(t.llr)} (SE {lr.num(t.se)}), {lr.p_text(t.p)}"
        d = (pooled[p] - pooled[BASELINE]).dropna()  # paired per target model
        entry += (f"; squared error minus the saturated-item prediction's {lr.num(d.mean(), sign=True)}, paired "
                  f"t({len(d) - 1}) = {lr.num(scipy_stats.ttest_1samp(d, 0).statistic)}, two-sided "
                  f"{lr.p_text(scipy_stats.ttest_1samp(d, 0).pvalue)}")
    lines.append(entry + ".")
caption = ("Squared error of each protocol's forecast of θ* for the 16 target models of 2025-II and 2026-I (8 per "
           "semester). Each small point is one target model's squared error averaged over its four domains (Science, "
           "Maths, Coding, Cyber); the large circle is the mean over models and the box its 95% t-interval. Judge "
           "protocols use each task's best judge on references (lowest leave-one-model-out error on the references), "
           "shown the target's reasoning. Stars test whether a protocol's forecast explains the true θ* better than the "
           "references' own distribution, with each target model (its four domains averaged) as the unit, 15 degrees "
           "of freedom. " + lr.TEST_TEXT + " n.s.: not significant. Each protocol's line also gives the two-sided paired "
           "t-test of the models' squared error against the saturated-item prediction's (not drawn).\n\n" + "\n".join(lines) + "\n")
(FINAL / "signal_by_protocol.txt").write_text(caption)

lines = []
for domain in sg.DOMAINS:
    lines.append(f"{gen.DOMAIN_LABELS[domain]}:")
    d = errors[errors.domain == domain]
    for semester, g in d.groupby("semester"):
        parts = []
        for p in [BASELINE, *sg.PAPER_PROTOCOLS]:
            row = g[g.protocol == p]
            if row.empty:
                continue
            row = row.iloc[0]
            part = f"{label(p)} {lr.num(row.value)} ({lr.num(row.lo)} to {lr.num(row.hi)})"
            parts.append(part)
        lines.append(f"  {semester} ({int(g.n.iloc[0])} targets): " + "; ".join(parts) + ".")
caption = ("Squared error over target semesters, one panel per domain. Each point is the mean squared error over a "
           "task's targets for the saturated-item prediction (grey) and each protocol; the box is its 95% "
           "percentile-bootstrap interval over the task's targets (2,000 resamples). Science and Maths run from 2024-I "
           "to 2026-I; Coding and Cyber start at 2025-II, their first semester. Judge protocols use each task's best "
           "judge on references. The squared-error axis is logarithmic, from 0.01 to 10; points above it are drawn as "
           "triangles on its edge. No test is drawn per semester: with 8 targets and 7 or 8 references per task it has "
           "little power; the protocols are tested over all 16 target models of 2025-II and 2026-I in the figure of "
           "squared error by protocol.\n\nMean squared error (95% interval):\n"
           + "\n".join(lines) + "\n")
(FINAL / "time_generalisation.txt").write_text(caption)

print("wrote", FINAL / "signal_by_protocol.txt", FINAL / "time_generalisation.txt")


# %% the channel and thirds figures, split into total above unique, in both metrics
copy(RESULTS / "channels/channel_contributions_split_error.png", "fig_channels_error")
copy(RESULTS / "positions/position_contributions_split_error.png", "fig_thirds_error")
copy(RESULTS / "channels/channel_shares.png", "channel_shares")
copy(RESULTS / "channels/channel_shares.png", "channel_shares", FINAL)
copy(RESULTS / "channels/channel_contributions_split_score.png", "fig_channels_split", SCORE_DIR)
copy(RESULTS / "positions/position_contributions_split_score.png", "fig_thirds_split", SCORE_DIR)

# %% no score on the axis: copied unchanged
copy(RESULTS / "generalisation/distance_above_top_binned.png", "capability_generalisation")
copy(RESULTS / "generalisation/reasoning_2025-I_scatter_math.png", "reasoning_2025-I_scatter_math")
copy(RESULTS / "generalisation/reasoning_2025-I_scatter_math.png", "reasoning_2025-I_scatter_math", FINAL)

# captions for the reasoning-target and channel-share figures (no tests: descriptive values only)
reasoning = gen.reasoning_rows(rows)
reasoning = reasoning[reasoning.domain == "math"]
refs_2025 = gen.reference_theta("math", "2025-I")
judge_2025 = lr.chosen_judges(forecasts, VARIANTS).query(
    "domain == 'math' and semester == '2025-I' and protocol == 'pairwise'").judge.iloc[0]
GROUPS = {"non-reasoning": "Non-reasoning targets", "reasoning": "Reasoning targets"}
lines = []
pairwise_2025 = reasoning[reasoning.protocol == "pairwise"]
for group, title in GROUPS.items():
    g = pairwise_2025[pairwise_2025.group == group].sort_values("theta")
    v = g.err.to_numpy(float)
    lines.append(f"{title} ({len(g)}): mean squared error {lr.num(v.mean())} (95% CI {lr.num(max(v.mean() - ci95(v), 0))} "
                 f"to {lr.num(v.mean() + ci95(v))}).")
    lines += [f"  {r.model}: θ* {lr.num(r.theta)}, predicted {lr.num(r.theta_hat)}, squared error {lr.num(r.err)}."
              for r in g.itertuples()]
# the tests on the simplified reasoning figures: two-sided t-tests of squared error
removed = forecasts.query("domain == 'math' and semester == '2025-I' and protocol == 'pairwise' and "
                          "variant == @VARIANTS['pairwise'] and condition == 'reasoning-hidden' and judge == @judge_2025")
shown_err = pairwise_2025.set_index("model").err
removed_err = removed.set_index("model").eval("(theta_hat - theta) ** 2")
reasoning_targets = [m for m in shown_err.index if m in gen.REASONING_2025_I]
non_reasoning_targets = [m for m in shown_err.index if m not in gen.REASONING_2025_I]


def welch(a: pd.Series, b: pd.Series) -> dict:
    t = scipy_stats.ttest_ind(a, b, equal_var=False)
    return dict(diff=float(a.mean() - b.mean()), t=float(t.statistic), df=float(t.df), p=float(t.pvalue))


paired = scipy_stats.ttest_rel(removed_err[reasoning_targets], shown_err[reasoning_targets])
reasoning_tests = {  # the non-reasoning targets' transcripts carry no reasoning: the shown run is theirs in both
    "Reasoning minus non-reasoning targets": welch(shown_err[reasoning_targets], shown_err[non_reasoning_targets]),
    "Reasoning targets with reasoning removed minus non-reasoning targets":
        welch(removed_err[reasoning_targets], shown_err[non_reasoning_targets]),
    "Reasoning targets, reasoning removed minus shown": dict(
        diff=float((removed_err[reasoning_targets] - shown_err[reasoning_targets]).mean()), t=float(paired.statistic),
        df=float(paired.df), p=float(paired.pvalue))}
test_lines = {k: f"{k}: mean squared error difference {lr.num(t['diff'], sign=True)}, t({t['df']:.1f}) = "
                 f"{lr.num(t['t'])}, {lr.p_text(t['p'])}." for k, t in reasoning_tests.items()}
print("\n".join(test_lines.values()))
NON_VS_SHOWN, NON_VS_REMOVED = (reasoning_tests[k]["p"] for k in list(reasoning_tests)[:2])
save(gen_plots.reasoning_pairwise_figure(rows, metric="error", tests=[(0, 1, NON_VS_SHOWN)]), "simplified_reasoning_error",
     final="simplified_reasoning")
REASONING_TEST_TEXT = (
    "Each bracket marks a test of squared error against the non-reasoning targets (* p < .05, ** p < .01, "
    "*** p < .001; n.s., not significant): Welch's t-test for two groups of targets, a paired t-test for the same "
    "targets with and without reasoning. Two-sided, uncorrected.")

(FINAL / "simplified_reasoning.txt").write_text(
    "The Pairwise judge on the Maths 2025-I targets, whose references (the 2024-II roster) are all non-reasoning "
    f"models; the task's best judge on references ({judge_2025}), shown the targets' reasoning. Left: predicted "
    "against true capability, the diagonal a perfect forecast. Right: each target's squared error, the large marker "
    "at the group's mean and the box its 95% t-interval, floored at 0. The reasoning targets are the five whose "
    "transcripts carry a reasoning trace (DeepSeek-R1 and MiniMax-M1 always reason; the three Qwen3 models were "
    "collected with reasoning enabled). " + REASONING_TEST_TEXT + "\n\n"
    + "\n".join(lines + ["", test_lines["Reasoning minus non-reasoning targets"]]) + "\n")

# the same figure with the same judge's reasoning-removed forecasts added (reasoning targets only)
save(gen_plots.reasoning_pairwise_figure(rows, metric="error", removed=removed,
                                   tests=[(0, 1, NON_VS_SHOWN), (0, 2, NON_VS_REMOVED)]), "simplified_reasoning_removed_error",
     final="simplified_reasoning_removed")
r = removed[removed.model.isin(gen.REASONING_2025_I)].assign(err=lambda d: (d.theta_hat - d.theta) ** 2).sort_values("theta")
v = r.err.to_numpy(float)
removed_lines = [f"Reasoning targets, reasoning removed ({len(r)}): mean squared error {lr.num(v.mean())} (95% CI "
                 f"{lr.num(max(v.mean() - ci95(v), 0))} to {lr.num(v.mean() + ci95(v))})."]
removed_lines += [f"  {x.model}: θ* {lr.num(x.theta)}, predicted {lr.num(x.theta_hat)}, squared error {lr.num(x.err)}."
                  for x in r.itertuples()]
(FINAL / "simplified_reasoning_removed.txt").write_text(
    "As the simplified reasoning figure, with the same judge's forecasts from transcripts whose reasoning is removed "
    "(hollow light-blue diamonds, right-hand slot; the reasoning shown in dark blue) for the five reasoning targets. The non-reasoning targets carry no "
    "reasoning, so their forecasts are unchanged and not drawn again. " + REASONING_TEST_TEXT
    + " The paired test of the reasoning targets with and without reasoning is reported here, not drawn.\n\n"
    + "\n".join(lines + removed_lines + [""] + list(test_lines.values())) + "\n")

lines = [f"Saturated-item prediction (the references' mean θ*): {lr.num(refs_2025.mean())} ({len(refs_2025)} references)."]
for p in sg.PAPER_PROTOCOLS:
    g = reasoning[reasoning.protocol == p].sort_values(["group", "theta"])
    lines.append(f"{label(p)}: " + "; ".join(f"{r.model} ({r.group}) θ* {lr.num(r.theta)}, predicted {lr.num(r.theta_hat)}"
                                            for r in g.itertuples()) + ".")
(FINAL / "reasoning_2025-I_scatter_math.txt").write_text(
    "Forecast against true θ* for the Maths 2025-I targets, one panel per protocol; the references (the 2024-II "
    "roster) are all non-reasoning models. Open circles: the three non-reasoning targets; filled diamonds: the five "
    "reasoning targets. The solid diagonal is a perfect forecast and the dashed horizontal line the saturated-item "
    "prediction, the references' mean θ*. Both axes are θ on the same scale; a forecast beyond the axis is drawn as a "
    "triangle on its edge. Judge protocols use the task's best judge on references. No test is drawn.\n\n"
    + "\n".join(lines) + "\n")

shares = pd.read_csv(RESULTS / "channels/channel_shares.csv", index_col=0)
lines = [f"{gen.DOMAIN_LABELS[d]}: " + ", ".join(f"{c} {100 * shares.loc[d, c]:.1f}%" for c in shares.columns) + "."
         for d in shares.index]
(FINAL / "channel_shares.txt").write_text(
    "Where the models' own words sit in their transcripts, one pie per domain: the share of the characters the "
    "models themselves wrote (role prefixes removed) in their reasoning (CoT), their messages to the user, and their "
    "tool calls and tool outputs (Tools), over the correct first-attempt transcripts of the references and targets on "
    "the channel analysis's items (10 per difficulty quartile) and semesters (Coding and Cyber 2026-I; Maths and "
    "Science 2025-II and 2026-I). Maths and Science have no tool use.\n\n" + "\n".join(lines) + "\n")
print("wrote the reasoning and channel-share captions")


# %% one target's forecast drawn as the distributions the log likelihood ratio compares
fig, values = gen_plots.prediction_example_figure(forecasts, VARIANTS, "cyber", "2025-II", "minimax-m2", ["pairwise", "glm_regex"])
save(fig, "prediction_example", final="prediction_example")
by = values.set_index("protocol")
n = int(values.references.iloc[0])
lines = [f"{label(p)}: forecast {lr.num(r.forecast)}, spread {lr.num(r.spread)}, squared error {lr.num(r.squared_error)}"
         + ("" if p == BASELINE else f", log likelihood ratio {lr.num(r.log_lr)}") + "." for p, r in by.iterrows()]
(FINAL / "prediction_example.txt").write_text(
    f"What one target's forecast looks like: minimax-m2 in Cyber 2025-II, true θ* {lr.num(by.theta.iloc[0])}. Grey diamonds are "
    f"the task's {n} references (the 2025-I roster) at their release dates; the black diamond is the target's true θ* at its "
    "release date. From that date each forecast is drawn as the distribution the log likelihood ratio uses: a Student t "
    f"with {n - 1} degrees of freedom, centred on the forecast (the line across the violin) and scaled by the forecast "
    "rule's root-mean-square leave-one-model-out error on the references (its spread). Grey: the saturated-item prediction, "
    f"the references' mean θ*; blue: the Pairwise judge (the task's best judge on references, {by.judge['pairwise']}); "
    "orange: Deterministic features. Every violin has the same width per unit of density, so a narrower distribution is "
    "wider at its peak; each is cut 3 spreads from its forecast or at the axis. The black line at θ* runs to the widest "
    "violin: where it crosses each violin's edge is that distribution's density at the truth, and a protocol's log "
    "likelihood ratio is the log of its density there over the saturated-item prediction's. The two protocols miss θ* by "
    "similar amounts in opposite directions, and the Pairwise judge's narrower distribution puts more density on it.\n\n"
    + "\n".join(lines) + "\n")
print("wrote", FINAL / "prediction_example.txt")
