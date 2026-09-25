"""Signal in saturated transcripts: which protocols beat the saturated-item prediction, and where.

One hierarchical Gamma GLM of squared forecast error over every protocol and domain on the
balanced semesters (2025-II, 2026-I), fitted with hibayes, next to a flat statsmodels fit as a
cross-check. Cell script (conventions as
scripts/analyses/results.py); reads analysis/results/forecasts.csv written by that runner.

    uv run python scripts/analyses/signal_glm.py

Design: docs/superpowers/specs/2026-09-16-signal-glm-design.md
"""

# %% settings
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis.challenges import likelihood_ratio as lr  # noqa: E402
from analysis.challenges import signal_glm  # noqa: E402
from plots import signal_glm as signal_glm_plots  # noqa: E402

RESULTS = Path("analysis/results")
JUDGES = "best"  # "best": the best judge on references per task; "all": every judge, with a judge random effect
SEMESTERS = tuple(os.environ.get("SIGNAL_SEMESTERS", ",".join(signal_glm.SEMESTERS)).split(","))  # e.g. SIGNAL_SEMESTERS=2026-I for the latest semester only
TAG = "signal_glm" if JUDGES == "best" else "signal_glm_all_judges"
if SEMESTERS != signal_glm.SEMESTERS:
    TAG += "_" + "_".join(SEMESTERS)
OUT = RESULTS / TAG  # rows.csv and hibayes' own state, diagnostics and plots
FIG = RESULTS / "figures" / TAG
CONFIG = Path("scripts/analyses/signal_glm/config.yaml")
# The same choice as scripts/analyses/results.py: one variant per protocol enters the model.
VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

# %% rows: one squared error per target x protocol
forecasts = pd.read_csv(RESULTS / "forecasts.csv")
table = signal_glm.rows(forecasts, VARIANTS, semesters=SEMESTERS, judges=JUDGES)
table.to_csv(OUT / "rows.csv", index=False)
print(table.groupby(["domain", "protocol"]).size().unstack())
print(table.groupby(["domain", "protocol_judge"]).size().unstack(0).dropna(how="all"))

# %% hierarchical fit through hibayes (load -> process -> model + checks -> communicate)
from hibayes.analysis import AnalysisConfig, communicate, load_data, model, process_data  # noqa: E402
from hibayes.platform import configure_computation_platform  # noqa: E402
from hibayes.ui import ModellingDisplay  # noqa: E402

config = AnalysisConfig.from_yaml(str(CONFIG))
config.data_loader.extracted_data = [str(OUT / "rows.csv")]  # the yaml names the best-judge table
display = ModellingDisplay()
configure_computation_platform(platform_config=config.platform, display=display)
state = load_data(config=config.data_loader, display=display)
state = process_data(analysis_state=state, config=config.data_process, display=display)
state = model(analysis_state=state, models_to_run_config=config.models, checker_config=config.checkers,
              platform_config=config.platform, display=display, out=OUT)
state = communicate(analysis_state=state, communicate_config=config.communicate, display=display, out=OUT)
state.save(path=OUT)
fitted, additive = state.models  # m2 (with the (protocol | domain) term), m1 (without)
summary = fitted.get_summary()
# hibayes' r_hat check reports "fail" because the pinned levels (protocol_effects[easy_items], interaction[0, :],
# sigma_protocol[0]) are constants with NaN r_hat; this is the check that counts.
for fit in (fitted, additive):
    rhat = fit.get_summary().r_hat.dropna()
    assert (rhat <= 1.01).all(), rhat[rhat > 1.01]
print(summary.loc[[i for i in summary.index if i.split("[")[0] in ("protocol_effects", "domain_effects", "sigma_protocol", "sigma_cell", "sigma_judge", "shape")]])

# %% fit diagnostics and the LOO comparison: m2 (protocol x domain) against m1 (protocol + domain)
idatas = {"m2 protocol x domain": fitted.inference_data, "m1 protocol + domain": additive.inference_data}
diag = pd.DataFrame({name: signal_glm.diagnostics(idata, fit.features) for (name, idata), fit in zip(idatas.items(), (fitted, additive))}).T
diag.to_csv(OUT / "diagnostics.csv", float_format="%.3f")
compare = signal_glm.loo_compare(idatas)
compare.to_csv(OUT / "loo_compare.csv", float_format="%.3f")
print(diag.round(3).to_string()); print(compare.round(3).to_string())
for name, fit in (("predictive", fitted), ("predictive_additive", additive)):
    fig = signal_glm_plots.predictive_figure(fit.inference_data, table)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
for name, fit in (("fit", fitted), ("fit_additive", additive)):  # observed vs predicted per row, 95% predictive coverage
    fig, coverage = signal_glm_plots.fit_figure(fit.inference_data, fit.features, table)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    coverage.to_csv(OUT / f"{name}_coverage.csv", index=False, float_format="%.3f")
    print(name); print(coverage.round(3).to_string(index=False))
fig, appendix_fit = signal_glm_plots.appendix_fit_figure(fitted.inference_data, fitted.features, table)
for ext in ("svg", "png"):
    fig.savefig(FIG / f"appendix_fit.{ext}", bbox_inches="tight", dpi=300)
plt.close(fig)
appendix_fit.to_csv(OUT / "appendix_fit.csv", index=False, float_format="%.3f")

# %% contrasts: Pairwise judge against every other protocol, pooled and per domain
contrasts = signal_glm.contrast_table(fitted.inference_data.posterior, fitted.coords)
contrasts.to_csv(RESULTS / f"{TAG}_contrasts.csv", index=False, float_format="%.4f")
print(contrasts.round(3).to_string(index=False))
for name, scale in (("contrasts", "ratio"), ("contrasts_score", "score")):
    fig = signal_glm_plots.contrast_figure(contrasts, scale)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

# %% ratio table: hierarchical next to the flat cross-check
ratios = signal_glm.ratio_table(fitted.inference_data.posterior, fitted.coords)
ratios = ratios.merge(signal_glm.flat_glm(table), on=["protocol", "domain"], how="left")
ratios.to_csv(RESULTS / f"{TAG}.csv", index=False)
errors = signal_glm.error_table(fitted.inference_data.posterior, fitted.coords)  # expected squared error, baseline included
errors.to_csv(RESULTS / f"{TAG}_expected_error.csv", index=False, float_format="%.4f")
signal_glm.wide_table(ratios).to_csv(RESULTS / f"{TAG}_wide.csv")  # protocol x domain, baseline row included
print(ratios.round(3).to_string(index=False))

# %% figures: ratios with intervals (hierarchical filled, flat open); posterior densities
for name, fig in (("ratios", signal_glm_plots.ratio_figure(ratios)), ("ratios_score", signal_glm_plots.ratio_figure(ratios, "score")), ("ratios_table", signal_glm_plots.table_figure(ratios)),
                  ("appendix_effects", signal_glm_plots.appendix_effect_figure(ratios)),
                  ("appendix_errors", signal_glm_plots.appendix_error_figure(errors)),
                  ("appendix_contrasts", signal_glm_plots.appendix_contrast_figure(contrasts, ratios)),
                  ("posteriors", signal_glm_plots.posterior_figure(fitted.inference_data.posterior, fitted.coords)),
                  ("posteriors_score", signal_glm_plots.posterior_figure(fitted.inference_data.posterior, fitted.coords, scale="score")),
                  ("convergence", signal_glm_plots.convergence_figure(fitted.inference_data, fitted.coords, table)),
                  ("fit_quality", signal_glm_plots.fit_quality_figure(fitted.inference_data, fitted.features, fitted.coords, table))):
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")

# %% figure one: prediction score per (target model, domain), averaged over domains per model; t-test over models
if JUDGES != "best":
    raise SystemExit("figure one and its tests take one forecast per target: done")
pooled = signal_glm.per_model(table)
scores = signal_glm.model_scores(table)
# stars: the log likelihood ratio against the references' distribution, a target model pooled over its domains
llr = pd.read_csv(RESULTS / "likelihood_ratio.csv")
llr = llr[llr.semester.isin(SEMESTERS)]


def llr_tests(judges: pd.DataFrame) -> pd.DataFrame:
    """Both shapes' tests, the chosen judges' rows (scripts/analyses/likelihood_ratio.py)."""
    picked = lr.select(llr, judges)
    return pd.concat([lr.pooled_tests(picked[picked["shape"] == shape], unit=("semester", "model")).assign(shape=shape)
                      for shape in lr.SHAPES], ignore_index=True)


both = llr_tests(lr.chosen_judges(forecasts, VARIANTS))
tests = both[both["shape"] == lr.SHAPE]
pooled.to_csv(OUT / "per_model.csv", float_format="%.4f")
scores.to_csv(OUT / "per_model_scores.csv", float_format="%.4f")
both.to_csv(RESULTS / f"{TAG.replace("signal_glm", "signal_llr_tests")}.csv", index=False, float_format="%.4g")
print(both.pivot_table(index="protocol", columns="shape", values=["llr", "p"]).round(4).to_string())
print(scores.mean(numeric_only=True).round(3).to_string())  # the mean prediction scores the figure draws
suffix = TAG.removeprefix("signal_glm")  # "" or "_<semesters>", like the other bar figures
for metric in ("mse", "delta", "ratio"):
    fig = signal_glm_plots.model_bar_figure(pooled, tests, metric)
    for ext in ("svg", "png"):
        fig.savefig(RESULTS / "figures" / "bars" / f"figure1_{metric}{suffix}.{ext}", bbox_inches="tight")
    plt.close(fig)
fig = signal_glm_plots.model_box_figure(scores, tests)  # the score as mean, 95% interval box and points
for ext in ("svg", "png"):
    fig.savefig(RESULTS / "figures" / "bars" / f"figure1_score_box{suffix}.{ext}", bbox_inches="tight")
plt.close(fig)

# %% figure one without self-judging: each target judged by the best judge from another developer family
no_self = signal_glm.model_scores(signal_glm.rows(forecasts, VARIANTS, semesters=SEMESTERS, judges="best_other_family"))
no_self_both = llr_tests(lr.chosen_judges(forecasts, VARIANTS, other_family=True))
no_self_tests = no_self_both[no_self_both["shape"] == lr.SHAPE]
no_self_both.to_csv(RESULTS / f"{TAG.replace("signal_glm", "signal_llr_tests")}_no_self_judging.csv", index=False, float_format="%.4g")
print(no_self_both.pivot_table(index="protocol", columns="shape", values=["llr", "p"]).round(4).to_string())
fig = signal_glm_plots.model_box_figure(no_self, no_self_tests)
for ext in ("svg", "png"):
    fig.savefig(RESULTS / "figures" / "bars" / f"figure1_score_no_self_judging{suffix}.{ext}", bbox_inches="tight")
plt.close(fig)
