"""
Tests whether transcripts of saturated items carry information about capability: each protocol's squared error and
log likelihood ratio against the Baseline for the targets of tasks.semesters, pooled over domains and per domain, and a
hierarchical Gamma model of how each protocol's error ratio to the Baseline varies with the domain. Judge protocols use
each task's selected judge (lowest leave-one-model-out error on the references), shown the target's reasoning.

Writes to <paths.outputs>/capability_signal/ (config.yaml):
    error_by_protocol.{svg,png,txt}        each target model's squared error averaged over its domains, per protocol,
                                           with the one-sample t-test of the mean log likelihood ratio; the .txt holds
                                           every plotted number, that test and the paired t-test against the
                                           Baseline's error
    error_by_protocol.csv                  those numbers, one row per protocol
    error_by_domain.csv                    mean squared error per protocol and domain, marked by its test
    prediction_example.{svg,png,txt}       one target's predictive distributions (config.yaml prediction_example)
    no_self_judging.{svg,png,txt}          error_by_protocol with each task's judge left out of its own references
    domain_glm_ratios.csv                  the Gamma model's error ratio to the Baseline per protocol and domain
    domain_glm_effects.{svg,png,txt}       those ratios
    domain_glm_contrasts.{svg,png,txt,csv} the Pairwise judge's error ratio to each other protocol, and each
                                           protocol's between-domain SD
    domain_glm/                            hibayes' rows, fit state and checks; diagnostics.csv (convergence, LOO and
                                           R^2 of both fits) and loo_compare.csv (with against without the
                                           (protocol | domain) term)
    breakdowns/forecasts_<protocol>.{svg,png}   predicted against measured theta* for every target, per domain and
                                                semester
    breakdowns/error_by_domain_by_semester.csv  error_by_domain for each semester on its own

Reads <paths.forecasts>/forecasts.csv and likelihood_ratio.csv (written by forecasts.py); prediction_example also
refits its listed protocols that have no judge on the references (the frozen tables, and the features or embeddings
those protocols use) and reads the release dates in paths.model_configs; no_self_judging refits the selected judges
from the frozen tables and the judge logs.
Run from the repository root:
    uv run python scripts/analysis/capability_signal.py [--config scripts/analysis/config.yaml] [--only error_by_protocol ...]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import plots, stats
from analysis.config import apply_paths, load_config
from analysis.capability_signal import by_protocol
from analysis.capability_signal import domain_glm as glm
from analysis.capability_signal import no_self_judging as refit
from analysis.capability_signal import prediction_example as example
from analysis.stats import BASELINE

BEST_JUDGE = ("Judge protocols use each task's best judge on references (lowest leave-one-model-out error on the "
              "references), shown the target's reasoning.")


def save(fig: plt.Figure, stem: Path) -> None:
    """
    Saves a figure as .svg and .png at 300 dpi and closes it.

    Inputs:
        fig: the figure.
        stem: the output path without its suffix.
    Outputs:
        stem.svg and stem.png.
    """
    for ext in ("svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem.with_suffix(".svg"))


def full_set_llr(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the log likelihood ratio rows the tests use: the targets of tasks.semesters, the Student t shape, the full
    reference set, and for judge protocols each task's selected judge.

    Inputs:
        config: the analysis configuration; uses protocols and tasks.semesters.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        llr: every target's log likelihood ratios (analysis/forecasts/likelihood_ratio.csv).
    Outputs:
        DataFrame with llr's columns (domain, semester, model, protocol, judge, shape, dropped, llr); one row per target
        and protocol (glm_regex, embedding, direct, pairwise).
    """
    chosen = by_protocol.chosen_llr(llr, forecasts, config["protocols"], config["tasks"]["semesters"])
    return chosen[chosen.dropped == "-"]


def error_by_protocol(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """
    Compares each protocol's squared error with the Baseline's, per target model averaged over its domains: the
    models' errors, their mean with its 95% t-interval, the paired t-test against the Baseline's error, and the
    two-sided one-sample t-test of the mean log likelihood ratio against 0 (a target model the unit).

    Inputs:
        config: the analysis configuration; uses protocols, tasks (domains, semesters) and display (paper_protocols,
            protocol_names, protocol_colours).
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        llr: every target's log likelihood ratios (analysis/forecasts/likelihood_ratio.csv).
        out: the output folder (<paths.outputs>/capability_signal).
    Outputs:
        in out: error_by_protocol.{svg,png}; error_by_protocol.csv (one row per protocol, the Baseline first: mse, lo,
        hi, models, and for the others delta, delta_t, delta_df, delta_p against the Baseline and llr, llr_se, llr_p);
        error_by_protocol.txt (the caption with every number).
    """
    protocols, domains = [BASELINE, *config["display"]["paper_protocols"]], config["tasks"]["domains"]
    tests = by_protocol.protocol_tests(full_set_llr(config, forecasts, llr))
    print(tests.round(4).to_string(index=False))
    errors = stats.target_errors(forecasts, config["protocols"], config["tasks"]["semesters"])
    pooled = by_protocol.per_model(errors, protocols, domains)
    save(by_protocol.model_error_figure(pooled, tests, protocols, config["display"]), out / "error_by_protocol")
    table = by_protocol.error_table(pooled, tests, protocols)
    table.to_csv(out / "error_by_protocol.csv", index=False)
    (out / "error_by_protocol.txt").write_text(
        by_protocol.error_caption(table, config["display"]["protocol_names"], BEST_JUDGE))


def error_by_domain(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """
    Tabulates each protocol's mean squared error per domain and pooled (the mean of the domains), marked with stars
    where the mean log likelihood ratio is positive and its two-sided one-sample t-test significant (per domain over
    its targets; pooled with a target model the unit).

    Inputs:
        config: the analysis configuration; uses protocols, tasks.semesters and display (table_protocol_names,
            domain_names).
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        llr: every target's log likelihood ratios (analysis/forecasts/likelihood_ratio.csv).
        out: the output folder (<paths.outputs>/capability_signal).
    Outputs:
        in out: error_by_domain.csv, columns Protocol, Pooled and one per domain name, each cell the error as text to
        two decimals with its stars; one row per protocol in display.table_protocol_names. Also printed.
    """
    rows = full_set_llr(config, forecasts, llr)
    errors = stats.target_errors(forecasts, config["protocols"], config["tasks"]["semesters"])
    table = by_protocol.domain_table(errors, rows, by_protocol.protocol_tests(rows), config["display"]["table_protocol_names"],
                                     config["display"]["domain_names"])
    table.to_csv(out / "error_by_domain.csv", index=False)
    print(table.to_string(index=False))


def prediction_example(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """
    Draws one target's predictions as the Student t distributions the log likelihood ratio compares, for the Baseline
    and the protocols in config.yaml prediction_example, beside the references' theta* at their release dates.

    Inputs:
        config: the analysis configuration; uses prediction_example (domain, semester, target, protocols), protocols,
            paths.model_configs and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); gives the judge
            protocols' prediction and spread.
        llr: unused; every capability-signal analysis takes it.
        out: the output folder (<paths.outputs>/capability_signal).
    Outputs:
        in out: prediction_example.{svg,png}; prediction_example.txt (the caption with each distribution's centre,
        spread, squared error and log likelihood ratio).
    """
    ex, names = config["prediction_example"], config["display"]["protocol_names"]
    refs, values = example.example_distributions(forecasts, config["protocols"], ex["domain"], ex["semester"], ex["target"],
                                                 ex["protocols"])
    print(values.round(3).to_string(index=False))
    fig = example.prediction_example_figure(values, refs, example.release_dates(Path(config["paths"]["model_configs"])), ex["target"],
                                            config["display"]["domain_markers"][ex["domain"]], config["display"])
    save(fig, out / "prediction_example")
    (out / "prediction_example.txt").write_text(example.example_caption(
        values, config["display"]["domain_names"][ex["domain"]], ex["semester"], ex["target"], names))


def no_self_judging(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """
    Repeats error_by_protocol with each task's selected pairwise and absolute-scoring judge left out of its own
    references: the judge's predictions are refitted on the other references (the judge choice is unchanged), and the
    test uses the log likelihood ratios computed without that reference.

    Inputs:
        config: the analysis configuration; uses protocols, tasks (domains, semesters) and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        llr: every target's log likelihood ratios (analysis/forecasts/likelihood_ratio.csv).
        out: the output folder (<paths.outputs>/capability_signal).
    Outputs:
        in out: no_self_judging.{svg,png}; no_self_judging.txt (the caption with every number).
    """
    protocols = [BASELINE, *config["display"]["paper_protocols"]]
    semesters, variants = config["tasks"]["semesters"], config["protocols"]
    best = stats.best_judges(forecasts, variants)
    refitted = refit.refit_without_judge(forecasts, best[best.semester.isin(semesters)],
                                                   config["tasks"]["domains"], semesters)
    rows = refit.judge_dropped_llr(by_protocol.chosen_llr(llr, forecasts, variants, semesters))
    tests = by_protocol.protocol_tests(rows)
    print(tests.round(4).to_string(index=False))
    pooled = by_protocol.per_model(stats.target_errors(refitted, variants, semesters), protocols, config["tasks"]["domains"])
    save(by_protocol.model_error_figure(pooled, tests, protocols, config["display"]), out / "no_self_judging")
    (out / "no_self_judging.txt").write_text(by_protocol.error_caption(
        by_protocol.error_table(pooled, tests, protocols), config["display"]["protocol_names"],
        "Judge protocols use each task's best judge on references, shown the target's reasoning, with the judge's "
        "forecasts refitted on the references without the judge; the stars test those refitted forecasts."))


def breakdowns(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """
    Writes the per-task detail behind the pooled results: predicted against measured theta* for every target, one
    figure per protocol, and error_by_domain's table for each semester on its own.

    Inputs:
        config: the analysis configuration; uses protocols, tasks.semesters and display (paper_protocols,
            table_protocol_names, domain_names).
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        llr: every target's log likelihood ratios (analysis/forecasts/likelihood_ratio.csv).
        out: the output folder (<paths.outputs>/capability_signal).
    Outputs:
        in out/breakdowns: forecasts_<protocol>.{svg,png} for each protocol in display.paper_protocols;
        error_by_domain_by_semester.csv (error_by_domain's columns and Semester; one row per protocol and semester).
    """
    semesters = config["tasks"]["semesters"]
    for p in config["display"]["paper_protocols"]:
        for ext in ("svg", "png"):
            plots.forecast_figure(forecasts, p, config["protocols"].get(p, "-"), out / "breakdowns" / f"forecasts_{p}.{ext}",
                                  semesters=semesters)
    rows = full_set_llr(config, forecasts, llr)
    errors = stats.target_errors(forecasts, config["protocols"], semesters)
    tables = []
    for semester in semesters:
        g = rows[rows.semester == semester]
        table = by_protocol.domain_table(errors[errors.semester == semester], g, by_protocol.protocol_tests(g),
                                         config["display"]["table_protocol_names"], config["display"]["domain_names"])
        tables.append(table.assign(Semester=semester))
    pd.concat(tables, ignore_index=True).to_csv(out / "breakdowns" / "error_by_domain_by_semester.csv", index=False)
    print("wrote", out / "breakdowns")


def domain_glm(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """
    Fits a hierarchical Gamma model of squared error (log link; protocol, domain, their interaction and a term per
    target and domain; the Baseline the reference level) with hibayes, with and without the interaction, and reports
    each protocol's error ratio to the Baseline per domain and the Pairwise judge's error ratio to each other protocol.

    Inputs:
        config: the analysis configuration; uses domain_glm (protocols, error_floor, priors, sampler,
            contrast_against), protocols, tasks and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        llr: unused; every capability-signal analysis takes it.
        out: the output folder (<paths.outputs>/capability_signal).
    Outputs:
        in out/domain_glm: hibayes' rows.csv, fit state, checks and summaries; diagnostics.csv (per fit: divergences,
        r_hat, ESS, LOO, Pareto k counts, R^2 on the log scale); loo_compare.csv. In out: domain_glm_ratios.csv (one row
        per protocol, the Baseline first: the ratio and its lo, hi per domain and pooled ("all"), then the
        between-domain SD sigma with sigma_lo, sigma_hi); domain_glm_contrasts.csv (per protocol and domain: against,
        ratio, lo, hi, p_below_1); domain_glm_effects.{svg,png,txt}; domain_glm_contrasts.{svg,png,txt}. Raises
        AssertionError when a free parameter's r_hat exceeds 1.01.
    """
    settings, domains = config["domain_glm"], config["tasks"]["domains"]
    paper, show = config["display"]["paper_protocols"], config["display"]
    errors = stats.target_errors(forecasts, config["protocols"], config["tasks"]["semesters"], protocols=tuple(settings["protocols"]))
    rows = glm.glm_rows(errors, settings["error_floor"])
    print(rows.groupby(["domain", "protocol"]).size().unstack())
    state = out / "domain_glm"
    fitted, additive = glm.fit(rows, state, settings["protocols"], domains, settings["priors"], settings["sampler"])
    fits = {"protocol x domain": fitted, "protocol + domain": additive}
    diagnostics = pd.DataFrame({name: glm.diagnostics(f.inference_data, f.features) for name, f in fits.items()}).T
    diagnostics.to_csv(state / "diagnostics.csv", float_format="%.3f")
    glm.loo_compare({name: f.inference_data for name, f in fits.items()}).to_csv(state / "loo_compare.csv", float_format="%.3f")
    print(diagnostics.round(3).to_string())

    ratios = glm.ratio_table(fitted.inference_data.posterior, fitted.coords)
    contrasts = glm.contrast_table(fitted.inference_data.posterior, fitted.coords, settings["contrast_against"])
    glm.wide_table(ratios).to_csv(out / "domain_glm_ratios.csv")
    contrasts.to_csv(out / "domain_glm_contrasts.csv", index=False, float_format="%.4f")
    print(ratios.round(3).to_string(index=False))
    save(glm.effects_figure(ratios, paper, show), out / "domain_glm_effects")
    (out / "domain_glm_effects.txt").write_text(glm.effects_caption(ratios, paper, show))
    save(glm.contrasts_figure(contrasts, ratios, paper, show), out / "domain_glm_contrasts")
    (out / "domain_glm_contrasts.txt").write_text(glm.contrasts_caption(contrasts, ratios, paper, show))


# in this order: hibayes runs last, so it cannot change the style of the figures drawn before it
ANALYSES = {"error_by_protocol": error_by_protocol, "error_by_domain": error_by_domain,
            "prediction_example": prediction_example, "no_self_judging": no_self_judging, "breakdowns": breakdowns,
            "domain_glm": domain_glm}


def main(config: dict, only: list[str] | None) -> None:
    """
    Runs the capability-signal analyses in order.

    Inputs:
        config: the analysis configuration (scripts/analysis/config.yaml).
        only: names of the analyses to run (keys of ANALYSES); None runs them all.
    Outputs:
        the files each analysis writes, in <paths.outputs>/capability_signal.
    """
    apply_paths(config)
    out = Path(config["paths"]["outputs"]) / "capability_signal"
    (out / "breakdowns").mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(Path(config["paths"]["forecasts"]) / "forecasts.csv")
    llr = pd.read_csv(Path(config["paths"]["forecasts"]) / "likelihood_ratio.csv")
    for name, analysis in ANALYSES.items():
        if only is None or name in only:
            print(f"== capability_signal: {name}", flush=True)
            analysis(config, forecasts, llr, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--only", nargs="+", choices=list(ANALYSES), help="run only these analyses")
    args = parser.parse_args()
    main(load_config(args.config), args.only)
