"""Do saturated transcripts carry capability information? Each protocol's squared error and log likelihood ratio
against the Baseline for the targets of the main analysis, and how they vary by domain.

Writes to <paths.outputs>/capability_signal/ (docs/manuscript_draft.pdf numbering):
    error_by_protocol.{svg,png,txt}        Figure 3: squared error per target model pooled over domains, with the
                                           one-sample test of the log likelihood ratio; the .txt holds every plotted
                                           number and both tests of Table 6
    error_by_protocol.csv                  Table 6
    error_by_domain.csv                    Table 3: mean squared error per protocol and domain, marked by its test
    prediction_example.{svg,png,txt}       Figure 10: one target's predictive distributions
    no_self_judging.{svg,png,txt}          Figure 21: Figure 3 with each task's judge left out of its own references
    domain_glm_ratios.csv                  Table 9: the domain GLM's error ratio to the Baseline per protocol and domain
    domain_glm_effects.{svg,png,txt}       Figure 16: those ratios
    domain_glm_contrasts.{svg,png,txt,csv} Figure 17: the Pairwise judge's error ratio to each other protocol, and each
                                           protocol's between-domain SD
    domain_glm/                            hibayes' rows, fit state and checks; diagnostics.csv (the convergence numbers
                                           of Appendix D.4) and loo_compare.csv (with against without the
                                           (protocol | domain) term)
    breakdowns/forecasts_<protocol>.{svg,png}   predicted against measured θ for every target, per domain and semester
    breakdowns/error_by_domain_by_semester.csv  Table 3 for each semester on its own

Reads <paths.forecasts>/forecasts.csv and likelihood_ratio.csv (scripts/analysis/forecasts.py); prediction_example
and no_self_judging refit forecasts, so they also read the frozen tables, judge logs, features and embeddings.
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
    for ext in ("svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem.with_suffix(".svg"))


def full_set_llr(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame) -> pd.DataFrame:
    """The tested log likelihood ratio rows of the main analysis, full reference set."""
    chosen = by_protocol.chosen_llr(llr, forecasts, config["protocols"], config["tasks"]["semesters"])
    return chosen[chosen.dropped == "-"]


def error_by_protocol(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """Figure 3 and Table 6."""
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
    """Table 3."""
    rows = full_set_llr(config, forecasts, llr)
    errors = stats.target_errors(forecasts, config["protocols"], config["tasks"]["semesters"])
    table = by_protocol.domain_table(errors, rows, by_protocol.protocol_tests(rows), config["display"]["table_protocol_names"],
                                     config["display"]["domain_names"])
    table.to_csv(out / "error_by_domain.csv", index=False)
    print(table.to_string(index=False))


def prediction_example(config: dict, forecasts: pd.DataFrame, llr: pd.DataFrame, out: Path) -> None:
    """Figure 10."""
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
    """Figure 21."""
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
    """Predicted against measured θ per target; Table 3 per semester."""
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
    """Table 9, Figures 16 and 17."""
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
