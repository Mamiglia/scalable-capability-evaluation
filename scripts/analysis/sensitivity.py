"""
How much the results depend on choices made in the analysis: which judge is used, how many items are judged, how many
references there are, where the saturation boundary is drawn, what each protocol costs, and whether an ensemble of
judges beats the single selected judge.

Writes to <paths.outputs>/sensitivity/ (config.yaml); each figure as .pdf, .png and .svg with its numbers as .csv:
    judge_choice            each target's squared error with the strongest, second and third strongest reference as
                            judge; judge_choice.csv (the plotted means and t-intervals), judge_choice_targets.csv (every
                            plotted target), judge_choice_medians.csv (medians over tasks), candidate_judges.csv (each
                            task's three candidate judges, ordered by their error on the references)
    item_count              squared error against the number of judged items, for the selected pairwise and absolute
                            scoring judges (item_count.csv); breakdowns/item_count_{pairwise,absolute}_by_judge for
                            every candidate judge
    reference_count         pairwise squared error as the references are cut to the strongest or the weakest few
                            (reference_count.csv)
    saturation_boundary     each protocol's error relative to the Baseline as the saturation boundary tightens;
                            saturation_boundary.csv (every protocol), saturation_boundary_pairwise.csv (pairwise judge)
    cost                    squared error against dollar cost per semester (cost.csv)
    ensemble                the mean prediction of the three candidate judges against the selected judge's
                            (ensemble.csv)
The item-count draws, reference subsets, refitted capabilities (theta*), cost curves and ensemble predictions are
cached in <paths.cache>/ and reused when present; delete them to recompute.

Reads analysis/forecasts/forecasts.csv (written by forecasts.py), the frozen tables and transcripts (paths.data), the
judge logs (logs/), the features and the embeddings. Launches no judging and calls no API: the prices are in
config.yaml.
Run from the repository root:
    uv run python scripts/analysis/sensitivity.py [--config scripts/analysis/config.yaml] [--only cost ensemble ...]
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis import stats
from analysis.config import apply_paths, load_config
from analysis.sensitivity import cost, ensemble, item_budget, judge_choice, saturation_boundary
from reference.freeze import TOP_K


def write_csv(table: pd.DataFrame, path: Path) -> None:
    """
    Writes the numbers behind a figure as a csv, rounded to four decimals (DataFrame.round, half to even: many medians
    fall exactly halfway, where "%.4f" formatting would round the other way).

    Inputs:
        table: the numbers.
        path: the .csv file to write.
    Outputs:
        path, written.
    """
    table.round(4).to_csv(path, index=False)


def pairwise_judges(forecasts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Each task's pairwise judge: of the three candidate judges, the one with the lowest leave-one-model-out error on the
    references.

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        config: the analysis configuration; uses protocols (the variant each protocol is scored in).
    Outputs:
        DataFrame with columns domain, semester, judge; one row per task.
    """
    best = stats.best_judges(forecasts, config["protocols"])
    return best[best.protocol == "pairwise"][["domain", "semester", "judge"]]


def pairwise_runs(forecasts: pd.DataFrame, config: dict) -> dict[tuple[str, str, str], Path]:
    """
    Where each candidate pairwise judge's verdicts are: the log folder of the run the main analysis scores (reasoning
    shown on the full train set where that run exists, otherwise the reasoning-hidden run).

    Inputs:
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        config: the analysis configuration; uses protocols.pairwise and paths.logs.
    Outputs:
        dict (domain, semester, judge) -> run folder, logs/pairwise/<view>/<domain>/<semester>/<judge>/<condition hash>.
    """
    return item_budget.core_run_dirs(forecasts, config["protocols"]["pairwise"], Path(config["paths"]["logs"]))


def run_judge_choice(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """
    Measures how much the choice among the three candidate judges matters: each target's squared error when the
    strongest, second or third strongest reference is the judge, for pairwise judging and absolute scoring.

    Inputs:
        config: the analysis configuration; uses protocols, tasks.semesters and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/sensitivity).
        cache: unused; every sensitivity analysis takes it.
    Outputs:
        in out: judge_choice.{pdf,png,svg}; judge_choice.csv (per protocol and judge rank: n targets, mean, the
        t-interval's lo, hi and half-width); judge_choice_targets.csv (per protocol, judge rank and target: the plotted
        squared error e and n_domains, the domains it is averaged over); judge_choice_medians.csv (median over tasks
        of the spread across judges, and of the gaps to absolute scoring and the Baseline); candidate_judges.csv (per
        task and judge protocol, the three candidates from lowest to highest error on the references).
    """
    variants, semesters, display = config["protocols"], config["tasks"]["semesters"], config["display"]
    f = judge_choice.judge_errors(forecasts, variants)
    medians = judge_choice.medians(f, semesters)
    medians.to_csv(out / "judge_choice_medians.csv", header=["mse"], float_format="%.3f")
    plotted = judge_choice.figure(f, semesters, display["protocol_names"], display["protocol_colours"], str(out / "judge_choice"))
    write_csv(plotted, out / "judge_choice.csv")
    write_csv(judge_choice.by_rank(f, semesters), out / "judge_choice_targets.csv")
    judge_choice.candidate_judges(forecasts, variants).to_csv(out / "candidate_judges.csv", index=False)


def run_item_count(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """
    Measures how the squared error depends on the number of judged items: for each task, random subsets of 3 to 100
    judged items, predictions recomputed from each subset, for pairwise judging and absolute scoring.

    Inputs:
        config: the analysis configuration; uses sensitivity.item_budget (subset sizes, draws, seed), tasks.domains,
            protocols and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/sensitivity).
        cache: folder for the per-subset errors, reused when present.
    Outputs:
        in out: item_count.{pdf,png,svg,csv} for each task's selected judges, and
        breakdowns/item_count_{pairwise,absolute}_by_judge.{pdf,png,svg,csv} for every candidate judge; each csv has
        one row per plotted point: protocol (item_count) or judge (its theta* rank, the breakdowns), semester, domain,
        n (judged items), median, lo and hi (2.5 and 97.5 percentiles over the item subsets) and baseline (the
        Baseline's MSE);
        in cache: item_count_pairwise.csv and item_count_direct.csv (one row per task, judge, subset size and draw).
    """
    budget, display, domains = config["sensitivity"]["item_budget"], config["display"], config["tasks"]["domains"]
    pair_csv, direct_csv = cache / "item_count_pairwise.csv", cache / "item_count_direct.csv"
    if not pair_csv.exists():
        item_budget.pairwise_item_draws(pairwise_runs(forecasts, config), domains, budget["sizes"], budget["draws"], budget["seed"]) \
            .to_csv(pair_csv, index=False, float_format="%.4f")
    if not direct_csv.exists():
        item_budget.direct_item_draws(domains, budget["sizes"], budget["draws"], budget["seed"]).to_csv(direct_csv, index=False, float_format="%.4f")
    pair, rate = pd.read_csv(pair_csv), pd.read_csv(direct_csv)
    best = stats.best_judges(forecasts, config["protocols"])
    selected = {p: best[best.protocol == p][["domain", "semester", "judge"]] for p in ("pairwise", "direct")}
    both = pd.concat([pair.merge(selected["pairwise"]).assign(row="Pairwise", mse=lambda d: d.linear, judge_rank=1),
                      rate.merge(selected["direct"]).assign(row="Absolute", mse=lambda d: d.direct, judge_rank=2)])
    shared = dict(x="n", xlabel="Judged items", xticks=(10, 50, 100), colours=display["domain_colours"], markers=display["domain_markers"],
                  names=display["domain_names"])
    table = item_budget.curves_figure(both, key="mse", path=str(out / "item_count"), **shared)
    write_csv(table.rename(columns={"row": "protocol"}), out / "item_count.csv")
    (out / "breakdowns").mkdir(exist_ok=True)
    for name, df, key in (("pairwise", pair, "linear"), ("absolute", rate, "direct")):
        path = out / "breakdowns" / f"item_count_{name}_by_judge"
        table = item_budget.curves_figure(df.assign(row="Top-" + df.judge_rank.astype(str) + " judge"), key=key, path=str(path), **shared)
        write_csv(table.rename(columns={"row": "judge"}), path.with_suffix(".csv"))


def run_reference_count(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """
    Measures how the pairwise squared error depends on the references: each task's targets placed using every subset of
    2, 4 or 6 references in place of all of them, summarised over the subsets keeping the strongest or the weakest.

    Inputs:
        config: the analysis configuration; uses sensitivity.reference_budget.sizes, tasks.domains and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/sensitivity).
        cache: folder for the per-subset errors, reused when present.
    Outputs:
        in out: reference_count.{pdf,png,svg}; reference_count.csv (one row per plotted point: references_kept
        ("Keep stronger" or "Keep weaker"), semester, domain, k (references kept), median, lo and hi (2.5 and 97.5
        percentiles over the kept subsets) and baseline (the Baseline's MSE with every reference)); in cache:
        reference_count.csv (one row per task and reference subset).
    """
    display, csv = config["display"], cache / "reference_count.csv"
    if not csv.exists():
        item_budget.reference_subsets(pairwise_runs(forecasts, config), config["tasks"]["domains"], pairwise_judges(forecasts, config),
                                      config["sensitivity"]["reference_budget"]["sizes"]) \
            .to_csv(csv, index=False, float_format="%.4f")
    ends = item_budget.keep_stronger_or_weaker(pd.read_csv(csv))
    table = item_budget.curves_figure(ends.assign(row="Keep " + ends.keep), key="linear", x="k", xlabel="Reference models",
                                      xticks=(2, 4, 6, 8), colours=display["domain_colours"], markers=display["domain_markers"],
                                      names=display["domain_names"], path=str(out / "reference_count"))
    write_csv(table.rename(columns={"row": "references_kept"}), out / "reference_count.csv")


def run_saturation_boundary(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """
    Measures how the results depend on the saturation boundary: items count as saturated only if the k strongest
    references (k = 3 to 8), or every model, solve them; theta* and every protocol are refitted at each level and
    each protocol's squared error is compared with the Baseline's.

    Inputs:
        config: the analysis configuration; uses sensitivity.saturation_levels, tasks, protocols, statistics (bootstrap
            draws and seed) and display.
        forecasts: unused; every sensitivity analysis takes it.
        out: the output folder (<paths.outputs>/sensitivity).
        cache: folder for the refitted theta*, reused when present.
    Outputs:
        in out: saturation_boundary.{pdf,png,svg}; saturation_boundary.csv (every protocol's error ratio to the
        Baseline per level and domain, with bootstrap intervals); saturation_boundary_pairwise.csv (the pairwise judge's
        ratios and item counts); in cache: saturation_theta.csv.
    """
    variants, display, levels = config["protocols"], config["display"], config["sensitivity"]["saturation_levels"]
    domains, statistics = config["tasks"]["domains"], config["statistics"]
    theta_csv = cache / "saturation_theta.csv"
    cached = pd.read_csv(theta_csv, dtype={"level": str}, float_precision="round_trip") if theta_csv.exists() else None
    theta, level_forecasts = saturation_boundary.level_forecasts(domains, config["tasks"]["semesters"], levels, cached, variants["embedding"])
    if cached is None:
        theta.to_csv(theta_csv, index=False)
    table = saturation_boundary.ratios(level_forecasts, variants, statistics["seed"], statistics["bootstrap_draws"])
    table.to_csv(out / "saturation_boundary.csv", index=False, float_format="%.4f")
    pairwise = saturation_boundary.pairwise_table(table, theta, domains, levels, display["domain_names"])
    pairwise.to_csv(out / "saturation_boundary_pairwise.csv", index=False, float_format="%.4f")
    saturation_boundary.figure(table, domains, levels, display["paper_protocols"], display["protocol_names"],
                               display["protocol_colours"], display["domain_names"], str(out / "saturation_boundary"))


def run_cost(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """
    Sets each protocol's squared error against its dollar cost per semester, pricing the tokens each judge used at the
    OpenRouter prices in config.yaml; pairwise judging and absolute scoring are traced over 10% to 100% of the items.

    Inputs:
        config: the analysis configuration; uses sensitivity.cost (item fractions, draws, seed, prices), tasks.domains,
            protocols, statistics and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/sensitivity).
        cache: folder for the cost curves, reused when present.
    Outputs:
        in out: cost.{pdf,png,svg}; cost.csv (one row per domain, protocol and item fraction: fraction (empty for
        the protocols without a judge and for the Baseline, "mean_ref"), cost (mean dollars per semester), mse and its
        bootstrap 95% interval lo, hi); in cache: cost_curves.csv (judge protocols per item fraction and draw) and
        cost_points.csv (the protocols without a judge).
    """
    c, display, statistics = config["sensitivity"]["cost"], config["display"], config["statistics"]
    curves_csv, points_csv = cache / "cost_curves.csv", cache / "cost_points.csv"
    if not curves_csv.exists():
        curves, points = cost.cost_curves(forecasts, stats.best_judges(forecasts, config["protocols"]), config["tasks"]["domains"],
                                          config["protocols"], c["fractions"], c["draws"], c["seed"], c["prices"], c["proxies"],
                                          c["embedding_price"])
        curves.to_csv(curves_csv, index=False, float_format="%.5f")
        points.to_csv(points_csv, index=False, float_format="%.5f")
    s = cost.summarise(pd.read_csv(curves_csv), pd.read_csv(points_csv), statistics["seed"], statistics["bootstrap_draws"])
    cost.figure(s, config["tasks"]["domains"], display["paper_protocols"], display["protocol_names"], display["protocol_colours"],
                display["domain_names"], str(out / "cost"))
    write_csv(s, out / "cost.csv")


def run_ensemble(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """
    Tests whether an ensemble of judges beats the selected one: per task, the squared error of the mean of the three
    candidate judges' pairwise predictions against the selected judge's, on the items all three judged.

    Inputs:
        config: the analysis configuration; uses tasks.domains, protocols and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/sensitivity).
        cache: folder for the ensemble predictions, reused when present.
    Outputs:
        in out: ensemble.{pdf,png,svg}; ensemble.csv (one row per task: domain, semester, the MSEs baseline, single
        (the selected judge, named in judge) and ensemble, and difference = ensemble - single); in cache:
        ensemble_forecasts.csv (every candidate's prediction per task and target).
    """
    display, csv = config["display"], cache / "ensemble_forecasts.csv"
    if not csv.exists():
        ensemble.ensemble_forecasts(pairwise_runs(forecasts, config), config["tasks"]["domains"]) \
            .to_csv(csv, index=False, float_format="%.4f")
    t = ensemble.task_errors(pd.read_csv(csv, keep_default_na=False, na_values=[""]), pairwise_judges(forecasts, config), TOP_K,
                             config["tasks"]["domains"])
    ensemble.figure(t, config["tasks"]["domains"], display["domain_colours"], display["domain_markers"], display["domain_names"], str(out / "ensemble"))
    write_csv(t.assign(difference=t.ensemble - t.single), out / "ensemble.csv")


ANALYSES = {"judge_choice": run_judge_choice, "item_count": run_item_count, "reference_count": run_reference_count,
            "saturation_boundary": run_saturation_boundary, "cost": run_cost, "ensemble": run_ensemble}


def main(config: dict, only: list[str] | None) -> None:
    """
    Runs the sensitivity analyses in order.

    Inputs:
        config: the analysis configuration (scripts/analysis/config.yaml).
        only: names of the analyses to run (keys of ANALYSES); None runs them all.
    Outputs:
        the files each analysis writes, in <paths.outputs>/sensitivity and <paths.cache>.
    """
    # the item-count, reference-count and ensemble code converts verdicts and ratings with these variants only
    assert config["protocols"]["pairwise"] == "linear" and config["protocols"]["direct"] == "raw", config["protocols"]
    apply_paths(config)
    out, cache = Path(config["paths"]["outputs"]) / "sensitivity", Path(config["paths"]["cache"])
    out.mkdir(parents=True, exist_ok=True), cache.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(Path(config["paths"]["forecasts"]) / "forecasts.csv")
    for name, run in ANALYSES.items():
        if only is None or name in only:
            print("==", name, flush=True)
            run(config, forecasts, out, cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--only", nargs="+", choices=list(ANALYSES), help="run only these analyses")
    args = parser.parse_args()
    main(load_config(args.config), args.only)
