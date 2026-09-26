"""How much do the analysis choices matter? The judge, the number of judged items and of references, the saturation
boundary, the cost, and an ensemble of judges (Appendix D; figure and table numbers of docs/manuscript_draft.pdf).

Writes to paths.outputs/sensitivity (config.yaml), each figure as .pdf, .png, .svg with a .txt of its plotted numbers:
    judge_choice            Figure 15: squared error per target with the strongest, 2nd and 3rd strongest reference as
                            judge; judge_choice_medians.csv: the Appendix D.3 medians; candidate_judges.csv: Table 8
    item_count              Figure 19: squared error against the number of judged items, the selected judge, pairwise
                            and absolute scoring; breakdowns/item_count_{pairwise,absolute}_by_judge: every candidate
    reference_count         Figure 20: squared error as the references are restricted to stronger or weaker ones
    saturation_boundary     Figure 25: error ratio to the baseline per saturation level; saturation_boundary.csv: every
                            protocol's ratios; saturation_boundary_pairwise.csv: Table 10
    cost                    Figure 27: squared error against cost per semester
    ensemble                Figure 28: the three candidate judges' mean prediction against the selected judge
The item-count draws, reference subsets, saturation refits (theta*), cost curves and ensemble forecasts are cached in
paths.cache and reused when present; delete them to recompute.

Reads analysis/forecasts/forecasts.csv (forecasts.py), the frozen tables and transcripts (paths.data), the judge logs
(logs/), features and embeddings. Launches no judging and calls no API (the cost figure's prices are frozen in the
configuration).
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


def write_txt(path: Path, caption: str, *tables: pd.DataFrame) -> None:
    """A figure's numbers: the caption line, then each table."""
    path.write_text(caption + "\n\n" + "\n\n".join(t.round(4).to_string(index=False) for t in tables) + "\n")


def pairwise_judges(forecasts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Each task's selected pairwise judge (domain, semester, judge)."""
    best = stats.best_judges(forecasts, config["protocols"])
    return best[best.protocol == "pairwise"][["domain", "semester", "judge"]]


def pairwise_runs(forecasts: pd.DataFrame, config: dict) -> dict[tuple[str, str, str], Path]:
    """(domain, semester, judge) -> the run directory of each pairwise judge's core-condition run."""
    return item_budget.core_run_dirs(forecasts, config["protocols"]["pairwise"], Path(config["paths"]["logs"]))


def run_judge_choice(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """Figure 15, the Appendix D.3 medians, Table 8."""
    variants, semesters, display = config["protocols"], config["tasks"]["semesters"], config["display"]
    f = judge_choice.judge_errors(forecasts, variants)
    medians = judge_choice.medians(f, semesters)
    medians.to_csv(out / "judge_choice_medians.csv", header=["mse"], float_format="%.3f")
    plotted = judge_choice.figure(f, semesters, display["protocol_names"], display["protocol_colours"], str(out / "judge_choice"))
    per_target = judge_choice.by_rank(f, semesters)
    write_txt(out / "judge_choice.txt",
              f"Squared error per target ({', '.join(semesters)}, mean over its domains) with the judge of each theta* rank "
              "among the three candidates; mean and 95% t-interval half-width per column (Figure 15); then every target's "
              "error; then the Appendix D.3 medians over tasks (MSE).",
              plotted, per_target, medians.rename("mse").reset_index().rename(columns={"index": "median over tasks"}))
    judge_choice.candidate_judges(forecasts, variants).to_csv(out / "candidate_judges.csv", index=False)


def run_item_count(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """Figure 19 and the per-judge breakdowns."""
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
    shared = dict(x="n", xlabel="Judged items", xticks=(10, 50, 100), colours=display["domain_line_colours"], markers=display["domain_markers"],
                  names=display["domain_names"])
    table = item_budget.curves_figure(both, key="mse", path=str(out / "item_count"), **shared)
    write_txt(out / "item_count.txt", "Squared error of the selected judge's predictions against the number of judged items "
              "(Figure 19): median over random item subsets, 2.5-97.5 percentile band, the task's baseline MSE.", table)
    (out / "breakdowns").mkdir(exist_ok=True)
    for name, df, key in (("pairwise", pair, "linear"), ("absolute", rate, "direct")):
        path = out / "breakdowns" / f"item_count_{name}_by_judge"
        table = item_budget.curves_figure(df.assign(row="Top-" + df.judge_rank.astype(str) + " judge"), key=key, path=str(path), **shared)
        write_txt(path.with_suffix(".txt"), f"As item_count, {name} judging by each candidate judge (row: its theta* rank).", table)


def run_reference_count(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """Figure 20."""
    display, csv = config["display"], cache / "reference_count.csv"
    if not csv.exists():
        item_budget.reference_subsets(pairwise_runs(forecasts, config), config["tasks"]["domains"], pairwise_judges(forecasts, config),
                                      config["sensitivity"]["reference_budget"]["sizes"]) \
            .to_csv(csv, index=False, float_format="%.4f")
    ends = item_budget.keep_stronger_or_weaker(pd.read_csv(csv))
    table = item_budget.curves_figure(ends.assign(row="Keep " + ends.keep), key="linear", x="k", xlabel="Reference models",
                                      xticks=(2, 4, 6, 8), colours=display["domain_line_colours"], markers=display["domain_markers"],
                                      names=display["domain_names"], path=str(out / "reference_count"))
    write_txt(out / "reference_count.txt", "Squared error of pairwise judging with k references kept (Figure 20): median over "
              "the third of the k-subsets keeping the strongest (or weakest) references, 2.5-97.5 percentile band, the task's "
              "baseline MSE (all references).", table)


def run_saturation_boundary(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """Figure 25, Table 10."""
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
    drawn = display["paper_protocols"]
    saturation_boundary.figure(table, domains, levels, drawn, display["protocol_names"], display["protocol_colours"],
                               display["domain_names"], str(out / "saturation_boundary"))
    write_txt(out / "saturation_boundary.txt", "MSE over the Baseline's MSE on the same targets per saturation level, with its "
              "bootstrap 95% interval over targets (Figure 25); domain 'all' pools every target (Table 10's pooled row).",
              table[table.protocol.isin(drawn)].drop(columns="judge"))


def run_cost(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """Figure 27."""
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
    write_txt(out / "cost.txt", "Per domain: mean cost per semester ($) and MSE over every target with its bootstrap 95% "
              "interval; judge protocols per fraction of the judged items; mean_ref is the baseline (Figure 27).", s)


def run_ensemble(config: dict, forecasts: pd.DataFrame, out: Path, cache: Path) -> None:
    """Figure 28."""
    display, csv = config["display"], cache / "ensemble_forecasts.csv"
    if not csv.exists():
        ensemble.ensemble_forecasts(pairwise_runs(forecasts, config), config["tasks"]["domains"]) \
            .to_csv(csv, index=False, float_format="%.4f")
    t = ensemble.task_errors(pd.read_csv(csv, keep_default_na=False, na_values=[""]), pairwise_judges(forecasts, config), TOP_K,
                             config["tasks"]["domains"])
    ensemble.figure(t, config["tasks"]["domains"], display["domain_line_colours"], display["domain_markers"], display["domain_names"], str(out / "ensemble"))
    write_txt(out / "ensemble.txt", "Per task, MSE over the targets of the baseline, of the selected pairwise judge and of the "
              "mean of the three candidates' predictions, on the items all three judged; difference = ensemble - single (Figure 28).",
              t.assign(difference=t.ensemble - t.single))


ANALYSES = {"judge_choice": run_judge_choice, "item_count": run_item_count, "reference_count": run_reference_count,
            "saturation_boundary": run_saturation_boundary, "cost": run_cost, "ensemble": run_ensemble}


def main(config: dict, only: list[str] | None) -> None:
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
