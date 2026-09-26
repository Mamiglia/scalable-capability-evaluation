"""Does the capability signal hold beyond the reference models? Five analyses, each selectable with --only.

Writes to paths.outputs/generalisation (config.yaml); docs/manuscript_draft.pdf numbering:
    time                 error_over_time.{svg,png,txt} (Figure 6) and error_over_time.csv (Table 7): each task's mean
                         squared error per protocol, 2024-I to 2026-I
    reasoning_targets    reasoning_targets.{svg,png,txt} (Figure 7): Maths 2025-I reasoning targets predicted from
                         non-reasoning references, with and without their reasoning, and the t-tests the text quotes
    capability_distance  capability_distance.{svg,png,txt} (Figure 8): squared error per bin of the target's distance
                         above the strongest reference; capability_distance.csv (the bins) and
                         capability_distance_slopes.csv (the slopes of squared error on distance the text quotes)
    frontier             closed_targets.{pdf,png,svg,txt} (Figure 9): closed-source Coding targets, reasoning hidden;
                         closed_targets_errors.csv, closed_targets_llr.csv
    cross_domain         cross_domain_errors.csv (every protocol per source and scored domain) and
                         cross_domain_pairwise_tests.csv (Appendix D.5, the numbers behind Figure 18)

Reads paths.forecasts/forecasts.csv and the frozen tables (paths.data); frontier also reads the closed-target judge
runs (logs/), transcripts, features and embeddings; cross_domain recomputes every protocol's forecasts with another
domain's θ* (about 25 minutes) unless paths.cache/cross_domain_forecasts.csv exists. Launches no judging.
Run from the repository root:
    uv run python scripts/analysis/generalisation.py [--config scripts/analysis/config.yaml] [--only time frontier]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import likelihood_ratio as lr
from analysis import stats
from analysis.config import apply_paths, load_config
from analysis.generalisation import capability_distance, cross_domain, frontier, reasoning_targets, time
from analysis.loading import domain
from capability.task import Task

# the paper's protocols in figure order: every forecast protocol but the baseline and Length (a diagnostic)
PROTOCOLS = [p for p in stats.PROTOCOLS if p not in (stats.BASELINE, "glm_length")]


def save(fig: plt.Figure, path: Path, formats: tuple[str, ...] = ("svg", "png")) -> None:
    """`fig` as path.<format> for each format, then closed."""
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.with_suffix(f".{formats[0]}"))


def run_time(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """Figure 6 and Table 7: each task's squared error, 2024-I to 2026-I."""
    domains, display, semester = config["tasks"]["domains"], config["display"], config["generalisation"]["reasoning_semester"]
    draws, seed = config["statistics"]["bootstrap_draws"], config["statistics"]["seed"]
    errors = stats.target_errors(forecasts, config["protocols"], tuple(config["tasks"]["all_semesters"]))
    tasks = time.task_errors(errors, draws, seed)
    tasks = tasks[tasks.protocol != "glm_length"]  # dropped after the bootstrap, whose draws it moves
    tasks.to_csv(out / "error_over_time.csv", index=False, float_format="%.4f")
    save(time.error_over_time_figure(tasks, domains, PROTOCOLS, display, semester), out / "error_over_time")
    (out / "error_over_time.txt").write_text(time.caption(tasks, domains, PROTOCOLS, display, draws))


def run_reasoning_targets(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """Figure 7: the Pairwise judge's Maths 2025-I reasoning targets, reasoning shown and removed."""
    g, variants = config["generalisation"], config["protocols"]
    task = dict(domain=g["reasoning_domain"], semester=g["reasoning_semester"], protocol="pairwise")
    judge = stats.best_judges(forecasts, variants).set_index(list(task)).judge[tuple(task.values())]
    shown = reasoning_targets.shown(stats.target_errors(forecasts, variants), **task, reasoning_models=g["reasoning_targets"])
    removed = reasoning_targets.removed(forecasts, **task, variant=variants["pairwise"], judge=judge,
                                        reasoning_models=g["reasoning_targets"])
    tests = reasoning_targets.group_tests(shown, removed)
    save(reasoning_targets.reasoning_targets_figure(shown, removed, tests), out / "reasoning_targets")
    (out / "reasoning_targets.txt").write_text(reasoning_targets.caption(shown, removed, tests, judge))


def run_capability_distance(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """Figure 8 and the slopes the text quotes: squared error against the distance above the strongest reference."""
    domains, protocols, display = config["tasks"]["domains"], PROTOCOLS, config["display"]
    draws, seed = config["statistics"]["bootstrap_draws"], config["statistics"]["seed"]
    errors = stats.target_errors(forecasts, config["protocols"], tuple(config["tasks"]["semesters"]))
    table = capability_distance.with_bins(capability_distance.distances(errors), config["generalisation"]["distance_bins"])
    sizes = table[table.protocol == stats.BASELINE].groupby(["domain", "bin"]).size()
    assert (sizes >= 3).all(), f"a bin with fewer than 3 targets: {sizes.to_dict()}"
    boxes = capability_distance.bin_errors(table, protocols, draws, seed)
    slopes = capability_distance.slope_tests(table, domains, protocols)
    boxes.to_csv(out / "capability_distance.csv", index=False, float_format="%.4f")
    slopes.to_csv(out / "capability_distance_slopes.csv", index=False, float_format="%.4g")
    save(capability_distance.capability_distance_figure(boxes, table, domains, protocols, display), out / "capability_distance")
    (out / "capability_distance.txt").write_text(
        capability_distance.caption(boxes, table, slopes, domains, protocols, display, draws))


def run_frontier(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """Figure 9: closed-source Coding targets against the open ones of the same task, reasoning hidden."""
    f, logs = config["frontier"], Path(config["paths"]["logs"])
    semester, vendors = f["semester"], f["closed_targets"]
    tasks = {g: Task.from_semester(semester, domain(name)) for g, name in f["domains"].items()}
    refs = [t.references.theta.sort_index() for t in tasks.values()]
    assert refs[0].equals(refs[1]), "the closed and open targets' reference sets differ"
    targets = {"closed": list(vendors), "open": list(tasks["open"].test.models.index)}
    errors, llr = [], []
    for group, task in tasks.items():
        runs = {p: str(logs / f[p]["log_dir"] / f["domains"][group] / semester / f[p]["judge"] / f[p]["condition"])
                for p in ("pairwise", "direct")}
        args = (task, targets[group], runs["pairwise"], runs["direct"], f["views"]["embedding"], f["views"]["glm_regex"])
        errors.append(frontier.target_errors(*args).assign(group=group))
        llr.append(frontier.target_llr(*args, lr.SHAPE)
                   .assign(group=group, domain=f["domains"]["open"], semester=semester))
    errors, llr = pd.concat(errors, ignore_index=True), pd.concat(llr, ignore_index=True)
    errors.to_csv(out / "closed_targets_errors.csv", index=False, float_format="%.4f")
    llr.to_csv(out / "closed_targets_llr.csv", index=False, float_format="%.5f")
    deltas, tests = frontier.error_deltas(errors), frontier.group_tests(llr)
    save(frontier.frontier_figure(deltas, tests, vendors, config["display"]), out / "closed_targets", ("pdf", "png", "svg"))
    (out / "closed_targets.txt").write_text(
        frontier.caption(deltas, tests, config["display"], f["pairwise"]["judge"], f["direct"]["judge"]))


def run_cross_domain(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """Appendix D.5 and the numbers behind Figure 18: every protocol with transcripts from one domain scored on another."""
    domains, semesters, variants = config["tasks"]["domains"], config["tasks"]["semesters"], config["protocols"]
    cache = Path(config["paths"]["cache"]) / "cross_domain_forecasts.csv"
    if cache.exists():  # written at full precision, so a reused table gives the numbers a clean run does
        print("reusing", cache)
        cross = pd.read_csv(cache, float_precision="round_trip")
    else:
        cross = cross_domain.forecasts(domains, semesters)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cross.to_csv(cache, index=False)
    print(cross_domain.check_diagonal(cross, forecasts, variants, semesters))
    rows = cross_domain.fixed_judge_rows(cross, variants)
    cross_domain.cell_errors(rows).to_csv(out / "cross_domain_errors.csv", index=False, float_format="%.4f")
    cross_domain.pairwise_tests(rows).to_csv(out / "cross_domain_pairwise_tests.csv", index=False, float_format="%.4g")
    print("wrote", out / "cross_domain_errors.csv", out / "cross_domain_pairwise_tests.csv")


ANALYSES = {"time": run_time, "reasoning_targets": run_reasoning_targets, "capability_distance": run_capability_distance,
            "frontier": run_frontier, "cross_domain": run_cross_domain}


def main(config: dict, only: list[str] | None = None) -> None:
    """Every analysis, or those `only` names, in paper order."""
    apply_paths(config)
    out = Path(config["paths"]["outputs"]) / "generalisation"
    out.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(Path(config["paths"]["forecasts"]) / "forecasts.csv")
    for name, analysis in ANALYSES.items():
        if only is None or name in only:
            print(f"== {name}", flush=True)
            analysis(config, forecasts, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--only", nargs="+", choices=list(ANALYSES), help="run only these analyses")
    args = parser.parse_args()
    main(load_config(args.config), args.only)
