"""
Tests whether the capability signal holds beyond the main analysis: over earlier target semesters, for reasoning
targets predicted from non-reasoning references, for targets far above the strongest reference, for closed-source
targets, and with one domain's transcripts scored against another domain's theta*. Each of the five analyses can be
selected with --only.

Writes to <paths.outputs>/generalisation/ (config.yaml):
    time                 error_over_time.{svg,png,txt} and error_over_time.csv: each task's mean squared error per
                         protocol with its bootstrap interval, 2024-I to 2026-I
    reasoning_targets    reasoning_targets.{svg,png,txt}: the Pairwise judge on the Maths 2025-I reasoning targets,
                         whose references are all non-reasoning models, with and without the targets' reasoning, and
                         the t-tests between the groups of targets
    capability_distance  capability_distance.{svg,png,txt}: squared error per bin of the target's distance above the
                         strongest reference; capability_distance.csv (the bins) and capability_distance_slopes.csv
                         (the slopes of squared error on distance)
    frontier             closed_targets.{pdf,png,svg,txt}: closed-source Coding targets, reasoning hidden;
                         closed_targets_errors.csv, closed_targets_llr.csv
    cross_domain         cross_domain_errors.csv (every protocol per source and scored domain) and
                         cross_domain_pairwise_tests.csv (the Pairwise judge's paired t-tests per source and scored
                         domain)

Reads <paths.forecasts>/forecasts.csv and the frozen tables (paths.data); frontier also reads the closed-target judge
runs (logs/), transcripts, features and embeddings; cross_domain recomputes every protocol's predictions with another
domain's theta* (about 25 minutes) unless <paths.cache>/cross_domain_forecasts.csv exists. Launches no judging.
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

# the protocols drawn, in order: every protocol with one prediction per target except the Baseline and Length
PROTOCOLS = [p for p in stats.PROTOCOLS if p not in (stats.BASELINE, "glm_length")]


def save(fig: plt.Figure, path: Path, formats: tuple[str, ...] = ("svg", "png")) -> None:
    """
    Saves a figure in each format at 300 dpi and closes it.

    Inputs:
        fig: the figure.
        path: the output path; its suffix is replaced by each format's.
        formats: the file formats.
    Outputs:
        path.<format> for each format.
    """
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.with_suffix(f".{formats[0]}"))


def run_time(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """
    Measures the prediction error over target semesters: each task's mean squared error per protocol (the Baseline
    included, Length left out) with a bootstrap interval over the task's targets, for every task of
    tasks.all_semesters.

    Inputs:
        config: the analysis configuration; uses tasks (domains, all_semesters), protocols, statistics (bootstrap draws
            and seed), display and generalisation.reasoning_semester (marked by an arrow).
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/generalisation).
    Outputs:
        in out: error_over_time.csv (per domain, semester and protocol: n, mse, baseline_mse, lo, hi);
        error_over_time.{svg,png}; error_over_time.txt (the caption with every task's error and interval).
    """
    domains, display, semester = config["tasks"]["domains"], config["display"], config["generalisation"]["reasoning_semester"]
    draws, seed = config["statistics"]["bootstrap_draws"], config["statistics"]["seed"]
    errors = stats.target_errors(forecasts, config["protocols"], tuple(config["tasks"]["all_semesters"]))
    tasks = time.task_errors(errors, draws, seed)
    tasks = tasks[tasks.protocol != "glm_length"]  # dropped after the bootstrap, whose draws it moves
    tasks.to_csv(out / "error_over_time.csv", index=False, float_format="%.4f")
    save(time.error_over_time_figure(tasks, domains, PROTOCOLS, display, semester), out / "error_over_time")
    (out / "error_over_time.txt").write_text(time.caption(tasks, domains, PROTOCOLS, display, draws))


def run_reasoning_targets(config: dict, forecasts: pd.DataFrame, out: Path) -> None:
    """
    Compares the Pairwise judge's squared error on the reasoning and the non-reasoning targets of a task whose
    references are all non-reasoning models (Maths 2025-I), and on the reasoning targets judged with their reasoning
    removed: Welch t-tests of each reasoning group against the non-reasoning targets, and a paired t-test of reasoning
    removed against shown.

    Inputs:
        config: the analysis configuration; uses generalisation (reasoning_domain, reasoning_semester,
            reasoning_targets) and protocols.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv); the task's selected
            pairwise judge's scored run gives the reasoning-shown errors, its reasoning-hidden run the removed ones.
        out: the output folder (<paths.outputs>/generalisation).
    Outputs:
        in out: reasoning_targets.{svg,png}; reasoning_targets.txt (the caption with every target's theta*, prediction
        and squared error, each group's mean and interval, and the three tests).
    """
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
    """
    Relates squared error to how far a target's theta* lies above its task's strongest reference: the targets of
    tasks.semesters in equal-count distance bins per domain, each bin's mean squared error per protocol with a
    bootstrap interval, and per domain and protocol the least-squares slope of squared error on distance.

    Inputs:
        config: the analysis configuration; uses tasks (domains, semesters), protocols, statistics,
            generalisation.distance_bins and display.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv).
        out: the output folder (<paths.outputs>/generalisation).
    Outputs:
        in out: capability_distance.csv (per domain, bin and protocol: x, the bin's mean distance, n, mean, lo, hi);
        capability_distance_slopes.csv (per domain and protocol: n, slope, se, p); capability_distance.{svg,png};
        capability_distance.txt (the caption with every bin, the Baseline's line and each slope test). Raises
        AssertionError when a bin has fewer than 3 targets.
    """
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
    """
    Predicts the closed-source Coding targets (frontier.closed_targets) and the open targets of the same task from the
    same references, with the reasoning hidden for every model and protocol, and compares each target's squared error
    with the Baseline's; per protocol and group (closed, open), a one-sample t-test of the mean log likelihood ratio.

    Inputs:
        config: the analysis configuration; uses frontier (semester, domains, the pairwise and direct runs, views,
            closed_targets), paths.logs and display.
        forecasts: unused; every generalisation analysis takes it.
        out: the output folder (<paths.outputs>/generalisation).
    Outputs:
        in out: closed_targets_errors.csv (per target and protocol, the Baseline included: model, protocol, theta,
        theta_hat, error, group); closed_targets_llr.csv (per target and protocol: model, shape, llr, dropped,
        protocol, group, domain, semester); closed_targets.{pdf,png,svg}; closed_targets.txt (the caption with every
        target's change in error and each group's test). Raises AssertionError when the two tasks' references differ.
    """
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
    """
    Scores every protocol's predictions from one domain's transcripts against another domain's theta*, for each
    ordered pair of domains on the tasks of tasks.semesters, each judge protocol keeping the source domain's own
    selected judge; first checks that a domain scored on itself reproduces forecasts.csv.

    Inputs:
        config: the analysis configuration; uses tasks (domains, semesters), protocols and paths.cache.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv), for that check.
        out: the output folder (<paths.outputs>/generalisation).
    Outputs:
        in out: cross_domain_errors.csv (per source domain, scored domain and protocol, semesters pooled: source,
        scored, protocol, n, mse, baseline_mse); cross_domain_pairwise_tests.csv (the Pairwise judge per source and
        scored domain: mse, baseline, own, ratio_own, p_own, p_baseline); in paths.cache: cross_domain_forecasts.csv
        (the recomputed predictions), reused when present.
    """
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
    """
    Runs the generalisation analyses in order.

    Inputs:
        config: the analysis configuration (scripts/analysis/config.yaml).
        only: names of the analyses to run (keys of ANALYSES); None runs them all.
    Outputs:
        the files each analysis writes, in <paths.outputs>/generalisation (and cross_domain's table in paths.cache).
    """
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
