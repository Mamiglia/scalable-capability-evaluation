"""
Builds the two tables every other analysis runner reads: each protocol's prediction of each target's theta*, and each
target's log likelihood ratio against the Baseline. The predictions cover every task of tasks.all_semesters, every
protocol and variant and every judge run on disk; the other runners pick the rows they report from them.

Writes to <paths.forecasts> (config.yaml):
    forecasts.csv          one row per domain, semester, protocol, variant, judge, condition and target (model):
                           theta_hat (the prediction), theta (the target's measured theta*), missing (the judgments the
                           judge run still lacks) and judge_loo_mse (the judge run's leave-one-model-out error on the
                           references; NaN for protocols without a judge, whose judge and condition are "-"). The
                           protocols include the accuracy oracle (accuracy_hard), Length (glm_length) and each judge
                           protocol's pooled run over its judges (pairwise_ensemble, direct_ensemble; judge "ensemble").
    likelihood_ratio.csv   one row per domain, semester, target, protocol, judge, dropped reference and shape: the log
                           likelihood ratio of the target's measured theta* under the protocol's predictive distribution
                           against the Baseline's. Protocols glm_regex and embedding (judge "-"), and pairwise and
                           direct for every judge's run closest to reasoning shown on the full train set, in the
                           variant config.yaml reports; dropped "-" is the full reference set; shape "normal" or "t".

Reads the frozen tables and transcripts (paths.data), judge logs (logs/), features and embeddings. Launches no judging.
Run from the repository root:
    uv run python scripts/analysis/forecasts.py [--config scripts/analysis/config.yaml]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import calibration, loading, matching, stats
from analysis import likelihood_ratio as lr
from analysis.config import apply_paths, load_config
from capability.evaluate import semester_names
from protocols import PROTOCOLS
from protocols.embeddings import EmbeddingRegression


def forecasts_table(domains: list[str], semesters: list[str]) -> pd.DataFrame:
    """
    Predicts every target of every task with every protocol: the protocols without a judge are fitted on the task's
    references, and every judge run on disk (cut to the frozen train items) is converted to theta_hat in each of its
    variants.

    Inputs:
        domains: domain names (configs/domain/<name>.yml).
        semesters: the target semesters; each domain is predicted on those of them it has.
    Outputs:
        DataFrame with columns domain, semester, protocol, variant, judge, condition, judge_loo_mse, model, theta_hat,
        theta, missing; one row per target, protocol, variant and judge run (judge and condition "-" for protocols
        without a judge; pooled judge runs as protocol pairwise_ensemble or direct_ensemble, judge "ensemble").
    """
    frames = []
    for name in domains:
        dom = loading.domain(name)
        protocols = calibration.transcript_protocols(dom)
        for task in loading.tasks(dom, [s for s in semesters if s in semester_names(dom)]):
            runs = [matching.match(run, task) for run in loading.judge_runs(dom, task)]
            frames.append(calibration.forecasts(dom, task, protocols, runs))
            print(name, task.semester, "forecasts done", flush=True)
    return pd.concat(frames, ignore_index=True)


def likelihood_ratio_table(forecasts: pd.DataFrame, domains: list[str], semesters: list[str],
                           variants: dict[str, str]) -> pd.DataFrame:
    """
    Computes each target's log likelihood ratio against the Baseline for Deterministic features, Embedding and every
    judge's scored run, on the full reference set and with each reference left out in turn, then checks that the
    full-reference predictions equal those in `forecasts` (each task's selected judge, the transcript protocols as
    they are).

    Inputs:
        forecasts: forecasts_table's output; per task and judge protocol, each judge's run closest to reasoning shown
            on the full train set, in the reported variant, is the one scored.
        domains: domain names (configs/domain/<name>.yml).
        semesters: the target semesters; each domain is scored on those of them it has.
        variants: protocol -> reported variant (config.yaml protocols); sets the judge protocols' conversion and the
            embedding variant.
    Outputs:
        DataFrame with columns domain, semester, model, protocol, judge, shape, dropped, llr; one row per target,
        protocol, judge ("-" for glm_regex and embedding), reference set (dropped: "-" for all references, else the
        one left out) and shape ("normal" or "t"). Raises AssertionError when the check fails.
    """
    core = stats.core_condition(forecasts)
    core = core[np.array([variants.get(p) == v for p, v in zip(core.protocol, core.variant)])]
    runs_used = core[core.protocol.isin(stats.JUDGED)][["domain", "semester", "protocol", "judge", "condition"]].drop_duplicates()
    frames, check = [], []
    for name in domains:
        dom = loading.domain(name)
        features = {"glm_regex": PROTOCOLS["glm_regex"], "embedding": EmbeddingRegression(variants["embedding"])}
        for task in loading.tasks(dom, [s for s in semesters if s in semester_names(dom)]):
            refs, truth = task.references.theta, task.truth
            rules = {(p, "-"): lr.feature_rule(f.estimator, pd.concat([f.get_features(task.references), f.get_features(task.test)]))
                     for p, f in features.items()}
            runs = [matching.match(run, task) for run in loading.judge_runs(dom, task)]
            used = runs_used[(runs_used.domain == name) & (runs_used.semester == task.semester)]
            for protocol, judge, condition in used[["protocol", "judge", "condition"]].itertuples(index=False):
                run = next(r for r in runs if (r.protocol, r.judge, r.condition) == (protocol, judge, condition))
                rules[(protocol, judge)] = lr.judge_rule(run.table, protocol, variants[protocol])
            for (protocol, judge), rule in rules.items():
                frames.append(lr.task_llr(rule, refs, truth).assign(domain=name, semester=task.semester, protocol=protocol, judge=judge))
                check.append(pd.DataFrame(dict(domain=name, semester=task.semester, protocol=protocol, judge=judge, model=truth.index,
                                               theta_hat=rule(refs, list(truth.index)).reindex(truth.index).to_numpy())))
            print(name, task.semester, "likelihood ratios done", flush=True)
    # the full-reference predictions must equal forecasts_table's (each task's best judge; other protocols as they are)
    ours = lr.select(pd.concat(check, ignore_index=True).assign(dropped="-"), lr.chosen_judges(forecasts, variants))
    theirs = stats.in_variants(stats.one_per_target(forecasts), variants)
    merged = ours.merge(theirs[["domain", "semester", "protocol", "model", "theta_hat"]],
                        on=["domain", "semester", "protocol", "model"], suffixes=("", "_forecasts"))
    gap = (merged.theta_hat - merged.theta_hat_forecasts).abs()
    assert len(merged) == len(ours) and gap.max() < 1e-3, merged[gap >= 1e-3]
    return pd.concat(frames, ignore_index=True)[["domain", "semester", "model", "protocol", "judge", "shape", "dropped", "llr"]]


def main(config: dict) -> None:
    """
    Builds both tables and writes them.

    Inputs:
        config: the analysis configuration (scripts/analysis/config.yaml); uses paths, tasks.domains,
            tasks.all_semesters and protocols.
    Outputs:
        <paths.forecasts>/forecasts.csv and <paths.forecasts>/likelihood_ratio.csv.
    """
    apply_paths(config)
    out = Path(config["paths"]["forecasts"])
    out.mkdir(parents=True, exist_ok=True)
    domains, semesters = config["tasks"]["domains"], config["tasks"]["all_semesters"]
    forecasts = forecasts_table(domains, semesters)
    forecasts.to_csv(out / "forecasts.csv", index=False, float_format="%.4f")
    llr = likelihood_ratio_table(forecasts, domains, semesters, config["protocols"])
    llr.to_csv(out / "likelihood_ratio.csv", index=False, float_format="%.5f")
    print("wrote", out / "forecasts.csv", out / "likelihood_ratio.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    main(load_config(parser.parse_args().config))
