"""The two tables every analysis group reads: each protocol's forecast of each target, and each target's log
likelihood ratio against the baseline.

Writes to paths.forecasts (config.yaml):
    forecasts.csv          one row per (domain, semester, protocol, variant, judge, condition, target): theta_hat, the
                           target's measured theta, and for judge protocols the judge's reference leave-one-model-out
                           error (judge_loo_mse). Every task of tasks.all_semesters, every judge run on disk, every variant.
    likelihood_ratio.csv   one row per (domain, semester, target, protocol, judge, dropped reference, shape): the log
                           likelihood ratio of the target's measured theta under the protocol's predictive distribution
                           against the baseline's. dropped "-" is the full reference set; judge "-" a transcript protocol.

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
from protocols import PROTOCOLS
from protocols.embeddings import EmbeddingRegression


def forecasts_table(domains: list[str], semesters: list[str]) -> pd.DataFrame:
    """Every protocol's forecast of every target, for every judge run and variant.

    domains: domain names (configs/domain/<name>.yml).
    semesters: the target semesters to forecast.
    """
    frames = []
    for name in domains:
        dom = loading.domain(name)
        protocols = calibration.transcript_protocols(dom)
        for task in loading.tasks(dom, semesters):
            runs = [matching.match(run, task) for run in loading.judge_runs(dom, task)]
            frames.append(calibration.forecasts(dom, task, protocols, runs))
            print(name, task.semester, "forecasts done", flush=True)
    return pd.concat(frames, ignore_index=True)


def likelihood_ratio_table(forecasts: pd.DataFrame, domains: list[str], semesters: list[str],
                           variants: dict[str, str]) -> pd.DataFrame:
    """Each target's log likelihood ratio per protocol against the baseline, with each reference left out in turn.

    forecasts: forecasts_table's output; its core-condition judge runs are the ones scored.
    variants: protocol -> reported variant; the judge protocols' entries pick the variant scored.
    """
    core = stats.core_condition(forecasts)
    core = core[np.array([variants.get(p) == v for p, v in zip(core.protocol, core.variant)])]
    runs_used = core[core.protocol.isin(stats.JUDGED)][["domain", "semester", "protocol", "judge", "condition"]].drop_duplicates()
    frames, check = [], []
    for name in domains:
        dom = loading.domain(name)
        features = {"glm_regex": PROTOCOLS["glm_regex"], "embedding": EmbeddingRegression(variants["embedding"])}
        for task in loading.tasks(dom, semesters):
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
    # the full-reference predictions must be the forecasts' own (each task's best judge, transcript protocols as they are)
    ours = lr.select(pd.concat(check, ignore_index=True).assign(dropped="-"), lr.chosen_judges(forecasts, variants))
    theirs = stats.in_variants(stats.one_per_target(forecasts), variants)
    merged = ours.merge(theirs[["domain", "semester", "protocol", "model", "theta_hat"]],
                        on=["domain", "semester", "protocol", "model"], suffixes=("", "_forecasts"))
    gap = (merged.theta_hat - merged.theta_hat_forecasts).abs()
    assert len(merged) == len(ours) and gap.max() < 1e-3, merged[gap >= 1e-3]
    return pd.concat(frames, ignore_index=True)[["domain", "semester", "model", "protocol", "judge", "shape", "dropped", "llr"]]


def main(config: dict) -> None:
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
