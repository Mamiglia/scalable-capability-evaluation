"""Does an ensemble of judges beat a single judge? Pairwise verdicts already in logs/judge, no judging is run.

    uv run python scripts/analyses/judge_ensemble.py [--recompute]

Judges = the top-3 references of each semester (data/meta/<domain>/fit.yml `top`), subjects' reasoning shown
(logs/judge/<domain>/<sem>/<judge>/{24d6cc32,66c170cd}, first that exists), restricted to the train items ALL three
judged, so a single judge and an ensemble see the same items. For every non-empty subset of the judges the targets
are placed by three estimators (JudgeLinear `linear`, pinned BT `bt`, JudgeCalibrated `cal`) after combining the
judges in one of four ways:

    pool        the judges' verdicts concatenated: every judge is one more vote on the same pair
    vote        one verdict per (item, pair, shown order): the majority pick (a split stays 0.5)
    instrument  pooled, but every (judge, bench) is calibrated as its own instrument (same as pool for pinned BT)
    mean        the mean of the single-judge forecasts

Writes analysis/judge/judge_ensemble.csv (one forecast per task, subset, method, estimator, target) and prints
the MSE tables, the constants (mean_ref, mean_frontier, top reference) and why the ensemble does or does not help:
the judges' verdict agreement, their accuracy on reference pairs alone and as a majority, and the correlation of
their forecast errors over the targets.
"""

import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).parent))
from plots.judge_ensemble import figure  # noqa: E402
from plots.palette import DOMAIN  # noqa: E402

from capability.task import Task
from configs.domain import DomainConfig
from configs.logs import logger
from protocols.pairwise import from_logs
from protocols.pairwise.bradley_terry import rank
from protocols.pairwise.calibrated import JudgeCalibrated, estimate_all, loo, shrink
from protocols.pairwise.linear import weighted_theta
from reference.freeze import TOP_K

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/judge_ensemble")
DOMAINS = ["science", "math", "coding", "cyber"]
CONDITIONS = ["24d6cc32", "66c170cd"]
KEY = ["bench", "item", "model_left", "model_right"]


def estimators(v: pd.DataFrame, refs: pd.Series, targets: list[str], prior: tuple[float, float]) -> dict[str, pd.Series]:
    sigma2 = float(((loo(v, refs) - refs) ** 2).mean())
    return {"linear": weighted_theta(v, refs, targets),
            "bt": rank(v, anchors=refs).set_index("model").ability.reindex(targets).fillna(refs.mean()),
            "cal": shrink(estimate_all(v, refs, targets), sigma2, prior)}


def combined(v: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "vote":  # np.round(0.5) = 0, so a split is kept at 0.5 by hand
        mean = v.groupby(KEY, as_index=False).pick_left.mean()
        return mean.assign(judge="vote", pick_left=np.where(mean.pick_left == 0.5, 0.5, mean.pick_left.round()))
    if how == "instrument":
        return v.assign(bench=v.judge + "|" + v.bench)
    return v  # pool


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts, verdicts = [], []
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            task = Task.from_semester(sem, domain)
            refs, targets = task.references.theta, list(task.test.models.index)
            prior = JudgeCalibrated.frontier_prior(SimpleNamespace(domain=domain), sem)
            per_judge = {}
            for judge in (j.lower() for j in judges):
                run = next((p for c in CONDITIONS if (p := Path("logs/judge", name, sem, judge, c)).exists()), None)
                if run is None:
                    print("missing", name, sem, judge, file=sys.stderr)
                    continue
                v = from_logs(run).sort_values("imported", kind="stable").drop_duplicates(KEY)
                per_judge[judge] = v.merge(task.references.items[["bench", "item"]]).assign(judge=judge)
            common = set.intersection(*(set(zip(v.bench, v.item)) for v in per_judge.values()))
            per_judge = {j: v[[k in common for k in zip(v.bench, v.item)]] for j, v in per_judge.items()}
            print(name, sem, list(per_judge), len(common), "items", file=sys.stderr)
            verdicts.append(pd.concat(per_judge.values()).assign(domain=name, semester=sem))

            def rows(subset, how, preds):
                return [dict(domain=name, semester=sem, judges="|".join(subset), k=len(subset), method=how, estimator=e,
                             model=m, forecast=f, truth=task.truth.get(m)) for e, p in preds.items() for m, f in p.items()]

            single = {j: estimators(v, refs, targets, prior) for j, v in per_judge.items()}
            for j, preds in single.items():
                forecasts += rows([j], "single", preds)
            for k in range(2, len(per_judge) + 1):
                for subset in combinations(per_judge, k):
                    pooled = pd.concat([per_judge[j] for j in subset])
                    for how in ["pool", "vote", "instrument"]:
                        forecasts += rows(subset, how, estimators(combined(pooled, how), refs, targets, prior))
                    forecasts += rows(subset, "mean", {e: pd.concat([single[j][e] for j in subset], axis=1).mean(axis=1) for e in single[subset[0]]})
            constants = {"mean_ref": refs.mean(), "mean_frontier": refs.nlargest(TOP_K).mean(), "top_ref": refs.max()}
            forecasts += rows([], "constant", {c: pd.Series(x, index=targets) for c, x in constants.items()})
            # the refs' theta*, to score the reference-pair verdicts below
            verdicts[-1] = verdicts[-1].assign(theta_left=verdicts[-1].model_left.map(refs), theta_right=verdicts[-1].model_right.map(refs))
    return pd.DataFrame(forecasts), pd.concat(verdicts)


def diagnostics(v: pd.DataFrame, f: pd.DataFrame) -> None:
    cells = ["domain", "semester"]
    out = []
    for cell, rows in v.groupby(cells):
        g = rows.set_index([*KEY, "judge"]).pick_left.unstack("judge").dropna()  # verdicts every judge of the task gave
        agree = np.mean([(g[a] == g[b]).mean() for a, b in combinations(g.columns, 2)])
        theta = rows.dropna(subset=["theta_left"]).drop_duplicates("model_left").set_index("model_left").theta_left
        left, right_ = g.index.get_level_values("model_left").map(theta), g.index.get_level_values("model_right").map(theta)
        picks = g[np.abs(left - right_) > 0.25]  # reference pairs only (NaN otherwise); below 0.25 no judge resolves the gap
        right = (left > right_)[np.abs(left - right_) > 0.25].astype(float)
        err = f[(f.domain == cell[0]) & (f.semester == cell[1]) & (f.method == "single") & (f.estimator == "linear")]
        err = err.assign(e=err.forecast - err.truth).pivot(index="model", columns="judges", values="e")
        corr = err.corr().to_numpy()[np.triu_indices(err.shape[1], 1)].mean()
        out.append(dict(zip(cells, cell), agreement=agree, ref_acc_single=picks.eq(right, axis=0).mean().mean(),
                        ref_acc_majority=(picks.mean(axis=1).round() == right).mean(), error_corr=corr, bias_share=(err.mean() ** 2).mean() / (err ** 2).mean().mean()))
    print("\nwhy: pairwise verdict agreement between judges; accuracy on reference pairs with |dtheta*| > 0.25, one judge vs"
          "\nthe majority; correlation of the judges' forecast errors over the targets (linear); share of the squared error"
          "\nthat is the cohort-level bias shared by all targets (an ensemble of judges with the same bias cannot remove it)")
    print(pd.DataFrame(out).set_index(cells).round(2).to_string())


def compare(mse: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """JudgeLinear: the ensemble = mean of the three judges' forecasts (the best and cheapest way to combine them)
    against the judge in use today, the top-1 reference, on the same items. Prints both per task and the paired
    difference (N = 14 tasks; the mean and its CI). Returns (one row per task: squared error and score of the top-1
    judge and of the ensemble, x position; the task's mean_ref MSE), which plots.judge_ensemble.figure draws."""
    cells = ["domain", "semester"]
    lin = mse[mse.estimator == "linear"].merge(mse[mse.estimator == "mean_ref"][[*cells, "se"]].rename(columns={"se": "baseline"}))
    lin = lin.assign(score=(100 - 100 * lin.se / lin.baseline).clip(lower=0))
    ens = lin[(lin.method == "mean") & (lin.k == 3)]
    top1 = lin[lin.method == "single"].merge(ens.assign(judges=ens.judges.str.split("|").str[0])[[*cells, "judges"]])  # fit.yml order: top-1 first
    tasks = ens[cells].assign(order=ens.domain.map({n.lower(): i for i, n in enumerate(DOMAIN)})).sort_values(["order", "semester"])
    tasks["x"] = np.arange(len(tasks)) + tasks.order * 0.6  # a gap between domains
    both = tasks.merge(top1[[*cells, "judges", "se", "score"]]).merge(ens[[*cells, "se", "score"]], on=cells, suffixes=("_top1", "_ens"))
    delta = both.score_ens - both.score_top1
    half = t.ppf(0.975, len(delta) - 1) * delta.sem()
    print("\nJudgeLinear, top-1 judge vs the 3-judge mean of forecasts (theta MSE and prediction score, same items)")
    print(both.drop(columns=["order", "x"]).assign(delta=delta).round(2).to_string(index=False))
    print(f"MSE mean {both.se_top1.mean():.2f} -> {both.se_ens.mean():.2f}, median {both.se_top1.median():.2f} -> {both.se_ens.median():.2f}; "
          f"score {delta.mean():+.1f} +- {half:.1f} points (legacy) (t 95% CI, N = {len(delta)} tasks), ensemble better in {(delta > 0).sum()}/{len(delta)}")
    return both, both.merge(lin[lin.method == "single"].drop_duplicates(cells)[[*cells, "baseline"]]).baseline


def main() -> int:
    csv, vcsv = OUT.with_suffix(".csv"), OUT.with_name(OUT.name + "_verdicts.csv.gz")
    if "--recompute" in sys.argv or not csv.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        f, v = compute()
        f.to_csv(csv, index=False, float_format="%.4f")
        v.to_csv(vcsv, index=False)
    f, v = pd.read_csv(csv, keep_default_na=False, na_values=[""]), pd.read_csv(vcsv, dtype={"item": str})
    f = f.dropna(subset=["truth"])
    cells = ["domain", "semester"]
    mse = f.assign(se=(f.forecast - f.truth) ** 2).groupby([*cells, "judges", "k", "method", "estimator"], dropna=False).se.mean().reset_index()

    const = mse[mse.method == "constant"].pivot(index=cells, columns="estimator", values="se")
    for est in ["linear", "cal", "bt"]:
        e = mse[mse.estimator == est]
        single = e[e.method == "single"].groupby(cells).se
        table = pd.DataFrame({"single_best": single.min(), "single_mean": single.mean(), "single_worst": single.max()})
        for how in ["pool", "vote", "instrument", "mean"]:
            table[f"{how}_3"] = e[(e.method == how) & (e.k == e.k.max())].set_index(cells).se
        table = table.join(const)
        print(f"\n{est}: theta MSE per task (single = one judge; *_3 = all three judges)")
        print(pd.concat([table, table.mean().rename(("MEAN", "")).to_frame().T, table.median().rename(("MEDIAN", "")).to_frame().T]).round(2).to_string())
        # how often the 3-judge ensemble beats the judges it is made of
        wins = {how: f"beats mean single {int((table[f'{how}_3'] < table.single_mean).sum())}/{len(table)}, best single "
                     f"{int((table[f'{how}_3'] < table.single_best).sum())}/{len(table)}" for how in ["pool", "vote", "instrument", "mean"]}
        print(*[f"  {how}: {w}" for how, w in wins.items()], sep="\n")

    print("\nMSE by number of judges (mean over tasks of the mean over subsets; band = min..max over subsets, i.e. which judges are in)")
    by_k = mse[mse.method.isin(["single", "mean", "instrument"])].assign(method=lambda d: d.method.replace("single", "mean"))
    by_k = pd.concat([by_k, mse[mse.method == "single"].assign(method="instrument")])
    agg = by_k.groupby(["estimator", "method", *cells, "k"]).se.agg(["mean", "min", "max"]).groupby(["estimator", "method", "k"]).mean()
    print(agg.round(2).to_string())
    diagnostics(v, f)
    figure(*compare(mse), OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
