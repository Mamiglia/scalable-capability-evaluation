"""Fast, non-Bayesian version of the three GLMs for screening waves and held-out designs.

    uv run python scripts/verdict_dissection/glm/screen.py <inputs_dir> <fits_dir> [--models ...]

Same partial pooling as model.py, fitted as one ridge-penalised regression: population slopes
plus per-group deviations (benchmark, or judge for the verdict model) with the deviations
penalised harder, item intercepts penalised (zero for an unseen item), position intercepts per
judge x benchmark free. Linear for the capability model, logistic for the pair models. The two
penalties are chosen by grouped cross-validation on the training rows. Writes the same
held_out_/predictions_/placement_ files as evaluate.py under <fits_dir>/<model>/<wave>/<design>/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import model_level, placement, report  # noqa: E402
from waves import MODELS, waves  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUP_COLUMN = {"capability_from_transcript": "bench", "stronger_model_from_pair": "bench",
                "judge_verdict_from_pair": "judge"}
POPULATION_PENALTIES = [0.1, 1.0, 10.0]      # ridge strength on the population slopes
DEVIATION_RATIOS = [3.0, 10.0, 30.0]         # how much harder the per-group deviations are penalised
INNER_FOLDS = 3


def design(table, predictors, model, group_levels, item_levels, deviation_ratio, use_accuracy_gap):
    """Sparse design: [population slopes | group deviations | item intercepts | judge intercepts | gap]."""
    x = table[predictors].to_numpy(np.float32)
    group = pd.Categorical(table[GROUP_COLUMN[model]], categories=group_levels).codes
    blocks = [sp.csr_matrix(x)]
    deviations = sp.hstack([sp.csr_matrix(x * (group == g)[:, None]) for g in range(len(group_levels))])
    blocks.append(deviations / np.sqrt(deviation_ratio))          # scaling a column raises its effective penalty
    if model == "capability_from_transcript":
        item = pd.Categorical(table.item, categories=item_levels).codes
        blocks.append(sp.csr_matrix((np.ones(len(table)), (np.arange(len(table)), item)), shape=(len(table), len(item_levels))))
    if model == "judge_verdict_from_pair":
        # position bias per judge x benchmark (it ranges 0.39-0.84 across cells); large scale = effectively unpenalised
        bench = pd.Categorical(table.bench, categories=sorted(table.bench.unique())).codes
        cell = group * len(set(bench)) + bench
        blocks.append(sp.csr_matrix((np.full(len(table), 30.0), (np.arange(len(table)), cell)),
                                    shape=(len(table), len(group_levels) * len(set(bench)))))
        if use_accuracy_gap:
            blocks.append(sp.csr_matrix(table.accuracy_difference.to_numpy(np.float32)[:, None] * 30.0))
    return sp.hstack(blocks).tocsr()


def coefficients(estimator, predictors, group_levels, deviation_ratio):
    """Population slope and each group's slope (population + deviation) per predictor, on the predictors' scale."""
    coef = np.asarray(estimator.coef_).ravel()
    n = len(predictors)
    population = coef[:n]
    deviations = coef[n:n + n * len(group_levels)].reshape(len(group_levels), n) / np.sqrt(deviation_ratio)
    table = pd.DataFrame({"population": population}, index=pd.Index(predictors, name="feature"))
    for g, level in enumerate(group_levels):
        table[level] = population + deviations[g]
    return table


def fit(x, y, gaussian, penalty):
    if gaussian:
        return Ridge(alpha=penalty, fit_intercept=True).fit(x, y)
    return LogisticRegression(C=1 / penalty, fit_intercept=False, max_iter=2000).fit(x, y)


def predict(estimator, x, gaussian):
    return estimator.predict(x) if gaussian else estimator.predict_proba(x)[:, 1]


def choose_penalties(table, predictors, model, levels, use_accuracy_gap, gaussian, target, inner_groups):
    """Grouped inner CV on the training rows over the penalty grid; returns (population penalty, deviation ratio)."""
    train = table[~table.test]
    y = train[target].to_numpy()
    best, best_score = None, -np.inf
    for ratio in DEVIATION_RATIOS:
        x = design(train, predictors, model, *levels, ratio, use_accuracy_gap)
        for penalty in POPULATION_PENALTIES:
            scores = []
            for fit_rows, score_rows in GroupKFold(INNER_FOLDS).split(x, y, inner_groups):
                predicted = predict(fit(x[fit_rows], y[fit_rows], gaussian, penalty), x[score_rows], gaussian)
                scores.append(-np.mean((predicted - y[score_rows]) ** 2) if gaussian
                              else np.mean((predicted > 0.5) == y[score_rows]))
            if np.mean(scores) > best_score:
                best, best_score = (penalty, ratio), np.mean(scores)
    return best


def run(table, predictors, model, use_accuracy_gap, out_dir, inner_group_column):
    gaussian = model == "capability_from_transcript"
    target = MODELS[model]["target"]
    levels = (sorted(table[GROUP_COLUMN[model]].unique()), sorted(table.item.unique()))
    train, test = table[~table.test], table[table.test]
    penalty, ratio = choose_penalties(table, predictors, model, levels, use_accuracy_gap, gaussian, target,
                                      train[inner_group_column].to_numpy())
    estimator = fit(design(train, predictors, model, *levels, ratio, use_accuracy_gap), train[target].to_numpy(), gaussian, penalty)
    predicted = predict(estimator, design(test, predictors, model, *levels, ratio, use_accuracy_gap), gaussian)
    observed = test[target].to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    group_column = GROUP_COLUMN[model]
    groups = {label: (test[group_column] == label).to_numpy() for label in levels[0]}
    scores = report(observed, predicted, groups, gaussian)
    identifiers = ["bench", "item", "model"] if gaussian else ["bench", "item", "model_left", "model_right"] + (
        ["judge"] if model == "judge_verdict_from_pair" else [])
    test[identifiers].assign(observed=observed, predicted=predicted).to_csv(out_dir / f"predictions_{model}.csv", index=False)
    if gaussian:
        rows = pd.concat([train, test])[["bench", "model", "test", target]].rename(columns={target: "observed"})
        scores = pd.concat([scores, model_level(observed, predicted, rows[rows.test])], ignore_index=True)
        if not set(test.model) <= set(train.model):
            train_predicted = predict(estimator, design(train, predictors, model, *levels, ratio, use_accuracy_gap), gaussian)
            placement(train_predicted, predicted, rows).to_csv(out_dir / f"placement_{model}.csv", index=False)
    scores.to_csv(out_dir / f"held_out_{model}.csv", index=False)
    pd.Series({"population_penalty": penalty, "deviation_ratio": ratio}).to_csv(out_dir / "penalties.csv")
    coefficients(estimator, predictors, levels[0], ratio).to_csv(out_dir / f"coefficients_{model}.csv")
    return scores.loc[0, "spearman" if gaussian else "accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("fits_dir", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--waves", nargs="+", help="feature sets to fit (default: all in waves.py)")
    parser.add_argument("--predictors", type=Path, default=HERE / "predictors.yaml", help="predictor list (predictors-dense.yaml for the embedding set)")
    parser.add_argument("--designs", nargs="+", help="held-out designs to fit (default: every directory in inputs_dir)")
    parser.add_argument("--drop-one", action="store_true", help="also fit each feature set with one predictor removed at a time")
    args = parser.parse_args()
    predictors = yaml.safe_load(args.predictors.read_text())
    feature_sets = {name: kept for name, kept in waves(predictors).items() if not args.waves or name in args.waves}
    for model in args.models:
        for design_dir in sorted(p for p in args.inputs_dir.iterdir() if p.is_dir() and (not args.designs or p.name in args.designs)):
            table = pd.read_parquet(design_dir / f"{model}.parquet")
            inner_group = "model" if model == "capability_from_transcript" else "model_left"   # inner CV groups by model
            for wave, kept in feature_sets.items():
                specs = {wave: False, f"{wave}+accuracy-gap": True} if model == "judge_verdict_from_pair" else {wave: False}
                variants = {name: (kept, gap) for name, gap in specs.items()}
                if args.drop_one:                                # the set minus one predictor, for every predictor
                    variants |= {f"{wave}-drop-{feature}": ([p for p in kept if p != feature], False) for feature in kept}
                for wave_name, (features, use_gap) in variants.items():
                    out_dir = args.fits_dir / model / wave_name / design_dir.name
                    if (out_dir / f"held_out_{model}.csv").exists():
                        continue                                # already fitted; delete the folder to redo
                    score = run(table, features, model, use_gap, out_dir, inner_group)
                    print(f"{model} {wave_name} {design_dir.name}: {score:.3f}", flush=True)


if __name__ == "__main__":
    main()
