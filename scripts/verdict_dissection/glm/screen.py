"""Fast, non-Bayesian version of the three GLMs for screening waves and held-out designs.

    uv run python scripts/verdict_dissection/glm/screen.py <inputs_dir> <fits_dir> [--models ...]

Same partial pooling as model.py, fitted as one ridge-penalised regression: population slopes
plus per-group deviations (benchmark, or judge for the verdict model) with the deviations
penalised harder, item intercepts penalised (zero for an unseen item), per-judge position
intercepts free. Linear for the capability model, logistic for the pair models. The two
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
        blocks.append(sp.csr_matrix((np.full(len(table), 30.0), (np.arange(len(table)), group)),
                                    shape=(len(table), len(group_levels))))   # large scale = effectively unpenalised
        if use_accuracy_gap:
            blocks.append(sp.csr_matrix(table.accuracy_difference.to_numpy(np.float32)[:, None] * 30.0))
    return sp.hstack(blocks).tocsr()


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
    return scores.loc[0, "spearman" if gaussian else "accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("fits_dir", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--waves", nargs="+", help="feature sets to fit (default: all in waves.py)")
    args = parser.parse_args()
    predictors = yaml.safe_load((HERE / "predictors.yaml").read_text())
    feature_sets = {name: kept for name, kept in waves(predictors).items() if not args.waves or name in args.waves}
    for model in args.models:
        for design_dir in sorted(p for p in args.inputs_dir.iterdir() if p.is_dir()):
            table = pd.read_parquet(design_dir / f"{model}.parquet")
            inner_group = "item" if design_dir.name.startswith("item-fold") else ("model" if model == "capability_from_transcript" else "model_left")
            for wave, kept in feature_sets.items():
                specs = {wave: False, f"{wave}+accuracy-gap": True} if model == "judge_verdict_from_pair" else {wave: False}
                for wave_name, use_gap in specs.items():
                    score = run(table, kept, model, use_gap, args.fits_dir / model / wave_name / design_dir.name, inner_group)
                    print(f"{model} {wave_name} {design_dir.name}: {score:.3f}", flush=True)


if __name__ == "__main__":
    main()
