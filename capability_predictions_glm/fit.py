"""Frequentist direct and antisymmetric pairwise capability GLMs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import AnalysisConfig


@dataclass
class DirectResult:
    target_quartile: str
    estimator: Ridge
    selected_features: tuple[str, ...]
    row_predictions: pd.DataFrame
    model_predictions: pd.DataFrame
    coefficients: pd.DataFrame
    tuning: pd.DataFrame
    diagnostics: dict[str, float]


@dataclass
class PairwiseResult:
    target_quartile: str
    estimator: LogisticRegression
    selected_features: tuple[str, ...]
    features: tuple[str, ...]
    benchmarks: tuple[str, ...]
    deviation_penalty: float
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    tuning: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: dict[str, float]


def model_group_splits(frame: pd.DataFrame, config: AnalysisConfig):
    """Inner folds that never put one model in fitting and validation rows."""
    groups = frame.model.to_numpy()
    if len(set(groups)) < config.inner_folds:
        raise ValueError("inner_folds exceeds the number of training models")
    return list(GroupKFold(config.inner_folds, shuffle=True, random_state=config.random_seed)
                .split(frame, groups=groups))


def _direct_levels(frame: pd.DataFrame) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    benches = tuple(sorted(frame.bench.unique()))
    items = tuple(map(tuple, frame[["bench", "item"]].drop_duplicates()
                      .sort_values(["bench", "item"]).to_numpy()))
    return benches, items


def _direct_design(
    frame: pd.DataFrame,
    features: list[str],
    levels: tuple[tuple[str, ...], tuple[tuple[str, str], ...]],
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    benches, items = levels
    n = len(frame)
    blocks, records = [], []
    for bench in benches:
        active = (frame.bench.to_numpy() == bench).astype(float)
        blocks.append(sparse.csr_matrix(active[:, None]))
        records.append({"term": "benchmark_intercept", "benchmark": bench, "item": None, "feature": None})
    for bench in benches:
        active = frame.bench.to_numpy() == bench
        for feature in features:
            values = frame[feature].to_numpy(float) * active
            blocks.append(sparse.csr_matrix(values[:, None]))
            records.append({"term": "benchmark_slope", "benchmark": bench, "item": None, "feature": feature})
    for bench, item in items:
        active = ((frame.bench.to_numpy() == bench) & (frame.item.to_numpy() == item)).astype(float)
        blocks.append(sparse.csr_matrix(active[:, None], shape=(n, 1)))
        records.append({"term": "item_intercept", "benchmark": bench, "item": item, "feature": None})
    return sparse.hstack(blocks, format="csr"), pd.DataFrame(records)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["bench", "model"]).model.transform("size").to_numpy(float)
    return 1 / counts


def _aggregate_direct(frame: pd.DataFrame, predicted_logit: np.ndarray) -> pd.DataFrame:
    rows = frame[["bench", "model", "target_accuracy"]].copy()
    rows["predicted_logit"] = predicted_logit
    out = (rows.groupby(["bench", "model"], as_index=False)
           .agg(observed_accuracy=("target_accuracy", "mean"),
                predicted_logit=("predicted_logit", "mean")))
    out["predicted_accuracy"] = expit(out.predicted_logit)
    return out


def _selected_features(
    coefficients: np.ndarray,
    column_map: pd.DataFrame,
    features: list[str],
) -> set[str]:
    active = column_map.assign(coefficient=np.asarray(coefficients).ravel())
    return set(active.loc[(active.feature.notna()) & (active.coefficient != 0), "feature"]) & set(features)


def _select_direct(
    train: pd.DataFrame,
    design: sparse.csr_matrix,
    column_map: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    config: AnalysisConfig,
) -> tuple[list[str], pd.DataFrame, float, float]:
    splits = model_group_splits(train, config)
    rows, best, best_loss = [], None, np.inf
    for alpha in config.direct_elastic_alphas:
        for l1_ratio in config.elastic_l1_ratios:
            losses = []
            for fit_rows, check_rows in splits:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                                   max_iter=config.max_iter, random_state=config.random_seed).fit(
                    design[fit_rows], y.iloc[fit_rows], sample_weight=_weights(train.iloc[fit_rows]))
                checked = _aggregate_direct(train.iloc[check_rows], model.predict(design[check_rows]))
                losses.append(mean_squared_error(
                    logit(checked.observed_accuracy.clip(config.target_clip, 1 - config.target_clip)),
                    checked.predicted_logit))
            loss = float(np.mean(losses))
            rows.append({"stage": "elastic_net", "record_type": "tuning", "alpha": alpha,
                         "l1_ratio": l1_ratio, "mean_model_logit_mse": loss})
            if loss < best_loss:
                best, best_loss = (alpha, l1_ratio), loss
    assert best is not None
    counts = dict.fromkeys(features, 0)
    for fit_rows, _ in splits:
        model = ElasticNet(alpha=best[0], l1_ratio=best[1], fit_intercept=False,
                           max_iter=config.max_iter, random_state=config.random_seed).fit(
            design[fit_rows], y.iloc[fit_rows], sample_weight=_weights(train.iloc[fit_rows]))
        for feature in _selected_features(model.coef_, column_map, features):
            counts[feature] += 1
    selected = [feature for feature in features
                if counts[feature] / len(splits) >= config.selection_min_fold_share]
    selected = selected or list(features)
    rows.extend({"stage": "elastic_net", "record_type": "stability", "alpha": best[0],
                 "l1_ratio": best[1], "feature": feature,
                 "fold_share": counts[feature] / len(splits), "retained": feature in selected}
                for feature in features)
    tuning = pd.DataFrame(rows)
    tuning["selected"] = ((tuning.record_type == "tuning") & (tuning.alpha == best[0])
                          & (tuning.l1_ratio == best[1]))
    return selected, tuning, best[0], best[1]


def fit_direct(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_quartile: str,
    config: AnalysisConfig,
) -> DirectResult:
    """Fit Q1 transcript features to one target-quartile model accuracy."""
    required = {"bench", "item", "model", "target_accuracy", *features}
    if missing := required - set(train) | required - set(test):
        raise ValueError(f"direct tables missing columns: {sorted(missing)}")
    levels = _direct_levels(train)
    full_x, full_map = _direct_design(train, features, levels)
    y = logit(train.target_accuracy.clip(config.target_clip, 1 - config.target_clip))
    selected, screen_tuning, elastic_alpha, elastic_l1 = _select_direct(
        train, full_x, full_map, y, features, config)
    train_x, column_map = _direct_design(train, selected, levels)
    tuning = []
    best_alpha, best_loss = None, np.inf
    for alpha in config.ridge_alphas:
        losses = []
        for fit_rows, check_rows in model_group_splits(train, config):
            estimator = Ridge(alpha=alpha, fit_intercept=False).fit(
                train_x[fit_rows], y.iloc[fit_rows], sample_weight=_weights(train.iloc[fit_rows]))
            checked = _aggregate_direct(train.iloc[check_rows], estimator.predict(train_x[check_rows]))
            losses.append(mean_squared_error(
                logit(checked.observed_accuracy.clip(config.target_clip, 1 - config.target_clip)),
                checked.predicted_logit))
        loss = float(np.mean(losses))
        tuning.append({"stage": "ridge", "record_type": "tuning", "alpha": alpha,
                       "mean_model_logit_mse": loss})
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    assert best_alpha is not None
    estimator = Ridge(alpha=best_alpha, fit_intercept=False).fit(
        train_x, y, sample_weight=_weights(train))
    test_x, _ = _direct_design(test, selected, levels)
    predicted_logit = estimator.predict(test_x)
    row_predictions = test[["bench", "item", "model", "target_accuracy"]].copy()
    row_predictions["predicted_logit"] = predicted_logit
    model_predictions = _aggregate_direct(test, predicted_logit).sort_values(["bench", "model"]).reset_index(drop=True)
    coefficients = column_map.assign(coefficient=np.asarray(estimator.coef_).ravel())
    tuning_table = pd.concat([screen_tuning, pd.DataFrame(tuning)], ignore_index=True, sort=False)
    tuning_table["selected"] = tuning_table.alpha == best_alpha
    tuning_table.loc[tuning_table.stage == "elastic_net", "selected"] = screen_tuning.selected.to_numpy()
    diagnostics = {"elastic_net_alpha": elastic_alpha, "elastic_net_l1_ratio": elastic_l1,
                   "ridge_alpha": best_alpha}
    return DirectResult(target_quartile, estimator, tuple(selected), row_predictions,
                        model_predictions, coefficients, tuning_table, diagnostics)


def fit_direct_fixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_quartile: str,
    config: AnalysisConfig,
    *,
    ridge_alpha: float,
) -> DirectResult:
    """Refit a direct specification with a frozen full-model ridge penalty."""
    levels = _direct_levels(train)
    train_x, column_map = _direct_design(train, features, levels)
    y = logit(train.target_accuracy.clip(config.target_clip, 1 - config.target_clip))
    estimator = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(
        train_x, y, sample_weight=_weights(train))
    test_x, _ = _direct_design(test, features, levels)
    predicted_logit = estimator.predict(test_x)
    row_predictions = test[["bench", "item", "model", "target_accuracy"]].copy()
    row_predictions["predicted_logit"] = predicted_logit
    model_predictions = _aggregate_direct(test, predicted_logit).sort_values(
        ["bench", "model"]).reset_index(drop=True)
    coefficients = column_map.assign(coefficient=np.asarray(estimator.coef_).ravel())
    return DirectResult(target_quartile, estimator, tuple(features), row_predictions,
                        model_predictions, coefficients, pd.DataFrame(),
                        {"ridge_alpha": ridge_alpha})


def _pair_levels(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(frame.bench.unique()))


def _pair_design(
    frame: pd.DataFrame,
    features: list[str],
    benches: tuple[str, ...],
    deviation_penalty: float,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    x = frame[[f"d_{feature}" for feature in features]].to_numpy(float)
    blocks = [sparse.csr_matrix(x)]
    records = [{"term": "population_slope", "benchmark": None, "feature": feature,
                "multiplier": 1.0} for feature in features]
    multiplier = 1 / np.sqrt(deviation_penalty)
    for bench in benches:
        active = frame.bench.to_numpy() == bench
        blocks.append(sparse.csr_matrix(x * active[:, None] * multiplier))
        records.extend({"term": "benchmark_deviation", "benchmark": bench, "feature": feature,
                        "multiplier": multiplier} for feature in features)
    return sparse.hstack(blocks, format="csr"), pd.DataFrame(records)


def _pair_splits(frame: pd.DataFrame, config: AnalysisConfig):
    groups = pd.MultiIndex.from_frame(frame[["bench", "item"]]).factorize()[0]
    if len(set(groups)) < config.inner_folds:
        raise ValueError("inner_folds exceeds the number of training items")
    return list(GroupKFold(config.inner_folds, shuffle=True, random_state=config.random_seed)
                .split(frame, groups=groups))


def _logistic(config: AnalysisConfig, c: float) -> LogisticRegression:
    return LogisticRegression(C=c, fit_intercept=False, solver="lbfgs",
                              max_iter=config.max_iter, random_state=config.random_seed)


def _elastic_logistic(config: AnalysisConfig, c: float, l1_ratio: float) -> LogisticRegression:
    return LogisticRegression(
        C=c, solver="saga", l1_ratio=l1_ratio,
        fit_intercept=False, max_iter=config.max_iter, random_state=config.random_seed,
    )


def _select_pairwise(
    train: pd.DataFrame,
    design: sparse.csr_matrix,
    column_map: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    config: AnalysisConfig,
) -> tuple[list[str], pd.DataFrame, float, float]:
    splits = _pair_splits(train, config)
    rows, best, best_loss = [], None, np.inf
    for c in config.pair_elastic_cs:
        for l1_ratio in config.elastic_l1_ratios:
            losses = []
            for fit_rows, check_rows in splits:
                model = _elastic_logistic(config, c, l1_ratio).fit(design[fit_rows], y[fit_rows])
                probability = model.predict_proba(design[check_rows])[:, 1]
                losses.append(log_loss(y[check_rows], probability, labels=[0, 1]))
            loss = float(np.mean(losses))
            rows.append({"stage": "elastic_net", "record_type": "tuning", "C": c,
                         "l1_ratio": l1_ratio, "mean_log_loss": loss})
            if loss < best_loss:
                best, best_loss = (c, l1_ratio), loss
    assert best is not None
    counts = dict.fromkeys(features, 0)
    for fit_rows, _ in splits:
        model = _elastic_logistic(config, best[0], best[1]).fit(design[fit_rows], y[fit_rows])
        for feature in _selected_features(model.coef_[0], column_map, features):
            counts[feature] += 1
    selected = [feature for feature in features
                if counts[feature] / len(splits) >= config.selection_min_fold_share]
    selected = selected or list(features)
    rows.extend({"stage": "elastic_net", "record_type": "stability", "C": best[0],
                 "l1_ratio": best[1], "feature": feature,
                 "fold_share": counts[feature] / len(splits), "retained": feature in selected}
                for feature in features)
    tuning = pd.DataFrame(rows)
    tuning["selected"] = ((tuning.record_type == "tuning") & (tuning.C == best[0])
                          & (tuning.l1_ratio == best[1]))
    return selected, tuning, best[0], best[1]


def fit_pairwise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_quartile: str,
    config: AnalysisConfig,
) -> PairwiseResult:
    """Fit an antisymmetric Q1 transcript-pair model for one target quartile."""
    feature_columns = {f"d_{feature}" for feature in features}
    required = {"bench", "item", "model_left", "model_right", "left_stronger", *feature_columns}
    if missing := required - set(train) | required - set(test):
        raise ValueError(f"pair tables missing columns: {sorted(missing)}")
    if not set(train.left_stronger.unique()) <= {0, 1} or train.left_stronger.nunique() < 2:
        raise ValueError("left_stronger must contain both binary outcomes")
    benches = _pair_levels(train)
    full_x, full_map = _pair_design(
        train, features, benches, config.benchmark_deviation_penalty)
    y = train.left_stronger.to_numpy(int)
    selected, screen_tuning, elastic_c, elastic_l1 = _select_pairwise(
        train, full_x, full_map, y, features, config)
    train_x, column_map = _pair_design(
        train, selected, benches, config.benchmark_deviation_penalty)
    tuning = []
    best_c, best_loss = None, np.inf
    for c in config.logistic_cs:
        losses = []
        for fit_rows, check_rows in _pair_splits(train, config):
            estimator = _logistic(config, c).fit(train_x[fit_rows], y[fit_rows])
            probability = estimator.predict_proba(train_x[check_rows])[:, 1]
            losses.append(log_loss(y[check_rows], probability, labels=[0, 1]))
        loss = float(np.mean(losses))
        tuning.append({"stage": "ridge", "record_type": "tuning", "C": c,
                       "mean_log_loss": loss})
        if loss < best_loss:
            best_c, best_loss = c, loss
    assert best_c is not None
    estimator = _logistic(config, best_c).fit(train_x, y)
    test_x, _ = _pair_design(test, selected, benches, config.benchmark_deviation_penalty)
    probability = estimator.predict_proba(test_x)[:, 1]
    observed = test.left_stronger.to_numpy(int)
    predictions = test[["bench", "item", "model_left", "model_right", "left_stronger"]].copy()
    predictions = predictions.rename(columns={"left_stronger": "observed"})
    predictions["probability"] = probability
    auc = roc_auc_score(observed, probability) if len(set(observed)) == 2 else np.nan
    metrics = pd.DataFrame({
        "metric": ["accuracy", "log_loss", "auc"],
        "value": [accuracy_score(observed, probability >= 0.5),
                  log_loss(observed, probability, labels=[0, 1]), auc],
    })
    natural = np.asarray(estimator.coef_).ravel() * column_map.multiplier.to_numpy()
    coefficients = column_map.assign(coefficient=np.asarray(estimator.coef_).ravel(), natural=natural)
    population = coefficients[coefficients.term == "population_slope"].set_index("feature").natural
    coefficients["effective"] = coefficients.natural
    deviations = coefficients.term == "benchmark_deviation"
    coefficients.loc[deviations, "effective"] += coefficients.loc[deviations, "feature"].map(population)
    tuning_table = pd.concat([screen_tuning, pd.DataFrame(tuning)], ignore_index=True, sort=False)
    tuning_table["selected"] = tuning_table.C == best_c
    tuning_table.loc[tuning_table.stage == "elastic_net", "selected"] = screen_tuning.selected.to_numpy()
    diagnostics = {"elastic_net_C": elastic_c, "elastic_net_l1_ratio": elastic_l1,
                   "ridge_C": best_c}
    return PairwiseResult(target_quartile, estimator, tuple(selected), tuple(selected), benches,
                          config.benchmark_deviation_penalty, predictions,
                          coefficients, tuning_table, metrics, diagnostics)


def fit_pairwise_fixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_quartile: str,
    config: AnalysisConfig,
    *,
    ridge_c: float,
) -> PairwiseResult:
    """Refit a pairwise specification with a frozen full-model ridge penalty."""
    benches = _pair_levels(train)
    train_x, column_map = _pair_design(
        train, features, benches, config.benchmark_deviation_penalty)
    y = train.left_stronger.to_numpy(int)
    estimator = _logistic(config, ridge_c).fit(train_x, y)
    test_x, _ = _pair_design(test, features, benches, config.benchmark_deviation_penalty)
    probability = estimator.predict_proba(test_x)[:, 1]
    observed = test.left_stronger.to_numpy(int)
    predictions = test[["bench", "item", "model_left", "model_right", "left_stronger"]].copy()
    predictions = predictions.rename(columns={"left_stronger": "observed"})
    predictions["probability"] = probability
    auc = roc_auc_score(observed, probability) if len(set(observed)) == 2 else np.nan
    metrics = pd.DataFrame({
        "metric": ["accuracy", "log_loss", "auc"],
        "value": [accuracy_score(observed, probability >= 0.5),
                  log_loss(observed, probability, labels=[0, 1]), auc],
    })
    raw = np.asarray(estimator.coef_).ravel()
    coefficients = column_map.assign(coefficient=raw,
                                     natural=raw * column_map.multiplier.to_numpy())
    population = coefficients[coefficients.term == "population_slope"].set_index("feature").natural
    coefficients["effective"] = coefficients.natural
    deviations = coefficients.term == "benchmark_deviation"
    coefficients.loc[deviations, "effective"] += coefficients.loc[deviations, "feature"].map(population)
    return PairwiseResult(target_quartile, estimator, tuple(features), tuple(features), benches,
                          config.benchmark_deviation_penalty, predictions, coefficients,
                          pd.DataFrame(), metrics, {"ridge_C": ridge_c})


def predict_pairwise(result: PairwiseResult, frame: pd.DataFrame) -> np.ndarray:
    """Predict any ordered pair rows with an already-fitted antisymmetric model."""
    design, _ = _pair_design(frame, list(result.features), result.benchmarks,
                             result.deviation_penalty)
    return result.estimator.predict_proba(design)[:, 1]
