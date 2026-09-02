"""Leakage-safe GPQA frames for bias-corrected cross-fitting."""

import numpy as np
import pandas as pd
from scipy.special import expit

from bias_model import POSTERIOR_KEYS, fit_calibration, infer_targets
from inputs import BIAS_FEATURES
from item_banks import irt_cells
from mc_cost.ranking import collect_pair_wins, components, fit_bt


PREDICTION_COVERAGE_COLUMNS = [
    "n_anchors", "n_calibration_models", "n_target_models", "n_judges",
    "n_sources", "n_easy_items", "n_comparison_edges", "n_rounds",
]
PREDICTION_COLUMNS = [
    "benchmark", "irt_model", "fold", "component", "method", "model",
    "predictor", "prediction", "prediction_p5", "prediction_p95", "observed",
    *PREDICTION_COVERAGE_COLUMNS,
]
COVERAGE_COLUMNS = [
    "benchmark", "irt_model", "fold", "component", "model", "n_anchors",
    "eligible", "reason", *PREDICTION_COVERAGE_COLUMNS[1:],
    "n_evidence_remaining", "n_evidence_excluded",
]
TARGET_EXCLUSION_COLUMNS = ["model", "reason", "count"]
CALIBRATION_COLUMNS = [
    "benchmark", "irt_model", "fold", "method", "judge", "source",
    "observed_rate", "predicted_rate", "brier", "log_loss",
    "interval_coverage_90", "n_rows",
]
COEFFICIENT_COLUMNS = [
    "benchmark", "irt_model", "fold", "parameter", "judge", "level",
    "feature", "mean", "p5", "p95",
]


def assign_exact_model_folds(models, n_splits=5, seed=0):
    """Assign every exact configuration ID to one deterministic fold."""
    models = np.array(sorted(set(models)), dtype=object)
    shuffled = np.random.default_rng(seed).permutation(models)
    return pd.DataFrame({
        "model": shuffled,
        "fold": np.arange(len(shuffled)) % n_splits,
    }).sort_values("model").reset_index(drop=True)


def _component_members(rows):
    """Map each compatible-source graph member to a deterministic component."""
    graph = components(rows[["left_model", "right_model"]].itertuples(index=False, name=None))
    graph = sorted(graph, key=lambda component: (-len(component), tuple(sorted(component))))
    return pd.DataFrame([
        {"model": model, "component": f"pooled:{index}"}
        for index, component in enumerate(graph)
        for model in sorted(component)
    ], columns=["model", "component"])


def _add_components(rows):
    members = _component_members(rows)
    result = rows.merge(
        members.rename(columns={"model": "left_model"}),
        on="left_model", validate="many_to_one",
    ).merge(
        members.rename(columns={"model": "right_model", "component": "right_component"}),
        on="right_model", validate="many_to_one",
    )
    if not result["component"].eq(result["right_component"]).all():
        raise ValueError("comparison endpoints must share a component")
    return result.drop(columns="right_component")


def anchor_coverage(rows, anchors):
    """Report pooled target components with at least two anchors."""
    anchor_models = set(anchors)
    members = _component_members(rows)
    result = []
    for component, group in members.groupby("component", sort=True):
        anchored = set(group["model"]) & anchor_models
        for model in sorted(set(group["model"]) - anchor_models):
            count = len(anchored)
            result.append({
                "model": model,
                "component": component,
                "n_anchors": count,
                "eligible": count >= 2,
                "reason": "" if count >= 2 else "fewer than 2 anchors",
            })
    return pd.DataFrame(result, columns=[
        "model", "component", "n_anchors", "eligible", "reason",
    ])


def build_fold_frames(
    votes, anchors, splits, folds, fold, irt_model,
    bin_column="half", bin_value="easy",
):
    """Return target-disjoint calibration rows and predictor-bin target rows."""
    targets = set(folds.loc[folds["fold"].eq(fold), "model"])
    if "benchmark" not in votes:
        raise ValueError("votes must include a benchmark column")
    votes = votes[votes["benchmark"].eq("gpqa")]
    split_scope = splits["benchmark"].eq("gpqa")
    if "irt_model" in splits:
        split_scope &= splits["irt_model"].eq(irt_model)
    if "fit" in splits:
        split_scope &= splits["fit"].eq("gpqa")
    gpqa_splits = splits[split_scope][["item", bin_column]]
    rows = votes.merge(
        gpqa_splits, left_on="item_id", right_on="item", how="inner", validate="many_to_one",
    ).drop(columns="item")
    calibration = rows[
        rows[bin_column].eq(bin_value)
        & ~rows["left_model"].isin(targets) & ~rows["right_model"].isin(targets)
    ].copy()
    anchor_scope = pd.Series(True, index=anchors.index)
    if "irt_model" in anchors:
        anchor_scope &= anchors["irt_model"].eq(irt_model)
    if "fit" in anchors:
        anchor_scope &= anchors["fit"].eq("gpqa")
    theta = anchors.loc[anchor_scope, ["model", "theta"]]
    calibration = calibration.merge(
        theta.rename(columns={"model": "left_model", "theta": "left_theta"}),
        on="left_model", validate="many_to_one",
    ).merge(
        theta.rename(columns={"model": "right_model", "theta": "right_theta"}),
        on="right_model", validate="many_to_one",
    )
    calibration["delta_theta"] = calibration["left_theta"] - calibration["right_theta"]
    target = rows[
        rows[bin_column].eq(bin_value)
        & (rows["left_model"].isin(targets) | rows["right_model"].isin(targets))
    ].copy()
    return _add_components(calibration), _add_components(target)


def calibrate_to_hard(train_predictor, train_hard, test_predictor):
    """Map a fold predictor to hard accuracy using calibration models only."""
    design = np.column_stack([
        np.ones(len(train_predictor)), np.asarray(train_predictor, dtype=float),
    ])
    intercept, slope = np.linalg.lstsq(
        design, np.asarray(train_hard, dtype=float), rcond=None,
    )[0]
    return np.clip(
        intercept + slope * np.asarray(test_predictor, dtype=float), 0, 1,
    )


def _gpqa(frame):
    frame = frame.copy()
    if "benchmark" in frame:
        frame = frame[frame["benchmark"].eq("gpqa")].copy()
    else:
        frame["benchmark"] = "gpqa"
    return frame


def _easy_frame(frame, name, bin_column="half", bin_value="easy"):
    if bin_column not in frame:
        raise ValueError(f"{name} frame must carry {bin_column}={bin_value!r}")
    frame = _gpqa(frame)
    if not frame[bin_column].eq(bin_value).all():
        raise ValueError(f"{name} frame must contain only {bin_value} rows")
    return frame


def filter_target_evidence(frame, calibration, target_models):
    """Drop target rows with category levels absent from calibration."""
    labels = {
        "judge": calibration.get("judges"),
        "source": calibration.get("sources"),
        "condition": calibration.get("conditions"),
    }
    if any(values is None for values in labels.values()):
        return frame.copy(), pd.DataFrame(columns=TARGET_EXCLUSION_COLUMNS)

    target_models = set(target_models)
    keep, excluded = [], []
    for index, row in frame.iterrows():
        reasons = [
            f"unseen {name}: {row[name]}"
            for name, values in labels.items()
            if row[name] not in set(values)
        ]
        if not reasons:
            keep.append(index)
            continue
        affected = {
            row["left_model"], row["right_model"],
        } & target_models
        excluded.extend(
            {"model": model, "reason": "; ".join(reasons)}
            for model in sorted(affected)
        )
    exclusions = (
        pd.DataFrame(excluded)
        .groupby(["model", "reason"], as_index=False, sort=True)
        .size().rename(columns={"size": "count"})
        if excluded else pd.DataFrame(columns=TARGET_EXCLUSION_COLUMNS)
    )
    return frame.loc[keep].copy(), exclusions


def _comparison_rows(calibration_frame, target_frame):
    rows = pd.concat([
        _gpqa(calibration_frame).assign(_origin="calibration"),
        _gpqa(target_frame).assign(_origin="target"),
    ], ignore_index=True, sort=False)
    return rows[
        rows["chosen_left"].notna()
        & rows["left_model"].ne(rows["right_model"])
    ].reset_index(drop=True)


def _fold_coverage(rows, anchors, target_models, fold, irt_model):
    members = _component_members(rows)
    anchor_models = set(anchors["model"])
    member_map = members.set_index("model")["component"].to_dict()
    result = []
    for model in sorted(target_models):
        component = member_map.get(model, "")
        component_models = set(
            members.loc[members["component"].eq(component), "model"]
        ) if component else set()
        component_rows = rows[
            rows["left_model"].isin(component_models)
            & rows["right_model"].isin(component_models)
        ]
        anchored = component_models & anchor_models
        pairs = component_rows[["left_model", "right_model"]].apply(
            lambda row: tuple(sorted(row)), axis=1,
        ) if not component_rows.empty else pd.Series(dtype=object)
        count = len(anchored)
        result.append({
            "benchmark": "gpqa", "irt_model": irt_model, "fold": fold,
            "component": component, "model": model, "n_anchors": count,
            "eligible": count >= 2,
            "reason": "" if count >= 2 else "fewer than 2 anchors",
            "n_calibration_models": count,
            "n_target_models": len(component_models & set(target_models)),
            "n_judges": component_rows["judge"].nunique(),
            "n_sources": component_rows["source"].nunique(),
            "n_easy_items": (
                component_rows["item_id"].nunique()
                if "item_id" in component_rows else 0
            ),
            "n_comparison_edges": pairs.nunique(),
            "n_rounds": (
                component_rows["round"].nunique()
                if "round" in component_rows else len(component_rows)
            ),
        })
    return pd.DataFrame(result, columns=COVERAGE_COLUMNS)


def _raw_elo_predictors(rows, anchors, coverage):
    """Fit item-counterbalanced BT scores and align each eligible component."""
    if rows.empty:
        return pd.DataFrame(columns=["model", "component", "raw_elo"])
    prepared = rows.reset_index(drop=True).copy()
    prepared["major"] = prepared[["left_model", "right_model"]].min(axis=1)
    prepared["minor"] = prepared[["left_model", "right_model"]].max(axis=1)
    prepared["picked_major"] = np.where(
        prepared["left_model"].eq(prepared["major"]),
        prepared["chosen_left"], 1 - prepared["chosen_left"],
    )
    identity = pd.Series("", index=prepared.index, dtype=object)
    for column in ("source", "round", "condition", "judge"):
        if column in prepared:
            identity = identity + "\0" + prepared[column].fillna("").astype(str)
    item = pd.Series(prepared.index.astype(str), index=prepared.index)
    if "item_id" in prepared:
        item = prepared["item_id"].fillna(item).astype(str)
    identity = identity + "\0" + item
    prepared["item_id"] = identity
    pair_wins = collect_pair_wins(
        prepared.to_dict("records"), ["benchmark"],
    )[("gpqa",)]
    theta = anchors.drop_duplicates("model").set_index("model")["theta"]
    rows_out = []
    members = _component_members(rows)
    eligible_components = set(coverage.loc[coverage["eligible"], "component"])
    for component in sorted(eligible_components):
        models = set(members.loc[members["component"].eq(component), "model"])
        wins = {
            pair: result for pair, result in pair_wins.items()
            if pair[0] in models and pair[1] in models
        }
        if not wins:
            continue
        strengths = fit_bt(wins)
        anchor_models = sorted(models & set(theta.index) & set(strengths))
        if len(anchor_models) < 2:
            continue
        design = np.column_stack([
            np.ones(len(anchor_models)),
            [strengths[model] for model in anchor_models],
        ])
        intercept, slope = np.linalg.lstsq(
            design, theta.loc[anchor_models].to_numpy(float), rcond=None,
        )[0]
        rows_out.extend({
            "model": model, "component": component,
            "raw_elo": intercept + slope * strength,
        } for model, strength in strengths.items())
    return pd.DataFrame(rows_out, columns=["model", "component", "raw_elo"])


def _model_values(frame, value_column, irt_model):
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    selected = _gpqa(frame)
    if irt_model and "irt_model" in selected:
        selected = selected[selected["irt_model"].eq(irt_model)]
    if value_column not in selected and "accuracy" in selected:
        subset = "easy" if value_column == "easy_accuracy" else "hard"
        if "subset" in selected:
            selected = selected[selected["subset"].eq(subset)]
        selected = selected.rename(columns={"accuracy": value_column})
    values = selected[["model", value_column]].dropna().drop_duplicates()
    if values["model"].duplicated().any():
        raise ValueError(f"conflicting {value_column} rows")
    return values.set_index("model")[value_column].astype(float)


def judge_ordering_rows(votes, hard_accuracy, irt_model=None):
    """Orient raw choices using only candidates' within-GPQA hard scores."""
    rows = _gpqa(votes)
    hard = _model_values(hard_accuracy, "hard_accuracy", irt_model)
    rows["left_hard_accuracy"] = rows["left_model"].map(hard)
    rows["right_hard_accuracy"] = rows["right_model"].map(hard)
    rows = rows[
        rows["chosen_left"].isin([0.0, 1.0])
        & rows["left_hard_accuracy"].notna()
        & rows["right_hard_accuracy"].notna()
    ].copy()
    true_sign = np.sign(
        rows["left_hard_accuracy"] - rows["right_hard_accuracy"],
    )
    choice_sign = 2 * rows["chosen_left"].to_numpy(float) - 1
    rows["left_is_stronger"] = true_sign > 0
    rows["judge_ordering_correct"] = (choice_sign * true_sign > 0).astype(int)
    return rows.reset_index(drop=True)


def _response_easy_accuracy(responses, easy_items, irt_model):
    if responses is None:
        return pd.Series(dtype=float)
    selected = _gpqa(irt_cells(responses))
    if isinstance(easy_items, pd.DataFrame):
        items = _gpqa(easy_items)
        if irt_model and "irt_model" in items:
            items = items[items["irt_model"].eq(irt_model)]
        if "fit" in items:
            items = items[items["fit"].eq("gpqa")]
        if "half" in items:
            items = items[items["half"].eq("easy")]
        item_column = "item" if "item" in items else "item_id"
        allowed = set(items[item_column].astype(str))
    else:
        allowed = set(map(str, easy_items))
    item_column = "item" if "item" in selected else "item_id"
    selected = selected[selected[item_column].astype(str).isin(allowed)]
    item_column = "item" if "item" in selected else "item_id"
    if selected.duplicated(["model", item_column]).any():
        raise ValueError("easy accuracy requires one canonical response per model/item")
    return selected.groupby("model")["correct"].mean().astype(float)


def posterior_predictive_summary(frame, calibration):
    """Summarize calibration-row posterior predictions by judge and source."""
    if not set(POSTERIOR_KEYS).issubset(calibration):
        return pd.DataFrame(columns=CALIBRATION_COLUMNS[4:])
    frame = _gpqa(frame).reset_index(drop=True)
    levels = {
        name: {value: index for index, value in enumerate(calibration[f"{name}s"])}
        for name in ("judge", "source", "condition")
    }
    encoded = {
        name: frame[name].map(mapping).to_numpy(dtype=int)
        for name, mapping in levels.items()
    }
    features = calibration["features"]
    design = frame.loc[:, features].to_numpy(float)
    if features:
        design = (
            design - np.asarray(calibration["feature_mean"])
        ) / np.asarray(calibration["feature_scale"])
    logits = np.asarray(calibration["alpha"])[:, encoded["judge"]]
    logits += np.asarray(calibration["source"])[
        :, encoded["judge"], encoded["source"]
    ]
    logits += np.asarray(calibration["condition"])[
        :, encoded["judge"], encoded["condition"]
    ]
    logits += np.asarray(calibration["discrimination"])[
        :, encoded["judge"]
    ] * frame["delta_theta"].to_numpy(float)[None, :]
    logits += np.sum(
        np.asarray(calibration["beta"])[:, encoded["judge"], :]
        * design[None, :, :], axis=-1,
    )
    probability = expit(logits).mean(axis=0)
    observed = frame["chosen_left"].to_numpy(float)
    scored = frame[["judge", "source"]].copy()
    scored["observed"] = observed
    scored["probability"] = probability
    scored["brier_row"] = (probability - observed) ** 2
    scored["log_loss_row"] = -(
        observed * np.log(np.clip(probability, 1e-12, 1))
        + (1 - observed) * np.log(np.clip(1 - probability, 1e-12, 1))
    )
    lower = (probability > 0.95).astype(float)
    upper = (probability > 0.05).astype(float)
    scored["covered"] = (observed >= lower) & (observed <= upper)
    return scored.groupby(["judge", "source"], as_index=False, sort=True).agg(
        observed_rate=("observed", "mean"),
        predicted_rate=("probability", "mean"),
        brier=("brier_row", "mean"),
        log_loss=("log_loss_row", "mean"),
        interval_coverage_90=("covered", "mean"),
        n_rows=("observed", "size"),
    )


def _coefficient_summary(calibration, fold, irt_model):
    if not set(POSTERIOR_KEYS).issubset(calibration):
        return pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    rows = []

    def add(parameter, values, judge, level="", feature=""):
        values = np.asarray(values, dtype=float)
        rows.append({
            "benchmark": "gpqa", "irt_model": irt_model, "fold": fold,
            "parameter": parameter, "judge": judge, "level": level,
            "feature": feature, "mean": values.mean(),
            "p5": np.percentile(values, 5), "p95": np.percentile(values, 95),
        })

    for judge_index, judge in enumerate(calibration["judges"]):
        add("alpha", calibration["alpha"][:, judge_index], judge)
        add(
            "discrimination", calibration["discrimination"][:, judge_index],
            judge,
        )
        for index, level in enumerate(calibration["sources"]):
            add("source", calibration["source"][:, judge_index, index], judge, level)
        for index, level in enumerate(calibration["conditions"]):
            add(
                "condition", calibration["condition"][:, judge_index, index],
                judge, level,
            )
        for index, feature in enumerate(calibration["features"]):
            add("beta", calibration["beta"][:, judge_index, index], judge, feature=feature)
    return pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS)


def _mapped_interval(train_predictor, train_hard, point, lower, upper):
    values = calibrate_to_hard(
        train_predictor, train_hard, [point, lower, upper],
    )
    return values[0], min(values[1:]), max(values[1:])


def run_fold(
    fold, calibration_frame, target_frame, anchors, hard_accuracy,
    *, easy_accuracy=None, responses=None, easy_items=None, irt_model=None,
    feature_columns=None, warmup=1000, samples=1000, seed=0, arm=None,
    target_accept=0.8,
    inference_options=None, feature_ablations=None,
    bin_column="half", bin_value="easy",
    fit_calibration_fn=fit_calibration, infer_targets_fn=infer_targets,
):
    """Run leakage-safe corrected and matched baseline predictions for one fold."""
    if arm not in (None, "corrected", "uncorrected"):
        raise ValueError(f"unknown fold arm: {arm}")
    calibration_frame = _easy_frame(
        calibration_frame, "calibration", bin_column, bin_value,
    )
    target_frame = _easy_frame(target_frame, "target", bin_column, bin_value)
    if responses is not None and easy_items is None:
        raise ValueError("easy_items is required when responses are supplied")
    if irt_model is None:
        forms = (
            sorted(anchors["irt_model"].dropna().unique())
            if "irt_model" in anchors else []
        )
        if len(forms) > 1:
            raise ValueError("irt_model is required when anchors contain multiple forms")
        irt_model = forms[0] if forms else ""
    if irt_model and "irt_model" in anchors:
        anchors = anchors[anchors["irt_model"].eq(irt_model)].copy()
    features = list(feature_columns) if feature_columns is not None else [
        f"delta_{name}" for name in BIAS_FEATURES
        if f"delta_{name}" in calibration_frame
    ]
    inference_options = dict(inference_options or {})
    target_models = set(inference_options.get("target_models", (
        set(target_frame["left_model"]) | set(target_frame["right_model"])
    ) - set(anchors["model"])))

    fits = {}
    if arm != "uncorrected":
        fits["corrected"] = fit_calibration_fn(
            calibration_frame, feature_columns=features,
            warmup=warmup, samples=samples, seed=seed,
            target_accept=target_accept,
        )
    if arm != "corrected":
        fits["uncorrected"] = fit_calibration_fn(
            calibration_frame, feature_columns=(),
            warmup=warmup, samples=samples, seed=seed,
            target_accept=target_accept,
        )

    target_evidence, target_exclusions = filter_target_evidence(
        target_frame, next(iter(fits.values())), target_models,
    )

    def infer(name, artifact):
        return infer_targets_fn(
            target_evidence, anchors, artifact, inference_name=name,
            warmup=warmup, samples=samples, seed=seed, **inference_options,
        )

    inferred = {}
    if "corrected" in fits:
        corrected_fit = fits["corrected"]
        inferred["corrected"] = infer("corrected", corrected_fit)
        for group, columns in (feature_ablations or {}).items():
            fitted = list(corrected_fit["features"])
            missing = sorted(set(columns) - set(fitted))
            if missing:
                raise ValueError(
                    f"ablation group {group}: features not in fit: {missing}"
                )
            ablated = dict(corrected_fit)
            ablated["beta"] = np.array(corrected_fit["beta"], copy=True)
            for column in columns:
                ablated["beta"][..., fitted.index(column)] = 0
            name = f"corrected_without_{group}"
            inferred[name] = infer(name, ablated)
    if "uncorrected" in fits:
        inferred["uncorrected"] = infer("uncorrected", fits["uncorrected"])

    calibration_outputs = []
    for method, artifact in fits.items():
        summary = posterior_predictive_summary(calibration_frame, artifact)
        if not summary.empty:
            calibration_outputs.append(summary.assign(
                benchmark="gpqa", irt_model=irt_model, fold=fold, method=method,
            )[CALIBRATION_COLUMNS])

    comparison_rows = _comparison_rows(calibration_frame, target_evidence)
    coverage = _fold_coverage(
        comparison_rows, anchors, target_models, fold, irt_model,
    )
    remaining = {
        model: int(
            (
                target_evidence["left_model"].eq(model)
                | target_evidence["right_model"].eq(model)
            ).sum()
        )
        for model in target_models
    }
    excluded = target_exclusions.groupby("model")["count"].sum()
    coverage["n_evidence_remaining"] = coverage["model"].map(remaining).fillna(0).astype(int)
    coverage["n_evidence_excluded"] = coverage["model"].map(excluded).fillna(0).astype(int)
    coverage = coverage.reindex(columns=COVERAGE_COLUMNS)
    raw = _raw_elo_predictors(comparison_rows, anchors, coverage)
    response_easy = _response_easy_accuracy(responses, easy_items, irt_model)

    # The hidden outcome enters only after every fit and target inference above.
    hard = _model_values(hard_accuracy, "hard_accuracy", irt_model)
    easy = (
        _model_values(easy_accuracy, "easy_accuracy", irt_model)
        if easy_accuracy is not None else response_easy
    )
    anchors_by_model = anchors.drop_duplicates("model").set_index("model")["theta"]
    training_models = sorted(set(anchors_by_model.index) & set(hard.index))
    inferred_targets = [set(values["model"]) for values in inferred.values()]
    raw_by_model = raw.drop_duplicates("model").set_index("model") if not raw.empty else raw
    eligible = set(coverage.loc[coverage["eligible"], "model"])
    matched = (
        target_models & eligible & set.intersection(*inferred_targets)
        & set(hard.index)
        & (set(easy.index) if easy_accuracy is not None or responses is not None else target_models)
    )
    raw_targets = set(raw_by_model.index) if not raw.empty else set()
    matched &= raw_targets

    prediction_rows = []
    if len(training_models) >= 2:
        train_theta = anchors_by_model.loc[training_models].to_numpy(float)
        train_hard = hard.loc[training_models].to_numpy(float)
        raw_training = [model for model in training_models if model in raw_targets]
        easy_training = [model for model in training_models if model in easy.index]
        coverage_by_model = coverage.set_index("model")
        for model in sorted(matched):
            common = {
                column: coverage_by_model.loc[model, column]
                for column in ["component", *PREDICTION_COVERAGE_COLUMNS]
            }
            common.update({
                "benchmark": "gpqa", "irt_model": irt_model, "fold": fold,
                "model": model, "observed": hard.loc[model],
            })
            for method, targets in inferred.items():
                values = targets.set_index("model").loc[model]
                prediction, p5, p95 = _mapped_interval(
                    train_theta, train_hard, values["theta"],
                    values["theta_p5"], values["theta_p95"],
                )
                prediction_rows.append({
                    **common, "method": method, "predictor": values["theta"],
                    "prediction": prediction, "prediction_p5": p5,
                    "prediction_p95": p95,
                })
            if arm == "uncorrected":
                continue
            if len(raw_training) >= 2:
                raw_prediction = calibrate_to_hard(
                    raw_by_model.loc[raw_training, "raw_elo"],
                    hard.loc[raw_training], [raw_by_model.loc[model, "raw_elo"]],
                )[0]
            else:
                raw_prediction = np.nan
            prediction_rows.append({
                **common, "method": "raw_elo",
                "predictor": raw_by_model.loc[model, "raw_elo"],
                "prediction": raw_prediction, "prediction_p5": raw_prediction,
                "prediction_p95": raw_prediction,
            })
            if easy_accuracy is not None or responses is not None:
                easy_prediction = (
                    calibrate_to_hard(
                        easy.loc[easy_training], hard.loc[easy_training],
                        [easy.loc[model]],
                    )[0] if len(easy_training) >= 2 else np.nan
                )
                easy_predictor = easy.loc[model]
            else:
                easy_prediction = easy_predictor = np.nan
            prediction_rows.append({
                **common, "method": "easy_accuracy", "predictor": easy_predictor,
                "prediction": easy_prediction, "prediction_p5": easy_prediction,
                "prediction_p95": easy_prediction,
            })
            mean = train_hard.mean()
            prediction_rows.append({
                **common, "method": "mean", "predictor": np.nan,
                "prediction": mean, "prediction_p5": mean, "prediction_p95": mean,
            })

    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    calibration_table = (
        pd.concat(calibration_outputs, ignore_index=True)
        if calibration_outputs else pd.DataFrame(columns=CALIBRATION_COLUMNS)
    )
    return {
        "predictions": predictions,
        "coverage": coverage,
        "calibration": calibration_table,
        "target_exclusions": target_exclusions,
        "coefficients": (
            _coefficient_summary(fits["corrected"], fold, irt_model)
            if "corrected" in fits else pd.DataFrame(columns=COEFFICIENT_COLUMNS)
        ),
    }


def reduce_folds(folds):
    """Concatenate independently fitted fold artifacts without refitting."""
    schemas = {
        "predictions": PREDICTION_COLUMNS,
        "coverage": COVERAGE_COLUMNS,
        "calibration": CALIBRATION_COLUMNS,
        "coefficients": COEFFICIENT_COLUMNS,
        "target_exclusions": TARGET_EXCLUSION_COLUMNS,
    }
    outputs = {}
    folds = list(folds)
    for name, columns in schemas.items():
        frames = [fold[name] for fold in folds if name in fold and not fold[name].empty]
        outputs[name] = (
            pd.concat(frames, ignore_index=True).reindex(columns=columns)
            if frames else pd.DataFrame(columns=columns)
        )
    return outputs
