#!/usr/bin/env python3
"""Sweep transcript nearest-neighbour and kernel regressors.

For each task, ``loo`` selects the configuration with the lowest in-task
leave-one-out error. ``rolling`` selects from errors on earlier tasks only,
with ``loo`` as the first-task fallback. ``oracle`` selects the one
configuration with the lowest pooled target error. Rolling is the recommended
paper result because it matches what a practitioner could choose at time t.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capability.evaluate import score, semester_names  # noqa: E402
from capability.task import Task  # noqa: E402
from configs.domain import DomainConfig  # noqa: E402
from protocols import PROTOCOLS  # noqa: E402
from protocols.embeddings import VARIANTS  # noqa: E402
from hillclimb import (  # noqa: E402
    aggregate,
    item_embeddings,
    per_item_mean,
    pool_scores,
    result_row,
    skip_estimator,
)

ALPHAS = np.logspace(-4, 2, 13)
KNN_KS = (1, 2, 3, 4, 5, 7)
GAMMAS = {"kr_rbf": (0.5, 1, 2, 4), "kr_lap": (0.5, 1, 2)}
def l2_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm != 0)
def preprocess(name: str, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if name == "raw":
        return train, test
    if name == "renorm":
        return l2_rows(train), l2_rows(test)
    if name == "center":
        mean = train.mean(axis=0)
        return l2_rows(train - mean), l2_rows(test - mean)
    k = int(name.removeprefix("pca"))
    if len(train) <= k:
        return None
    model = PCA(n_components=k, random_state=0).fit(train)
    return l2_rows(model.transform(train)), l2_rows(model.transform(test))
def knn_k(name: str) -> int:
    return int(name.split("_")[1][1:])
def estimator(name: str, x: np.ndarray):
    if name.startswith("knn_"):
        return KNeighborsRegressor(n_neighbors=knn_k(name), metric="cosine", weights=name.rsplit("_", 1)[1])
    if name == "kr_lin":
        base = KernelRidge(kernel="linear")
    elif name == "kr_poly2":
        base = KernelRidge(kernel="polynomial", degree=2, coef0=1)
    else:
        scale = float(np.median(pdist(x, metric="sqeuclidean"))) if len(x) > 1 else 0.0
        scale = scale if scale > 0 else np.finfo(float).eps
        family, multiplier = name.rsplit("_g", 1)
        kernel = "rbf" if family == "kr_rbf" else "laplacian"
        base = KernelRidge(kernel=kernel, gamma=float(multiplier) / scale)
    return GridSearchCV(base, {"alpha": ALPHAS}, cv=LeaveOneOut(), scoring="neg_mean_squared_error")
def can_cv(name: str, n: int) -> bool:
    if name.startswith("knn_"):
        return n > knn_k(name)
    return n >= 3
def cv_predictions(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(cross_val_predict(estimator(name, x), x, y, cv=LeaveOneOut())).reshape(-1)
def family(name: str) -> str:
    if name.startswith("knn_"):
        return "knn"
    if name.startswith("kr_"):
        return "kernel_ridge"
    if name.startswith("ens_"):
        return "ensemble"
    return "item_knn"
def model_task(task: Task, variant: str, preproc: str, name: str, ridge_forecast: pd.Series) -> tuple[pd.Series, float]:
    train_df = aggregate(task, task.train, "train", variant, "mean")
    test_df = aggregate(task, task.test, "test", variant, "mean").reindex(columns=train_df.columns)
    y_all = task.train.theta.reindex(train_df.index).to_numpy(float)
    finite = np.isfinite(train_df.to_numpy(float)).all(axis=1) & np.isfinite(y_all)
    pred = pd.Series(np.nan, index=task.test.models.index)
    if not finite.any():
        return pred, np.nan
    x, y = train_df.to_numpy(float)[finite], y_all[finite]
    test_array = test_df.to_numpy(float)
    test_ok = np.isfinite(test_array).all(axis=1)
    transformed = preprocess(preproc, x, test_array[test_ok]) if test_ok.any() else preprocess(preproc, x, np.empty((0, x.shape[1])))
    if transformed is None:
        return pred, np.nan
    x, xtest = transformed
    member = "knn_k3_distance" if name.startswith("ens_") else name
    cv_mse = np.nan
    if can_cv(member, len(y)):
        cv = cv_predictions(member, x, y)
        cv_mse = float(np.mean((cv - y) ** 2))
    if name.startswith("ens_"):
        if skip_estimator("knn3_cos", len(y)):
            return pred, cv_mse
        knn = estimator(member, x).fit(x, y)
        knn_pred = pd.Series(knn.predict(xtest), index=test_df.index[test_ok]) if test_ok.any() else pd.Series(dtype=float)
        if name == "ens_knn_kr":
            model = estimator("kr_lin", x).fit(x, y)
            second = pd.Series(model.predict(xtest), index=test_df.index[test_ok]) if test_ok.any() else pd.Series(dtype=float)
        else:
            second = ridge_forecast.reindex(test_df.index[test_ok])
        fitted = (knn_pred + second) / 2
    else:
        if (name.startswith("knn_") and len(y) < knn_k(name)) or (not name.startswith("knn_") and len(y) < 2):
            return pred, cv_mse
        model = estimator(name, x).fit(x, y)
        fitted = pd.Series(model.predict(xtest), index=test_df.index[test_ok]) if test_ok.any() else pd.Series(dtype=float)
    if len(fitted):
        pred.loc[fitted.index] = fitted
    return pred, cv_mse
def item_forecast(target: pd.DataFrame, references: dict[str, pd.DataFrame], theta: pd.Series, k: int) -> float:
    estimates = []
    for item, target_row in target.iterrows():
        target_vector = target_row.to_numpy(float)
        if not np.isfinite(target_vector).all():
            continue
        candidates = []
        for model, table in references.items():
            if item not in table.index or not np.isfinite(theta.get(model, np.nan)):
                continue
            reference_row = table.loc[item]
            if isinstance(reference_row, pd.DataFrame):
                reference_row = reference_row.iloc[0]
            reference_vector = reference_row.to_numpy(float)
            denominator = np.linalg.norm(target_vector) * np.linalg.norm(reference_vector)
            if denominator:
                candidates.append((float(target_vector @ reference_vector / denominator), float(theta[model])))
        if candidates:
            candidates.sort(reverse=True)
            selected = candidates[:k]
            weights = np.maximum([x[0] for x in selected], 0) + 1e-6
            estimates.append(float(np.average([x[1] for x in selected], weights=weights)))
    return float(np.mean(estimates)) if estimates else np.nan
def item_task(task: Task, variant: str, k: int) -> tuple[pd.Series, float]:
    train = item_embeddings(task.train, variant, "train")
    test = item_embeddings(task.test, variant, "test")
    theta = task.train.theta
    pred = pd.Series({model: item_forecast(table, train, theta, k) for model, table in test.items()}, dtype=float)
    pred = pred.reindex(task.test.models.index)
    errors = []
    for held_out, table in train.items():
        if not len(table):
            continue
        references = {model: other for model, other in train.items() if model != held_out}
        estimate = item_forecast(table, references, theta, k)
        if np.isfinite(estimate) and np.isfinite(theta.get(held_out, np.nan)):
            errors.append((estimate - theta[held_out]) ** 2)
    return pred, float(np.mean(errors)) if errors else np.nan
def specs(quick: bool) -> list[tuple[str, str, str, str, str]]:
    variants = ["full"] if quick else [*VARIANTS, "both"]
    preprocs = ["raw", "center"] if quick else ["raw", "renorm", "center", "pca2", "pca4", "pca6"]
    estimators = (["knn_k3_distance", "knn_k5_distance", "kr_lin"] if quick else
                  [f"knn_k{k}_{w}" for k in KNN_KS for w in ("uniform", "distance")] +
                 ([] if quick else ["kr_lin", "kr_poly2", "ens_knn_kr", "ens_knn_len",
                                    *[f"kr_rbf_g{g:g}" for g in GAMMAS["kr_rbf"]],
                                    *[f"kr_lap_g{g:g}" for g in GAMMAS["kr_lap"]]]))
    result = [(f"{v}/{p}/{e}", family(e), v, p, e) for v in variants for p in preprocs for e in estimators]
    item_variants = ["full"] if quick else ["full", "answer"]
    item_ks = (3,) if quick else (1, 3, 5)
    result.extend((f"{v}/raw/item_knn_k{k}", "item_knn", v, "raw", f"item_knn_k{k}")
                  for v in item_variants for k in item_ks)
    return result
def evaluate_spec(tasks: list[Task], spec: tuple[str, str, str, str, str], ridge_forecasts: list[pd.Series]):
    config, config_family, variant, preproc, name = spec
    forecasts, cv_by_task, task_mse = [], [], []
    for task, ridge in zip(tasks, ridge_forecasts):
        if name.startswith("item_knn_k"):
            forecast, cv = item_task(task, variant, int(name.removeprefix("item_knn_k")))
        else:
            forecast, cv = model_task(task, variant, preproc, name, ridge)
        forecasts.append(forecast)
        cv_by_task.append(cv)
        task_mse.append(score(forecast, task.truth)["mse"])
    per_semester, all_score = pool_scores(tasks, forecasts)
    cv_values = [x for x in cv_by_task if np.isfinite(x)]
    base = result_row(variant, preproc, name, "train", float(np.mean(cv_values)) if cv_values else np.nan,
                      all_score, per_semester)
    row = {"config": config, "family": config_family, "variant": variant, "preproc": preproc,
           "cv_mse": base["cv_mse"], "n": base["n"], "mse_all": base["mse_all"]}
    row.update({f"mse_{s}": base[f"mse_{s}"] for s in per_semester})
    return row, forecasts, cv_by_task, task_mse
def selection(tasks: list[Task], details: dict[str, dict], table: pd.DataFrame, semesters: list[str]) -> pd.DataFrame:
    names = list(details)
    loo = []
    for index in range(len(tasks)):
        usable = [(details[name]["cv"][index], name) for name in names if np.isfinite(details[name]["cv"][index])]
        loo.append(min(usable)[1] if usable else names[0])
    rolling = [loo[0]]
    for index in range(1, len(tasks)):
        scores = {}
        for name in names:
            scores[name] = pool_scores(tasks[:index], details[name]["forecast"][:index])[1]["mse"]
        usable = [(value, name) for name, value in scores.items() if np.isfinite(value)]
        rolling.append(min(usable)[1] if usable else loo[index])
    oracle = table.set_index("config").loc[names, "mse_all"].astype(float).idxmin()

    def row(rule: str, chosen: list[str], rolling_part: bool) -> dict:
        forecasts = [details[name]["forecast"][i] for i, name in enumerate(chosen)]
        _, all_score = pool_scores(tasks, forecasts)
        result = {"rule": rule, "mse_all": all_score["mse"],
                  "mse_rolling_part": np.nan if not rolling_part else pool_scores(tasks[1:], forecasts[1:])[1]["mse"]}
        result.update({semester: chosen[i] for i, semester in enumerate(semesters)})
        return result

    return pd.DataFrame([row("loo", loo, False), row("rolling", rolling, True),
                         row("oracle", [oracle] * len(tasks), False)])
def main() -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--semester", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=Path("/fast/jyeon/smalljudge_embeddings/hillclimb"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    domain = DomainConfig.from_cli(args.domain)
    semesters = args.semester or semester_names(domain)
    tasks = [Task.from_semester(s, domain) for s in semesters]
    baseline_forecasts = {name: [PROTOCOLS[name](task).reindex(task.truth.index) for task in tasks]
                          for name in ("chars", "ridge")}
    started = time.time()
    rows, details = [], {}
    sweep = specs(args.quick)
    registered = [
        ("registered/emb_kr", "kernel_ridge", "full", "renorm", "kr_lin"),
        ("registered/emb_knn", "knn", "full", "raw", "knn_k3_distance"),
    ]
    registered_names = {item[0] for item in registered}
    for config, config_family, variant, preproc, name in sweep + registered:
        try:
            source = baseline_forecasts["ridge"] if name == "ens_knn_len" else [pd.Series(dtype=float)] * len(tasks)
            row, forecasts, cv, task_mse = evaluate_spec(tasks, (config, config_family, variant, preproc, name), source)
            rows.append(row)
            if config not in registered_names:
                details[config] = {"forecast": forecasts, "cv": cv, "mse": task_mse}
        except Exception as error:  # noqa: BLE001
            message = str(error).splitlines()[0] if str(error) else type(error).__name__
            print(f"skip {config}: {message}")
    for name, forecasts in baseline_forecasts.items():
        per_semester, all_score = pool_scores(tasks, forecasts)
        base = result_row("", name, "", "", np.nan, all_score, per_semester)
        row = {"config": f"baseline/{name}", "family": "baseline", "variant": "", "preproc": "",
               "cv_mse": np.nan, "n": base["n"], "mse_all": base["mse_all"]}
        row.update({f"mse_{s}": base[f"mse_{s}"] for s in per_semester})
        rows.append(row)
    columns = ["config", "family", "variant", "preproc", "cv_mse", "n", "mse_all"] + [f"mse_{s}" for s in semesters]
    table = pd.DataFrame(rows).reindex(columns=columns).sort_values("mse_all")
    config_table = table[table.family != "baseline"]
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"hillclimb2_{domain.name}.csv"
    selection_table = selection(tasks, details, config_table, semesters)
    selection_output = args.out / f"hillclimb2_{domain.name}_selection.csv"
    table.to_csv(output, index=False, float_format="%.4f")
    selection_table.to_csv(selection_output, index=False, float_format="%.4f")
    print("\n-- 20 best configs by mse_all --")
    print(config_table.head(20).round(3).to_string(index=False))
    print("\n-- 10 best configs by cv_mse --")
    print(config_table.sort_values("cv_mse").head(10).round(3).to_string(index=False))
    print("\n-- selection --")
    print(selection_table.to_string(index=False))
    print("\n-- per-task loo and rolling choices --")
    print(pd.DataFrame({"semester": semesters, "loo": selection_table.loc[selection_table.rule == "loo", semesters].iloc[0].to_numpy(),
                        "rolling": selection_table.loc[selection_table.rule == "rolling", semesters].iloc[0].to_numpy()}).to_string(index=False))
    print("\n-- baselines --")
    print(table[table.family == "baseline"].round(3).to_string(index=False))
    print(f"\n{len(config_table)} configs evaluated in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
