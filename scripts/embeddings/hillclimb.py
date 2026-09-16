#!/usr/bin/env python3
"""Hill-climb configurations of the transcript-embedding protocol on one domain.

    python scripts/embeddings/hillclimb.py configs/domain/science.yml [--semester S ...]
        [--variant full answer] [--out /fast/jyeon/smalljudge_embeddings/hillclimb] [--quick]

protocols.embeddings.EmbeddingRegression (emb_full / emb_answer) is one fixed feature aggregation
(mean embedding over a model's correct train transcripts) and estimator (RidgeCV on standardised
features). This sweeps text variant, item -> model aggregation, and estimator around that choice,
scored like capability.evaluate: MSE per semester task and pooled over every target ("all"),
against the frozen theta*. It also reports leave-one-model-out cross-validation MSE on the
training cohorts alone (cv_mse), so a configuration can be chosen without looking at the targets.

Output columns: variant (full / answer / both), aggregation (mean / centered / renorm), estimator
(an sklearn family, or "item_ridge"), train (train = references+historical, refs = references
only; baselines leave estimator/train empty and put the baseline protocol's name in aggregation),
cv_mse (leave-one-model-out MSE on the training cohort, NaN if too few usable rows), n (pooled
targets scored), mse_all (pooled MSE over every target), and one mse_<semester> column per
semester task. mse_all uses theta*, the number you'd be tempted to select on, so treat it as a
diagnostic, not a selection target; cv_mse never touches theta* and is the honest, target-free
selection criterion. Results go to <out>/hillclimb_<domain.name>.csv, sorted by mse_all.

Environment: SMALL_JUDGE_EMBEDDINGS gives the embedding cache root (read by protocols.embeddings at
import time); PYTHONPATH=src. Runs as an HTCondor CPU job through scripts/embeddings/cpu.sub.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from capability.evaluate import evaluate, score, semester_names  # noqa: E402
from capability.task import Cohort, Task  # noqa: E402
from configs.domain import DomainConfig  # noqa: E402
from protocols import PROTOCOLS  # noqa: E402
from protocols.embeddings import VARIANTS, embedding_path, read_embeddings  # noqa: E402

ALPHAS = (0.1, 1, 10, 100, 1000)  # item_ridge candidates, after the 64-d PCA projection
BASELINES = ("acc_train", "acc_easy", "acc_all", "chars", "ridge", "embedding", "emb_ens")
ESTIMATORS = {
    "ridgecv_std": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 15))),
    "ridgecv_raw": lambda: RidgeCV(alphas=np.logspace(-4, 3, 15)),
    **{f"pca{k}_ridge": (lambda k=k: make_pipeline(StandardScaler(), PCA(n_components=k),
                          RidgeCV(alphas=np.logspace(-2, 4, 13)))) for k in (2, 4, 8)},
    **{f"pls{k}": (lambda k=k: PLSRegression(n_components=k, scale=True)) for k in (1, 2, 3)},
    "knn3_cos": lambda: KNeighborsRegressor(n_neighbors=3, metric="cosine", weights="distance"),
    "kr_cos": lambda: make_pipeline(Normalizer(), GridSearchCV(KernelRidge(kernel="linear"), {"alpha": np.logspace(-3, 1, 9)},
                                                                cv=LeaveOneOut(), scoring="neg_mean_squared_error")),
    "svr_lin": lambda: make_pipeline(StandardScaler(), GridSearchCV(SVR(kernel="linear", epsilon=0.1),
                                     {"C": [0.1, 1.0, 10.0, 100.0]}, cv=LeaveOneOut(), scoring="neg_mean_squared_error")),
}
_item_cache: dict[tuple, dict[str, pd.DataFrame]] = {}
_mean_cache: dict[tuple, pd.DataFrame] = {}


def skip_estimator(name: str, n: int) -> bool:
    """Too few usable training rows for this estimator: <= its component count, <3 for knn, else <2."""
    if name.startswith("pca") and n <= int(name[3:name.index("_")]):
        return True
    if name.startswith("pls") and n <= int(name[3:]):
        return True
    return n < 3 if name == "knn3_cos" else n < 2


def item_embeddings(cohort: Cohort, variant: str, role: str) -> dict[str, pd.DataFrame]:
    """model -> item-level embeddings (index (bench, item)), restricted to this cohort's items.

    Cached on (cohort.semester, role, variant); `role` distinguishes same-semester cohorts, which
    Cohort itself does not label (read_embeddings is already lru_cached, this just skips re-filtering).
    """
    key = (cohort.semester, role, variant)
    if key not in _item_cache:
        result = {}
        for model in cohort.models.index:
            parts = []
            for bench, item_ids in cohort.items.groupby("bench").item:
                path = cohort.models.loc[model, f"path_{bench}"]
                if isinstance(path, str):
                    table = read_embeddings(embedding_path(path, variant))
                    table = table[table.index.isin(set(item_ids))]
                    table.index = pd.MultiIndex.from_arrays([[bench] * len(table), table.index], names=["bench", "item"])
                    parts.append(table)
            result[model] = pd.concat(parts) if parts else pd.DataFrame()
        _item_cache[key] = result
    return _item_cache[key]


def per_item_mean(task: Task, variant: str) -> pd.DataFrame:
    """Mean embedding per (bench, item) across the task's training models (references+historical), never the targets."""
    key = (task.semester, variant)
    if key not in _mean_cache:
        cohorts = (("references", task.references), ("historical", task.historical))
        frames = [t for role, cohort in cohorts for t in item_embeddings(cohort, variant, role).values() if len(t)]
        combined = pd.concat(frames) if frames else pd.DataFrame()
        _mean_cache[key] = combined.groupby(level=[0, 1]).mean() if len(combined) else combined
    return _mean_cache[key]


def aggregate(task: Task, cohort: Cohort, role: str, variant: str, method: str) -> pd.DataFrame:
    """One row per cohort model: item embeddings reduced by `method`, over `variant` ("both" concatenates)."""
    def base(v: str) -> pd.DataFrame:
        item_level = item_embeddings(cohort, v, role)
        pim = per_item_mean(task, v) if method == "centered" else None
        rows = {m: ((t - pim.reindex(t.index)).mean() if method == "centered" else t.mean()) if len(t) else pd.Series(dtype=np.float32)
                for m, t in item_level.items()}
        table = pd.DataFrame(rows).T.reindex(cohort.models.index)
        return table if table.shape[1] else pd.DataFrame(np.nan, index=cohort.models.index, columns=[0])

    table = pd.concat([base("full").add_prefix("full_"), base("answer").add_prefix("answer_")], axis=1) if variant == "both" else base(variant)
    if method == "renorm":
        norm = np.linalg.norm(table.to_numpy(), axis=1, keepdims=True)
        norm[norm == 0] = 1
        table = pd.DataFrame(table.to_numpy() / norm, index=table.index, columns=table.columns)
    return table


def item_training_rows(task: Task, cohort: Cohort, role: str, variant: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Item-level rows (float array) and the owning model per row; "centered" skips the model mean."""
    def base(v: str) -> tuple[np.ndarray, np.ndarray]:
        item_level = item_embeddings(cohort, v, role)
        pim = per_item_mean(task, v) if method == "centered" else None
        xs, models = [], []
        for model, table in item_level.items():
            if len(table):
                rows = table - pim.reindex(table.index) if method == "centered" else table
                xs.append(rows.to_numpy(dtype=float))
                models += [model] * len(rows)
        return (np.concatenate(xs), np.array(models)) if xs else (np.empty((0, 0)), np.empty(0, dtype=object))

    if variant != "both":
        return base(variant)
    xf, mf = base("full")
    xa, ma = base("answer")
    if len(mf) != len(ma) or not np.array_equal(mf, ma):  # embed.py writes both variants from the same sample rows
        return np.empty((0, 0)), np.empty(0, dtype=object)
    return np.hstack([xf, xa]), mf


def result_row(variant: str, aggregation: str, estimator: str, train: str, cv_mse: float,
                all_score: dict, per_semester: dict) -> dict:
    row = {"variant": variant, "aggregation": aggregation, "estimator": estimator, "train": train,
           "cv_mse": cv_mse, "n": all_score["n"], "mse_all": all_score["mse"]}
    row.update({f"mse_{s}": m for s, m in per_semester.items()})
    return row


def pool_scores(tasks: list[Task], forecasts: list[pd.Series]) -> tuple[dict, dict]:
    """Per-semester and pooled "all" MSE. forecasts[i] shares tasks[i].test.models.index by construction,
    so the pooled pair aligns positionally (score()'s .align short-circuits on already-equal indexes)."""
    per_semester = {t.semester: score(p.reindex(t.truth.index), t.truth)["mse"] for t, p in zip(tasks, forecasts)}
    all_score = score(pd.concat(forecasts), pd.concat(t.truth for t in tasks))
    return per_semester, all_score


def evaluate_config(tasks: list[Task], variant: str, aggregation: str, estimator_name: str, train_role: str) -> dict:
    """One row: model-level fit/predict per task, pooled "all" score, pooled leave-one-model-out cv_mse."""
    forecasts, cv_errors = [], []
    for task in tasks:
        cohort, role = (task.train, "train") if train_role == "train" else (task.references, "references")
        x = aggregate(task, cohort, role, variant, aggregation)
        y = cohort.theta.reindex(x.index)
        finite = np.isfinite(x.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
        x_fit, y_fit, n = x.to_numpy()[finite], y.to_numpy()[finite], int(finite.sum())

        x_test = aggregate(task, task.test, "test", variant, aggregation)
        pred = pd.Series(np.nan, index=task.test.models.index)
        if not skip_estimator(estimator_name, n):
            model = ESTIMATORS[estimator_name]().fit(x_fit, y_fit)
            test_ok = np.isfinite(x_test.to_numpy()).all(axis=1)
            if test_ok.any():
                pred.loc[x_test.index[test_ok]] = np.asarray(model.predict(x_test.to_numpy()[test_ok])).reshape(-1)
            if n >= 3 and not skip_estimator(estimator_name, n - 1):
                cv_pred = np.asarray(cross_val_predict(ESTIMATORS[estimator_name](), x_fit, y_fit, cv=LeaveOneOut())).reshape(-1)
                cv_errors.extend(((cv_pred - y_fit) ** 2).tolist())
        forecasts.append(pred)

    per_semester, all_score = pool_scores(tasks, forecasts)
    cv_mse = float(np.mean(cv_errors)) if cv_errors else np.nan
    return result_row(variant, aggregation, estimator_name, train_role, cv_mse, all_score, per_semester)


def evaluate_item_ridge(tasks: list[Task], variant: str, aggregation: str, train_role: str) -> dict:
    """Ridge on item rows, one fit per task (pooling tasks would leak other tasks' targets into the fit).

    Item rows are first projected on the training rows' top-64 principal components (unsupervised),
    so the leave-one-model-out alpha search stays cheap; test transcripts are predicted and averaged per model.
    """
    forecasts, cv_errors = [], []
    for task in tasks:
        cohort, role = (task.train, "train") if train_role == "train" else (task.references, "references")
        x, models = item_training_rows(task, cohort, role, variant, aggregation)
        pred = pd.Series(np.nan, index=task.test.models.index)
        if len(models):
            y = np.array([cohort.theta[m] for m in models], dtype=float)
            finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
            x, y, models = x[finite], y[finite], models[finite]
        if len(models) and len(np.unique(models)) >= 3:
            pca = PCA(n_components=min(64, len(x) - 1, x.shape[1]), random_state=0).fit(x)
            xp = pca.transform(x)

            def err(xx, yy, alpha, tr, te):  # squared error of the held-out model's mean transcript prediction
                return (Ridge(alpha=alpha).fit(xx[tr], yy[tr]).predict(xx[te]).mean() - yy[te][0]) ** 2

            folds = list(LeaveOneGroupOut().split(xp, y, models))
            for tr, te in folds:  # nested: the held-out model's alpha is chosen on the other models' folds only
                inner = list(LeaveOneGroupOut().split(xp[tr], y[tr], models[tr]))
                alpha = min(ALPHAS, key=lambda a: np.mean([err(xp[tr], y[tr], a, i_tr, i_te) for i_tr, i_te in inner]))
                cv_errors.append(err(xp, y, alpha, tr, te))
            best_alpha = min(ALPHAS, key=lambda a: np.mean([err(xp, y, a, tr, te) for tr, te in folds]))
            fitted = Ridge(alpha=best_alpha).fit(xp, y)
            x_test, test_models = item_training_rows(task, task.test, "test", variant, aggregation)
            if len(test_models):
                keep = np.isfinite(x_test).all(axis=1)
                if keep.any():
                    scores = pd.Series(fitted.predict(pca.transform(x_test[keep])), index=test_models[keep])
                    pred.update(scores.groupby(level=0).mean())
        forecasts.append(pred)

    per_semester, all_score = pool_scores(tasks, forecasts)
    cv_mse = float(np.mean(cv_errors)) if cv_errors else np.nan
    return result_row(variant, aggregation, "item_ridge", train_role, cv_mse, all_score, per_semester)


def baseline_rows(domain: DomainConfig, semesters: list[str]) -> list[dict]:
    """chars / ridge / emb_full / emb_answer, scored the same way, for context (no cv_mse)."""
    rows = []
    for name in BASELINES:
        result = evaluate(domain, PROTOCOLS[name], "", semesters).set_index("semester")
        all_score = {"n": int(result.loc["all", "n"]), "mse": result.loc["all", "mse"]}
        per_semester = {s: result.loc[s, "mse"] for s in semesters}
        rows.append(result_row("baseline", name, "", "", np.nan, all_score, per_semester))
    return rows


def main() -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--semester", nargs="+", default=None, help="default: every semester task")
    parser.add_argument("--variant", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                        help="base variants to load; 'both' is added automatically when both are requested")
    parser.add_argument("--out", type=Path, default=Path("/fast/jyeon/smalljudge_embeddings/hillclimb"))
    parser.add_argument("--quick", action="store_true", help="small sweep for a fast sanity check")
    args = parser.parse_args()

    domain = DomainConfig.from_cli(args.domain)
    semesters = args.semester or semester_names(domain)
    tasks = [Task.from_semester(s, domain) for s in semesters]

    if args.quick:
        variants, aggregations = ["full"], ["mean"]
        estimator_names, train_roles, run_item_ridge = ["ridgecv_std", "pls2", "knn3_cos"], ["train"], False
    else:
        variants = list(args.variant)
        if "full" in variants and "answer" in variants:
            variants.append("both")
        aggregations, estimator_names = ["mean", "centered", "renorm"], list(ESTIMATORS)
        train_roles, run_item_ridge = ["train", "refs"], True

    def try_append(rows: list[dict], label: str, fn, *fn_args) -> None:
        try:
            rows.append(fn(*fn_args))
        except Exception as error:  # noqa: BLE001  one bad config should not kill the sweep
            print(f"skip {label}: {str(error).splitlines()[0] if str(error) else type(error).__name__}")

    started = time.time()
    rows = []
    for variant in variants:
        for aggregation in aggregations:
            for train_role in train_roles:
                for estimator_name in estimator_names:
                    label = f"variant={variant} aggregation={aggregation} estimator={estimator_name} train={train_role}"
                    try_append(rows, label, evaluate_config, tasks, variant, aggregation, estimator_name, train_role)
                if run_item_ridge and aggregation in ("mean", "centered"):
                    label = f"variant={variant} aggregation={aggregation} estimator=item_ridge train={train_role}"
                    try_append(rows, label, evaluate_item_ridge, tasks, variant, aggregation, train_role)
    rows += baseline_rows(domain, semesters)

    table = pd.DataFrame(rows).sort_values("mse_all")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"hillclimb_{domain.name}.csv"
    table.to_csv(out_path, index=False, float_format="%.4f")

    print("\n-- 25 best by mse_all --")
    print(table.head(25).round(3).to_string())
    print("\n-- 10 best by cv_mse (rows scored on every target) --")
    print(table[table.n == table.n.max()].sort_values("cv_mse").head(10).round(3).to_string())
    print("\n-- baselines --")
    print(table[table.variant == "baseline"].round(3).to_string())
    print(f"\n{len(table)} configs evaluated in {time.time() - started:.0f}s; wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
