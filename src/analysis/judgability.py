"""Which train items are easy to judge: per-item discrimination of the pairwise verdicts.

Reference-vs-reference verdicts only (theta* known on both sides), pooled over judges and
semesters. Per (domain, bench, item) a 2PL-style logistic fit
    P(pick_left) = sigmoid(a_i * (theta*_left - theta*_right) + b_i)
where `a_i` (slope) is how sharply the judge's pick tracks the true gap and `accuracy` is the
plain share of verdicts that pick the stronger reference. Joined to the frozen item table
(guess, a, b = the capability IRT's discrimination and difficulty) and the references'
first-attempt correctness on the item.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from analysis import loading
from capability.task import Cohort
from configs.domain import DomainConfig
from protocols.transcript_features.features import FEATURES
from protocols.transcript_features.reader import features_path, read_features
from reference import tables


def reference_verdicts(dom: DomainConfig) -> pd.DataFrame:
    """Every pairwise verdict between two references on the frozen train items, all
    semesters and roster judges: bench, item, judge, semester, gap (theta*_left - theta*_right),
    pick_left."""
    out = []
    for task in loading.tasks(dom):
        theta = task.references.models.theta
        for run in loading.pairwise_runs(dom, task):
            v = run.table.merge(task.references.items[["bench", "item"]])
            v = v[v.model_left.isin(theta.index) & v.model_right.isin(theta.index)]
            v = v.sort_values("imported", kind="stable").drop_duplicates(["bench", "item", "model_left", "model_right"])
            gap = theta.reindex(v.model_left).to_numpy() - theta.reindex(v.model_right).to_numpy()
            out.append(v.assign(semester=task.semester, judge=run.judge, condition=run.condition, gap=gap)[
                ["bench", "item", "judge", "semester", "condition", "model_left", "model_right", "gap", "pick_left"]])
    return pd.concat(out, ignore_index=True)


def transcript_features(cohort: Cohort, reasoning: str) -> pd.DataFrame:
    """FEATURES of every correct first-attempt transcript of the cohort's models on its items,
    in the `reasoning` view ("visible" or "hidden"): model, bench, item, FEATURES."""
    frames = []
    for model in cohort.models.index:
        for bench, item_ids in cohort.items.groupby("bench").item:
            path = cohort.models.loc[model, f"path_{bench}"]
            if isinstance(path, str):
                f = read_features(features_path(path), set(item_ids), reasoning=reasoning)[FEATURES]
                frames.append(f.reset_index().assign(model=model, bench=bench))
    return pd.concat(frames, ignore_index=True)


def with_feature_gaps(verdicts: pd.DataFrame, dom: DomainConfig) -> pd.DataFrame:
    """verdicts plus one `d_<feature>` column per FEATURES entry: left minus right transcript,
    in the view the judge saw (reasoning-shown runs -> visible, reasoning-hidden -> hidden)."""
    out = []
    for semester, v in verdicts.groupby("semester"):
        task = next(t for t in loading.tasks(dom, [semester]))
        for view in ("visible", "hidden"):
            part = v[v.condition.str.startswith("reasoning-shown") == (view == "visible")]
            if part.empty:
                continue
            f = transcript_features(task.references, view).set_index(["model", "bench", "item"])[FEATURES]
            left = f.reindex(pd.MultiIndex.from_frame(part[["model_left", "bench", "item"]])).to_numpy()
            right = f.reindex(pd.MultiIndex.from_frame(part[["model_right", "bench", "item"]])).to_numpy()
            out.append(part.assign(**{f"d_{c}": left[:, i] - right[:, i] for i, c in enumerate(FEATURES)}))
    return pd.concat(out).sort_index()


def item_fit(verdicts: pd.DataFrame) -> pd.DataFrame:
    """One row per (bench, item): n verdicts, slope `a_judge`, intercept `bias`, `accuracy`
    (share picking the stronger side, ties on gap excluded), `mean_gap` (mean |gap| seen)."""
    rows = []
    for (bench, item), g in verdicts.groupby(["bench", "item"]):
        y, x = g.pick_left.to_numpy(), g.gap.to_numpy()
        row = dict(bench=bench, item=item, n=len(g), mean_gap=float(np.abs(x).mean()),
                   accuracy=float((np.sign(x) == np.sign(y - 0.5))[x != 0].mean()))
        if y.min() != y.max():
            # ponytail: light ridge (C=10) keeps separated items finite; unpenalised if that ever matters
            m = LogisticRegression(C=10.0).fit(x[:, None], y)
            row.update(a_judge=float(m.coef_[0, 0]), bias=float(m.intercept_[0]))
        rows.append(row)
    return pd.DataFrame(rows)


def item_covariates(dom: DomainConfig) -> pd.DataFrame:
    """Per (bench, item): guess, a, b from the frozen table and `model_acc`, the mean
    first-attempt correctness over every model in the frozen responses."""
    items = tables.load(dom)[1][["bench", "item", "guess", "a", "b"]]
    resp = tables.load_responses(dom)
    acc = resp[resp.repeat == 0].groupby(["bench", "item"]).correct.mean().rename("model_acc").reset_index()
    return items.merge(acc, how="left")


def table(dom: DomainConfig) -> pd.DataFrame:
    """item_fit joined to item_covariates, with a `domain` column."""
    return item_fit(reference_verdicts(dom)).merge(item_covariates(dom), how="left").assign(domain=dom.name)
