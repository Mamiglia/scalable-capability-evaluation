"""Per-transcript elo from the pairwise verdicts (spec step 1).

Bradley-Terry as sparse logistic regression: one coefficient per
transcript plus a global slot-1 position term, P(slot1 picked) =
sigmoid(b + elo_1 - elo_2). C is picked by judge-grouped CV on held-out
verdict log-loss. Comparisons stay within an item, so elos are centered
per item and read "better than the other transcripts of this item".

Writes out_dir/elo.csv: text_id, item, model, elo, n_comparisons,
sorted by item then descending elo.

Usage:
  uv run python scripts/analyses/judge_interp/elo.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def text_table(rows) -> pd.DataFrame:
    """(text_id, item) -> author model, read off both slots. One elo
    entity per (text_id, item): a few bare-answer texts recur verbatim
    across items and would otherwise share a coefficient."""
    slot1_major = rows["slot_of_major"] == "1"
    frames = [pd.DataFrame({
        "text_id": rows[f"text_id_slot{s}"],
        "item": rows["item"],
        "model": np.where(slot1_major == (s == 1), rows["major"], rows["minor"]),
    }) for s in (1, 2)]
    texts = pd.concat(frames).drop_duplicates().reset_index(drop=True)
    assert not texts.duplicated(["text_id", "item"]).any(), "conflicting author"
    return texts


def design_matrix(rows, entity_index) -> sp.csr_matrix:
    """One verdict per row: +1 at the slot-1 entity, -1 at the slot-2
    entity, and a constant column for the slot-1 position term."""
    n = len(rows)
    def entities(slot):
        return [entity_index[k] for k in zip(rows[f"text_id_slot{slot}"], rows["item"])]
    cols = np.stack([entities(1), entities(2),
                     np.full(n, len(entity_index))], axis=1).ravel()
    data = np.tile(np.array([1.0, -1.0, 1.0]), n)
    return sp.csr_matrix((data, (np.repeat(np.arange(n), 3), cols)),
                         shape=(n, len(entity_index) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(ROOT / cfg["rows_parquet"])
    texts = text_table(rows)
    entity_index = {k: i for i, k in enumerate(zip(texts["text_id"], texts["item"]))}
    X = design_matrix(rows, entity_index)
    y = rows["y"].to_numpy()
    print(f"{len(rows)} verdicts, {len(texts)} elo entities "
          f"({texts['text_id'].nunique()} unique texts), "
          f"{rows['judge'].nunique()} judges, {texts['item'].nunique()} items")

    bt = cfg["bt"]
    total = {c: 0.0 for c in bt["c_grid"]}
    for train, test in GroupKFold(bt["cv_folds"]).split(X, y, groups=rows["judge"]):
        for c in bt["c_grid"]:
            model = LogisticRegression(C=c, fit_intercept=False,
                                       max_iter=bt["max_iter"]).fit(X[train], y[train])
            total[c] += log_loss(y[test], model.predict_proba(X[test])[:, 1]) * len(test)
    nll = {c: v / len(y) for c, v in total.items()}
    best = min(nll, key=nll.get)
    print(f"held-out nll by C (always-0.5 reference {np.log(2):.4f}):",
          {c: round(v, 4) for c, v in nll.items()}, f"-> C={best}")

    model = LogisticRegression(C=best, fit_intercept=False,
                               max_iter=bt["max_iter"]).fit(X, y)
    texts["elo"] = model.coef_[0][: len(texts)]
    texts["elo"] -= texts.groupby("item")["elo"].transform("mean")
    both = pd.concat([rows[[f"text_id_slot{s}", "item"]]
                      .rename(columns={f"text_id_slot{s}": "text_id"}) for s in (1, 2)])
    counts = both.value_counts()
    texts["n_comparisons"] = [counts[k] for k in zip(texts["text_id"], texts["item"])]

    capability = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    by_model = texts.groupby("model")["elo"].mean()
    rho = spearmanr(by_model, capability[by_model.index]).statistic
    print(f"slot-1 position term {model.coef_[0][-1]:+.3f}, "
          f"within-item elo sd {texts['elo'].std():.3f}, "
          f"spearman(mean elo by author, capability logit) {rho:.3f}")

    texts = texts.sort_values(["item", "elo"], ascending=[True, False])
    texts.to_csv(out / "elo.csv", index=False)
    print(f"-> {out / 'elo.csv'}")


if __name__ == "__main__":
    main()
