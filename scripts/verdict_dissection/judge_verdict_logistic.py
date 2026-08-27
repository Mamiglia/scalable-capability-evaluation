"""Plain logistic regression for judge verdicts from feature-table differences, with and without extra blocks.

    uv run python scripts/verdict_dissection/judge_verdict_logistic.py --base raw/surface-textual.parquet --extra raw/clause-network.parquet:^topo_ ...

One fit per judge x benchmark per design (item folds: models seen; strict "new models and items"
folds from glm/splits.yaml). Features are left-minus-right differences of each table's columns,
z-scaled on the training rows; no intercept per judge x benchmark beyond the fitted one (position
bias). Answer-only transcripts are excluded. Prints held-out accuracy per block set, per benchmark,
micro-averaged over the held-out rows of the folds.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection/glm"))
from judge.dataset import has_reasoning   # noqa: E402
from prepare import held_out, SPLITS      # noqa: E402
FREEZE = ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66"
FEATURES_ROOT = Path("/Volumes/T7/judge-dissection/transcript-features")
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
STRICT = [f"new-models-and-items-{r}" for r in range(SPLITS["item_folds"])]


def load_pairs():
    pairs = pd.read_parquet(FREEZE / "pairs.parquet",
                            columns=["bench", "judge", "item", "model_left", "model_right", "pick_left",
                                     "content_hash_l", "content_hash_r"])
    store = pd.read_parquet(FREEZE / "transcript_store.parquet", columns=["content_hash", "transcript"]).drop_duplicates("content_hash")
    answer_only = set(store.content_hash[~store.transcript.map(has_reasoning)])
    pairs = pairs[~pairs.content_hash_l.isin(answer_only) & ~pairs.content_hash_r.isin(answer_only)].reset_index(drop=True)
    pairs["item"] = np.where(pairs.bench.str.startswith("aime"), pairs.bench.str[4:] + "/" + pairs.item, pairs.item)
    pairs["bench"] = pairs.bench.str.replace(r"aime\d+", "aime", regex=True)
    return pairs


def block(spec, pairs):
    """'table.parquet[:regex]' -> left-minus-right differences of the matching columns."""
    path, _, pattern = spec.partition(":")
    table = pd.read_parquet(FEATURES_ROOT / path).drop(columns=["transcript_key"], errors="ignore")
    columns = [c for c in table.columns if c not in KEYS and (not pattern or re.search(pattern, c))]
    values = table.drop_duplicates("content_hash").set_index("content_hash")[columns].astype(float)
    diff = values.reindex(pairs.content_hash_l).values - values.reindex(pairs.content_hash_r).values
    return pd.DataFrame(np.nan_to_num(diff), columns=[f"{Path(path).stem}:{c}" for c in columns])


def fit_predict(X, y, train, test):
    scaler = StandardScaler().fit(X[train])
    model = LogisticRegression(C=0.1, max_iter=3000).fit(scaler.transform(X[train]), y[train])
    return (model.predict(scaler.transform(X[test])) == y[test]).astype(float)


def evaluate(pairs, X, design):
    """Correct/incorrect per held-out row, over the folds of one design family."""
    hits = pd.Series(np.nan, index=pairs.index)
    for (bench, judge), group in pairs.groupby(["bench", "judge"]):
        idx = group.index.to_numpy()
        y = group.pick_left.to_numpy()
        if design == "item-folds":
            for train, test in GroupKFold(SPLITS["item_folds"]).split(idx, y, group.item.to_numpy()):
                hits.iloc[idx[test]] = fit_predict(X[idx], y, train, test)
        else:
            for fold in STRICT:
                role = held_out(fold, group.bench, group.item, group.model_left, group.model_right)
                train, test = np.where(role == "train")[0], np.where(role == "test")[0]
                if len(test) and len(set(y[train])) == 2:
                    hits.iloc[idx[test]] = fit_predict(X[idx], y, train, test)   # rows can be held out in >1 fold: last wins
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", default=["raw/surface-textual.parquet"])
    parser.add_argument("--extra", nargs="+", default=[], help="blocks added one at a time on top of the base, then all together")
    parser.add_argument("--designs", nargs="+", default=["item-folds", "strict"])
    args = parser.parse_args()
    pairs = load_pairs()
    base = pd.concat([block(s, pairs) for s in args.base], axis=1)
    extras = {s: block(s, pairs) for s in args.extra}
    sets = {"base": base} | {f"base + {s}": pd.concat([base, x], axis=1) for s, x in extras.items()}
    if len(extras) > 1:
        sets["base + all extras"] = pd.concat([base, *extras.values()], axis=1)
    for design in args.designs:
        print(f"\n== {design}")
        rows = []
        for name, X in sets.items():
            hits = evaluate(pairs, X.values, design)
            row = {"features": name, "columns": X.shape[1], "mean": np.nan}
            row |= {b: hits[pairs.bench == b].mean() for b in sorted(pairs.bench.unique())}
            row["mean"] = np.mean([row[b] for b in sorted(pairs.bench.unique())])
            rows.append(row)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
