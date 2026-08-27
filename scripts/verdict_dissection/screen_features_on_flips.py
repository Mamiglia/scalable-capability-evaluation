"""Screen feature tables on the within-model flip set: can they tell which rollout pair the judge preferred?

    uv run python scripts/verdict_dissection/screen_features_on_flips.py <features_root>/raw/derivation.parquet [more tables...]

The flip set: same two models, same item, same shown order; the two rollout pairs got opposite verdicts.
For each judge the contrast is (rollout pair 1 minus rollout pair 0) of the left-minus-right feature
differences, and the label is which rollout pair went left. Reports per-feature sign agreement and an
item-grouped 5-fold logistic accuracy over all columns of each table (and over all tables together).
Chance is 0.5; model identity, item and position cancel by construction.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from judge.dataset import has_reasoning   # noqa: E402
FREEZE = ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66"
GROUP = ["bench", "judge", "item", "model_left", "model_right"]
STRONG_JUDGES = ["o4-mini", "deepseek-v4-flash-0731", "deepseek-r1-0528", "gpt-oss-120b", "gemma-4-31b-it"]


def flip_set():
    pairs = pd.read_parquet(FREEZE / "pairs.parquet",
                            columns=GROUP + ["epoch_left", "epoch_right", "pick_left", "content_hash_l", "content_hash_r"])
    store = pd.read_parquet(FREEZE / "transcript_store.parquet", columns=["content_hash", "transcript"]).drop_duplicates("content_hash")
    answer_only = set(store.content_hash[~store.transcript.map(has_reasoning)])
    pairs = pairs[~pairs.content_hash_l.isin(answer_only) & ~pairs.content_hash_r.isin(answer_only)]
    pairs = pairs[pairs.epoch_left == pairs.epoch_right]
    two = pairs.groupby(GROUP).filter(lambda d: len(d) == 2).sort_values(GROUP + ["epoch_left"])
    first, second = two.groupby(GROUP).nth(0).set_index(GROUP), two.groupby(GROUP).nth(1).set_index(GROUP)
    flipped = first.pick_left != second.pick_left
    first, second = first[flipped], second[flipped]
    return first, second, (second.pick_left > first.pick_left).astype(int).values


def contrast(table, first, second):
    """(pair 1 left−right) − (pair 0 left−right) for every feature column."""
    features = [c for c in table.columns if c not in ("bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash")]
    values = table.drop_duplicates("content_hash").set_index("content_hash")[features].astype(float)
    diff = lambda side: values.reindex(side.content_hash_l).values - values.reindex(side.content_hash_r).values
    return pd.DataFrame(np.nan_to_num(diff(second) - diff(first)), columns=features, index=first.index)


def accuracy(X, y, items):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return cross_val_score(model, X, y, cv=GroupKFold(5), groups=items).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="+", type=Path)
    parser.add_argument("--strong-judges", action="store_true", help="only the five judges that agree with each other")
    args = parser.parse_args()
    first, second, y = flip_set()
    if args.strong_judges:
        keep = first.index.get_level_values("judge").isin(STRONG_JUDGES)
        first, second, y = first[keep], second[keep], y[keep]
    items = first.index.get_level_values("item").astype(str).values
    benches = first.index.get_level_values("bench").str.replace(r"aime\d+", "aime", regex=True).values
    print(f"{len(y)} flips" + (" (strong judges)" if args.strong_judges else ""))
    combined = []
    for path in args.tables:
        X = contrast(pd.read_parquet(path), first, second)
        combined.append(X)
        print(f"\n== {path.name}: all {X.shape[1]} columns, item-grouped CV accuracy {accuracy(X.values, y, items):.3f}")
        for bench in sorted(set(benches)):
            m = benches == bench
            print(f"   {bench:10s} n={m.sum():5d} {accuracy(X.values[m], y[m], items[m]):.3f}")
        rows = []
        for f in X.columns:
            s = np.sign(X[f].values); nonzero = s != 0
            rows.append((f, nonzero.mean(), (s[nonzero] == np.where(y[nonzero] == 1, 1, -1)).mean() if nonzero.any() else np.nan))
        print(pd.DataFrame(rows, columns=["feature", "nonzero", "sign_agreement"]).sort_values("sign_agreement", ascending=False)
              .round(3).to_string(index=False))
    if len(combined) > 1:
        X = pd.concat(combined, axis=1)
        print(f"\n== all tables together ({X.shape[1]} columns): {accuracy(X.values, y, items):.3f}")


if __name__ == "__main__":
    main()
