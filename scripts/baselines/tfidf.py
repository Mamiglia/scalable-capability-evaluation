"""TF-IDF baseline for the stronger-model task: logistic regression on tfidf(left) - tfidf(right).

    uv run python scripts/baselines/tfidf.py <output_dir>

Same rows and held-out designs as the stronger-model GLM (scripts/verdict_dissection/glm/prepare.py):
one row per distinct transcript pair, ties in benchmark accuracy dropped, a pair held out when
either transcript is. The vectoriser is fitted on the training transcripts only (so a held-out
model's vocabulary is never seen); one logistic regression per benchmark with no intercept, since
the differences are antisymmetric; C chosen on inner folds grouped by item. Text is the assistant
spans of the rendered transcript, as for the surface features.

Writes <output_dir>/stronger_model_from_pair/tfidf/<design>/predictions_stronger_model_from_pair.csv
and held_out_stronger_model_from_pair.csv in the GLM fits' layout, so glm/collate.py and
glm/compare_to_judges.py read them unchanged (pass <output_dir> as the pair fits).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection/glm"))
from features_basic import _assistant_text          # noqa: E402
from prepare import PAIR_IDENTITY, designs, held_out  # noqa: E402

MODEL = "stronger_model_from_pair"
TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
VECTORISER = dict(ngram_range=(1, 2), min_df=5, max_features=50_000, sublinear_tf=True, dtype=np.float32)


def load_pairs(config):
    """One row per distinct transcript pair; left_stronger by the models' accuracy over all rollouts."""
    pairs = pd.read_parquet(Path(config["output_root"]) / "pair.parquet", columns=PAIR_IDENTITY).drop_duplicates()
    accuracy = pd.read_parquet(Path(config["features_root"]) / "raw/model-benchmark-accuracy.parquet",
                               columns=["bench", "model", "accuracy"])
    for side in ("left", "right"):
        pairs = pairs.merge(accuracy.rename(columns={"model": f"model_{side}", "accuracy": f"accuracy_{side}"}),
                            on=["bench", f"model_{side}"], how="left", validate="many_to_one")
    if pairs[["accuracy_left", "accuracy_right"]].isna().any().any():
        raise ValueError("a model is missing its benchmark accuracy")
    pairs = pairs[pairs.accuracy_left != pairs.accuracy_right]
    return pairs.assign(left_stronger=(pairs.accuracy_left > pairs.accuracy_right).astype(int)).reset_index(drop=True)


def load_transcripts(config):
    """Assistant text of every canonical transcript, one per transcript key."""
    identity = pd.read_parquet(Path(config["features_root"]) / "raw/identity.parquet",
                               columns=TRANSCRIPT_KEYS + ["sample_epoch", "transcript", "is_canonical"])
    identity = identity[identity.is_canonical].sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    return (identity[TRANSCRIPT_KEYS].assign(text=identity.transcript.map(_assistant_text))
            .reset_index(drop=True))


def transcript_rows(pairs, transcripts, side):
    """Row of `transcripts` holding each pair's left or right transcript."""
    keys = {k: f"{k}_{side}" if k in ("model", "epoch", "log") else k for k in TRANSCRIPT_KEYS}
    index = transcripts[TRANSCRIPT_KEYS].reset_index().rename(columns=keys)
    matched = pairs.merge(index, on=list(keys.values()), how="left", validate="many_to_one")
    if matched["index"].isna().any():
        raise ValueError(f"{int(matched['index'].isna().sum())} pairs have no {side} transcript")
    return matched["index"].to_numpy(int)


def fit_predict(x_train, y_train, groups_train, x_test, inner_folds=3, seed=0):
    """P(left stronger) for the test rows. No intercept; C chosen on inner folds grouped by item."""
    best, best_score = None, -1.0
    for c in C_GRID:
        score = 0.0
        for fit, check in GroupKFold(inner_folds, shuffle=True, random_state=seed).split(x_train, y_train, groups_train):
            model = LogisticRegression(C=c, fit_intercept=False, max_iter=1000).fit(x_train[fit], y_train[fit])
            score += model.score(x_train[check], y_train[check])
        if score > best_score:
            best, best_score = c, score
    model = LogisticRegression(C=best, fit_intercept=False, max_iter=1000).fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1], best


def run_design(design, pairs, transcripts, left, right):
    """Held-out predictions for one design, fitted per benchmark."""
    test = held_out(design, pairs.bench, pairs.item, pairs.model_left, pairs.model_right)
    seen = ~held_out(design, transcripts.bench, transcripts.item, transcripts.model)
    out = []
    for bench, block in pairs.groupby("bench"):
        docs = (transcripts.bench == bench).to_numpy()
        if not test[block.index].any():
            continue
        vectoriser = TfidfVectorizer(**VECTORISER).fit(transcripts.text[docs & seen])
        local = pd.Series(np.arange(docs.sum()), index=transcripts.index[docs])
        x = vectoriser.transform(transcripts.text[docs])
        diff = x[local[left[block.index]].to_numpy()] - x[local[right[block.index]].to_numpy()]
        train, y = ~test[block.index], block.left_stronger.to_numpy()
        predicted, c = fit_predict(diff[train], y[train], block.item.to_numpy()[train], diff[~train])
        print(f"  {bench}: vocabulary {len(vectoriser.vocabulary_)}, C {c}, "
              f"accuracy {((predicted > 0.5) == y[~train]).mean():.3f} on {int((~train).sum())} held-out pairs", flush=True)
        out.append(block.loc[~train, ["bench", "item", "model_left", "model_right"]]
                   .assign(observed=y[~train], predicted=predicted))
    return pd.concat(out, ignore_index=True)


def report(predictions):
    """Accuracy and log loss overall and per benchmark, as glm/evaluate.py writes them."""
    rows = []
    for name, group in [("all", predictions)] + list(predictions.groupby("bench")):
        p, y = group.predicted.clip(1e-6, 1 - 1e-6), group.observed
        rows.append({"group": name, "n": len(group), "accuracy": float(((p > 0.5) == y).mean()),
                     "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--designs", nargs="+", default=list(designs()), choices=list(designs()))
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    pairs, transcripts = load_pairs(config), load_transcripts(config)
    left, right = transcript_rows(pairs, transcripts, "left"), transcript_rows(pairs, transcripts, "right")
    print(f"{len(pairs)} pairs, {len(transcripts)} transcripts")
    for design in args.designs:
        print(design, flush=True)
        predictions = run_design(design, pairs, transcripts, left, right)
        design_dir = args.output_dir / MODEL / "tfidf" / design
        design_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(design_dir / f"predictions_{MODEL}.csv", index=False)
        scores = report(predictions)
        scores.to_csv(design_dir / f"held_out_{MODEL}.csv", index=False)
        print(f"  all: accuracy {scores.accuracy.iloc[0]:.3f}, log loss {scores.log_loss.iloc[0]:.3f}")


if __name__ == "__main__":
    main()
