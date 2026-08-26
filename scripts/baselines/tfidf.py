"""TF-IDF baseline for the stronger-model task: logistic regression on tfidf(left) - tfidf(right).

    uv run python scripts/baselines/tfidf.py <output_dir>

Same rows and held-out designs as the stronger-model GLM (scripts/verdict_dissection/glm/prepare.py):
one row per distinct transcript pair, ties in benchmark accuracy dropped, a pair held out when
either transcript is, dropped when only one side is unseen; AIME years pooled. The vectoriser is fitted on the training transcripts only (so a held-out
model's vocabulary is never seen); one logistic regression per benchmark with no intercept, since
the differences are antisymmetric; C chosen on inner folds grouped by item. Text is the assistant
spans of the rendered transcript, as for the surface features.

Writes <output_dir>/stronger_model_from_pair/tfidf/<design>/ in the GLM fits' layout (see pairs.py).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from pairs import designs, held_out, load_config, load_pairs, load_transcripts, transcript_rows, write_design

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
VECTORISER = dict(ngram_range=(1, 2), min_df=5, max_features=50_000, sublinear_tf=True, dtype=np.float32)


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


def run_design(status, seen, pairs, transcripts, left, right):
    """Held-out predictions for one design, fitted per benchmark. `status` is each pair's train / test /
    drop, `seen` marks the transcripts the vectoriser may see."""
    out = []
    for bench, block in pairs[status != "drop"].groupby("bench"):
        test = status[block.index] == "test"
        docs = (transcripts.bench == bench).to_numpy()
        if not test.any() or test.all():                    # nothing to predict, or nothing to fit on
            continue
        vectoriser = TfidfVectorizer(**VECTORISER).fit(transcripts.text[docs & seen])
        local = pd.Series(np.arange(docs.sum()), index=transcripts.index[docs])
        x = vectoriser.transform(transcripts.text[docs])
        diff = x[local[left[block.index]].to_numpy()] - x[local[right[block.index]].to_numpy()]
        train, y = ~test, block.left_stronger.to_numpy()
        predicted, c = fit_predict(diff[train], y[train], block.item.to_numpy()[train], diff[~train])
        print(f"  {bench}: vocabulary {len(vectoriser.vocabulary_)}, C {c}, "
              f"accuracy {((predicted > 0.5) == y[~train]).mean():.3f} on {int((~train).sum())} held-out pairs", flush=True)
        out.append(block.loc[~train, ["bench", "item", "model_left", "model_right"]]
                   .assign(observed=y[~train], predicted=predicted))
    return pd.concat(out, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--designs", nargs="+", default=list(designs()), choices=list(designs()))
    args = parser.parse_args()
    config = load_config()
    pairs, transcripts = load_pairs(config), load_transcripts(config)
    left, right = transcript_rows(pairs, transcripts, "left"), transcript_rows(pairs, transcripts, "right")
    print(f"{len(pairs)} pairs, {len(transcripts)} transcripts")
    for design in args.designs:
        print(design, flush=True)
        status = held_out(design, pairs.bench, pairs.item, pairs.model_left, pairs.model_right)
        seen = held_out(design, transcripts.bench, transcripts.item, transcripts.model) == "train"
        scores = write_design(args.output_dir, "tfidf", design, run_design(status, seen, pairs, transcripts, left, right))
        print(f"  all: accuracy {scores.accuracy.iloc[0]:.3f}, log loss {scores.log_loss.iloc[0]:.3f}")


if __name__ == "__main__":
    main()
