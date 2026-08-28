"""TF-IDF model of each judge's verdict: logistic regression on tfidf(left) - tfidf(right) plus a
position intercept, one model per judge x benchmark.

    uv run python scripts/baselines/tfidf_verdict.py <output_dir> [--designs ...]

Rows are the verdict draws of the judge GLM (one per judge x shown order x rollout pair); held-out
designs as glm/prepare.py. The vectoriser is fitted on the training transcripts of the benchmark
only; C chosen on inner folds grouped by item. Writes
<output_dir>/judge_verdict_from_pair/tfidf/<design>/{predictions,held_out}_judge_verdict_from_pair.csv
and <output_dir>/judge_verdict_from_pair/tfidf/ngrams_<design>.csv (the 40 largest coefficients
either way per judge x benchmark, so the direction the judge rewards can be read off).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from pairs import PAIR_IDENTITY, designs, held_out, load_config, load_transcripts, transcript_rows
from tfidf import C_GRID, VECTORISER

MODEL = "judge_verdict_from_pair"
TOP_NGRAMS = 40


def load_verdicts(config):
    columns = PAIR_IDENTITY + ["judge", "pick_left"]
    return pd.read_parquet(Path(config["output_root"]) / "pair.parquet", columns=columns).reset_index(drop=True)


def fit_predict(x_train, y_train, groups_train, x_test, inner_folds=3, seed=0):
    """P(pick left) for the test rows, intercept free (position bias); C chosen on inner folds grouped by item."""
    best, best_score = None, -1.0
    folds = list(GroupKFold(inner_folds, shuffle=True, random_state=seed).split(x_train, y_train, groups_train))
    for c in C_GRID:
        score = 0.0
        for fit, check in folds:
            model = LogisticRegression(C=c, max_iter=1000).fit(x_train[fit], y_train[fit])
            score += model.score(x_train[check], y_train[check])
        if score > best_score:
            best, best_score = c, score
    model = LogisticRegression(C=best, max_iter=1000).fit(x_train, y_train)
    return model, model.predict_proba(x_test)[:, 1], best


def run_design(status, seen, verdicts, transcripts, left, right):
    """Held-out predictions for one design, one logistic model per judge x benchmark."""
    out, ngrams = [], []
    for bench, block in verdicts[status != "drop"].groupby("bench"):
        test = status[block.index] == "test"
        docs = (transcripts.bench == bench).to_numpy()
        if not test.any() or test.all():
            continue
        vectoriser = TfidfVectorizer(**VECTORISER).fit(transcripts.text[docs & seen])
        names = vectoriser.get_feature_names_out()
        local = pd.Series(np.arange(docs.sum()), index=transcripts.index[docs])
        x = vectoriser.transform(transcripts.text[docs])
        diff = sp.csr_matrix(x[local[left[block.index]].to_numpy()] - x[local[right[block.index]].to_numpy()])
        for judge, rows in block.groupby("judge"):
            here = block.index.get_indexer(rows.index)
            train, y = ~test[here], rows.pick_left.to_numpy()
            if train.all() or not train.any():
                continue
            model, predicted, c = fit_predict(diff[here][train], y[train], rows.item.to_numpy()[train], diff[here][~train])
            print(f"  {bench} {judge}: C {c}, accuracy {((predicted > 0.5) == y[~train]).mean():.3f} "
                  f"on {int((~train).sum())} held-out verdicts", flush=True)
            out.append(rows.loc[~train, ["bench", "item", "judge", "model_left", "model_right"]]
                       .assign(observed=y[~train], predicted=predicted))
            coef = model.coef_.ravel()
            top = np.argsort(-np.abs(coef))[:TOP_NGRAMS]
            ngrams.append(pd.DataFrame({"bench": bench, "judge": judge, "ngram": names[top], "coefficient": coef[top]}))
    return pd.concat(out, ignore_index=True), pd.concat(ngrams, ignore_index=True)


def report(predictions):
    rows = []
    for name, group in [("all", predictions)] + list(predictions.groupby("judge")):
        p, y = group.predicted.clip(1e-6, 1 - 1e-6), group.observed
        rows.append({"group": name, "n": len(group), "accuracy": float(((p > 0.5) == y).mean()),
                     "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--designs", nargs="+", default=[d for d in designs() if d.startswith("new-models-and-items")],
                        choices=list(designs()))
    args = parser.parse_args()
    config = load_config()
    verdicts, transcripts = load_verdicts(config), load_transcripts(config)
    left, right = transcript_rows(verdicts, transcripts, "left"), transcript_rows(verdicts, transcripts, "right")
    print(f"{len(verdicts)} verdicts, {len(transcripts)} transcripts")
    for design in args.designs:
        print(design, flush=True)
        status = held_out(design, verdicts.bench, verdicts.item, verdicts.model_left, verdicts.model_right)
        seen = held_out(design, transcripts.bench, transcripts.item, transcripts.model) == "train"
        predictions, ngrams = run_design(status, seen, verdicts, transcripts, left, right)
        design_dir = args.output_dir / MODEL / "tfidf" / design
        design_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(design_dir / f"predictions_{MODEL}.csv", index=False)
        scores = report(predictions)
        scores.to_csv(design_dir / f"held_out_{MODEL}.csv", index=False)
        ngrams.to_csv(design_dir.parent / f"ngrams_{design}.csv", index=False)
        print(f"  all: accuracy {scores.accuracy.iloc[0]:.3f}, log loss {scores.log_loss.iloc[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
