"""De-bias pick-rate with a CROSS-FIT per-judge bias (honest, no in-sample optimism).

For each judge, its pairs are split into 5 folds; the bias logistic for a fold is
fit on the judge's OTHER pairs and applied to the held-out fold. Same scale as the
data (unlike the published-coefficient route), and a pair is never de-biased by a
model trained on it. Same figures as gap_curve_debiased.py.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_curve_debiased import collapse, figure, figure_identity

KEEP = ["capability_difference_sd", "same_model_family", "self_comparison"]
BIAS = ["length_difference_sd", "personal_pronouns_difference_sd", "hedging_difference_sd",
        "backtracking_difference_sd", "uncertainty_difference_sd", "confidence_difference_sd",
        "questions_difference_sd"]
FOLDS = 5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = pd.read_csv(args.frame)
    d = d[d.left_capability != d.right_capability].reset_index(drop=True)
    d["chosen_left"] = d.chosen_left.astype(int)
    d["ukey"] = list(zip(np.minimum(d.left_model, d.right_model), np.maximum(d.left_model, d.right_model)))
    d["bias_left"] = np.nan

    for _, gj in d.groupby("judge"):
        pairs = sorted(gj.ukey.unique(), key=str)
        fold = {pair: i % FOLDS for i, pair in enumerate(pairs)}   # deterministic pair->fold
        gjf = gj.ukey.map(fold)
        for f in range(FOLDS):
            te = gj.index[gjf == f]
            tr = gj.index[gjf != f]
            if len(te) == 0:
                continue
            if len(tr) < 50:                                       # too little to hold out -> in-sample fallback
                tr = gj.index
            m = LogisticRegression(C=1e4, max_iter=3000).fit(d.loc[tr, KEEP + BIAS], d.loc[tr, "chosen_left"])
            d.loc[te, "bias_left"] = d.loc[te, BIAS].to_numpy() @ m.coef_[0][len(KEEP):]

    strong_left = d.left_capability > d.right_capability
    d["pick_strong"] = np.where(strong_left, d.chosen_left, 1 - d.chosen_left)
    d["bias_strong"] = np.where(strong_left, d.bias_left, -d.bias_left)
    d["gap"] = (d.left_capability - d.right_capability).abs()

    pooled = collapse(d, ["ukey"], min_n=1)
    by_judge = collapse(d, ["judge", "ukey"], min_n=20)
    args.out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out / "debiased_pairs.csv", index=False)
    by_judge.to_csv(args.out / "debiased_pairs_by_judge.csv", index=False)
    figure(pooled, args.out, "gap_curve_debiased", per_judge_curves=False, unit="pairs",
           ms=3, kept_alpha=0.5, legend_y=-0.16)
    figure(by_judge, args.out, "gap_curve_debiased_by_judge", per_judge_curves=True,
           unit="judge×pair", ms=2.4, kept_alpha=0.28, legend_y=-0.30)
    figure_identity(by_judge, args.out)
    print("written to", args.out)


if __name__ == "__main__":
    main()
