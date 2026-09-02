"""De-bias pick-rate using the PUBLISHED per-judge bias coefficients (coefficients.csv),
not an in-sample refit. Same figures as gap_curve_debiased.py.

Bias log-odds toward the stronger model = sum_k beta_jk * delta_k_stronger_sd, with
beta from coefficients.csv (parameter 'beta', averaged over folds per judge) and the
seven oriented presentation deltas re-standardised on this frame (the published
standardisation scale isn't recoverable locally -- an approximation of magnitude only).
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gap_curve_debiased import collapse, figure, figure_identity  # reuse plotting

FEATMAP = {  # coefficients.csv 'beta' feature -> left-minus-right column in the frame
    "delta_log_chars": "length_difference",
    "delta_personal_pronouns": "personal_pronouns_difference",
    "delta_hedging": "hedging_difference",
    "delta_backtracking": "backtracking_difference",
    "delta_uncertainty": "uncertainty_difference",
    "delta_confidence": "confidence_difference",
    "delta_questions": "questions_difference",
}
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--coeffs", type=Path,
                   default=HERE.parents[1] / "judge_results/glm_correction/tables/coefficients.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = pd.read_csv(args.frame)
    d = d[d.left_capability != d.right_capability].copy()

    # published per-judge betas, fold-averaged, indexed by normalised judge name
    coef = pd.read_csv(args.coeffs)
    beta = coef[coef.parameter.eq("beta")].groupby(["judge", "feature"])["mean"].mean().unstack("feature")
    beta.index = [norm(j) for j in beta.index]
    beta = beta[~beta.index.duplicated()]  # gemma4-31b vs gemma-4-31b-it collapse to same norm; keep first

    # oriented (stronger-minus-weaker), re-standardised deltas
    sign = np.sign(d.capability_difference.to_numpy())
    feats = list(FEATMAP.values())
    X = np.zeros((len(d), len(feats)))
    for k, col in enumerate(feats):
        v = d[col].to_numpy() * sign
        X[:, k] = (v - v.mean()) / v.std(ddof=0)

    jn = d.judge.map(norm)
    matched = sorted(set(jn) & set(beta.index))
    missing = sorted(set(jn) - set(beta.index))
    B = beta.reindex(jn)[list(FEATMAP.keys())].fillna(0.0).to_numpy()  # (n,7) per-verdict betas
    d["bias_strong"] = (X * B).sum(1)

    d["pick_strong"] = np.where(d.left_capability > d.right_capability, d.chosen_left, 1 - d.chosen_left)
    d["gap"] = (d.left_capability - d.right_capability).abs()
    d["ukey"] = list(zip(np.minimum(d.left_model, d.right_model), np.maximum(d.left_model, d.right_model)))

    print(f"judges matched to published betas: {len(matched)} | unmatched (bias=0): {missing}")
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
