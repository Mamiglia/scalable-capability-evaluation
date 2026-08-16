"""Plot the full hierarchical GLM for base and fine-tuned judges."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import t


TERMS = ("intercept", "accuracy_gap", "length_gap", "fine_tuned",
         "fine_tuned:accuracy_gap", "fine_tuned:length_gap")
EFFECTS = (
    ("GPQA accuracy (+10 pp)", 1, 4),
    ("Length (2× longer)", 2, 5),
    ("Position (side 1)", 0, 3),
)

# Same order and definitions as the manuscript's full hierarchical GLM.
REGRESSORS = (
    ("capability_gap_sd", "Capability gap", "Capability"),
    ("truth_side", "Position", "Position"),
    ("truth_side_x_total_length", "Position × length", "Position"),
    ("total_length_sd", "Combined length", "Length"),
    ("length_difference_stronger_sd", "Length gap", "Length"),
    ("same_model_family", "Same family", "Control"),
    ("personal_pronouns_difference_stronger_sd", "Pronouns", "Style"),
    ("hedging_difference_stronger_sd", "Hedging", "Style"),
    ("backtracking_difference_stronger_sd", "Backtracking", "Style"),
    ("uncertainty_difference_stronger_sd", "Uncertainty", "Style"),
    ("confidence_difference_stronger_sd", "Confidence", "Style"),
    ("questions_difference_stronger_sd", "Questions", "Style"),
)


def load_full_posterior(posterior_path, processed_path):
    """Summarize the completed 4-condition, 12-regressor hierarchical fit."""
    import xarray as xr

    with xr.open_dataset(posterior_path, group="posterior") as posterior:
        values = posterior["judge_predictor_effects"].values
    draws = values.reshape(-1, values.shape[-2], values.shape[-1])
    if draws.shape[1:] != (4, len(REGRESSORS)):
        raise ValueError(f"Expected 4 judges × {len(REGRESSORS)} regressors, got {draws.shape[1:]}")

    processed = pd.read_parquet(processed_path, columns=["judge", "item"])
    rows = []
    for judge, base_index, ft_index, condition_names in (
        ("Qwen3-8B", 0, 1, ("qwen_base", "qwen_ft")),
        ("Llama-3.2-3B", 2, 3, ("llama_base", "llama_ft")),
    ):
        sample = processed[processed.judge.isin(condition_names)]
        for index, (predictor, effect, group) in enumerate(REGRESSORS):
            base, ft = draws[:, base_index, index], draws[:, ft_index, index]
            change = ft - base

            def summary(vector):
                return float(vector.mean()), *np.quantile(vector, [0.025, 0.975]).tolist()

            base_m, base_lo, base_hi = summary(base)
            ft_m, ft_lo, ft_hi = summary(ft)
            delta, delta_lo, delta_hi = summary(change)
            tail = min(float((change <= 0).mean()), float((change >= 0).mean()))
            rows.append(dict(
                judge=judge, predictor=predictor, effect=effect, group=group,
                base_estimate=base_m, base_ci_low=base_lo, base_ci_high=base_hi,
                ft_estimate=ft_m, ft_ci_low=ft_lo, ft_ci_high=ft_hi,
                change=delta, change_ci_low=delta_lo, change_ci_high=delta_hi,
                p_change=min(1.0, 2 * tail), n_rows=len(sample), n_items=sample.item.nunique(),
            ))
    return pd.DataFrame(rows)


def prepare_rows(verdicts, pairs, accuracies, fine_tuned):
    """Join verdicts to transcripts and create directional, interpretable gaps."""
    text = pairs[["pair_uid", "order", "text_1", "text_2"]]
    out = verdicts.merge(text, on=["pair_uid", "order"], how="left", validate="one_to_one")
    acc_1, acc_2 = out.model_1.map(accuracies), out.model_2.map(accuracies)
    if out[["text_1", "text_2"]].isna().any().any() or acc_1.isna().any() or acc_2.isna().any():
        raise ValueError("Missing transcript or GPQA accuracy for a verdict row")
    n_1 = out.text_1.str.count(r"\S+").clip(lower=1)
    n_2 = out.text_2.str.count(r"\S+").clip(lower=1)
    return pd.DataFrame({
        "item": out.item,
        "pair_uid": out.pair_uid,
        "choose_1": (out.logit_1 > out.logit_2).astype(int),
        "accuracy_gap": (acc_1 - acc_2) / 0.10,
        "length_gap": np.log2(n_1 / n_2),
        "fine_tuned": int(fine_tuned),
    })


def _design(frame):
    return np.column_stack([
        np.ones(len(frame)), frame.accuracy_gap, frame.length_gap, frame.fine_tuned,
        frame.accuracy_gap * frame.fine_tuned,
        frame.length_gap * frame.fine_tuned,
    ]).astype(float)


def fit_clustered_logit(frame):
    """Unpenalized logistic MLE with CR1 item-clustered sandwich covariance."""
    x, y = _design(frame), frame.choose_1.to_numpy(float)

    def objective(beta):
        eta = x @ beta
        return np.logaddexp(0, eta).sum() - y @ eta

    def gradient(beta):
        return x.T @ (expit(x @ beta) - y)

    result = minimize(objective, np.zeros(x.shape[1]), jac=gradient, method="BFGS",
                      options={"gtol": 1e-8, "maxiter": 1000})
    if not result.success and np.linalg.norm(gradient(result.x), ord=np.inf) > 1e-5:
        raise RuntimeError(f"GLM did not converge: {result.message}")
    beta = result.x
    p = expit(x @ beta)
    bread = np.linalg.pinv(x.T @ ((p * (1 - p))[:, None] * x))
    _, cluster = np.unique(frame.item, return_inverse=True)
    scores = np.zeros((cluster.max() + 1, x.shape[1]))
    np.add.at(scores, cluster, x * (y - p)[:, None])
    groups, n, k = len(scores), len(x), x.shape[1]
    correction = groups / (groups - 1) * (n - 1) / (n - k)
    cov = correction * bread @ (scores.T @ scores) @ bread
    return beta, cov


def summarize_fit(beta, cov, judge, n_rows, n_items):
    """Return base, fine-tuned, and paired-change estimates with 95% CIs."""
    critical = t.ppf(0.975, n_items - 1)
    rows = []
    for effect, base_i, change_i in EFFECTS:
        base = np.eye(len(beta))[base_i]
        change = np.eye(len(beta))[change_i]
        ft = base + change

        def estimate(vector):
            value = float(vector @ beta)
            se = float(np.sqrt(vector @ cov @ vector))
            return value, value - critical * se, value + critical * se, se

        base_m, base_lo, base_hi, _ = estimate(base)
        ft_m, ft_lo, ft_hi, _ = estimate(ft)
        delta, delta_lo, delta_hi, delta_se = estimate(change)
        p_change = float(2 * t.sf(abs(delta / delta_se), n_items - 1)) if delta_se else np.nan
        rows.append(dict(
            judge=judge, effect=effect,
            base_estimate=base_m, base_ci_low=base_lo, base_ci_high=base_hi,
            ft_estimate=ft_m, ft_ci_low=ft_lo, ft_ci_high=ft_hi,
            change=delta, change_ci_low=delta_lo, change_ci_high=delta_hi,
            p_change=p_change, n_rows=n_rows, n_items=n_items,
        ))
    return pd.DataFrame(rows)


def format_results_markdown(table):
    """Format all base, fine-tuned, and paired-change estimates for the manuscript."""
    def interval(row, prefix):
        return (f"{row[f'{prefix}_estimate']:+.3f} "
                f"[{row[f'{prefix}_ci_low']:+.3f}, {row[f'{prefix}_ci_high']:+.3f}]")

    lines = [
        "| Judge | Predictor | Base β [95% CI] | Fine-tuned β [95% CI] | "
        "Δβ [95% CI] | p(Δ) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        delta = f"{row.change:+.3f} [{row.change_ci_low:+.3f}, {row.change_ci_high:+.3f}]"
        lines.append(
            f"| {row.judge} | {row.effect} | {interval(row, 'base')} | "
            f"{interval(row, 'ft')} | {delta} | {row.p_change:.2e} |")
    samples = table[["judge", "n_rows", "n_items"]].drop_duplicates()
    sample_note = "; ".join(
        f"{row.judge}: {int(row.n_rows):,} verdicts across "
        f"{int(row.n_items)} held-out items" for _, row in samples.iterrows())
    lines.extend([
        "",
        "Hierarchical Bayesian binomial GLM; equal-tail 95% credible intervals. "
        "Continuous predictors are z-scored; style predictors are stronger−weaker "
        "lexicon-hit rates per 1,000 words. p(Δ) is twice the smaller posterior tail mass.",
        "",
        sample_note + ".",
    ])
    return "\n".join(lines) + "\n"


def plot_coefficients(table, out_stub):
    """Compact coefficient plot with paired base-to-fine-tuned model estimates."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyses"))
    import matplotlib.pyplot as plt
    from plots import INK, MUTED, save, style

    fig, ax = plt.subplots(figsize=(4.4, 2.65), layout="constrained")
    centres = np.arange(len(REGRESSORS))
    indexed = table.set_index(["effect", "judge"])
    model_colours = {"Qwen3-8B": "#2a78d6", "Llama-3.2-3B": "#4a3aa7"}
    offsets = {"Qwen3-8B": (-0.24, -0.08), "Llama-3.2-3B": (0.08, 0.24)}
    for judge in table.judge.drop_duplicates():
        colour = model_colours[judge]
        for x, (_, effect, _) in zip(centres, REGRESSORS, strict=True):
            row = indexed.loc[(effect, judge)]
            base_x, ft_x = x + np.asarray(offsets[judge])
            ax.plot([base_x, ft_x], [row.base_estimate, row.ft_estimate],
                    color=colour, lw=1, zorder=2)
            for point_x, prefix, face in ((base_x, "base", "white"),
                                          (ft_x, "ft", colour)):
                value, low, high = (row[f"{prefix}_estimate"],
                                    row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"])
                ax.errorbar(point_x, value, yerr=[[value - low], [high - value]],
                            fmt="o", color=colour, mfc=face, mec=colour, mew=0.8,
                            ms=3.4, elinewidth=1, capsize=0, barsabove=True, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(-0.5, len(REGRESSORS) - 0.5)
    ax.set_xticks(centres, [label for _, label, _ in REGRESSORS], rotation=90)
    ax.set_ylim(-3.5, 3.5)
    ax.set_yticks([-3.5, -2, 0, 2, 3.5])
    ax.set_ylabel("Coefficient (log-odds)")
    style(ax)
    ax.tick_params(axis="x", labelrotation=90)
    groups = [group for _, _, group in REGRESSORS]
    starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    ends = starts[1:] + [len(groups)]
    for start, end in zip(starts, ends, strict=True):
        ax.text((start + end - 1) / 2, 1.015, groups[start], color=MUTED, fontsize=6.5,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
        if start:
            ax.axvline(start - 0.5, color="#dddddd", lw=0.6, zorder=0)
    handles = []
    for judge, short in (("Qwen3-8B", "Qwen"), ("Llama-3.2-3B", "Llama")):
        colour = model_colours[judge]
        handles.extend([
            plt.Line2D([], [], marker="o", ls="", mfc="white", mec=colour,
                       color=colour, label=f"{short} base"),
            plt.Line2D([], [], marker="o", ls="", mfc=colour, mec=colour,
                       color=colour, label=f"{short} FT"),
        ])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="outside lower center",
               fontsize=6.5,
               handlelength=0.8, handletextpad=0.2, columnspacing=0.55)
    save(fig, out_stub)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-root", default="analysis/hibayes/fine_tuning/full",
        help="Completed full hierarchical fine-tuning GLM",
    )
    parser.add_argument("--out-stub", default="judge_results/judge_ft/length_bias_glm")
    args = parser.parse_args()
    fit_root, out_stub = Path(args.fit_root), Path(args.out_stub)
    posterior = fit_root / "models/judge_feature_logistic_model_full/inference_data.nc"
    table = load_full_posterior(posterior, fit_root / "processed_data.parquet")
    out_stub.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_stub.with_suffix(".csv"), index=False)
    out_stub.with_suffix(".md").write_text(format_results_markdown(table))
    plot_coefficients(table, out_stub)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {out_stub}.csv/.md/.png/.svg")


if __name__ == "__main__":
    main()
