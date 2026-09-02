"""Corrected per-bin predictions and partition figures beside the original methods.

Corrected theta comes from the partition-calibrated fold artifacts; per-bin
predictions use the same out-of-sample linear map as the halves pipeline
(anchor theta -> anchor bin accuracy, applied to held-out targets).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bias_crossfit import calibrate_to_hard
from correct_biases import read_fold_artifact

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#7f7f7f"
SCHEMES = {
    "tertile": ("T2", "T3"),
    "quartile": ("Q2", "Q3", "Q4"),
    "quintile": ("P2", "P3", "P4", "P5"),
}


def corrected_theta(scheme_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        artifact = read_fold_artifact(
            scheme_dir / "folds" / "fixed_3pl" / f"fold-{fold}" / "corrected.npz",
        )
        predictions = artifact["predictions"]
        corrected = predictions[predictions["method"].eq("corrected")]
        rows.append(corrected[["model", "predictor"]].assign(fold=fold))
    return pd.concat(rows, ignore_index=True)


def bin_accuracy(responses: pd.DataFrame, items: pd.DataFrame,
                 scheme: str, bins: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for target_bin in [*bins, f"{bins[0]}–{bins[-1]}"]:
        members = set(bins) if "–" in target_bin else {target_bin}
        bin_items = set(items.loc[items[scheme].isin(members), "item"])
        scored = responses[responses["item"].isin(bin_items)]
        accuracy = scored.groupby("model")["correct"].mean()
        frames.append(pd.DataFrame({
            "model": accuracy.index, "target_bin": target_bin,
            "observed": accuracy.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def corrected_predictions(theta: pd.DataFrame, anchors: pd.DataFrame,
                          observed: pd.DataFrame) -> pd.DataFrame:
    anchor_theta = anchors.set_index("model")["theta"]
    rows = []
    for target_bin, bin_rows in observed.groupby("target_bin"):
        bin_observed = bin_rows.set_index("model")["observed"]
        for fold, fold_rows in theta.groupby("fold"):
            targets = set(fold_rows["model"])
            train = sorted(set(anchor_theta.index) & set(bin_observed.index) - targets)
            predicted = calibrate_to_hard(
                anchor_theta.loc[train], bin_observed.loc[train],
                fold_rows["predictor"],
            )
            rows.append(pd.DataFrame({
                "model": fold_rows["model"].to_numpy(),
                "target_bin": target_bin,
                "method": "corrected",
                "prediction": predicted,
                "observed": fold_rows["model"].map(bin_observed).to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)


def original_predictions(old: pd.DataFrame, scheme: str) -> pd.DataFrame:
    sel = old[
        old["irt_model"].eq("fixed_3pl") & old["fit"].eq("gpqa")
        & old["partition"].eq(scheme) & old["judge"].eq("consensus")
        & old["method"].isin(["elo", "easy_accuracy"]) & old["reason"].isna()
    ]
    return sel[["model", "target_bin", "method", "prediction", "observed"]]


def scatter_grid(table: pd.DataFrame, scheme: str, out: Path) -> None:
    bins = [b for b in table["target_bin"].unique() if "–" not in b]
    bins = sorted(bins) + [b for b in table["target_bin"].unique() if "–" in b]
    methods = [("elo", "Elo", GREY), ("easy_accuracy", "Easy accuracy", ORANGE),
               ("corrected", "Corrected", BLUE)]
    fig, axes = plt.subplots(
        len(bins), 3, figsize=(4.6, 1.35 * len(bins) + 0.5),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, target_bin in enumerate(bins):
        for column, (method, label, colour) in enumerate(methods):
            ax = axes[row][column]
            cell = table[table["target_bin"].eq(target_bin)
                         & table["method"].eq(method)]
            ax.plot([0, 1], [0, 1], "--", lw=0.8, c="grey", zorder=0)
            ax.plot(cell["observed"], cell["prediction"], "o", ms=2.5,
                    c=colour, mec="none", alpha=0.75)
            if row == 0:
                ax.set_title(f"{label}\n(n={cell['model'].nunique()})",
                             loc="left", fontsize=6.5, color="dimgrey")
            if column == 0:
                ax.set_ylabel(target_bin, rotation=0, labelpad=12, color="dimgrey")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
    fig.supxlabel("Observed accuracy", fontsize=8)
    fig.supylabel("Predicted accuracy", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{scheme}_scatter.{suffix}", dpi=300)
    plt.close(fig)


def rank_scatter_grid(table: pd.DataFrame, scheme: str, out: Path) -> None:
    bins = [b for b in table["target_bin"].unique() if "–" not in b]
    bins = sorted(bins) + [b for b in table["target_bin"].unique() if "–" in b]
    methods = [("elo", "Elo", GREY), ("easy_accuracy", "Easy accuracy", ORANGE),
               ("corrected", "Corrected", BLUE)]
    fig, axes = plt.subplots(
        len(bins), 3, figsize=(4.6, 1.35 * len(bins) + 0.5),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, target_bin in enumerate(bins):
        for column, (method, label, colour) in enumerate(methods):
            ax = axes[row][column]
            cell = table[table["target_bin"].eq(target_bin)
                         & table["method"].eq(method)].dropna(subset=["prediction"])
            ranks = cell["prediction"].rank()
            ax.plot(ranks, cell["observed"], "o", ms=2.5, c=colour,
                    mec="none", alpha=0.75)
            if row == 0:
                ax.set_title(f"{label}\n(n={cell['model'].nunique()})",
                             loc="left", fontsize=6.5, color="dimgrey")
            if column == 0:
                ax.set_ylabel(target_bin, rotation=0, labelpad=12, color="dimgrey")
            ax.set_yticks([0, 0.5, 1])
    fig.supxlabel("Predictor rank", fontsize=8)
    fig.supylabel("Observed accuracy", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{scheme}_rank_scatter.{suffix}", dpi=300)
    plt.close(fig)


def easy_vs_rest_scatter(table: pd.DataFrame, scheme: str, out: Path) -> None:
    pooled = table[table["target_bin"].str.contains("–")]
    scatter_grid_frame = pooled.assign(target_bin="rest")
    methods = [("elo", "Elo", GREY), ("easy_accuracy", "Easy accuracy", ORANGE),
               ("corrected", "Corrected", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.9), sharex=True, sharey=True)
    for ax, (method, label, colour) in zip(axes, methods):
        cell = scatter_grid_frame[scatter_grid_frame["method"].eq(method)]
        ax.plot([0, 1], [0, 1], "--", lw=0.8, c="grey", zorder=0)
        ax.plot(cell["observed"], cell["prediction"], "o", ms=2.5,
                c=colour, mec="none", alpha=0.75)
        ax.set_title(f"{label}\n(n={cell['model'].nunique()})",
                     loc="left", fontsize=6.5, color="dimgrey")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    fig.supxlabel("Observed accuracy (all harder items)", fontsize=8)
    fig.supylabel("Predicted", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{scheme}_easy_vs_rest.{suffix}", dpi=300)
    plt.close(fig)


def prediction_by_slice(table: pd.DataFrame, observed: pd.DataFrame,
                        scheme: str, out: Path) -> None:
    bins = sorted(b for b in table["target_bin"].unique() if "–" not in b)
    methods = [("elo", "Elo", GREY), ("easy_accuracy", "Easy accuracy", ORANGE),
               ("corrected", "Corrected", BLUE)]
    metrics = [("tau_a", "Kendall tau-a", (-1.05, 1.05)),
               ("mae", "MAE", (0, None))]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.1))
    for ax, (metric, ylabel, limits) in zip(axes, metrics):
        for offset, (method, label, colour) in zip((-0.18, 0.0, 0.18), methods):
            values = []
            for target_bin in bins:
                cell = table[table["target_bin"].eq(target_bin)
                             & table["method"].eq(method)].dropna(
                                 subset=["prediction", "observed"])
                if metric == "tau_a":
                    values.append(kendall_tau_a(cell["prediction"], cell["observed"]))
                else:
                    values.append(float(
                        (cell["prediction"] - cell["observed"]).abs().mean()))
            ax.plot(np.arange(len(bins)) + offset, values, "o", ms=3.5,
                    c=colour, mec="none", label=label)
        if metric == "tau_a":
            ax.axhline(0, ls="--", lw=0.8, c="grey")
        ax.set_xticks(range(len(bins)), bins)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Item slice")
        ax.set_ylim(*(limit for limit in limits))
    axes[0].legend(frameon=False, ncol=3, loc="lower center",
                   bbox_to_anchor=(1.1, -0.62), fontsize=6.5)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{scheme}_prediction_by_slice.{suffix}", dpi=300)
    plt.close(fig)


def kendall_tau_a(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    concordant = sum(
        np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        for i in range(n) for j in range(i + 1, n)
    )
    return concordant / (n * (n - 1) / 2)


def tau_summary(table: pd.DataFrame, scheme: str, out: Path) -> pd.DataFrame:
    rows = []
    for (target_bin, method), cell in table.groupby(["target_bin", "method"]):
        cell = cell.dropna(subset=["prediction", "observed"])
        if cell["model"].nunique() < 3:
            continue
        rows.append({
            "scheme": scheme, "target_bin": target_bin, "method": method,
            "tau_a": kendall_tau_a(cell["prediction"], cell["observed"]),
            "mae": float((cell["prediction"] - cell["observed"]).abs().mean()),
            "n_models": cell["model"].nunique(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes-root", type=Path, required=True,
                        help="dir containing analysis/bias_correction_<scheme>")
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--old-partition-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    responses = pd.read_parquet(args.responses)
    all_items = pd.read_csv(args.reference / "items.csv")
    items = all_items[all_items["irt_model"].eq("fixed_3pl")]
    anchors = pd.read_csv(args.reference / "anchors.csv")
    anchors = anchors[anchors["irt_model"].eq("fixed_3pl")]
    old = pd.read_csv(args.old_partition_predictions)
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for scheme, bins in SCHEMES.items():
        scheme_dir = args.schemes_root / f"analysis/bias_correction_{scheme}"
        theta = corrected_theta(scheme_dir)
        observed = bin_accuracy(responses, items, scheme, bins)
        table = pd.concat([
            corrected_predictions(theta, anchors, observed),
            original_predictions(old, scheme),
        ], ignore_index=True)
        table.to_csv(args.out / f"{scheme}_predictions.csv", index=False)
        # direct comparison: every method on the models the original methods cover
        matched_models = (
            set(table.loc[table["method"].eq("elo"), "model"])
            & set(table.loc[table["method"].eq("corrected"), "model"])
        )
        matched = table[table["model"].isin(matched_models)]
        scatter_grid(matched, scheme, args.out)
        rank_scatter_grid(matched, scheme, args.out)
        easy_vs_rest_scatter(matched, scheme, args.out)
        prediction_by_slice(matched, observed, scheme, args.out)
        summaries.append(tau_summary(matched, scheme, args.out).assign(pool="matched"))
        summaries.append(tau_summary(table, scheme, args.out).assign(pool="full"))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.out / "partition_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
