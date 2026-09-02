#!/usr/bin/env python3
"""Fit group-only arms and draw the two GLM predictor figures.

With no flags, redraws every figure whose inputs already exist. ``--fit``
runs all missing reduced-draw group arms; names after the flag restrict the
run, e.g. ``--fit only_style``. Fits are resumable at whole-arm granularity.
"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "scripts/hibayes/config_correctness.yaml"
FIT_ROOT = ROOT / "analysis/hibayes/run_correctness_predictors"
CONFIG_ROOT = ROOT / "analysis/hibayes/correctness_predictor_configs"
IMPORTANCE_CSV = ROOT / "analysis/hibayes/correctness_predictor_importance.csv"
EFFECTS_CSV = ROOT / "analysis/hibayes/correctness_effects.csv"
OUT = ROOT / "manuscript/draft1/figures/03-hierarchical-glm"
HIBAYES = ROOT / "analysis/hibayes/.venv/bin/hibayes-full"

PREDICTORS = (
    "capability_gap_sd",
    "truth_side",
    "truth_side_x_total_length",
    "total_length_sd",
    "length_difference_stronger_sd",
    "same_model_family",
    "personal_pronouns_difference_stronger_sd",
    "hedging_difference_stronger_sd",
    "backtracking_difference_stronger_sd",
    "uncertainty_difference_stronger_sd",
    "confidence_difference_stronger_sd",
    "questions_difference_stronger_sd",
)
LABELS = {
    "truth_side": "Position",
    "truth_side_x_total_length": "Position × length",
    "total_length_sd": "Combined length",
    "length_difference_stronger_sd": "Length gap",
    "capability_gap_sd": "Capability gap",
    "same_model_family": "Same family",
    "personal_pronouns_difference_stronger_sd": "Pronouns",
    "hedging_difference_stronger_sd": "Hedging",
    "backtracking_difference_stronger_sd": "Backtracking",
    "uncertainty_difference_stronger_sd": "Uncertainty",
    "confidence_difference_stronger_sd": "Confidence",
    "questions_difference_stronger_sd": "Questions",
}
REGRESSOR_DEFINITIONS = {
    "capability_gap_sd": "Absolute external-capability (ECI) gap",
    "truth_side": "+1 stronger answer left; −1 stronger answer right",
    "truth_side_x_total_length": "Position × combined length",
    "total_length_sd": "log(1 + left chars + right chars)",
    "length_difference_stronger_sd": "log(1 + stronger chars) − log(1 + weaker chars)",
    "same_model_family": "1 if distinct answer models share a family; otherwise 0",
    "personal_pronouns_difference_stronger_sd": "I, me, we, our, us",
    "hedging_difference_stronger_sd": "may, might, likely, I think, etc.",
    "backtracking_difference_stronger_sd": "wait, actually, let me reconsider, etc.",
    "uncertainty_difference_stronger_sd": "unsure, not sure, unclear, guess, etc.",
    "confidence_difference_stronger_sd": "obviously, certainly, definitely, etc.",
    "questions_difference_stronger_sd": "Question-mark characters",
}
STYLE_CALCULATION = "zscore(rate stronger − rate weaker); rate = 1,000 × hits / words"
REGRESSOR_CALCULATIONS = {
    "capability_gap_sd": "zscore(abs(ECI left − ECI right))",
    "truth_side": "+1 if stronger is left; −1 if stronger is right",
    "truth_side_x_total_length": "truth_side × zscore(log1p(left chars + right chars))",
    "total_length_sd": "zscore(log1p(left chars + right chars))",
    "length_difference_stronger_sd": "zscore(log1p(stronger chars) − log1p(weaker chars))",
    "same_model_family": "1 if distinct models share a family; otherwise 0",
    **{predictor: STYLE_CALCULATION for predictor in PREDICTORS[6:]},
}
PREDICTOR_GROUPS = {
    "truth_side": "Position",
    "truth_side_x_total_length": "Position",
    "total_length_sd": "Length",
    "length_difference_stronger_sd": "Length",
    "capability_gap_sd": "Capability",
    "same_model_family": "Control",
    **{name: "Style" for name in PREDICTORS[6:]},
}
GROUP_PREDICTORS = {
    "Capability": ["capability_gap_sd"],
    "Length": ["truth_side_x_total_length", "total_length_sd", "length_difference_stronger_sd"],
    "Style": list(PREDICTORS[6:]),
    "Position": ["truth_side", "truth_side_x_total_length"],
}
GROUP_COLOURS = {
    "Position": "#8250c4",
    "Length": "#2a78d6",
    "Capability": "#eb6834",
    "Control": "#777777",
    "Style": "#1b8f69",
}
JUDGES = (
    "qwen3-0.6b",
    "llama-3.2-3b",
    "qwen3-8b",
    "qwen3-14b",
    "gpt-oss-20b",
    "qwen3-32b",
    "gpt-oss-120b",
)
JUDGE_LABELS = {
    "qwen3-0.6b": "Qwen3 0.6B",
    "llama-3.2-3b": "Llama 3.2 3B",
    "qwen3-8b": "Qwen3 8B",
    "qwen3-14b": "Qwen3 14B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "qwen3-32b": "Qwen3 32B",
    "gpt-oss-120b": "GPT-OSS 120B",
}
JUDGE_COLOURS = ("#777777", "#8250c4", "#2a78d6", "#eda100", "#1b8f69", "#eb6834", "#c44e87")


def make_config(base: dict, predictors: list[str], tag: str, samples: int, warmup: int, chains: int) -> dict:
    """Return one lean, LOO-producing fit config."""
    config = deepcopy(base)
    model = config["model"]["models"][0]["config"]
    model["tag"] = tag
    model["predictors"] = predictors
    model["fit"] = {
        "samples": samples,
        "warmup": warmup,
        "chains": chains,
        "seed": 0,
        "target_accept": 0.95,
    }
    config["check"]["checkers"] = ["divergences", "r_hat", "ess_bulk", "ess_tail", "bfmi", "loo"]
    config["communicate"]["communicators"] = []
    return config


DEFINITIVE_LOO = {
    "full": ROOT / "analysis/hibayes/run_correctness_full/models/judge_feature_logistic_model_full/diagnostics/inference_data_loo_i.csv",
    **{
        f"minus_{group.lower()}": ROOT / f"analysis/hibayes/run_correctness_minus_{group.lower()}"
        / f"models/judge_feature_logistic_model_minus_{group.lower()}/diagnostics/inference_data_loo_i.csv"
        for group in GROUP_PREDICTORS
    },
}


def arm_predictors() -> dict[str, list[str]]:
    return {f"only_{group.lower()}": predictors for group, predictors in GROUP_PREDICTORS.items()}


def loo_file(tag: str) -> Path | None:
    if tag in DEFINITIVE_LOO:
        return DEFINITIVE_LOO[tag] if DEFINITIVE_LOO[tag].exists() else None
    matches = list((FIT_ROOT / tag / "models").glob("*/diagnostics/inference_data_loo_i.csv"))
    return matches[0] if len(matches) == 1 else None


def loo_accuracy(path: Path) -> float:
    return float((np.exp(pd.read_csv(path)["loo_i"].to_numpy()) > 0.5).mean())


def loo_accuracies(path: Path, judges=JUDGES) -> pd.Series:
    correct = np.exp(pd.read_csv(path)["loo_i"].to_numpy()) > 0.5
    frame = pd.read_parquet(path.parents[3] / "processed_data.parquet", columns=["judge"])
    if len(frame) != len(correct):
        raise ValueError(f"LOO rows ({len(correct)}) do not match processed rows ({len(frame)}) at {path}")
    accuracies = pd.Series(correct, index=frame.index).groupby(frame["judge"]).mean().reindex(judges)
    if accuracies.isna().any():
        raise ValueError(f"Missing judge accuracies at {path}: {accuracies[accuracies.isna()].index.tolist()}")
    return accuracies


def mean_ci(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        raise ValueError("A 95% CI requires at least three replication units")
    return float(values.mean()), float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def importance_rows(accuracies: dict[str, pd.Series], groups=GROUP_PREDICTORS, judges=JUDGES) -> list[dict]:
    rows = []
    for group in groups:
        tag = group.lower()
        for judge in judges:
            full = accuracies["full"][judge]
            without = accuracies[f"minus_{tag}"][judge]
            rows.append(
                {
                    "group": group,
                    "judge": judge,
                    "full_accuracy": full,
                    "without_accuracy": without,
                    "only_accuracy": accuracies[f"only_{tag}"][judge],
                    "ablation_delta_accuracy": without - full,
                }
            )
    return rows


def regressor_rows(effects: pd.DataFrame, judges=JUDGES, predictors=PREDICTORS) -> list[dict]:
    block = effects[(effects["arm"] == "full") & effects["level"].isin(judges)]
    means = block.groupby("predictor")["mean"].agg(["mean", "count"])
    if any(means.loc[predictor, "count"] != len(judges) for predictor in predictors):
        raise ValueError("Full-arm effects do not contain one row per requested judge and predictor")
    return [
        {
            "regressor": LABELS[predictor],
            "mean_beta": means.loc[predictor, "mean"],
            "definition": REGRESSOR_DEFINITIONS[predictor],
            "calculation": REGRESSOR_CALCULATIONS[predictor],
        }
        for predictor in predictors
    ]


def fit_arms(requested: list[str], samples: int, warmup: int, chains: int) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    arms = arm_predictors()
    unknown = set(requested) - set(arms)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "XDG_CACHE_HOME": str(ROOT / "analysis/hibayes/.cache"),
        "MPLCONFIGDIR": str(ROOT / "analysis/hibayes/.cache/matplotlib"),
    }
    for tag in requested:
        if loo_file(tag):
            print(f"skip complete arm: {tag}")
            continue
        config_path = CONFIG_ROOT / f"{tag}.yaml"
        config_path.write_text(yaml.safe_dump(make_config(base, arms[tag], tag, samples, warmup, chains), sort_keys=False))
        run = FIT_ROOT / tag
        if run.exists():
            raise RuntimeError(f"Incomplete arm exists at {run}; inspect it before retrying")
        run.parent.mkdir(parents=True, exist_ok=True)
        log = FIT_ROOT / f"{tag}.log"
        print(f"fit {tag} -> {run}", flush=True)
        with log.open("w") as stream:
            subprocess.run([str(HIBAYES), "--config", str(config_path), "--out", str(run), "--no-tui"],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def decorate_groups(axis, coloured: bool) -> None:
    groups = [PREDICTOR_GROUPS[p] for p in PREDICTORS]
    starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    ends = starts[1:] + [len(groups)]
    for start, end in zip(starts, ends, strict=True):
        group = groups[start]
        if end - start > 1:
            axis.text((start + end - 1) / 2, 1.015, group,
                      color=GROUP_COLOURS[group] if coloured else "#666666", fontsize=6.5,
                      ha="center", va="bottom", transform=axis.get_xaxis_transform())
        if start:
            axis.axvline(start - 0.5, color="#dddddd", lw=0.6, zorder=0)


def finish_axis(axis, coloured_groups: bool = False) -> None:
    for side in ("left", "bottom"):
        axis.spines[side].set_position(("outward", 4))
    axis.tick_params(length=2.5, width=0.8)
    axis.set_xlim(-0.5, len(PREDICTORS) - 0.5)
    axis.set_xticks(range(len(PREDICTORS)), [LABELS[p] for p in PREDICTORS], rotation=90)
    decorate_groups(axis, coloured_groups)


def save(figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_importance(frame: pd.DataFrame) -> None:
    style()
    x = np.arange(len(GROUP_PREDICTORS))
    figure, (accuracy_axis, change_axis) = plt.subplots(
        1, 2, figsize=(3.84, 2.65), gridspec_kw={"wspace": 0.65}
    )
    judge_handles = []
    for judge, colour in zip(JUDGES, JUDGE_COLOURS, strict=True):
        block = frame[frame["judge"] == judge].set_index("group").loc[list(GROUP_PREDICTORS)]
        for axis, column in ((accuracy_axis, "only_accuracy"), (change_axis, "ablation_delta_accuracy")):
            (line,) = axis.plot(x, block[column], marker="o", ms=2.8, lw=1, color=colour, alpha=0.75,
                                label=JUDGE_LABELS[judge])
        judge_handles.append(line)
    for i, group in enumerate(GROUP_PREDICTORS):
        block = frame[frame["group"] == group].set_index("judge").loc[list(JUDGES)]
        for axis, column in ((accuracy_axis, "only_accuracy"), (change_axis, "ablation_delta_accuracy")):
            values = block[column].to_numpy()
            mean, half_width = mean_ci(values)
            axis.errorbar(i, mean, yerr=half_width, fmt="D", ms=3.2, color="#222222", mfc="white",
                          mec="#222222", elinewidth=1, capsize=0, zorder=3)
    accuracy_axis.set_ylabel("Single variable model accuracy")
    accuracy_axis.set_ylim(0.5, 0.8)
    accuracy_axis.set_yticks([0.5, 0.6, 0.7, 0.8])
    accuracy_axis.axhline(0.5, color="#888888", ls="--", lw=0.7, zorder=0)
    change_axis.set_ylabel("Regressor ablation Δaccuracy")
    change_axis.set_ylim(-0.1, 0.01)
    change_axis.set_yticks([-0.1, -0.05, 0, 0.01])
    change_axis.axhline(0, color="#888888", ls="--", lw=0.7, zorder=0)
    for axis in (accuracy_axis, change_axis):
        for side in ("left", "bottom"):
            axis.spines[side].set_position(("outward", 4))
        axis.tick_params(length=2.5, width=0.8)
        axis.set_xlim(-0.5, len(GROUP_PREDICTORS) - 0.5)
    for axis in (accuracy_axis, change_axis):
        axis.set_xticks(x, GROUP_PREDICTORS, rotation=90)
    mean_handle = plt.Line2D([], [], ls="", marker="D", mfc="white", mec="#222222", ms=3.2,
                             color="#222222", label="Mean ± 95% CI")
    figure.legend(handles=[*judge_handles, mean_handle], loc="lower center", bbox_to_anchor=(0.5, -0.005),
                  frameon=False, ncol=3, handlelength=1.3, columnspacing=0.8)
    figure.subplots_adjust(left=0.15, right=0.99, top=0.96, bottom=0.52)
    save(figure, "glm_predictor_importance")


def plot_weights(effects: pd.DataFrame) -> None:
    style()
    block = effects[(effects["arm"] == "full") & effects["level"].isin(JUDGES)]
    counts = block.groupby(["level", "predictor"]).size()
    if any(counts.get((judge, predictor), 0) != 1 for judge in JUDGES for predictor in PREDICTORS):
        raise ValueError("Full-arm effects do not contain exactly one row per judge and predictor")
    figure, axis = plt.subplots(figsize=(3.62, 2.65))
    x = np.arange(len(PREDICTORS))
    for judge, colour in zip(JUDGES, JUDGE_COLOURS, strict=True):
        values = block[block["level"] == judge].set_index("predictor").loc[list(PREDICTORS), "mean"]
        axis.plot(x, values, marker="o", ms=2.4, lw=0.9, color=colour, label=JUDGE_LABELS[judge])
    matrix = np.stack([
        block[block["level"] == judge].set_index("predictor").loc[list(PREDICTORS), "mean"].to_numpy()
        for judge in JUDGES
    ])
    summary = [mean_ci(matrix[:, i]) for i in range(len(PREDICTORS))]
    means, half_widths = np.asarray(summary).T
    axis.errorbar(x, means, yerr=half_widths, color="#222222", marker="D", mfc="white", mec="#222222",
                  mew=0.8, ms=3, lw=1, elinewidth=0.8, capsize=0, zorder=5, label="Mean ± 95% CI")
    axis.axhline(0, color="#888888", ls="--", lw=0.7, zorder=0)
    axis.set_ylabel("Fitted coefficient (log-odds)")
    axis.set_ylim(-0.6, 2.1)
    axis.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])
    finish_axis(axis)
    figure.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
                  handlelength=1.5, columnspacing=1.1)
    figure.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.52)
    save(figure, "glm_judge_weights")


def plot_regressor_table(effects: pd.DataFrame) -> None:
    style()
    rows = regressor_rows(effects)
    figure, axis = plt.subplots(figsize=(3.96, 3.15))
    axis.axis("off")
    axis.text(0.01, 0.97, "Regressor", weight="bold", fontsize=7, transform=axis.transAxes)
    axis.text(0.41, 0.97, "Mean β", weight="bold", fontsize=7, ha="right", transform=axis.transAxes)
    axis.text(0.45, 0.97, "Definition / calculation", weight="bold", fontsize=7, transform=axis.transAxes)
    axis.plot([0.01, 0.99], [0.945, 0.945], color="#555555", lw=0.8, transform=axis.transAxes)
    for index, row in enumerate(rows):
        y = 0.91 - index * 0.066
        axis.text(0.01, y, row["regressor"], fontsize=6.6, va="center", transform=axis.transAxes)
        axis.text(0.41, y, f'{row["mean_beta"]:+.3f}', fontsize=6.6, ha="right", va="center",
                  transform=axis.transAxes)
        axis.text(0.45, y, row["definition"], fontsize=6.4, va="center", transform=axis.transAxes)
    axis.plot([0.01, 0.99], [0.095, 0.095], color="#555555", lw=0.8, transform=axis.transAxes)
    axis.text(0.01, 0.058, "β is the mean of seven judge-specific posterior means.",
              fontsize=6.2, color="#555555", transform=axis.transAxes)
    axis.text(0.01, 0.022, "Continuous predictors are z-scored; style = stronger−weaker hits per 1,000 words.",
              fontsize=6.2, color="#555555", transform=axis.transAxes)
    save(figure, "glm_regressor_table")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="*", metavar="ARM", help="fit named group-only arms; no names means all four")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    arms = arm_predictors()
    if args.fit is not None:
        fit_arms(list(arms) if not args.fit else args.fit, args.samples, args.warmup, args.chains)

    effects = pd.read_csv(EFFECTS_CSV)
    plot_weights(effects)
    plot_regressor_table(effects)
    table = pd.DataFrame(regressor_rows(effects))
    table["mean_beta"] = table["mean_beta"].round(3)
    table.to_csv(OUT / "glm_regressor_table.csv", index=False)
    required = list(DEFINITIVE_LOO) + list(arms)
    missing = [tag for tag in required if loo_file(tag) is None]
    if missing:
        print(f"Weight figure written; importance figure awaits {len(missing)} arms: {', '.join(missing)}")
        return
    accuracies = {tag: loo_accuracies(loo_file(tag)) for tag in required}
    importance = pd.DataFrame(importance_rows(accuracies))
    importance.to_csv(IMPORTANCE_CSV, index=False)
    plot_importance(importance)
    print(f"Wrote {IMPORTANCE_CSV} and both figures under {OUT}")


if __name__ == "__main__":
    main()
