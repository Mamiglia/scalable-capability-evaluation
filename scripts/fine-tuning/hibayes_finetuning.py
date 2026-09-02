#!/usr/bin/env python3
"""HiBayes coefficient and regressor-group analysis for base vs fine-tuned judges."""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import t
import yaml


ROOT = Path(__file__).resolve().parents[2]
HIBAYES_SCRIPTS = ROOT / "scripts" / "hibayes"
sys.path.insert(0, str(HIBAYES_SCRIPTS))
from correctness_predictor_figures import (  # noqa: E402
    GROUP_PREDICTORS,
    LABELS,
    PREDICTOR_GROUPS,
    PREDICTORS,
)
from correctness_report import load_arm  # noqa: E402
from style_features import feature_rates, load_lexicon  # noqa: E402


STYLE_LEXICON = load_lexicon(HIBAYES_SCRIPTS / "style_lexicon.yaml")
STYLE_CATEGORIES = tuple(STYLE_LEXICON)
CHECKPOINTS = (
    ("qwen_base", "out", "base"),
    ("qwen_ft", "out", "ft"),
    ("llama_base", "out_llama3b", "base"),
    ("llama_ft", "out_llama3b", "ft"),
)
JUDGE_ORDER = tuple(checkpoint[0] for checkpoint in CHECKPOINTS)
ARCHITECTURES = {
    "Qwen3-8B": ("qwen_base", "qwen_ft"),
    "Llama-3.2-3B": ("llama_base", "llama_ft"),
}
MODEL_COLOURS = {"Qwen3-8B": "#2a78d6", "Llama-3.2-3B": "#4a3aa7"}
RUN_ROOT = ROOT / "analysis" / "hibayes" / "fine_tuning"
FRAME_PATH = RUN_ROOT / "gpqa_correctness.csv"
HIBAYES = ROOT / "analysis" / "hibayes" / ".venv" / "bin" / "hibayes-full"
FIGURE_STUB = ROOT / "manuscript" / "draft1" / "figures" / "06-fine-tuning" / "hibayes_regressor_ablation"
TABLE_ROOT = ROOT / "manuscript" / "draft1" / "tables"


def model_family(name):
    match = re.match(r"[a-z]+", name.lower())
    return match.group() if match else name.lower()


def standardise(data, columns):
    for column in columns:
        scale = data[column].std()
        if not scale:
            raise ValueError(f"Cannot standardise constant feature: {column}")
        data[f"{column}_sd"] = (data[column] - data[column].mean()) / scale


@lru_cache(maxsize=None)
def _style_rates(text):
    return feature_rates(text, STYLE_LEXICON)


def prepare_checkpoint_rows(verdicts, pairs, accuracies, judge):
    """Create correctness-oriented raw features for one judge checkpoint."""
    data = verdicts.merge(
        pairs[["pair_uid", "order", "text_1", "text_2"]],
        on=["pair_uid", "order"], how="left", validate="one_to_one",
    )
    acc_1, acc_2 = data.model_1.map(accuracies), data.model_2.map(accuracies)
    if data[["text_1", "text_2"]].isna().any().any() or acc_1.isna().any() or acc_2.isna().any():
        raise ValueError("Missing transcript or GPQA accuracy for a verdict row")
    truth_side = np.sign(acc_1 - acc_2)
    if not pd.Series(truth_side).isin([-1.0, 1.0]).all():
        raise ValueError("Equal GPQA accuracies make correctness and stronger-oriented features undefined")
    choose_1 = data.logit_1 > data.logit_2
    left_rates = pd.DataFrame([_style_rates(text) for text in data.text_1], index=data.index)
    right_rates = pd.DataFrame([_style_rates(text) for text in data.text_2], index=data.index)
    left_chars, right_chars = data.text_1.str.len(), data.text_2.str.len()
    frame = pd.DataFrame({
        "is_correct": (choose_1 == (truth_side > 0)).astype(int),
        "judge": judge,
        "item": data.item,
        "transcript_condition": "eval",
        "model_pair": ["|".join(sorted(pair))
                       for pair in zip(data.model_1, data.model_2, strict=True)],
        "truth_side": truth_side,
        "left_chars": left_chars,
        "right_chars": right_chars,
        "capability_gap": (acc_1 - acc_2).abs(),
        "total_length": np.log1p(left_chars + right_chars),
        "length_difference_stronger": (
            np.log1p(left_chars) - np.log1p(right_chars)) * truth_side,
        "same_model_family": (
            (data.model_1 != data.model_2)
            & (data.model_1.map(model_family) == data.model_2.map(model_family))
        ).astype(int),
    })
    for category in STYLE_CATEGORIES:
        frame[f"{category}_difference_stronger"] = (
            left_rates[category] - right_rates[category]) * truth_side
    return frame


def prepare_frame(root=ROOT / "judge_results" / "judge_ft"):
    """Stack all four checkpoints and reproduce the main GLM feature scaling."""
    root = Path(root)
    pairs = pd.read_parquet(root / "pairs.parquet")
    accuracies = pd.read_csv(root / "models_ft.csv").set_index("model").accuracy
    frames = []
    for judge, subdir, tag in CHECKPOINTS:
        verdicts = pd.read_json(root / subdir / f"verdicts_{tag}.jsonl", lines=True)
        frames.append(prepare_checkpoint_rows(verdicts, pairs, accuracies, judge))
    frame = pd.concat(frames, ignore_index=True)
    scaled = ["capability_gap", "total_length", "length_difference_stronger",
              *[f"{category}_difference_stronger" for category in STYLE_CATEGORIES]]
    standardise(frame, scaled)
    frame["truth_side_x_total_length"] = frame.truth_side * frame.total_length_sd
    if not np.isfinite(frame[list(PREDICTORS)].to_numpy()).all():
        raise ValueError("HiBayes predictors contain non-finite values")
    return frame.sort_values(["judge", "item", "model_pair"]).reset_index(drop=True)


def arm_predictors():
    arms = {"full": list(PREDICTORS)}
    arms.update({f"minus_{group.lower()}": [p for p in PREDICTORS if p not in dropped]
                 for group, dropped in GROUP_PREDICTORS.items()})
    return arms


def make_config(frame_path, tag, predictors, samples, warmup, chains):
    return {
        "data_loader": {"paths": {"extracted_data": [str(frame_path)]}},
        "data_process": {"processors": [
            {"extract_observed_feature": {"feature_name": "is_correct"}},
            {"extract_features": {
                "categorical_features": ["judge", "item", "transcript_condition"],
                "continuous_features": list(PREDICTORS),
                "category_order": {"judge": list(JUDGE_ORDER),
                                   "transcript_condition": ["eval"]},
            }},
        ]},
        "model": {"path": "scripts/hibayes/model.py", "models": [{
            "name": "judge_feature_logistic_model",
            "config": {
                "tag": tag,
                "pool_judge_effects": True,
                "predictors": predictors,
                "fit": {"samples": samples, "warmup": warmup, "chains": chains,
                        "seed": 0, "target_accept": 0.95},
            },
        }]},
        "check": {"checkers": ["divergences", "r_hat", "ess_bulk", "ess_tail", "bfmi", "loo"]},
        "communicate": {"communicators": []},
        "platform": {"device_type": "cpu", "chain_method": "parallel"},
    }


def loo_path(run):
    paths = list(Path(run).glob("models/*/diagnostics/inference_data_loo_i.csv"))
    return paths[0] if len(paths) == 1 else None


def fit_arms(frame_path=FRAME_PATH, samples=1000, warmup=1000, chains=2):
    """Fit the full model and one model dropping each manuscript regressor group."""
    config_root = RUN_ROOT / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "XDG_CACHE_HOME": str(ROOT / "analysis" / "hibayes" / ".cache"),
        "MPLCONFIGDIR": str(ROOT / "analysis" / "hibayes" / ".cache" / "matplotlib"),
    }
    for tag, predictors in arm_predictors().items():
        run = RUN_ROOT / tag
        if loo_path(run):
            print(f"skip complete arm: {tag}")
            continue
        if run.exists():
            raise RuntimeError(f"Incomplete fit exists at {run}; inspect before retrying")
        config_path = config_root / f"{tag}.yaml"
        config_path.write_text(yaml.safe_dump(
            make_config(frame_path, tag, predictors, samples, warmup, chains), sort_keys=False))
        print(f"fit {tag} -> {run}", flush=True)
        with (RUN_ROOT / f"{tag}.log").open("w") as log:
            subprocess.run([str(HIBAYES), "--config", str(config_path), "--out", str(run), "--no-tui"],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def paired_item_ablation(full_elpd, reduced_elpd, frame, group):
    """LOO accuracy loss from an ablation, paired over the 40 held-out items."""
    if len(full_elpd) != len(reduced_elpd) or len(frame) != len(full_elpd):
        raise ValueError("LOO arrays and processed frame differ in length")
    data = frame[["judge", "item"]].copy()
    data["delta"] = ((np.exp(reduced_elpd) > 0.5).astype(float)
                     - (np.exp(full_elpd) > 0.5).astype(float))
    per_item = data.groupby(["judge", "item"], observed=True).delta.mean()
    rows = []
    for judge, values in per_item.groupby(level="judge"):
        values = values.to_numpy()
        n = len(values)
        mean = values.mean()
        half = t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
        rows.append({"judge": judge, "group": group, "estimate": mean,
                     "ci_low": mean - half, "ci_high": mean + half, "n_items": n})
    return pd.DataFrame(rows)


def _summary(draws):
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(draws)), float(low), float(high)


def posterior_comparisons(predictors, judges, effects, intercepts):
    """Posterior base, fine-tuned, and within-architecture change summaries."""
    index = {judge: i for i, judge in enumerate(judges)}
    rows = []
    terms = [("intercept", "Baseline", "Baseline", intercepts)]
    terms += [(name, LABELS[name], PREDICTOR_GROUPS.get(name, "Control"), effects[:, :, i])
              for i, name in enumerate(predictors)]
    for architecture, (base_name, ft_name) in ARCHITECTURES.items():
        for raw, label, group, draws in terms:
            base, ft = draws[:, index[base_name]], draws[:, index[ft_name]]
            change = ft - base
            base_mean, base_low, base_high = _summary(base)
            ft_mean, ft_low, ft_high = _summary(ft)
            change_mean, change_low, change_high = _summary(change)
            rows.append({
                "architecture": architecture, "predictor_id": raw,
                "predictor": label, "group": group,
                "base_mean": base_mean, "base_low": base_low, "base_high": base_high,
                "ft_mean": ft_mean, "ft_low": ft_low, "ft_high": ft_high,
                "change_mean": change_mean, "change_low": change_low,
                "change_high": change_high, "p_change_gt_0": float((change > 0).mean()),
            })
    return pd.DataFrame(rows)


def format_posterior_markdown(table):
    lines = [
        "| Architecture | Group | Predictor | Base β [95% CrI] | Fine-tuned β [95% CrI] | "
        "Δβ [95% CrI] | P(Δ>0) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    interval = lambda row, prefix: (  # noqa: E731
        f"{row[f'{prefix}_mean']:+.3f} "
        f"[{row[f'{prefix}_low']:+.3f}, {row[f'{prefix}_high']:+.3f}]")
    for _, row in table.iterrows():
        lines.append(
            f"| {row.architecture} | {row.group} | {row.predictor} | "
            f"{interval(row, 'base')} | {interval(row, 'ft')} | "
            f"{interval(row, 'change')} | {row.p_change_gt_0:.3f} |")
    lines.extend(["", "Posterior means and equal-tail 95% credible intervals from the full HiBayes model.", ""])
    return "\n".join(lines)


def plot_ablation(table, out_stub):
    """Four-group LOO ablation plot with paired base-to-fine-tuned estimates."""
    sys.path.insert(0, str(ROOT / "scripts" / "analyses"))
    import matplotlib.pyplot as plt
    from plots import MUTED, save, style

    groups = list(GROUP_PREDICTORS)
    centres = np.arange(len(groups))
    indexed = table.set_index(["group", "judge"])
    y_min, y_max = -0.02, 0.01
    clipped = []
    fig, ax = plt.subplots(figsize=(3.1, 1.85), layout="constrained")
    offsets = {"Qwen3-8B": (-0.24, -0.08), "Llama-3.2-3B": (0.08, 0.24)}
    for architecture, (base, ft) in ARCHITECTURES.items():
        colour = MODEL_COLOURS[architecture]
        for x, group in zip(centres, groups, strict=True):
            base_row, ft_row = indexed.loc[(group, base)], indexed.loc[(group, ft)]
            base_x, ft_x = x + np.asarray(offsets[architecture])
            ax.plot([base_x, ft_x], [base_row.estimate, ft_row.estimate],
                    color=colour, lw=1, zorder=2)
            for point_x, row, face in ((base_x, base_row, "white"),
                                       (ft_x, ft_row, colour)):
                ax.errorbar(point_x, row.estimate,
                            yerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                            fmt="o", color=colour, mfc=face, mec=colour, mew=0.8,
                            ms=3.4, elinewidth=1, capsize=0, barsabove=True, zorder=3)
                if row.ci_high < y_min:
                    clipped.append((point_x, row.estimate, colour, face))
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xticks(centres, groups)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([y_min, -0.01, 0, y_max])
    ax.set_ylabel("Ablation Δaccuracy")
    style(ax)
    for index, (point_x, estimate, colour, face) in enumerate(clipped):
        marker_y = y_min + 0.0005
        ax.plot(point_x, marker_y, marker="v", ls="", ms=3.4, color=colour,
                mfc=face, mec=colour, mew=0.8, clip_on=False, zorder=4)
        ax.annotate(f"−{abs(estimate):.2f}", (point_x, marker_y), xytext=(0, 4 + 5 * index),
                    textcoords="offset points", color=colour, fontsize=6,
                    ha="center", va="bottom", rotation=90, zorder=5)
    handles = []
    for architecture, (base, ft) in ARCHITECTURES.items():
        colour = MODEL_COLOURS[architecture]
        short = "Qwen" if architecture.startswith("Qwen") else "Llama"
        handles.extend([
            plt.Line2D([], [], marker="o", ls="", mfc="white", mec=colour,
                       color=colour, label=f"{short} base"),
            plt.Line2D([], [], marker="o", ls="", mfc=colour, mec=colour,
                       color=colour, label=f"{short} FT"),
        ])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="outside lower center",
               fontsize=6.5, handlelength=0.8, handletextpad=0.2, columnspacing=0.55)
    save(fig, out_stub)


def report(run_root=RUN_ROOT, figure_stub=FIGURE_STUB, table_root=TABLE_ROOT):
    arms = {tag: load_arm(Path(run_root) / tag) for tag in arm_predictors()}
    full = arms["full"]
    frame = pd.read_parquet(Path(run_root) / "full" / "processed_data.parquet")
    rows = []
    for group in GROUP_PREDICTORS:
        reduced = arms[f"minus_{group.lower()}"]
        reduced_frame = pd.read_parquet(Path(run_root) / f"minus_{group.lower()}" / "processed_data.parquet")
        if not frame[["judge", "item"]].equals(reduced_frame[["judge", "item"]]):
            raise ValueError(f"{group} ablation changed the row order")
        rows.append(paired_item_ablation(full["elpd"], reduced["elpd"], frame, group))
    ablation = pd.concat(rows, ignore_index=True)
    coefficients = posterior_comparisons(
        full["predictors"], full["judges"], full["judge_predictor_effects"], full["judge_intercepts"])
    Path(figure_stub).parent.mkdir(parents=True, exist_ok=True)
    Path(table_root).mkdir(parents=True, exist_ok=True)
    ablation.to_csv(Path(table_root) / "fine_tuning_hibayes_ablation.csv", index=False)
    coefficients.to_csv(Path(table_root) / "fine_tuning_hibayes_coefficients.csv", index=False)
    (Path(table_root) / "fine_tuning_hibayes_coefficients.md").write_text(
        format_posterior_markdown(coefficients))
    plot_ablation(ablation, figure_stub)
    print(ablation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()
    run_all = not (args.prepare or args.fit or args.report)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if run_all or args.prepare:
        frame = prepare_frame()
        frame.to_csv(FRAME_PATH, index=False)
        print(f"wrote {len(frame):,} rows to {FRAME_PATH}")
    if run_all or args.fit:
        fit_arms(samples=args.samples, warmup=args.warmup, chains=args.chains)
    if run_all or args.report:
        report()


if __name__ == "__main__":
    main()
