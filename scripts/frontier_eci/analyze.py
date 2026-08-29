"""Predict Q2/Q3/Q4 accuracy from Q1 judgments for the frozen ECI frontier."""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log_samples
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from judge.util import extract_score, extract_transcript  # noqa: E402
from ranking import rank  # noqa: E402
from ranking.scoring import resample_items  # noqa: E402

JUDGE_LABELS = {
    "gpt-4o-2024-11-20": "GPT-4o (Nov 2024)",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5.2": "GPT-5.2",
    "o4-mini": "o4-mini",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "consensus": "Equal-weight consensus",
}
COLORS = {
    "gpt-4o-2024-11-20": "#4477AA",
    "gemini-2.5-pro": "#EE6677",
    "gpt-5.2": "#228833",
    "o4-mini": "#CCBB44",
    "deepseek-v4-flash-0731": "#AA3377",
    "consensus": "#44AA99",
}
SHORT = {
    "gpt-4o-1120": "GPT-4o",
    "claude-3.7-sonnet": "Claude 3.7",
    "o3": "o3",
    "gpt-5": "GPT-5",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "claude-opus-4.6": "Opus 4.6",
    "claude-opus-4.7": "Opus 4.7",
    "kimi-k3": "Kimi K3",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def sigmoid(ability, scale, slope, offset):
    return scale / (1 + np.exp(-(slope * ability + offset)))


def calibrate(ability: pd.Series, actual: pd.Series, anchors, targets):
    """Fit on explicit historical anchors, then predict the requested targets."""
    anchors = ability.index.intersection(actual.dropna().index).intersection(anchors)
    if len(anchors) < 4:
        raise ValueError(f"only {len(anchors)} non-frontier calibration anchors")
    x, y = ability[anchors].to_numpy(), actual[anchors].to_numpy()
    params, _ = curve_fit(
        sigmoid, x, y, p0=(max(float(y.max()), 0.05), 1.0, 0.0),
        bounds=([0.01, 0.0, -20.0], [1.0, 50.0, 20.0]), maxfev=20_000,
    )
    return pd.Series(sigmoid(ability.reindex(targets), *params), index=targets), len(anchors)


def release_key(name):
    return re.sub(r"[^a-z0-9.]", "", name.split("/")[-1].lower())


def release_date(releases, model):
    value = releases.get(model, releases.get(release_key(model)))
    return pd.Timestamp(value) if value is not None else pd.NaT


def prefix_verdicts(verdicts, target, releases):
    """Only edges whose two answer models existed by the target's release."""
    cutoff = release_date(releases, target)
    left = verdicts.model_left.map(lambda model: release_date(releases, model))
    right = verdicts.model_right.map(lambda model: release_date(releases, model))
    return verdicts[(left <= cutoff) & (right <= cutoff)].copy()


def load_releases(experiment):
    releases = dict(experiment["releases"])
    for path in (ROOT / "configs/models").rglob("*.yml"):
        spec = yaml.safe_load(path.read_text()) or {}
        if not spec.get("release_date") or not spec.get("name"):
            continue
        name = spec["name"].split("/")[-1]
        date = str(spec["release_date"])[:10]
        releases.setdefault(name, date)
        releases.setdefault(release_key(name), date)
        if spec.get("slug"):
            slug = spec["slug"].split("/")[-1]
            releases.setdefault(slug, date)
            releases.setdefault(release_key(slug), date)
    for name, date in list(releases.items()):
        releases.setdefault(release_key(name), date)
    return releases


def merge_rollouts(canonical: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    """Use frozen canonical rows whenever that benchmark/model already exists."""
    canonical = canonical[canonical.is_canonical].copy()
    canonical["actual_source"] = "frozen_canonical"
    frontier = frontier.copy()
    frontier["actual_source"] = "frontier_side_import"
    present = pd.MultiIndex.from_frame(canonical[["bench", "model"]]).unique()
    keys = pd.MultiIndex.from_frame(frontier[["bench", "model"]])
    return pd.concat([canonical, frontier[~keys.isin(present)]], ignore_index=True)


def combine_verdicts(hf: pd.DataFrame, new: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    columns = ["bench", "judge", "item", "model_left", "model_right",
               "epoch_left", "epoch_right", "judge_epoch", "pick_left"]
    table = pd.concat([hf[columns], new[columns]], ignore_index=True)
    table = table[table.judge.isin(judges)]
    key = columns[:-1]
    return table.drop_duplicates(key, keep="last").reset_index(drop=True)


def consensus_verdicts(verdicts):
    """Average rounds within judge, then judges, for each exact shown comparison."""
    keys = ["bench", "item", "model_left", "model_right", "epoch_left", "epoch_right"]
    per_judge = verdicts.groupby(keys + ["judge"], as_index=False).pick_left.mean()
    out = per_judge.groupby(keys, as_index=False).pick_left.mean()
    out["judge"], out["judge_epoch"] = "consensus", 1
    return out


def predict_block(verdicts, actual, target, releases, resamples, rng):
    if verdicts.empty:
        return {"model": target, "actual": actual.get(target, np.nan), "predicted": np.nan,
                "predicted_low": np.nan, "predicted_high": np.nan,
                "n_anchors": 0, "n_bootstrap": 0, "n_prefix_verdicts": 0}
    ability = rank(verdicts).set_index("model").ability
    cutoff = release_date(releases, target)
    anchors = [model for model in ability.index if release_date(releases, model) < cutoff]
    if target not in ability or len(ability.index.intersection(actual.index).intersection(anchors)) < 4:
        return {"model": target, "actual": actual.get(target, np.nan), "predicted": np.nan,
                "predicted_low": np.nan, "predicted_high": np.nan,
                "n_anchors": len(ability.index.intersection(actual.index).intersection(anchors)),
                "n_bootstrap": 0, "n_prefix_verdicts": len(verdicts)}
    point, n_anchors = calibrate(ability, actual, anchors, [target])
    draws = []
    for _ in range(resamples):
        try:
            draw_ability = rank(resample_items(verdicts, rng)).set_index("model").ability
            draws.append(calibrate(draw_ability, actual, anchors, [target])[0])
        except (RuntimeError, ValueError):
            continue
    if not draws:
        raise RuntimeError("every item-bootstrap calibration failed")
    draws = pd.concat(draws, axis=1)
    return {"model": target, "actual": actual.get(target, np.nan),
            "predicted": point[target],
            "predicted_low": draws.loc[target].quantile(0.025),
            "predicted_high": draws.loc[target].quantile(0.975),
            "n_anchors": n_anchors, "n_bootstrap": draws.shape[1],
            "n_prefix_verdicts": len(verdicts)}


def reasoning_blocks(sample):
    messages = getattr(sample, "messages", None) or []
    if not messages and sample.output and sample.output.choices:
        messages = [sample.output.choices[0].message]
    blocks = []
    for message in messages:
        content = getattr(message, "content", [])
        if isinstance(content, list):
            blocks.extend(block for block in content if getattr(block, "type", None) == "reasoning")
    return blocks


def transcript_features(config_root, quartiles, candidates):
    rows = []
    for bench in sorted(quartiles.bench.unique()):
        cfg = yaml.safe_load((config_root / bench / "gpt-4o-2024-11-20.yaml").read_text())
        q1 = set(quartiles.query("bench == @bench and item_quartile == 1").item.astype(str))
        for model, path in cfg["transcripts"].items():
            if not Path(path).exists():
                continue
            parts = Path(path).parts
            epoch = int(parts[4])
            for sample in read_eval_log_samples(path, all_samples_required=False):
                if str(sample.id) not in q1 or not extract_score(sample):
                    continue
                visible = extract_transcript(sample, include_reasoning=False)
                full = extract_transcript(sample, include_reasoning=True)
                blocks = reasoning_blocks(sample)
                rows.append({
                    "bench": bench, "model": model, "item": str(sample.id), "epoch": epoch,
                    "visible_chars": len(visible), "cot_added_chars": max(0, len(full) - len(visible)),
                    "reasoning_blocks": len(blocks),
                    "redacted_reasoning": any(bool(getattr(block, "redacted", False)) for block in blocks),
                })
    detail = pd.DataFrame(rows)
    summary = (detail[detail.model.isin(candidates)].groupby(["bench", "model"])
               .agg(n_correct_q1=("item", "size"), visible_chars_median=("visible_chars", "median"),
                    cot_added_chars_median=("cot_added_chars", "median"),
                    reasoning_block_share=("reasoning_blocks", lambda x: float((x > 0).mean())),
                    redacted_reasoning_share=("redacted_reasoning", "mean"))
               .reset_index())
    return detail, summary


def diagnostics(verdicts, new_verdicts, features, quartiles):
    q1 = quartiles.query("item_quartile == 1")[["bench", "item"]]
    table = verdicts.merge(q1, on=["bench", "item"], how="inner")
    table["pair_a"] = table[["model_left", "model_right"]].min(axis=1)
    table["pair_b"] = table[["model_left", "model_right"]].max(axis=1)
    table["chosen"] = np.where(table.pick_left.astype(bool), table.model_left, table.model_right)
    groups = (table.groupby(["bench", "judge", "item", "pair_a", "pair_b", "judge_epoch"])
              .chosen.agg(n="size", choices="nunique").reset_index())
    consistency = (groups[groups.n >= 2].assign(consistent=lambda d: d.choices == 1)
                   .groupby(["bench", "judge"]).consistent.mean())
    out = (table.groupby(["bench", "judge"])
           .agg(n_verdicts=("item", "size"), n_items=("item", "nunique"),
                first_position_rate=("pick_left", "mean")).reset_index())
    out["consistent_pair_rate"] = [consistency.get((r.bench, r.judge), np.nan) for r in out.itertuples()]

    exact = features.groupby(["bench", "model", "item", "epoch"], as_index=False).visible_chars.median()
    new = new_verdicts.merge(q1, on=["bench", "item"], how="inner")
    left = exact.rename(columns={"model": "model_left", "epoch": "epoch_left", "visible_chars": "left_chars"})
    right = exact.rename(columns={"model": "model_right", "epoch": "epoch_right", "visible_chars": "right_chars"})
    new = new.merge(left, on=["bench", "model_left", "item", "epoch_left"], how="left")
    new = new.merge(right, on=["bench", "model_right", "item", "epoch_right"], how="left")
    new = new.dropna(subset=["left_chars", "right_chars"])
    new = new[new.left_chars != new.right_chars]
    new["picked_longer"] = np.where(new.pick_left.astype(bool),
                                     new.left_chars > new.right_chars, new.right_chars > new.left_chars)
    longer = new.groupby(["bench", "judge"]).picked_longer.mean()
    out["picked_longer_rate_new"] = [longer.get((r.bench, r.judge), np.nan) for r in out.itertuples()]
    return out


def save(fig, path):
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_predictions(predictions, candidates, releases, judges, benches, out, name):
    fig, axes = plt.subplots(len(benches), 3, figsize=(7.2, 2.55 * len(benches)),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(candidates))
    labels = [f"{str(pd.Timestamp(releases[m]).year)[2:]}Q{pd.Timestamp(releases[m]).quarter}"
              for m in candidates]
    for row, bench in enumerate(benches):
        for col, quartile in enumerate((2, 3, 4)):
            ax = axes[row, col]
            block = predictions.query("bench == @bench and target_quartile == @quartile")
            actual = block.groupby("model").actual.first().reindex(candidates)
            ax.plot(x, actual, "o-", color="#222222", lw=1.2, ms=3, label="Actual")
            for judge in judges:
                values = block[block.judge == judge].set_index("model").predicted.reindex(candidates)
                ax.plot(x, values, "o-", color=COLORS[judge], lw=0.8, ms=2.2,
                        label=JUDGE_LABELS[judge])
            ax.set_title(f"{'GPQA' if bench == 'gpqa' else 'SWE-mini'} · Q{quartile}")
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            ax.axhline(0.5, color="#dddddd", lw=0.5, zorder=0)
            if col == 0:
                ax.set_ylabel("Accuracy")
            ax.set_xticks(x)
            ax.set_xticklabels(labels if row == len(benches) - 1 else [])
            if row == len(benches) - 1:
                ax.tick_params(axis="x", labelrotation=45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.72, bottom=0.2, wspace=0.24, hspace=0.28)
    save(fig, out / name)


def plot_error_heatmap(predictions, candidates, judges, benches, out):
    errors = (predictions.assign(error=lambda d: d.predicted - d.actual)
              .groupby(["bench", "judge", "model"]).error.mean())
    values = [abs(float(x)) for x in errors.dropna()]
    limit = max(0.1, min(0.5, max(values)))
    shown = [model for model in candidates
             if model in errors.index.get_level_values("model")]
    fig, axes = plt.subplots(1, len(benches), figsize=(5.2 * len(benches), 2.9),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, bench in zip(axes, benches):
        matrix = errors.loc[bench].unstack("model").reindex(index=judges, columns=shown)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#f2f2f2")
        image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        for y in range(len(judges)):
            for x in range(len(shown)):
                value = matrix.iloc[y, x]
                ax.text(x, y, "—" if pd.isna(value) else f"{value:+.2f}",
                        ha="center", va="center", fontsize=5.5)
        ax.set_title("GPQA" if bench == "gpqa" else "SWE-mini")
        ax.set_xticks(range(len(shown)), [SHORT[m] for m in shown], rotation=45, ha="right")
        ax.set_yticks(range(len(judges)), [JUDGE_LABELS[j] for j in judges])
    fig.subplots_adjust(left=0.27, right=0.82, bottom=0.28, top=0.84, wspace=0.12)
    cax = fig.add_axes([0.86, 0.25, 0.025, 0.52])
    fig.colorbar(image, cax=cax, label="Predicted − actual accuracy")
    save(fig, out / "frontier_error_heatmap")


def plot_visibility(summary, model_errors, candidates, benches, out):
    table = summary.merge(model_errors, on=["bench", "model"], validate="one_to_one")
    fig, axes = plt.subplots(1, len(benches), figsize=(3.6 * len(benches), 2.7),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, bench in zip(axes, benches):
        block = table[table.bench == bench].set_index("model").reindex(candidates).reset_index()
        block = block.dropna(subset=["visible_chars_median", "mean_error"])
        encrypted = block.redacted_reasoning_share > 0.5
        for flag, marker, label in ((False, "o", "No redacted reasoning"), (True, "s", "Redacted/encrypted reasoning")):
            rows = block[encrypted == flag]
            ax.scatter(np.log10(rows.visible_chars_median), rows.mean_error, marker=marker,
                       s=24, facecolor="white", edgecolor="#333333", lw=0.8, label=label)
        for row in block.itertuples():
            ax.annotate(SHORT[row.model], (np.log10(row.visible_chars_median), row.mean_error),
                        xytext=(3, 2), textcoords="offset points", fontsize=5.5)
        rho = spearmanr(np.log10(block.visible_chars_median), block.mean_error).statistic
        ax.text(0.03, 0.96, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top")
        ax.axhline(0, color="#999999", lw=0.7)
        ax.set_title("GPQA" if bench == "gpqa" else "SWE-mini")
        ax.set_xlabel("Median visible transcript length (log10 chars)")
    axes[0].set_ylabel("Mean prediction error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.82, wspace=0.16)
    save(fig, out / "visibility_vs_prediction_error")


def plot_frontier_style(table, judge, bench, quartile, out, style,
                        prediction_label="predicted", filename_prefix=""):
    """Original frontier figure grammar: release time, judge cutoff, actual vs prediction."""
    table = table.sort_values("target_release").copy()
    table["target_release"] = pd.to_datetime(table.target_release)
    judge_date = pd.Timestamp(table.judge_release.dropna().iloc[0])
    predicted = table.dropna(subset=["predicted"])
    actual = table.dropna(subset=["actual"])
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.axvspan(judge_date, table.target_release.max() + pd.Timedelta(days=400),
               color="#f2f2f2", lw=0, zorder=0)
    for row in predicted.itertuples():
        ax.plot([row.target_release, row.target_release], [row.actual, row.predicted],
                ":", color="k", lw=0.8)
    if style == "band":
        band = ax.fill_between(predicted.target_release, predicted.predicted_low,
                               predicted.predicted_high, color="#2a78d6", alpha=0.18, lw=0)
        line, = ax.plot(predicted.target_release, predicted.predicted, color="#2a78d6", lw=1)
        prediction_handle = (band, line)
    else:
        prediction_handle, = ax.plot(predicted.target_release, predicted.predicted, "o",
                                     ms=3, color="#2a78d6", clip_on=False)
    actual_handle, = ax.plot(actual.target_release, actual.actual, "o", ms=3,
                             color="#333333", clip_on=False)
    ax.axvline(judge_date, color="grey", lw=0.8, ls="--")
    ax.text(judge_date, 1.0, f" {JUDGE_LABELS[judge]} released", color="grey",
            fontsize=6, va="top", ha="left")
    first, last = min(table.target_release.min(), judge_date), max(table.target_release.max(), judge_date)
    ticks = pd.date_range(first.replace(month=1, day=1), last + pd.DateOffset(months=6), freq="6MS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick.strftime("%b %Y") for tick in ticks])
    ax.set_xlim(ticks[0] - pd.Timedelta(days=30), ticks[-1] + pd.Timedelta(days=30))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(f"Q{quartile} hard-item accuracy")
    ax.set_xlabel(f"{'GPQA' if bench == 'gpqa' else 'SWE-mini'}: model release date")
    ax.spines["left"].set_bounds(0, 1)
    ax.legend([actual_handle, prediction_handle],
              ["actual (best model per quarter)", prediction_label],
              handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower left",
              bbox_to_anchor=(0, 1.02), frameon=False, handletextpad=0.5)
    save(fig, out / f"{filename_prefix}{judge}_{bench}_q{quartile}_frontier_{style}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tables", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benches", nargs="+", choices=("gpqa", "swe_bench_mini"),
                        default=("gpqa", "swe_bench_mini"))
    parser.add_argument("--out", type=Path, default=ROOT / "judge_results/frontier/eci-semiannual/analysis")
    args = parser.parse_args()
    experiment = yaml.safe_load((ROOT / "configs/frontier_eci/experiment.yaml").read_text())
    candidates = experiment["candidates"]
    releases = load_releases(experiment)
    checkpoint = [slug.split("/")[-1] for slug in experiment["judges"]["checkpoint"]]
    bridge = [slug.split("/")[-1] for slug in experiment["judges"]["bridge"]]
    judges = checkpoint + bridge
    quartiles = pd.read_csv(ROOT / "configs/frontier_eci/item_quartiles.csv", dtype={"item": str})
    quartiles = quartiles[quartiles.bench.isin(args.benches)]
    hf = pd.read_parquet(ROOT / "judge_results/frontier/eci-semiannual/hf/verdicts.parquet")
    new_verdicts = pd.read_parquet(args.new_tables / "verdicts.parquet")
    verdicts = combine_verdicts(hf, new_verdicts, judges)
    verdicts_for_prediction = pd.concat([verdicts, consensus_verdicts(verdicts)], ignore_index=True)
    analysis_judges = judges + ["consensus"]

    canonical = pd.read_parquet(ROOT / "judge_results/frontier/eci-semiannual/canonical-source/identity.parquet",
                                columns=["bench", "model", "item", "epoch", "correct", "is_canonical"])
    frontier = pd.read_parquet(args.new_tables / "rollouts.parquet")
    rollouts = merge_rollouts(canonical, frontier).merge(quartiles[["bench", "item", "item_quartile"]],
                                                         on=["bench", "item"], how="inner")
    actual_sources = rollouts.groupby(["bench", "model"]).actual_source.first()
    accuracy = rollouts.groupby(["bench", "item_quartile", "model"]).correct.mean()
    q1 = quartiles.query("item_quartile == 1")[["bench", "item"]]
    q1_verdicts = verdicts_for_prediction.merge(q1, on=["bench", "item"], how="inner")

    rng = np.random.default_rng(args.seed)
    predictions = []
    for bench in args.benches:
        for judge in analysis_judges:
            judged = q1_verdicts.query("bench == @bench and judge == @judge")
            if judged.empty:
                raise ValueError(f"no Q1 verdicts for {bench}/{judge}")
            for target in candidates:
                prefix = prefix_verdicts(judged, target, releases)
                for target_quartile in (2, 3, 4):
                    actual = accuracy.loc[(bench, target_quartile)]
                    row = predict_block(prefix, actual, target, releases, args.resamples, rng)
                    predictions.append({"bench": bench, "judge": judge,
                                        "target_quartile": target_quartile, **row})
    predictions = pd.DataFrame(predictions)
    predictions["actual_source"] = [actual_sources.get((row.bench, row.model))
                                    for row in predictions.itertuples()]
    predictions["target_release"] = predictions.model.map(lambda model: release_date(releases, model))
    predictions["judge_release"] = predictions.judge.map(lambda judge: release_date(releases, judge))
    predictions["post_judge_release"] = predictions.target_release > predictions.judge_release
    predictions["error"] = predictions.predicted - predictions.actual
    predictions["abs_error"] = predictions.error.abs()

    scored = predictions.dropna(subset=["predicted", "actual"])
    metrics = (scored.groupby(["bench", "judge", "target_quartile"])
               .agg(mae=("abs_error", "mean"), bias=("error", "mean"),
                    spearman=("predicted", lambda x: spearmanr(x, scored.loc[x.index, "actual"]).statistic),
                    n_models=("model", "size"), n_anchors_min=("n_anchors", "min"),
                    n_anchors_max=("n_anchors", "max"),
                    n_bootstrap_min=("n_bootstrap", "min")).reset_index())
    temporal_metrics = (scored[scored.judge != "consensus"]
                        .groupby(["bench", "judge", "target_quartile", "post_judge_release"])
                        .agg(mae=("abs_error", "mean"), bias=("error", "mean"),
                             n_models=("model", "size")).reset_index())
    model_errors = (scored[scored.judge == "consensus"].groupby(["bench", "model"])
                    .agg(mean_error=("error", "mean"), mae=("abs_error", "mean"),
                         worst_abs_error=("abs_error", "max")).reset_index())
    features, feature_summary = transcript_features(ROOT / "configs/frontier_eci/runs", quartiles, candidates)
    judge_diagnostics = diagnostics(verdicts, new_verdicts, features, quartiles)

    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    predictions.to_csv(args.out / "predictions.csv", index=False)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    temporal_metrics.to_csv(args.out / "metrics_by_judge_time.csv", index=False)
    model_errors.to_csv(args.out / "model_errors.csv", index=False)
    feature_summary.to_csv(args.out / "transcript_visibility.csv", index=False)
    judge_diagnostics.to_csv(args.out / "judge_diagnostics.csv", index=False)
    plot_predictions(predictions, candidates, experiment["releases"], checkpoint + ["consensus"],
                     args.benches, figures, "frontier_predictions_checkpoint_judges")
    plot_predictions(predictions, candidates, experiment["releases"], bridge + ["consensus"],
                     args.benches, figures, "frontier_predictions_bridge_judges")
    plot_error_heatmap(predictions, candidates, judges, args.benches, figures)
    plot_visibility(feature_summary, model_errors, candidates, args.benches, figures)
    frontier_figures = figures / "frontier_style"
    frontier_figures.mkdir(exist_ok=True)
    for (bench, judge, quartile), table in predictions[predictions.judge != "consensus"].groupby(
            ["bench", "judge", "target_quartile"]):
        for style in ("band", "points"):
            plot_frontier_style(table, judge, bench, quartile, frontier_figures, style)
    (args.out / "summary.json").write_text(json.dumps({
        "verdicts": len(verdicts), "q1_verdicts": len(q1_verdicts),
        "judges": judges, "consensus": "equal judge weight per exact verdict cell",
        "candidates": candidates, "resamples": args.resamples,
    }, indent=2) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
