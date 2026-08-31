"""Analyze matched CoT-on/off verdicts and the non-reasoning-only subset."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.frontier_eci.analyze import save  # noqa: E402
from scripts.frontier_eci.canonical_analysis import cap_pair_items  # noqa: E402
from scripts.frontier_eci.rank_regression import (  # noqa: E402
    _atomic_write_json, _combined_hard, combined_frontier_grid,
    fit_combined_with_bootstrap, load_rank_analysis_inputs,
)
from scripts.judge.verdict_tables import read_verdicts  # noqa: E402


def match_conditions(off: pd.DataFrame, cot: pd.DataFrame) -> pd.DataFrame:
    keys = ["bench", "judge", "item", "model_left", "model_right",
            "epoch_left", "epoch_right", "judge_epoch"]
    matched = off[keys + ["pick_left"]].merge(
        cot[keys + ["pick_left"]], on=keys, how="inner", validate="one_to_one",
        suffixes=("_off", "_cot"))
    matched["flipped"] = matched.pick_left_off.ne(matched.pick_left_cot)
    return matched.sort_values(keys).reset_index(drop=True)


def restrict_to_manifest(protocol: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    table = protocol.copy()
    pair = np.sort(table[["model_left", "model_right"]].astype(str).to_numpy(), axis=1)
    table["opponent"], table["target"] = pair[:, 0], pair[:, 1]
    keys = ["bench", "judge", "target", "opponent", "item"]
    return table.merge(manifest[keys].drop_duplicates(), on=keys, how="inner")[protocol.columns]


def metrics(points: pd.DataFrame, value: str) -> pd.DataFrame:
    table = points.dropna(subset=["actual", value]).copy()
    table["error"] = table[value] - table.actual
    rows = []
    for (bench, judge), block in table.groupby(["bench", "judge"], sort=True):
        rho = (spearmanr(block[value], block.actual).statistic
               if block[value].nunique() > 1 and block.actual.nunique() > 1 else np.nan)
        rows.append({
            "bench": bench, "judge": judge, "n_models": len(block),
            "mae": block.error.abs().mean(), "bias": block.error.mean(),
            "spearman": rho,
        })
    return pd.DataFrame(rows)


def write_cot_figures(off: pd.DataFrame, cot: pd.DataFrame, out: Path) -> None:
    keys = ["bench", "judge", "model"]
    columns = keys + ["actual", "target_release", "judge_release",
                      "fitted", "fitted_low", "fitted_high"]
    table = off[columns].merge(
        cot[columns], on=keys, validate="one_to_one", suffixes=("_off", "_cot"))
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge), block in table.groupby(["bench", "judge"], sort=True):
        block = block.sort_values("target_release_off").copy()
        x = pd.to_datetime(block.target_release_off)
        judge_date = pd.Timestamp(block.judge_release_off.iloc[0])
        fig, ax = plt.subplots(figsize=(1.75, 1.55))
        handles = []
        for suffix, color, label in (("off", "#2a78d6", "CoT off"),
                                     ("cot", "#eb6834", "CoT on")):
            low = block[f"fitted_low_{suffix}"].clip(0, 1)
            high = block[f"fitted_high_{suffix}"].clip(0, 1)
            value = block[f"fitted_{suffix}"].clip(0, 1)
            band = ax.fill_between(x, low, high, color=color, alpha=0.16, lw=0)
            line, = ax.plot(x, value, color=color, lw=1)
            handles.append(((band, line), label))
        actual, = ax.plot(x, block.actual_off, "o", ms=3.5, color="#222222", zorder=4)
        release = ax.axvline(judge_date, color="#777777", lw=0.8, ls="--")
        first = min(x.min(), judge_date).replace(month=1, day=1)
        last = max(x.max(), judge_date).replace(month=1, day=1) + pd.DateOffset(years=1)
        ticks = pd.date_range(first, last, freq="YS")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(first, last)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Model release")
        ax.set_ylabel("Combined hard accuracy")
        name = "GPQA" if bench == "gpqa" else "SWE-mini"
        ax.text(0, 1.02, f"{name} · Q2–Q4", color="#777777", fontsize=7,
                ha="left", va="bottom", transform=ax.transAxes)
        ax.spines["left"].set_bounds(0, 1)
        ax.spines["bottom"].set_bounds(mdates.date2num(first), mdates.date2num(last))
        for spine in ("left", "bottom"):
            ax.spines[spine].set_position(("outward", 5))
        ax.legend(
            [actual, handles[0][0], handles[1][0], release],
            ["Actual", handles[0][1], handles[1][1], "Judge release"],
            handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower left",
            bbox_to_anchor=(0, 1.08), frameon=False, handletextpad=0.5,
            borderaxespad=0)
        save(fig, out / f"cot-on-off_{judge}_{bench}_q2-q4_frontier")


def write_non_reasoning_figures(points: pd.DataFrame, out: Path) -> None:
    labels = {"gpt-4o-1120": "GPT-4o", "gpt-4.1": "GPT-4.1",
              "claude-3.7-sonnet": "Claude 3.7", "mistral-medium-3": "Mistral Medium 3"}
    order = list(labels)
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge), block in points.groupby(["bench", "judge"], sort=True):
        block = block.set_index("model").reindex(order).dropna(subset=["actual"])
        x = np.arange(len(block))
        fig, ax = plt.subplots(figsize=(1.75, 1.55))
        value = block.selected_fitted.clip(0, 1)
        low, high = block.fitted_low.clip(0, 1), block.fitted_high.clip(0, 1)
        predicted = ax.errorbar(
            x + 0.06, value, yerr=np.vstack([(value - low).clip(lower=0),
                                              (high - value).clip(lower=0)]),
            fmt="o", ms=3, lw=1, capsize=0, color="#2a78d6")
        actual, = ax.plot(x - 0.06, block.actual, "o", ms=3.5, color="#222222")
        for position, truth, fitted in zip(x, block.actual, value):
            ax.plot([position, position], [truth, fitted], ":", color="#222222", lw=1)
        ax.set_xlim(-0.5, len(block) - 0.5)
        ax.set_xticks(x, [labels[model] for model in block.index], rotation=90, ha="center")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Combined hard accuracy")
        name = "GPQA" if bench == "gpqa" else "SWE-mini"
        ax.text(0, 1.02, name, color="#777777", fontsize=7, ha="left", va="bottom",
                transform=ax.transAxes)
        ax.spines["left"].set_bounds(0, 1)
        ax.spines["bottom"].set_bounds(0, len(block) - 1)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_position(("outward", 5))
        residual, = ax.plot([], [], ":", color="#222222", lw=1)
        ax.legend([actual, predicted, residual], ["Actual", "Predicted", "Residual"],
                  loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False,
                  handletextpad=0.5, borderaxespad=0)
        save(fig, out / f"non-reasoning_{judge}_{bench}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--canonical-logs", type=Path,
                        default=ROOT / "logs/frontier-eci-canonical-anchors")
    parser.add_argument("--cot-logs", type=Path,
                        default=ROOT / "logs/reasoning_ablation/cot_on")
    parser.add_argument("--topup-logs", type=Path,
                        default=ROOT / "logs/reasoning_ablation/non_reasoning_topup")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "configs/reasoning_ablation/cot_on/selected_pair_items.csv")
    parser.add_argument("--result-root", type=Path,
                        default=ROOT / "judge_results/frontier/reasoning_ablation/analysis")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config, hard, protocol, _, _ = load_rank_analysis_inputs(args.config, args.canonical_logs)
    judges = ["gpt-5.2", "gemini-2.5-pro"]
    manifest = pd.read_csv(args.manifest, dtype={"item": str})
    off = restrict_to_manifest(protocol[protocol.judge.isin(judges)], manifest)
    cot, _ = read_verdicts([args.cot_logs])
    matched = match_conditions(off, cot)
    keys = ["bench", "judge", "item", "model_left", "model_right",
            "epoch_left", "epoch_right", "judge_epoch"]
    off_matched = matched[keys].assign(pick_left=matched.pick_left_off)
    cot_matched = matched[keys].assign(pick_left=matched.pick_left_cot)

    cot_root = args.result_root / "cot_on_off"
    cot_root.mkdir(parents=True, exist_ok=True)
    matched.to_csv(cot_root / "matched_verdicts.csv", index=False)
    targets = list(config["candidates"])
    releases = config["releases"]
    combined = _combined_hard(hard)
    actual = (combined.groupby(["bench", "model"], sort=True)
              .agg(actual=("correct", "mean"), n_items=("correct", "size")).reset_index())
    grids = {}
    for condition, rows in (("cot_off", off_matched), ("cot_on", cot_matched)):
        abilities, ranking, fits, points = fit_combined_with_bootstrap(
            rows, hard, resamples=args.resamples, seed=args.seed,
            workers=args.workers, pair_limit=None)
        grid = combined_frontier_grid(
            points, actual, targets, judges, releases, sorted(hard.bench.unique()))
        condition_root = cot_root / condition
        condition_root.mkdir(parents=True, exist_ok=True)
        abilities.to_csv(condition_root / "abilities.csv", index=False)
        ranking.to_csv(condition_root / "ranking_blocks.csv", index=False)
        fits.to_csv(condition_root / "fits.csv", index=False)
        points.to_csv(condition_root / "training_fitted.csv", index=False)
        grid.to_csv(condition_root / "frontier_fitted.csv", index=False)
        metric = metrics(grid, "fitted").assign(condition=condition)
        metric.to_csv(condition_root / "metrics.csv", index=False)
        grids[condition] = grid
    write_cot_figures(grids["cot_off"], grids["cot_on"], cot_root / "figures")

    classification = pd.read_csv(
        ROOT / "judge_results/frontier/reasoning_ablation/preflight/model_reasoning.csv")
    reasoning = set(classification.loc[classification.is_reasoning, "model"])
    for suffix in ("off", "cot"):
        picked = np.where(matched[f"pick_left_{suffix}"].astype(bool),
                          matched.model_left, matched.model_right)
        matched[f"picked_reasoning_{suffix}"] = pd.Series(picked).isin(reasoning)
    matched.to_csv(cot_root / "matched_verdicts.csv", index=False)
    flip_summary = (matched.groupby(["bench", "judge"], sort=True)
                    .agg(n=("flipped", "size"), flip_rate=("flipped", "mean"),
                         reasoning_pick_off=("picked_reasoning_off", "mean"),
                         reasoning_pick_cot=("picked_reasoning_cot", "mean")).reset_index())
    flip_summary.to_csv(cot_root / "verdict_summary.csv", index=False)

    topup, _ = read_verdicts([args.topup_logs])
    non_reasoning_models = {"gpt-4o-1120", "gpt-4.1", "claude-3.7-sonnet", "mistral-medium-3"}
    non_reasoning = pd.concat([protocol, topup], ignore_index=True)
    non_reasoning = non_reasoning[
        non_reasoning.judge.isin(judges)
        & non_reasoning.model_left.isin(non_reasoning_models)
        & non_reasoning.model_right.isin(non_reasoning_models)]
    non_reasoning = cap_pair_items(non_reasoning, 8)
    nr_hard = hard[hard.model.isin(non_reasoning_models)]
    abilities, ranking, fits, points = fit_combined_with_bootstrap(
        non_reasoning, nr_hard, resamples=args.resamples, seed=args.seed,
        workers=args.workers, pair_limit=None)
    nr_root = args.result_root / "non_reasoning"
    nr_root.mkdir(parents=True, exist_ok=True)
    abilities.to_csv(nr_root / "abilities.csv", index=False)
    ranking.to_csv(nr_root / "ranking_blocks.csv", index=False)
    fits.to_csv(nr_root / "fits.csv", index=False)
    points.to_csv(nr_root / "fitted.csv", index=False)
    metrics(points, "selected_fitted").to_csv(nr_root / "metrics.csv", index=False)
    write_non_reasoning_figures(points, nr_root / "figures")

    summary = {
        "matched_verdict_rows": len(matched),
        "cot_fit_blocks": sum(len(pd.read_csv(cot_root / c / "fits.csv"))
                              for c in ("cot_off", "cot_on")),
        "non_reasoning_models": sorted(non_reasoning_models),
        "non_reasoning_fit_blocks": len(fits),
        "resamples": args.resamples,
    }
    _atomic_write_json(summary, args.result_root / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
