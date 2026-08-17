"""Gap curves for the isolated 2026-08-17 Hugging Face judging snapshot."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / "data" / "hf-judging-2026-08-17"
RAW = ROUND / "raw"
DERIVED = ROUND / "derived"
FIGURES = ROOT / "judge_results" / "hf-judging-2026-08-17"
BENCHMARK_SCORES = ROUND / "benchmark-scores"
BINS = tuple(range(0, 61, 5))

EXPECTED_MISSING = {
    ("GPQA", "deepseek-r1-0528"),
    ("GPQA", "kimi-k2.6"),
    ("GPQA", "ling-2.6-flash"),
}


def orient_accuracy(accuracy, major_score, minor_score):
    return accuracy if major_score > minor_score else 1 - accuracy


def keep_judge(judge):
    """Standing analysis rule: Phi-family models are never judges."""
    return not judge.casefold().startswith("phi")


def load_benchmark_scores(root=BENCHMARK_SCORES):
    totals = defaultdict(lambda: [0.0, 0])
    for path in root.glob("*/*/*/*/logs.json"):
        bench, _, model, _, _ = path.relative_to(root).parts
        for log in json.loads(path.read_text()).values():
            if log.get("status") != "success" or log.get("invalidated") or "results" not in log:
                continue
            score = next(s for s in log["results"]["scores"] if "accuracy" in s["metrics"])
            n = score["scored_samples"]
            totals[(bench.upper(), model)][0] += score["metrics"]["accuracy"]["value"] * n
            totals[(bench.upper(), model)][1] += n
    return {key: correct / n for key, (correct, n) in totals.items()}


def summarize(rows, edges=BINS):
    grouped = defaultdict(list)
    for row in rows:
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= row["gap"] < hi:
                grouped[(row["eval"], lo, hi, row["judge"])].append(row)
                break

    per_judge = []
    for (eval_name, lo, hi, judge), cells in sorted(grouped.items()):
        n = sum(cell["n"] for cell in cells)
        per_judge.append({
            "eval": eval_name,
            "gap_lo": lo,
            "gap_hi": hi,
            "gap_mid": (lo + hi) / 2,
            "judge": judge,
            "n_pairs": len({cell["pair"] for cell in cells}),
            "n_verdicts": n,
            "accuracy": sum(cell["correct"] for cell in cells) / n,
        })

    by_bin = defaultdict(list)
    for row in per_judge:
        by_bin[(row["eval"], row["gap_lo"], row["gap_hi"])].append(row)

    summary = []
    for (eval_name, lo, hi), judges in sorted(by_bin.items()):
        values = [row["accuracy"] for row in judges]
        mean = statistics.fmean(values)
        half = (t.ppf(0.975, len(values) - 1) * statistics.stdev(values)
                / math.sqrt(len(values))) if len(values) >= 3 else None
        source = [cell for cell in rows
                  if cell["eval"] == eval_name and lo <= cell["gap"] < hi]
        summary.append({
            "eval": eval_name,
            "gap_lo": lo,
            "gap_hi": hi,
            "gap_mid": (lo + hi) / 2,
            "accuracy": mean,
            "ci_low": max(0, mean - half) if half is not None else None,
            "ci_high": min(1, mean + half) if half is not None else None,
            "n_judges": len(values),
            "n_pairs": len({cell["pair"] for cell in source}),
            "n_verdicts": sum(cell["n"] for cell in source),
        })
    return summary, per_judge


def load_rows(raw=RAW, benchmark_root=BENCHMARK_SCORES):
    benchmark_scores = load_benchmark_scores(benchmark_root)
    rows, missing = [], set()
    for path in sorted(raw.glob("*/logs.json")):
        dirname = path.parent.name
        bench = next((name for name in ("gpqa", "mbpp") if f"-{name}-" in dirname), None)
        if bench is None:
            continue
        eval_name = bench.upper()
        judge = dirname.removeprefix("pilot-").split(f"-{bench}-", 1)[0]
        if not keep_judge(judge):
            continue
        for log in json.loads(path.read_text()).values():
            if log.get("status") != "success" or log.get("invalidated"):
                continue
            meta = log["metadata"]
            major = Path(meta["major_path"]).parts[-3]
            minor = Path(meta["minor_path"]).parts[-3]
            if (eval_name, major) not in benchmark_scores or (eval_name, minor) not in benchmark_scores:
                missing.update((eval_name, model) for model in (major, minor)
                               if (eval_name, model) not in benchmark_scores)
                continue
            score = next(s for s in log["results"]["scores"] if s["name"] == "judge_scorer")
            n = score["scored_samples"]
            accuracy = orient_accuracy(score["metrics"]["accuracy"]["value"],
                                       benchmark_scores[(eval_name, major)],
                                       benchmark_scores[(eval_name, minor)])
            rows.append({
                "eval": eval_name,
                "judge": judge,
                "pair": "|".join(sorted((major, minor))),
                "gap": 100 * abs(benchmark_scores[(eval_name, major)]
                                   - benchmark_scores[(eval_name, minor)]),
                "correct": accuracy * n,
                "n": n,
            })

    unexpected = missing - EXPECTED_MISSING
    if unexpected:
        raise ValueError(f"Missing benchmark scores for unexpected models: {sorted(unexpected)}")
    if not rows:
        raise ValueError(f"No usable result rows in {raw}")
    print("Excluded from per-benchmark curves (missing source scores): "
          + ", ".join(f"{bench}/{model}" for bench, model in sorted(missing)))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, per_judge, out=FIGURES / "gap_curve_by_eval"):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.9), sharex=True, sharey=True)
    for ax, eval_name in zip(axes, ("GPQA", "MBPP")):
        for judge in sorted({row["judge"] for row in per_judge if row["eval"] == eval_name}):
            cells = [row for row in per_judge
                     if row["eval"] == eval_name and row["judge"] == judge]
            ax.plot([row["gap_mid"] for row in cells], [row["accuracy"] for row in cells],
                    color="#8a8a86", alpha=0.32, lw=0.6, marker="o", ms=1.8)
        cells = [row for row in summary if row["eval"] == eval_name]
        xs = [row["gap_mid"] for row in cells]
        ys = [row["accuracy"] for row in cells]
        ax.plot(xs, ys, color="#111111", lw=1, marker="o", ms=3, zorder=4)
        for row in cells:
            if row["ci_low"] is not None:
                ax.errorbar(row["gap_mid"], row["accuracy"],
                            yerr=[[row["accuracy"] - row["ci_low"]],
                                  [row["ci_high"] - row["accuracy"]]],
                            color="#111111", lw=1, capsize=0, zorder=3)
        ax.axhline(0.5, color="#8a8a86", lw=0.8, ls="--")
        ax.set_title(eval_name, loc="left", color="#66645f", fontsize=7)
        ax.set_xlabel("Accuracy gap (pp)")
        ax.set_xlim(0, 60)
        ax.set_xticks((0, 20, 40, 60))
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.5, 1))
        ax.grid(axis="y", color="#d3d2c9", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 5))
    axes[0].set_ylabel("Judge accuracy")
    fig.legend(handles=[
        Line2D([], [], color="#8a8a86", alpha=0.6, lw=0.6, marker="o", ms=2,
               label="Individual judge"),
        Line2D([], [], color="#111111", lw=1, marker="o", ms=3,
               label="Mean ± 95% CI"),
        Line2D([], [], color="#8a8a86", lw=0.8, ls="--", label="Chance"),
    ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01),
       handlelength=1.4, columnspacing=0.8)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.32, top=0.92, wspace=0.24)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main():
    rows = load_rows()
    summary, per_judge = summarize(rows)
    write_csv(DERIVED / "gap_curve_summary.csv", summary)
    write_csv(DERIVED / "gap_curve_per_judge.csv", per_judge)
    plot(summary, per_judge)
    print(f"{len(rows)} order-level cells; {len({r['pair'] for r in rows})} pairs; "
          f"{len({r['judge'] for r in rows})} judges -> {FIGURES}")


if __name__ == "__main__":
    main()
