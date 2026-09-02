"""Figure set for the SWE-bench-Verified-Mini and Intercode-CTF pilot judging.

Consumes the viewer datasets baked by swe_intercode_viewer_export.py (pairs/,
transcripts/, index.json) and writes the usual per-round tables and figures to
judge_results/pilot-swebench/ and judge_results/pilot-intercode/:

  summary.csv, models.csv, verdicts.csv
  figures/0_model_accuracy       solve rate per transcript model
  figures/1_run_overview         every run: judge accuracy vs pair gap
  figures/2a_gap_curve_by_judge  per-judge dots + binned curve
  figures/2b_gap_curve_pooled    binned mean +/- 95% CI across judges
  figures/3_judge_strength       judge x pair accuracy heatmap
  figures/9_position_bias        P(picked slot 1) per judge
  figures/10_length_bias         P(picked the longer transcript) per judge
  figures/11_capability_vs_length  solve rate vs mean transcript length

Also holds the shared judging-log loaders used by the exporter. Truncation,
release-date, and IRT item figures from the GPQA rounds do not apply here
(no truncation arms, release dates, or item parameters for these benches).

Usage:
  uv run python scripts/judge/swe_intercode_figures.py
"""

import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analyses"))

import numpy as np
from scipy.stats import t
from inspect_ai.log import read_eval_log

from plots import (COL, HEAT, INK, INK2, MUTED, PALETTE, WIDE, chance, dot_ci,
                   legend, plt, save, style)

JUDGING = ROOT / "data" / "pilot-judging" / "pilot"
BENCHES = {
    "swe_bench_mini": (ROOT / "data/swe_bench_mini/477dc96c", "SWE-bench Verified Mini"),
    "intercode": (ROOT / "data/intercode/463eb861", "Intercode CTF"),
}
DATASETS = {  # bench -> (viewer dataset id, dropdown label)
    "swe_bench_mini": ("swe-mini-pilot", "SWE-bench Mini pilot"),
    "intercode": ("intercode-pilot", "Intercode CTF pilot"),
}
OUT_DIRS = {"swe_bench_mini": "pilot-swebench", "intercode": "pilot-intercode"}
VIEWER_DATA = ROOT / "interactive_dashboards" / "transcript_viewer" / "data"
BINS = tuple(range(0, 101, 10))

# weak -> strong; fixed slots, never re-mapped
JUDGES = ["Llama-3.1-8B-Instruct", "gemma-4-31b-it", "ling-2.6-flash",
          "gpt-oss-120b", "deepseek-r1-0528", "o4-mini", "deepseek-v4-flash-0731"]


# --- shared judging-log loaders (also imported by the viewer exporter) ----

def load_solve_rates(rollout_root: Path) -> dict:
    """model -> solve rate pooled over both rollout epochs (verdict-weighted)."""
    totals = defaultdict(lambda: [0.0, 0])
    for path in rollout_root.glob("*/*/logs.json"):
        model = path.parts[-3]
        for log in json.loads(path.read_text()).values():
            if log.get("status") != "success" or log.get("invalidated") or "results" not in log:
                continue
            score = next(s for s in log["results"]["scores"]
                         if {"accuracy", "mean"} & set(s["metrics"]))
            metric = score["metrics"].get("accuracy") or score["metrics"]["mean"]
            n = score["scored_samples"]
            totals[model][0] += metric["value"] * n
            totals[model][1] += n
    return {model: solved / n for model, (solved, n) in totals.items()}


def judging_logs(bench: str):
    """Newest successful log per (judge, pair, rollout epoch)."""
    picked = {}
    for path in sorted(glob.glob(f"{JUDGING}/{bench}/*/*/*.eval")):
        header = read_eval_log(path, header_only=True)
        if header.status != "success":
            continue
        judge, pair = Path(path).parts[-3:-1]
        epoch = Path(header.eval.metadata["major_path"]).parts[-2]
        picked[(judge, pair, epoch)] = (path, header)
    return picked


# --- viewer-dataset digestion ---------------------------------------------

def load_dataset(bench: str):
    """index.json + flat verdict rows from the baked viewer dataset."""
    data = VIEWER_DATA / DATASETS[bench][0]
    index = json.loads((data / "index.json").read_text())
    lengths = {}  # model -> {item: chars}
    for path in (data / "transcripts").glob("*.json"):
        lengths[path.stem] = {item: len(text)
                              for item, text in json.loads(path.read_text()).items()}
    verdicts = []
    for path in (data / "pairs").glob("*.json"):
        strong, weak, _ = path.stem.split("__")
        gap = abs(index["models"][strong]["accuracy"] - index["models"][weak]["accuracy"])
        for item, judges in json.loads(path.read_text()).items():
            len_s = lengths.get(strong, {}).get(item)
            len_w = lengths.get(weak, {}).get(item)
            for judge, cell in judges.items():
                for order in cell["orders"]:
                    if order is None:
                        continue
                    picked_strong = order["picked_major"]
                    picked_longer = None
                    if picked_strong is not None and len_s and len_w and len_s != len_w:
                        picked_longer = int((len_s > len_w) == bool(picked_strong))
                    verdicts.append({
                        "item": item, "epoch": item.rsplit("__ep", 1)[1],
                        "judge": judge, "major": strong, "minor": weak,
                        "gap": 100 * gap, "slot_of_major": order["slot_of_major"],
                        "pick": order["pick"], "picked_major": picked_strong,
                        "picked_longer": picked_longer,
                        "confidence": order["confidence"],
                        "len_major": len_s, "len_minor": len_w,
                    })
    return index, verdicts


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sem_ci(values):
    m = statistics.fmean(values)
    if len(values) < 3:
        return m, m, m
    half = t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return m, m - half, m + half


def wilson(c, n, z=1.96):
    if not n:
        return None, None
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def binned(rows, key):
    """(bin mid, rate over rows, n) per occupied bin; rows need gap + key."""
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        cell = [r[key] for r in rows if lo <= r["gap"] < hi and r[key] is not None]
        if cell:
            out.append(((lo + hi) / 2, statistics.fmean(cell), len(cell)))
    return out


# --- figures ---------------------------------------------------------------

def judge_color(index, judge):
    return index["colors"].get(judge, "#888888")


def fig_model_accuracy(index, out):
    members = sorted(((meta["accuracy"], m) for m, meta in index["models"].items()
                      if meta["arm"] != "judge" and meta["accuracy"] is not None))
    fig, ax = plt.subplots(figsize=(COL, 0.17 * len(members) + 0.8))
    for y, (acc, m) in enumerate(members):
        ax.plot(acc, y, "o", ms=4.5, color=PALETTE[0], zorder=3)
    ax.set_yticks(range(len(members)))
    ax.set_yticklabels([m for _, m in members], fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("solve rate (both epochs pooled)")
    style(ax, grid_axis="x")
    save(fig, out / "0_model_accuracy")


def fig_run_overview(index, runs, out):
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for judge in JUDGES:
        mine = [r for r in runs if r["judge"] == judge]
        if not mine:
            continue
        ax.scatter([100 * r["d_acc"] for r in mine], [r["score"] for r in mine],
                   s=14, color=judge_color(index, judge), label=judge, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
    chance(ax)
    ax.set_xlabel("solve-rate gap (pp)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1)
    legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.02), fontsize=6.5)
    style(ax)
    save(fig, out / "1_run_overview")


def fig_gap_curve_by_judge(index, runs, verdicts, out):
    fig, axes = plt.subplots(2, 4, figsize=(WIDE, 3.4), sharex=True, sharey=True)
    for ax, judge in zip(axes.flat, JUDGES):
        color = judge_color(index, judge)
        mine = [r for r in runs if r["judge"] == judge and r["d_acc"] > 0]
        ax.scatter([100 * r["d_acc"] for r in mine], [r["score"] for r in mine],
                   s=[6 + r["n_items"] / 8 for r in mine], color=color, alpha=0.45,
                   edgecolors="white", linewidths=0.4, zorder=2)
        cells = [v for v in verdicts if v["judge"] == judge and v["gap"] > 0]
        pts = binned(cells, "picked_major")
        for x, y, n in pts:
            lo, hi = wilson(y * n, n)
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], color=color, marker="o",
                        ms=3.5, capsize=0, lw=1, zorder=4)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1, zorder=3)
        chance(ax)
        ax.set_title(judge, fontsize=7, color=INK2, loc="left")
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.5, 1))
        ax.set_xlim(0, 100)
        ax.set_xticks((0, 50, 100))
        style(ax)
    spare = axes.flat[-1]
    spare.axis("off")
    spare.text(0.5, 0.5, "dot area scales with\nitems in the run\n\nbinned line\n± 95% Wilson CI",
               ha="center", va="center", color=MUTED, fontsize=7)
    for ax in axes[1][:3]:
        ax.set_xlabel("solve-rate gap (pp)")
    for ax in axes[:, 0]:
        ax.set_ylabel("judge accuracy")
    save(fig, out / "2a_gap_curve_by_judge")


def fig_gap_curve_pooled(index, verdicts, out):
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    per_judge = {}
    for judge in JUDGES:
        cells = [v for v in verdicts if v["judge"] == judge and v["gap"] > 0]
        pts = binned(cells, "picked_major")
        if pts:
            per_judge[judge] = {x: y for x, y, _ in pts}
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED,
                    alpha=0.35, lw=0.6, marker="o", ms=1.8, zorder=2)
    xs, ys, prev = [], [], None
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mid = (lo + hi) / 2
        values = [curve[mid] for curve in per_judge.values() if mid in curve]
        if not values:
            prev = None
            continue
        if prev is not None and prev != lo:
            xs.append(math.nan)
            ys.append(math.nan)
        m, ci_lo, ci_hi = sem_ci(values)
        if len(values) >= 3:
            ax.errorbar(mid, m, yerr=[[m - max(ci_lo, 0)], [min(ci_hi, 1) - m]],
                        color=INK, lw=1, capsize=0, zorder=4)
        xs.append(mid)
        ys.append(m)
        prev = hi
    ax.plot(xs, ys, color=INK, lw=1, marker="o", ms=3, zorder=4)
    chance(ax)
    ax.set_xlabel("solve-rate gap (pp)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1)
    ax.set_yticks((0, 0.5, 1))
    ax.set_xlim(0, 100)
    ax.text(1.0, 1.02, "mean ± 95% CI across judges", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=7)
    style(ax)
    save(fig, out / "2b_gap_curve_pooled")


def fig_judge_strength(index, runs, out):
    acc = {m: meta["accuracy"] for m, meta in index["models"].items()}
    pairs = sorted({(r["major"], r["minor"]) for r in runs},
                   key=lambda p: acc[p[0]] - acc[p[1]], reverse=True)
    judges = [j for j in JUDGES if any(r["judge"] == j for r in runs)]
    M = np.full((len(pairs), len(judges)), np.nan)
    for r in runs:
        M[pairs.index((r["major"], r["minor"])), judges.index(r["judge"])] = r["score"]

    fig, ax = plt.subplots(figsize=(0.62 * len(judges) + 2.6, 0.26 * len(pairs) + 1.4))
    im = ax.imshow(M, cmap=HEAT, vmin=0, vmax=1, aspect="auto")
    for i, (m, n) in enumerate(pairs):
        for k, j in enumerate(judges):
            if j in (m, n) and np.isfinite(M[i, k]):
                ax.add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK, lw=1.4, zorder=4))
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{m} v {n}  ·  Δ{acc[m] - acc[n]:.2f}" for m, n in pairs],
                       fontsize=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color=INK2 if 0.2 < M[i, j] < 0.8 else "white")
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("judge accuracy (0.5 = chance)", color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.set_title("outlined = judge is a pair member · Δ0.00 pairs keep config order",
                 color=MUTED, fontsize=7, loc="right")
    save(fig, out / "3_judge_strength")


def judge_rate_rows(ax, index, verdicts, value):
    """Per-judge dot+CI rows of a per-run rate; CI across runs."""
    present = []
    for judge in JUDGES:
        runs = defaultdict(list)
        for v in verdicts:
            if v["judge"] == judge:
                runs[(v["major"], v["minor"])].append(v)
        rates = [value(rows) for rows in runs.values() if value(rows) is not None]
        if not rates:
            continue
        m, lo, hi = sem_ci(rates)
        dot_ci(ax, len(present), m, lo, hi, judge_color(index, judge))
        ax.annotate(f"n={len(rates)} runs", xy=(hi + 0.008, len(present)),
                    va="center", ha="left", color=MUTED, fontsize=7)
        present.append(judge)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=6.5)
    ax.invert_yaxis()
    chance(ax, 0.5, axis="x")
    style(ax, grid_axis="x")


def fig_position_bias(index, verdicts, out):
    def slot1(rows):
        picks = [v["pick"] for v in rows if v["pick"] in ("1", "2")]
        return (sum(p == "1" for p in picks) / len(picks)) if picks else None
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    judge_rate_rows(ax, index, verdicts, slot1)
    ax.set_xlabel("P(picked slot 1)")
    ax.text(1.0, 1.02, "95% CI over runs", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=7)
    save(fig, out / "9_position_bias")


def fig_length_bias(index, verdicts, out):
    def longer(rows):
        cells = [v["picked_longer"] for v in rows if v["picked_longer"] is not None]
        return statistics.fmean(cells) if cells else None
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    judge_rate_rows(ax, index, verdicts, longer)
    ax.set_xlabel("P(picked the longer transcript)")
    ax.text(1.0, 1.02, "95% CI over runs · equal-length items excluded",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7)
    save(fig, out / "10_length_bias")


def fig_capability_vs_length(index, verdicts, out):
    lens = defaultdict(list)
    for v in verdicts:
        if v["len_major"]:
            lens[v["major"]].append(v["len_major"])
        if v["len_minor"]:
            lens[v["minor"]].append(v["len_minor"])
    pts = sorted((index["models"][m]["accuracy"], statistics.fmean(ls) / 1000, m)
                 for m, ls in lens.items() if index["models"][m]["accuracy"] is not None)
    fig, ax = plt.subplots(figsize=(WIDE, 3.4))
    offsets = [(0, 6), (0, -10), (26, 3), (-26, 3), (0, 13), (0, -17), (40, -8), (-40, -8)]
    short = lambda m: m.removeprefix("nvidia-").removesuffix("-bf16")
    for i, (x, y, m) in enumerate(pts):
        ax.plot(x, y, "o", ms=4.5, color=PALETTE[0], zorder=3)
        ax.annotate(short(m), xy=(x, y), xytext=offsets[i % 8], textcoords="offset points",
                    ha="center", color=INK2, fontsize=5.5)
    ax.set_yscale("log")
    ax.set_xlabel("solve rate")
    ax.set_ylabel("mean transcript length (k chars)")
    ax.margins(x=0.08, y=0.2)
    ax.text(1.0, 1.02, "judged items only", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=7)
    style(ax)
    save(fig, out / "11_capability_vs_length")


# --- main ------------------------------------------------------------------

def export_bench(bench):
    out = ROOT / "judge_results" / OUT_DIRS[bench]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    index, verdicts = load_dataset(bench)
    runs = index["runs"]

    write_csv(out / "summary.csv", [
        {k: r[k] for k in ("judge", "major", "minor", "d_acc", "score", "score_lo",
                           "score_hi", "n_items", "slot_a", "disagree_rate",
                           "noanswer_rate", "weak_judge")} for r in runs])
    write_csv(out / "models.csv", [
        {"model": m, **{k: meta[k] for k in ("label", "accuracy", "acc_swe_bench_mini",
                                             "acc_intercode", "acc_gpqa", "acc_mbpp",
                                             "eci", "arm")}}
        for m, meta in sorted(index["models"].items())])
    write_csv(out / "verdicts.csv", verdicts)

    fig_model_accuracy(index, figures)
    fig_run_overview(index, runs, figures)
    fig_gap_curve_by_judge(index, runs, verdicts, figures)
    fig_gap_curve_pooled(index, verdicts, figures)
    fig_judge_strength(index, runs, figures)
    fig_position_bias(index, verdicts, figures)
    fig_length_bias(index, verdicts, figures)
    fig_capability_vs_length(index, verdicts, figures)
    ties = sorted({(r["major"], r["minor"]) for r in runs if r["d_acc"] == 0})
    print(f"{OUT_DIRS[bench]}: {len(runs)} runs, {len(verdicts)} verdict rows"
          + (f"; excluded from curves (solve-rate tie): {ties}" if ties else ""))


def main():
    for bench in BENCHES:
        export_bench(bench)


if __name__ == "__main__":
    main()
