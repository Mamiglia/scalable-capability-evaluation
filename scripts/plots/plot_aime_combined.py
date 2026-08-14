#!/usr/bin/env python3
"""One AIME plot: every model's accuracy chained across aime2024 -> aime2025 -> aime2026.

X is the eval, Y is mean accuracy, and each model is a polyline in its vendor's
color connecting its three points. Only models evaluated on 2+ of the evals are
chained (so partial runs still show).

Run:  uv run python scripts/plot_aime_combined.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from inspect_ai.log import read_eval_log

from scripts.plots.plot_benchmarks import resolve, eval_accuracy, VENDOR_PREFIX

DATA = Path("data")
OUT = Path("analysis/benchmark_plots")
EVALS = ["aime2024", "aime2025", "aime2026"]
X_LABELS = {"aime2024": "AIME 2024", "aime2025": "AIME 2025", "aime2026": "AIME 2026"}


def accuracy_by_model(bench: str) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for path in glob.glob(str(DATA / bench / "*" / "*" / "*" / "*.eval")):
        acc = eval_accuracy(path)
        if acc is not None:
            per.setdefault(Path(path).parts[-3], []).append(acc)
    return {m: sum(v) / len(v) for m, v in per.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = {b: accuracy_by_model(b) for b in EVALS}

    models = sorted({m for s in series.values() for m in s})
    chains = [m for m in models if sum(m in s for s in series.values()) >= 2]

    vendors = sorted({resolve(m)[0] for m in chains})
    palette = plt.cm.tab20([i / max(1, len(vendors) - 1) for i in range(len(vendors))])
    color_of = {v: palette[i] for i, v in enumerate(vendors)}

    fig, ax = plt.subplots(figsize=(11, 6))
    seen: set[str] = set()
    for m in chains:
        vendor, date = resolve(m)
        if not date:
            continue
        x = float(mdates.datestr2num([date])[0])
        ys = [series[EVALS[i]].get(m) for i in range(len(EVALS))]
        present = [(i, y) for i, y in enumerate(ys) if y is not None]
        label = vendor if vendor not in seen else "_nolegend_"
        seen.add(vendor)
        # chain the present points vertically at the model's release date
        ax.plot([x] * len(present), [y for _, y in present],
                marker="o", ms=5, lw=1.0, alpha=0.7, color=color_of[vendor], label=label)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Release date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.set_ylim(0, 1.02)
    ax.set_title(f"AIME combined — per-model accuracy by release date, 3 evals chained ({len(chains)} models)")
    ax.legend(title="Vendor", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / "aime_combined.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    npartial = len(models) - len(chains)
    print(f"[ok] {len(chains)} models chained | {npartial} skipped (in <2 evals) -> {out.name}")


if __name__ == "__main__":
    main()
