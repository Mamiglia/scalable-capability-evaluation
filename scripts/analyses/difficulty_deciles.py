"""Item difficulty b_i from the domain's 1PL fit (`uv run domain <domain.yml>`), by decile.

    uv run python scripts/analyses/difficulty_deciles.py science
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from configs.domain import DomainConfig  # noqa: E402

BENCH_COLOR = {"gpqa": "#2a78d6", "arc_challenge": "#1baf7a", "hle": "#eb6834", "mmlu_pro_math": "#4a3aa7",
               "aime2024": "#2a78d6", "aime2025": "#2a78d6", "aime2026": "#2a78d6",
               "apex_2025": "#eda100", "apex_shortlist": "#eda100"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})


def offset(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 4))


def main(domain: str):
    cfg = DomainConfig.from_cli(f"configs/domain/{domain}.yml")
    items = cfg.load_items()
    items["decile"] = pd.qcut(items.b.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    out = Path(f"analysis/{domain}_collection")
    share = items.groupby(["decile", "bench"]).size().unstack(fill_value=0)
    share.to_csv(out / "difficulty_deciles.csv")
    print(items.groupby("decile").b.agg(["min", "median", "max", "count"]).round(2).to_string())
    print(share.to_string())

    benches = [b for b in BENCH_COLOR if b in set(items.bench)]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    ax = axes[0]
    bins = np.linspace(np.floor(items.b.min()), np.ceil(items.b.max()), 25)
    ax.hist([items[items.bench == b].b for b in benches], bins=bins, stacked=True,
            color=[BENCH_COLOR[b] for b in benches], label=benches)
    for edge in items.groupby("decile").b.max().values[:-1]:
        ax.axvline(edge, lw=.5, ls=":", color="grey")
    ax.set_xlabel("$b_i$ (dotted: decile edges)"); ax.set_ylabel("Items")
    ax.set_xlim(bins[0], bins[-1]); ax.set_xticks([bins[0], 0, bins[-1]])
    offset(ax)
    ax = axes[1]
    frac = share.div(share.sum(1), axis=0)
    bottom = np.zeros(len(frac))
    for b in benches:
        ax.bar(frac.index, frac[b], bottom=bottom, color=BENCH_COLOR[b], width=.8)
        bottom += frac[b].values
    ax.set_xlabel("Difficulty decile"); ax.set_ylabel("Share of items")
    ax.set_xticks([1, 5, 10]); ax.set_xlim(.4, 10.6); ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1])
    offset(ax)
    fig.legend(loc="lower center", ncol=len(benches), frameon=False, bbox_to_anchor=(.5, -.1))
    fig.tight_layout()
    fig.savefig(out / "difficulty_deciles.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "difficulty_deciles.svg", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "science")
