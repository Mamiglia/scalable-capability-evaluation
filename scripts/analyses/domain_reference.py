"""Reference fit of a domain: theta* per model by era, item parameters, quartile mix, score curves.

Reads data/meta/domains/<domain>/{items.csv, reference_models.csv, reference_items.csv}
(`uv run domain` then `uv run reference`). Writes analysis/<domain>_collection/.

    uv run python scripts/analyses/domain_reference.py math
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analyses")
from configs.domain import DomainConfig  # noqa: E402
from difficulty_deciles import BENCH_COLOR, offset  # noqa: E402
from domain_collection import ERA_COLOR, ROSTER  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})


def main(domain: str):
    cfg = DomainConfig.from_cli(f"configs/domain/{domain}.yml")
    out = Path(f"analysis/{domain}_collection")
    items = cfg.load_items()
    models = pd.read_csv(cfg.dir / "reference_models.csv").set_index("model")
    ref_items = pd.read_csv(cfg.dir / "reference_items.csv")
    era_of = {m: era for era, ms in ROSTER.items() for m, _ in ms}
    roster = [m for ms in ROSTER.values() for m, _ in ms if m in models.index]
    benches = [b for b in BENCH_COLOR if b in set(items.bench)]
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=c, label=e) for e, c in ERA_COLOR.items()]

    # Fig 1: theta* per model with 95% interval, roster order, coloured by era.
    t = models.loc[roster]
    t.assign(era=[era_of[m] for m in roster]).to_csv(out / "reference_theta_roster.csv", float_format="%.3f")
    print(t.assign(era=[era_of[m] for m in roster]).round(2).to_string())
    fig, ax = plt.subplots(figsize=(2.4, 0.115 * len(roster) + 0.5))
    y = np.arange(len(roster))[::-1]
    for yi, m in zip(y, roster):
        c = ERA_COLOR[era_of[m]]
        ax.plot([t.theta_lo[m], t.theta_hi[m]], [yi, yi], lw=1, color=c)
        ax.plot(t.theta[m], yi, "o", ms=3, color=c, mec="none")
    ax.set_yticks(y); ax.set_yticklabels(roster); ax.set_ylim(-.7, len(roster) - .3)
    lo, hi = np.floor(t.theta_lo.min()), np.ceil(t.theta_hi.max())
    ticks = np.arange(np.ceil(lo / 2) * 2, hi + 1, 2)  # even ticks, both ends inside the limits
    ax.set_xlim(lo, hi); ax.set_xticks(ticks); ax.set_xlabel(r"$\theta^*$ (95% interval)")
    ax.axvline(0, lw=.8, ls="--", color="grey")
    offset(ax)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.6, -.04))
    fig.tight_layout()
    fig.savefig(out / "reference_theta.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "reference_theta.svg", bbox_inches="tight")

    # Fig 2: theta* by era (points + era mean), 2PL item a vs b by bench, quartile mix per bench.
    fig, axes = plt.subplots(1, 3, figsize=(4.4, 1.9))
    ax = axes[0]
    eras = [e for e in ROSTER if any(m in models.index for m, _ in ROSTER[e])]
    for i, e in enumerate(eras):
        v = t.theta[[m for m, _ in ROSTER[e] if m in t.index]]
        ax.plot(np.full(len(v), i) + np.linspace(-.18, .18, len(v)), v, "o", ms=2.5, color=ERA_COLOR[e], mec="none", alpha=.8)
        ax.plot([i - .3, i + .3], [v.mean()] * 2, lw=1, color=ERA_COLOR[e])
    ax.set_xticks(range(len(eras))); ax.set_xticklabels(eras); ax.tick_params(axis="x", rotation=90); ax.set_ylabel(r"$\theta^*$")
    ax.set_ylim(lo, hi); ax.set_yticks(ticks); offset(ax)
    ax = axes[1]
    for b in benches:
        sub = ref_items[ref_items.bench == b]
        ax.plot(sub.b, sub.a, "o", ms=2, color=BENCH_COLOR[b], mec="none", alpha=.6, label=b)
    ax.set_xlabel("Item $b$"); ax.set_ylabel("Item $a$"); offset(ax)
    ax = axes[2]
    mix = items.groupby(["bench", "quartile"]).size().unstack(fill_value=0).reindex(benches)
    mix.to_csv(out / "quartile_mix.csv")
    frac = mix.div(mix.sum(1), axis=0)
    bottom = np.zeros(len(benches))
    shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for q, c in zip([1, 2, 3, 4], shades):
        ax.bar(range(len(benches)), frac[q], bottom=bottom, color=c, width=.8, label=f"Q{q}")
        bottom += frac[q].values
    ax.set_xticks(range(len(benches))); ax.set_xticklabels(benches); ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1]); ax.set_ylabel("Share of items"); offset(ax)
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out / "reference_items.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "reference_items.svg", bbox_inches="tight")

    # Fig 3: expected bench score as a function of theta (the theta -> score curve per bench), with the
    # roster's theta* as ticks; shows where each bench saturates.
    grid = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    for b in benches:
        sub = ref_items[ref_items.bench == b]
        if len(sub) == 0:
            continue
        p = sub.guess.values[:, None] + (1 - sub.guess.values[:, None]) / (1 + np.exp(-sub.a.values[:, None] * (grid[None, :] - sub.b.values[:, None])))
        ax.plot(grid, p.mean(0), lw=1, color=BENCH_COLOR[b], label=b)
    ax.plot(t.theta, np.full(len(t), -.04), "|", ms=4, color="grey")
    ax.set_xlim(lo, hi); ax.set_xticks(ticks); ax.set_ylim(-.06, 1); ax.set_yticks([0, .5, 1])
    ax.set_xlabel(r"$\theta$ (ticks: roster $\theta^*$)"); ax.set_ylabel("Expected score"); offset(ax)
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.55, -.2))
    fig.tight_layout()
    fig.savefig(out / "score_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "score_curves.svg", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "science")
