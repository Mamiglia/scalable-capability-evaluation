"""Per release era: distribution of P(correct) per item, and the size of a hypothetical EASY set
under three rules (mean P(correct) > .8, > .9, and the era's 3 best models solve it on every repeat).

    uv run python scripts/analyses/easy_set_by_era.py science
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analyses")
from configs.domain import DomainConfig  # noqa: E402
from evals.responses import responses  # noqa: E402
from domain_collection import DOMAINS, ERA_COLOR, ROSTER  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})
RULES = ["P>.8", "P>.9", "top2=1", "top3=1"]
RULE_COLOR = ["#e87ba4", "#8c564b", "#17becf", "#111111"]  # hues not used by the eras


def offset(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 4))


def main(domain: str):
    out = Path(f"analysis/{domain}_collection")
    _, _, panels = DOMAINS[domain]
    label = {b: lab for lab, bs in panels for b in bs}
    era_of = {m: era for era, ms in ROSTER.items() for m, _ in ms}
    resp = responses(DomainConfig.from_cli(f"configs/domain/{domain}.yml"))
    resp = resp[resp.model.isin(era_of)].assign(era=lambda d: d.model.map(era_of), bench=lambda d: d.bench.map(label))
    resp = resp.dropna(subset=["bench"])
    benches = [lab for lab, _ in panels if lab in set(resp.bench)]
    eras = [e for e in ROSTER if e in set(resp.era)]

    # P(correct) per item and era: mean over the era's models (each model weighted equally).
    per_model = resp.groupby(["era", "bench", "item", "model"]).correct.mean().reset_index()
    p = per_model.groupby(["era", "bench", "item"]).correct.mean().rename("p").reset_index()
    # the era's 3 best models = highest mean accuracy over the domain's pooled items
    best = (per_model.groupby(["era", "model"]).correct.mean().reset_index()
            .sort_values("correct", ascending=False).groupby("era").head(3))
    for k in (2, 3):
        top = per_model.merge(best.groupby("era").head(k)[["era", "model"]])
        p = p.merge(top.groupby(["era", "bench", "item"]).correct.min().rename(f"top{k}").reset_index())
    rows = []
    for (era, bench), g in p.groupby(["era", "bench"]):
        rows.append(dict(era=era, bench=bench, items=len(g), **{"P>.8": int((g.p > .8).sum()),
                    "P>.9": int((g.p > .9).sum()), "top2=1": int((g.top2 >= 1).sum()),
                    "top3=1": int((g.top3 >= 1).sum())}))
    counts = pd.DataFrame(rows).set_index(["bench", "era"])
    counts = counts.loc[[(b, e) for b in benches for e in eras if (b, e) in counts.index]]
    counts.to_csv(out / "easy_set_by_era.csv")
    print("3 best models per era:\n" + best.groupby("era").model.apply(list).to_string())
    print(counts.to_string())

    fig, axes = plt.subplots(2, len(benches), figsize=(1.15 * len(benches) + 0.6, 2.9), squeeze=False)
    for j, bench in enumerate(benches):
        ax = axes[0, j]
        for era in eras:
            v = np.sort(p[(p.era == era) & (p.bench == bench)].p.values)
            if len(v) == 0:
                continue
            ax.step(np.r_[0, v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], lw=1, color=ERA_COLOR[era], label=era)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.set_xlabel("P(correct)")
        ax.text(0, 1.02, bench, transform=ax.transAxes, color="grey", ha="left", va="bottom")
        offset(ax)
        ax = axes[1, j]
        sub = counts.loc[bench] if bench in counts.index.get_level_values(0) else None
        x = np.arange(len(eras))
        for k, rule in enumerate(RULES):
            y = [sub[rule][e] if sub is not None and e in sub.index else 0 for e in eras]
            ax.bar(x + (k - 1.5) * .2, y, width=.2, color=RULE_COLOR[k], label=rule)
        ax.set_xticks(x); ax.set_xticklabels(eras); ax.tick_params(axis="x", rotation=90)
        ax.set_ylim(0, 210); ax.set_yticks([0, 100, 200])
        offset(ax)
    axes[0, 0].set_ylabel("Items (cum. share)")
    axes[1, 0].set_ylabel("EASY items")
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.14))
    fig.tight_layout()
    fig.savefig(out / "easy_set_by_era.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "easy_set_by_era.svg", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "science")
