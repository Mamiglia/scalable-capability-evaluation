"""Per-model accuracy on each bench of a domain (after the domain's item filters), with the
expected numbers the roster was drawn up with. Writes analysis/<domain>_collection/.

    uv run python scripts/analyses/domain_collection.py science
    uv run python scripts/analyses/domain_collection.py math
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from configs.domain import DomainConfig  # noqa: E402
from evals.responses import responses  # noqa: E402

# era -> [(model, expected accuracy or None)]; the same roster for every domain.
ROSTER = {
    "2023": [("mixtral-8x7b-instruct-v0.1", .306), ("llama-2-70b-chat-hf", .263), ("llama-2-13b-chat-hf", .240),
             ("mpt-30b", .222), ("qwen-72b-chat", None), ("nous-hermes-2-yi-34b", None), ("gpt-4-0613", .307),
             ("gpt-3.5-turbo-0125", None)],
    "2024-I": [("wizardlm-2-8x22b-hf", None), ("qwen2-72b-instruct", .408), ("meta-llama-3-70b-instruct", .406),
               ("gemma-2-27b-it", .365), ("qwen1.5-72b-chat", None), ("phi-3-medium-128k-instruct", .276),
               ("gpt-4o-2024-05-13", .489), ("gpt-4-turbo-2024-04-09", .466)],
    "2024-II": [("phi-4", .561), ("qwen2.5-72b-instruct", .491), ("deepseek-v3", None),
                ("llama-3.3-70b-instruct", .474), ("qwen2.5-32b-instruct", .461), ("hermes-3-llama-3.1-405b", None),
                ("open-mistral-nemo-2407", .299), ("llama-3.1-8b-instruct", .259)],
    "2025-I": [("deepseek-r1-0528", .769), ("qwen3-235b-a22b", .707), ("deepseek-v3-0324", .676),
               ("qwen3-30b-a3b", None), ("qwen3-32b", .631), ("minimax-m1", None),
               ("mistral-small-3.2-24b-instruct", None), ("gemma-3-27b-it", .439)],
    "2025-II": [("kimi-k2-thinking", .879), ("glm-4.5", .841), ("glm-4.7", .835), ("deepseek-v3.2", .824),
                ("qwen3-235b-a22b-thinking-2507", .801), ("minimax-m2", None), ("gpt-oss-120b", .715),
                ("gpt-oss-20b", .588)],
    "2026-I": [("glm-5.1", .899), ("kimi-k2.6", .897), ("glm-5.2", .884), ("kimi-k2.5", .876), ("gemma-4-31b-it", .865),
               ("qwen3.5-397b-a17b", .859), ("qwen3.6-27b", .859), ("nvidia-nemotron-3-ultra-550b-a55b-bf16", .837)],
}
ERA_COLOR = dict(zip(ROSTER, ["#9e9e9e", "#eda100", "#eb6834", "#1baf7a", "#2a78d6", "#4a3aa7"]))
# domain -> (bench the expected column refers to, bench the other benches are plotted against,
#            [(panel label, [benches pooled into it])])
DOMAINS = {
    "science": ("gpqa", "gpqa", [("ARC", ["arc_challenge"]), ("GPQA", ["gpqa"]), ("HLE gold", ["hle"])]),
    "math": ("mmlu_pro_math", "aime2024",
             [("MMLU-Pro", ["mmlu_pro_math"]), ("AIME", ["aime2024", "aime2025", "aime2026"]),
              ("APEX", ["apex_2025", "apex_shortlist"]), ("HLE gold", ["hle", "hle_gold"])]),
    "coding": ("swe_bench_mini", "swe_bench_mini",
               [("SWE-bench mini", ["swe_bench_mini"]), ("Terminal-Bench 2.1", ["terminal_bench_2_1"]),
                ("AgentBench OS", ["agent_bench_os"])]),
    "cyber": ("cybench", "cybench", [("CyBench", ["cybench"]), ("InterCode", ["intercode"])]),
}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})


def offset(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 4))


def main(domain: str):
    out = Path(f"analysis/{domain}_collection")
    anchor, against, panels = DOMAINS[domain]
    resp = responses(DomainConfig.from_cli(f"configs/domain/{domain}.yml"))
    label = {b: label for label, bs in panels for b in bs}
    resp["raw_bench"] = resp.bench
    resp["bench"] = resp.bench.map(label)
    resp = resp.dropna(subset=["bench"])
    anchor, against = label[anchor], label[against]
    roster = pd.DataFrame([(m, e, era) for era, ms in ROSTER.items() for m, e in ms], columns=["model", "expected", "era"])
    resp = resp[resp.model.isin(roster.model)]
    roster = roster[roster.model.isin(resp.model)]  # coding: no pre-2025 agentic roster
    per_rep = resp.groupby(["bench", "model", "repeat"]).correct.mean().reset_index()
    acc = per_rep.groupby(["bench", "model"]).correct.agg(["mean", "std", "count"]).reset_index()
    wide = acc.pivot(index="model", columns="bench", values="mean").reindex(roster.model)
    table = roster.set_index("model").join(wide).join(
        acc.pivot(index="model", columns="bench", values="count").add_prefix("n_"))
    table["anchor_minus_expected"] = table[anchor] - table.expected
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "accuracy.csv", float_format="%.3f")
    print(table.round(3).to_string())

    # Fig 1: per-bench accuracy per model, rows sorted by the anchor bench (best on top), repeats as small points.
    order = table[[b for b, _ in panels]].dropna(how="all")[anchor].sort_values().index.tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(1.25 * len(panels) + 1.5, 5.6), sharey=True)
    for ax, (bench, _) in zip(axes, panels):
        sub = per_rep[per_rep.bench == bench]
        for i, m in enumerate(order):
            era = roster.set_index("model").era[m]
            pts = sub[sub.model == m].correct.values
            if len(pts) == 0:
                continue
            ax.plot(pts, np.full(len(pts), i), ".", ms=2.5, color=ERA_COLOR[era], alpha=.5)
            ax.plot(pts.mean(), i, "o", ms=3.5, color=ERA_COLOR[era], mec="none")
        guess = resp[resp.bench == bench].guess.mean()
        if guess > 0:
            ax.axvline(guess, lw=.8, ls="--", color="grey")
        ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", ".5", "1"])
        ax.text(0, 1.01, bench, transform=ax.transAxes, color="grey", ha="left", va="bottom")
        offset(ax)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
    axes[0].set_ylim(-.7, len(order) - .3)
    axes[1].set_xlabel("Accuracy")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=c, label=e) for e, c in ERA_COLOR.items() if e in set(roster.era)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.55, -.005))
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out / "accuracy_per_model.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "accuracy_per_model.svg", bbox_inches="tight")

    # Fig 2: collected vs expected on the anchor bench, and every other bench against the anchor.
    pairs = ([("expected", anchor, "Expected", anchor)] if table.expected.notna().any() else []) + [(against, b, against, b) for b, _ in panels if b != against]
    fig, axes = plt.subplots(1, len(pairs), figsize=(1.1 * len(pairs) + 0.4, 1.5))
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        for era, c in ERA_COLOR.items():
            t = table[table.era == era]
            ax.plot(t[x], t[y], "o", ms=3, color=c, mec="none", alpha=.85)
        if x == "expected":
            ax.plot([0, 1], [0, 1], lw=.8, ls="--", color="grey")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.set_xlabel(xl); ax.set_ylabel(yl); offset(ax)
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, -.08))
    fig.tight_layout()
    fig.savefig(out / "bench_agreement.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "bench_agreement.svg", bbox_inches="tight")
    print("\nPearson r:", {f"{x}~{y}": round(table[[x, y]].dropna().corr().iloc[0, 1], 3) for x, y, *_ in pairs})

    # Fig 3: every bench unpooled, models x benches heatmap of mean accuracy (repeats averaged).
    raw = resp.groupby(["raw_bench", "model", "repeat"]).correct.mean().groupby(["raw_bench", "model"]).mean()
    grid = raw.unstack("raw_bench").reindex(roster.model)
    grid = grid[[b for _, bs in panels for b in bs if b in grid.columns]]
    grid.to_csv(out / "accuracy_per_bench.csv", float_format="%.3f")
    fig, ax = plt.subplots(figsize=(0.32 * grid.shape[1] + 1.9, 0.115 * grid.shape[0] + 0.5))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}".lstrip("0") or "0", ha="center", va="center", fontsize=5,
                        color="white" if v > .6 else "black")
    ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels(grid.columns, rotation=90)
    ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index)
    for lab, m in zip(ax.get_yticklabels(), grid.index):
        lab.set_color(ERA_COLOR[roster.set_index("model").era[m]])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02, ticks=[0, .5, 1]); cb.set_label("Accuracy"); cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "accuracy_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "accuracy_heatmap.svg", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "science")
