"""How one LLM judge behaves on one semester task, from its verdicts (no judging is run).

    uv run python scripts/analyses/judge_figures.py configs/domain/science.yml configs/judges/gemma4-31b.yml 2024-I

Reads logs/judge/<domain>/<semester>/<judge>/<hash>/ and writes
analysis/<domain>/judge_<judge>_<semester>.{png,svg}:
  a  P(target picked) per (target, reference), pooled over benches, items and shown orders
  b  the same rate against the theta* gap, with the Bradley-Terry curve sigmoid(gap) the fit assumes
  c  theta-hat (references pinned) against theta*
  d  position bias: P(pick the transcript shown first), per pair, split by which model was first
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from protocols.pairwise import JudgeBT, from_logs

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a9a"


def offset(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 4))


def short(name: str) -> str:
    """Model name without the suffixes that carry no identity on a tick label."""
    name = re.sub(r"-20\d\d-\d\d-\d\d$|-instruct|-chat|-it$|-hf$|-v0\.1$|-128k", "", name)
    return name.replace("meta-llama-3", "llama-3").replace("nous-hermes-2-yi", "hermes-2-yi")


def main(domain_path: str, judge_path: str, semester: str) -> int:
    domain = DomainConfig.from_cli(Path(domain_path))
    protocol = JudgeBT(JudgeConfig.from_cli(Path(judge_path)), domain)
    task = Task.from_semester(semester, domain)
    verdicts = from_logs(protocol.log_dir(semester)).merge(task.train.items[["bench", "item"]])
    references, targets = task.train.models.index, task.test.models.index
    theta = pd.concat([task.train.theta, task.truth])

    # one row per verdict as (target, reference, target_picked, target_shown_first)
    is_target_left = verdicts.model_left.isin(targets)
    rows = pd.DataFrame({
        "target": np.where(is_target_left, verdicts.model_left, verdicts.model_right),
        "reference": np.where(is_target_left, verdicts.model_right, verdicts.model_left),
        "picked": np.where(is_target_left, verdicts.pick_left, 1 - verdicts.pick_left),
        "target_first": is_target_left,
        "bench": verdicts.bench,
    })
    rows = rows[rows.target.isin(targets) & rows.reference.isin(references)]
    protocol.fit(task.train)
    theta_hat = protocol.predict(task.test)

    fig, axes = plt.subplots(2, 2, figsize=(4.4, 3.8), gridspec_kw={"height_ratios": [1.25, 1]})
    a, b, c, d = axes.flat

    # a. win-rate matrix, both axes ordered by theta*
    order_t = theta[targets].sort_values(ascending=False).index
    order_r = theta[references].sort_values(ascending=False).index
    rate = rows.groupby(["target", "reference"]).picked.mean().unstack().reindex(index=order_t, columns=order_r)
    im = a.imshow(rate.to_numpy(float), cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(len(order_r)), [short(m) for m in order_r], rotation=90, ha="center", fontsize=5)
    a.set_yticks(range(len(order_t)), [short(m) for m in order_t], fontsize=5)
    a.tick_params(length=0)
    a.set_xlabel("Reference (by θ*)"), a.set_ylabel("Target (by θ*)")
    cbar = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03, ticks=[0, 0.5, 1])
    cbar.set_label("P(target picked)")
    cbar.outline.set_visible(False)

    # b. rate against the theta* gap, with the curve the Bradley-Terry fit assumes
    pair = rows.groupby(["target", "reference"]).picked.agg(["mean", "size"]).reset_index()
    pair["gap"] = theta[pair.target].to_numpy() - theta[pair.reference].to_numpy()
    b.scatter(pair.gap, pair["mean"], s=6, lw=0, color=BLUE, alpha=0.7, label="pair")
    x = np.linspace(-1, 3, 100)
    b.plot(x, 1 / (1 + np.exp(-x)), lw=1, color="k", label="σ(gap)")
    b.axhline(0.5, ls="--", lw=0.8, color=GREY)
    b.set_xlim(-1, 3), b.set_xticks([-1, 0, 1, 2, 3]), b.set_ylim(0, 1), b.set_yticks([0, 0.5, 1])
    b.set_xlabel("θ* target − θ* reference"), b.set_ylabel("P(target picked)")
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3)

    # c. theta-hat against theta*
    lo, hi = -3, 3
    c.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=GREY)
    c.scatter(task.train.theta, task.train.theta, s=12, marker="s", facecolor="white", edgecolor="k", lw=0.6, label="reference (pinned)")
    c.scatter(task.truth[targets], theta_hat[targets], s=12, color=ORANGE, lw=0, label="target")
    off = [m for m in theta_hat[targets].sort_values(ascending=False).index if theta_hat[m] - task.truth[m] > 1]
    for i, m in enumerate(off):  # stacked so the labels of the over-shot targets do not collide
        c.annotate(short(m), (task.truth[m], theta_hat[m]), fontsize=5, xytext=(4, 6 - 7 * i), textcoords="offset points")
    c.set_xlim(lo, hi), c.set_ylim(lo, hi), c.set_xticks([lo, 0, hi]), c.set_yticks([lo, 0, hi])
    c.set_xlabel("θ*"), c.set_ylabel("θ̂ (judge)")
    c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3)

    # d. position bias: P(pick the first-shown transcript), per pair, by who was shown first
    first = rows.assign(picked_first=np.where(rows.target_first, rows.picked, 1 - rows.picked))
    per_pair = first.groupby(["target", "reference", "target_first"]).picked_first.mean().unstack()
    d.scatter(per_pair[True], per_pair[False], s=6, lw=0, color=BLUE, alpha=0.7)
    d.plot([0, 1], [1, 0], ls="--", lw=0.8, color=GREY)  # no position bias: P(first | target first) = 1 - P(first | reference first)
    d.set_xlim(0, 1), d.set_ylim(0, 1), d.set_xticks([0, 0.5, 1]), d.set_yticks([0, 0.5, 1])
    d.set_xlabel("P(pick 1st | target 1st)"), d.set_ylabel("P(pick 1st | reference 1st)")
    for ax in (b, c, d):
        offset(ax)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)

    out = Path("analysis") / domain.name / f"judge_{protocol.config.judge.shortname.lower()}_{semester}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print("wrote", out.with_suffix(".png"))

    n_pairs = per_pair.shape[0]
    bias = (per_pair[True] + per_pair[False] - 1).mean()  # >0: prefers the first-shown transcript
    consistent = first.groupby(["target", "reference", "bench"]).apply(
        lambda g: g.groupby("target_first").picked.mean().round().nunique() == 1, include_groups=False).mean()
    print(f"{len(rows)} verdicts, {n_pairs} pairs; mean first-position preference {bias:+.2f}; "
          f"pair-bench majority verdict consistent across orders in {consistent:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
