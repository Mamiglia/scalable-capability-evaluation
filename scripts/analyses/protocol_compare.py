"""Protocols side by side per domain and semester, scored on all targets and on the frontier (top-3 targets by theta*); no judging is run.

    uv run python scripts/analyses/protocol_compare.py [--recompute]

Judge = the top reference of each semester (data/meta/<domain>/fit.yml `top[0]`), subjects' reasoning shown.
Pairwise: JudgeLinear on logs/judge/<domain>/<sem>/<judge>/{24d6cc32,66c170cd,7ea04f2b} (first that exists; 2024
semesters have no reasoning models, so the reasoning-off run is the same condition). Direct: DirectRegression on
logs/direct/<domain>/<sem>/<judge>/. Length+prose: PROTOCOLS["glm_regex"] (glm_features/, generality-labs/glm_features).
Embedding: PROTOCOLS["embedding"] (embeddings/, generality-labs/transcript-embeddings). Every protocol fits on all the
references; "frontier" only restricts the targets scored. MSE as % of the mean_ref baseline on the same targets.
Writes analysis/judge/protocol_compare.{csv,png,svg}.
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from capability.evaluate import score
from capability.task import Task
from configs.domain import DomainConfig
from configs.logs import logger
from protocols import PROTOCOLS
from protocols.direct_judge import DirectRegression
from protocols.direct_judge import from_logs as direct_logs
from protocols.pairwise import from_logs as pair_logs
from protocols.pairwise.linear import JudgeLinear

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/protocol_compare")
DOMAINS = ["science", "math", "coding", "cyber"]
SERIES = {"pairwise": ("Pairwise", "#2a78d6"), "direct": ("Direct", "#eb6834"),
          "glm": ("Length+prose", "#1baf7a"), "embedding": ("Embedding", "#4a3aa7")}
CONDITIONS = ["24d6cc32", "66c170cd", "7ea04f2b"]
TOP = 3  # frontier = the top-3 targets by theta*
YMIN, YMAX = 0, 200  # % of ref-mean; values above are drawn as an arrow at the edge


def pairwise(task: Task, run: Path) -> pd.Series:
    v = pair_logs(run).sort_values("imported", kind="stable").drop_duplicates(["bench", "item", "model_left", "model_right"])
    p = JudgeLinear.__new__(JudgeLinear)
    p.verdicts = lambda train, test: v
    p.fit(task.references)
    return p.predict(task.test)


def direct(task: Task, run: Path) -> pd.Series:
    p = DirectRegression()
    p.scores = lambda train, test: direct_logs(run)
    p.fit(task.references)
    return p.predict(task.test)


def compute() -> pd.DataFrame:
    rows = []
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            judge = judges[0].lower()
            task = Task.from_semester(sem, domain)
            forecasts = {"baseline": PROTOCOLS["mean_ref"](task), "glm": PROTOCOLS["glm_regex"](task),
                         "embedding": PROTOCOLS["embedding"](task)}
            pair_run = next((Path("logs/judge", name, sem, judge, c) for c in CONDITIONS if Path("logs/judge", name, sem, judge, c).exists()), None)
            if pair_run:
                forecasts["pairwise"] = pairwise(task, pair_run)
            direct_dir = Path("logs/direct", name, sem, judge)
            if direct_dir.exists():
                forecasts["direct"] = direct(task, next(direct_dir.iterdir()))
            for k in SERIES:
                if k not in forecasts:
                    print("missing", k, name, sem, judge, file=sys.stderr)
            frontier = task.truth.nlargest(TOP)
            for targets, truth in (("all", task.truth), ("frontier", frontier)):
                rows.append(dict(domain=name, semester=sem, judge=judge, targets=targets, n_targets=len(truth),
                                 **{k: score(f, truth)["mse"] for k, f in forecasts.items()}))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 4, figsize=(4.4, 1.9), sharey=True, gridspec_kw={"wspace": 0.12},
                             width_ratios=[df[df.targets == "all"].domain.eq(d).sum() for d in DOMAINS])
    dodge = np.linspace(-0.3, 0.3, len(SERIES))
    for ax, name in zip(axes, DOMAINS):
        d = df[df.domain == name]
        semesters = sorted(d.semester.unique())
        for (key, (_, color)), dx in zip(SERIES.items(), dodge):
            for x, sem in enumerate(semesters):
                cell = d[d.semester == sem].set_index("targets")
                y = 100 * cell[key] / cell.baseline
                pair = y.clip(YMIN, YMAX)
                if abs(pair["all"] - pair["frontier"]) > 2:  # arrow from all targets to the top-3: down = better
                    ax.annotate("", (x + dx, pair["frontier"]), (x + dx, pair["all"]), zorder=1,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=color, lw=1, shrinkA=2, shrinkB=2.5))
                for targets, mfc in (("all", color), ("frontier", "white")):
                    marker = "^" if y[targets] > YMAX else "v" if y[targets] < YMIN else "o"
                    ax.plot(x + dx, pair[targets], marker, mfc=mfc, mec=color, ms=3.5, zorder=3, clip_on=False)
                if y["all"] > YMAX:  # off the chart: the all-targets value above the arrow
                    ax.text(x + dx, YMAX * 1.06, f"{y['all']:.0f}", ha="center", va="bottom", fontsize=4, color=color)
        ax.axhline(100, color="#9a9a9a", lw=0.8, ls="--", zorder=0)
        for x in range(1, len(semesters)):
            ax.axvline(x - 0.5, color="#eeeeee", lw=0.5, zorder=0)
        ax.set_xticks(range(len(semesters)), semesters)
        ax.set_xlim(-0.6, len(semesters) - 0.4)
        ax.text(0, 1.02, name, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
        ax.tick_params(axis="x", length=0, labelsize=5, labelrotation=90)
    axes[0].set_ylim(YMIN, YMAX)
    axes[0].set_yticks([0, 50, 100, 150, 200])
    axes[0].set_ylabel("MSE (% ref-mean)")
    handles = [plt.Line2D([], [], marker="o", color=c, ls="", ms=3.5, label=label) for label, c in SERIES.values()]
    handles += [plt.Line2D([], [], marker="o", color="black", ls="", ms=3.5, label="All targets"),
                plt.Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=3.5, label=f"Top-{TOP} targets"),
                plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Ref. mean")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98),
               handletextpad=0.3, columnspacing=0.8)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")


def main() -> int:
    csv = OUT.with_suffix(".csv")
    if "--recompute" in sys.argv or not csv.exists():
        df = compute()
        df.to_csv(csv, index=False, float_format="%.4f")
    df = pd.read_csv(csv)
    pct = df[list(SERIES)].div(df.baseline, axis=0) * 100
    print(pd.concat([df[["domain", "semester", "judge", "targets", "n_targets"]], pct.round(0)], axis=1).to_string(index=False))
    plot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
