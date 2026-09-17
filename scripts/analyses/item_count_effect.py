"""Effect of the number of judged items on the strongest judge's pairwise MSE, per domain and semester (no judging is run).

    uv run python scripts/analyses/item_count_effect.py [--recompute]

Judge = the top reference of each semester (data/meta/<domain>/fit.yml `top[0]`), subjects' reasoning shown
(logs/judge/<domain>/<sem>/<judge>/{24d6cc32,66c170cd}, first that exists). For n in a doubling grid up to the
number of judged train items, DRAWS random subsets of n (bench, item)s are taken from the judged verdicts, and
the targets are placed from those verdicts alone by the per-bench calibration (JudgeLinear); the CSV also keeps
pinned Bradley-Terry (JudgeBT). MSE over all targets as % of the mean_ref baseline; the band is the 2.5-97.5
percentile over draws (uncertainty from which items were judged). Writes analysis/judge/item_count_effect.{csv,png,svg}.
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
from protocols.pairwise import from_logs
from protocols.pairwise.bradley_terry import rank
from protocols.pairwise.linear import weighted_theta

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/item_count_effect")
DOMAINS = ["science", "math", "coding", "cyber"]
CONDITIONS = ["24d6cc32", "66c170cd"]
SEMESTERS = {"2024-I": "#2a78d6", "2024-II": "#eb6834", "2025-I": "#1baf7a", "2025-II": "#eda100", "2026-I": "#4a3aa7"}
SIZES = [3, 6, 12, 25, 50, 100]
DRAWS = 30
YMAX = 200


def forecasts(verdicts: pd.DataFrame, task: Task) -> dict[str, pd.Series]:
    refs, targets = task.references.theta, list(task.test.models.index)
    bt = rank(verdicts, anchors=refs).set_index("model").ability.reindex(targets)
    return {"bt": bt, "linear": weighted_theta(verdicts, refs, targets)}


def compute() -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(0)
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            judge = judges[0].lower()
            run = next((p for c in CONDITIONS if (p := Path("logs/judge", name, sem, judge, c)).exists()), None)
            if run is None:
                print("missing", name, sem, judge, file=sys.stderr)
                continue
            task = Task.from_semester(sem, domain)
            v = from_logs(run).sort_values("imported", kind="stable").drop_duplicates(["bench", "item", "model_left", "model_right"])
            v = v.merge(task.references.items[["bench", "item"]])
            items = v[["bench", "item"]].drop_duplicates().reset_index(drop=True)
            baseline = score(PROTOCOLS["mean_ref"](task), task.truth)["mse"]
            for n in [s for s in SIZES if s < len(items)] + [len(items)]:
                for draw in range(1 if n == len(items) else DRAWS):
                    keep = items.iloc[rng.choice(len(items), n, replace=False)]
                    mse = {k: score(f, task.truth)["mse"] for k, f in forecasts(v.merge(keep), task).items()}
                    rows.append(dict(domain=name, semester=sem, judge=judge, n_judged=len(items), n=n, draw=draw,
                                     baseline=baseline, **mse))
            print(name, sem, judge, len(items), "items", file=sys.stderr)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 4, figsize=(4.4, 1.5), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25})
    for key in ["linear"]:
        for ax, name in zip(axes, DOMAINS):
            for sem, color in SEMESTERS.items():
                d = df[(df.domain == name) & (df.semester == sem)]
                if d.empty:
                    continue
                y = (100 * d[key] / d.baseline).groupby(d.n)
                lo, mid, hi = y.quantile(0.025), y.median(), y.quantile(0.975)
                ax.fill_between(mid.index, lo.clip(upper=YMAX), hi.clip(upper=YMAX), color=color, alpha=0.12, lw=0)
                ax.plot(mid.index, mid.clip(upper=YMAX), color=color, lw=1, zorder=3, clip_on=False)
                for x, m in mid.items():  # off the chart: an arrow at the edge
                    ax.plot(x, min(m, YMAX), "^" if m > YMAX else "o", color=color, ms=2.5, zorder=3, clip_on=False)
            ax.axhline(100, color="#9a9a9a", lw=0.8, ls="--", zorder=0)
            if True:
                ax.text(0, 1.04, name, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_position(("outward", 4))
    ax = axes[0]
    ax.set_ylabel("MSE (% ref-mean)")
    ax.set_xscale("log"), ax.set_xlim(3, 100), ax.set_xticks([3, 10, 30, 100], [3, 10, 30, 100]), ax.set_xticks([], minor=True)
    ax.set_ylim(0, YMAX), ax.set_yticks([0, 50, 100, 150, 200])
    fig.supxlabel("Judged items", fontsize=8, y=-0.2)
    handles = [plt.Line2D([], [], color=c, lw=1, marker="o", ms=2.5, label=s) for s, c in SEMESTERS.items()]
    handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Ref. mean"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.98),
               handletextpad=0.3, columnspacing=0.8)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")


def main() -> int:
    csv = OUT.with_suffix(".csv")
    if "--recompute" in sys.argv or not csv.exists():
        compute().to_csv(csv, index=False, float_format="%.4f")
    df = pd.read_csv(csv)
    summary = df.assign(bt=100 * df.bt / df.baseline, linear=100 * df.linear / df.baseline)
    print(summary.groupby(["domain", "semester", "n"])[["bt", "linear"]].median().round(0).unstack("n").to_string())
    plot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
