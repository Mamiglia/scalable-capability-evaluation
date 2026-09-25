"""Effect of the number of judged items on the strongest judge's pairwise MSE, per domain and semester (no judging is run).

    uv run python scripts/analyses/item_count_effect.py [--recompute] [--last K] [--direct | --both]
    # --last K: only the last K semesters (files tagged _lastK); --direct: direct judging by the top-1 judge (files tagged _direct);
    # --both: the top-1 judge, a Pairwise row and an Absolute (direct) row, squared error, from the two cached tables (files tagged _both)

Judges = the top-3 references of each semester (data/meta/<domain>/fit.yml `top`, one figure row per rank), subjects' reasoning shown
(logs/judge/<domain>/<sem>/<judge>/{24d6cc32,66c170cd}, first that exists). For n in 3, 6, 10, 20, ... up to the
number of judged train items, DRAWS random subsets of n (bench, item)s are taken from the judged verdicts, and
the targets are placed from those verdicts alone by the per-bench calibration (JudgeLinear). MSE over all targets
as % of the mean_ref baseline; the figure plots the prediction score = 100 - that percentage (100 = perfect, 0 = no
better than the reference mean). One panel per semester, one line per domain (colour as in reasoning_effect.py, marker
as in scripts/analyses/palette.py), each stopping at the task's number of judged items; the band is the 2.5-97.5
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
from configs.judge import DirectJudgeConfig
from configs.logs import logger
from palette import DOMAIN  # domain markers, scripts/analyses/palette.py
from protocols import PROTOCOLS
from protocols.direct_judge import JudgeDirect, from_logs as direct_logs
from protocols.direct_judge.scoring import calibrate
from protocols.pairwise import from_logs
from protocols.pairwise.linear import weighted_theta

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/item_count_effect")
DOMAINS = ["science", "math", "coding", "cyber"]
COLOR = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#4a3aa7"}  # as in reasoning_effect.py
CONDITIONS = ["24d6cc32", "66c170cd"]
SIZES = [3, 6, *range(10, 101, 10)]
DRAWS = 30
XMIN = 10  # the figure starts here


def forecasts(verdicts: pd.DataFrame, task: Task) -> dict[str, pd.Series]:
    refs, targets = task.references.theta, list(task.test.models.index)
    return {"linear": weighted_theta(verdicts, refs, targets)}


def compute() -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(0)
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            task = Task.from_semester(sem, domain)
            baseline = score(PROTOCOLS["mean_ref"](task), task.truth)["mse"]
            for judge_rank, judge in enumerate((j.lower() for j in judges), 1):
                run = next((p for c in CONDITIONS if (p := Path("logs/judge", name, sem, judge, c)).exists()), None)
                if run is None:
                    print("missing", name, sem, judge, file=sys.stderr)
                    continue
                v = from_logs(run).sort_values("imported", kind="stable").drop_duplicates(["bench", "item", "model_left", "model_right"])
                v = v.merge(task.references.items[["bench", "item"]])
                items = v[["bench", "item"]].drop_duplicates().reset_index(drop=True)
                for n in [s for s in SIZES if s < len(items)] + [len(items)]:
                    for draw in range(1 if n == len(items) else DRAWS):
                        keep = items.iloc[rng.choice(len(items), n, replace=False)]
                        mse = {k: score(f, task.truth)["mse"] for k, f in forecasts(v.merge(keep), task).items()}
                        rows.append(dict(domain=name, semester=sem, judge=judge, judge_rank=judge_rank, n_judged=len(items),
                                         n=n, draw=draw, baseline=baseline, **mse))
                print(name, sem, judge_rank, judge, len(items), "items", file=sys.stderr)
    return pd.DataFrame(rows)


def compute_direct() -> pd.DataFrame:
    """The same subsampling for direct judging: the top-1 reference's 1-10 ratings (logs/direct/<domain>/<sem>/<judge>/),
    mean rating -> theta* calibrated on the references (protocols.direct_judge.scoring.calibrate, as in protocol_compare.py)."""
    rows, rng = [], np.random.default_rng(0)
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            judge, task = judges[0].lower(), Task.from_semester(sem, domain)
            runs = JudgeDirect(DirectJudgeConfig.from_cli(Path(f"configs/judges/{judge}.yml")), domain).log_dirs(sem)
            if not runs:
                print("missing", name, sem, judge, file=sys.stderr)
                continue
            refs, targets = task.references.theta, task.truth.index
            r = pd.concat([direct_logs(d) for d in runs], ignore_index=True)  # every bench set's directory (JudgeDirect.log_dirs)
            r = r.drop_duplicates(["model", "bench", "item", "epoch"]).merge(task.references.items[["bench", "item"]])
            r = r[(r.semester == sem) & r.model.isin(refs.index.union(targets))]
            items = r[["bench", "item"]].drop_duplicates().reset_index(drop=True)
            baseline = score(PROTOCOLS["mean_ref"](task), task.truth)["mse"]
            for n in [s for s in SIZES if s < len(items)] + [len(items)]:
                for draw in range(1 if n == len(items) else DRAWS):
                    keep = items.iloc[rng.choice(len(items), n, replace=False)]
                    theta_hat = calibrate(r.merge(keep), anchors=refs).reindex(targets)
                    rows.append(dict(domain=name, semester=sem, judge=judge, judge_rank=1, n_judged=len(items), n=n, draw=draw,
                                     baseline=baseline, direct=score(theta_hat, task.truth)["mse"]))
            print(name, sem, judge, len(items), "items", file=sys.stderr)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, key: str = "linear", tag: str = "", x: str = "n", xlabel: str = "Judged items",
         xticks: tuple = (XMIN, 50, 100), out: Path = OUT, mse: bool = False) -> None:
    """Rows = judge rank among the references, one panel per semester, one line per domain (colour + palette marker): the task's median prediction score over draws,
    with the 2.5-97.5 percentile band over draws.

    A line stops at the task's number of judged items; scores below 0 sit on the floor. mse: plot the squared error (theta^2,
    log axis) instead, with the reference-mean baseline of each task as a dashed line in the domain's colour."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    df = df[df[x] >= xticks[0]]  # the axis starts at its first tick (items: fewer than XMIN is mostly draw noise)
    if "row" not in df:  # figure rows: the judge's rank among the references, unless the caller labels them
        df = df.assign(row="Top-" + df.judge_rank.astype(str) + " judge")
    semesters, ranks = sorted(df.semester.unique()), list(dict.fromkeys(df.sort_values("judge_rank", kind="stable").row))
    fig, axes = plt.subplots(len(ranks), len(semesters), figsize=(0.3 + 1.2 * len(semesters), 0.3 + 1.1 * len(ranks)),
                             sharex=True, sharey=True, squeeze=False, gridspec_kw={"wspace": 0.4, "hspace": 0.35})
    draws = (df[key] if mse else 100 - 100 * df[key] / df.baseline).groupby([df.row, df.semester, df.domain, df[x]])
    floor = (1e-9, np.inf) if mse else (0, 100)
    score, lo, hi = draws.median(), draws.quantile(0.025), draws.quantile(0.975)  # band: which items were judged
    for row, rank in zip(axes, ranks):
        for ax, sem in zip(row, semesters):
            for name, marker in DOMAIN.items():
                cell = (rank, sem, name.lower())
                if cell in score.index.droplevel(x):
                    y = score[cell].clip(*floor)
                    ax.fill_between(y.index, lo[cell].clip(*floor), hi[cell].clip(*floor), color=COLOR[name.lower()], alpha=0.15, lw=0)
                    if mse:
                        base = df[(df.row == rank) & (df.semester == sem) & (df.domain == name.lower())].baseline.iloc[0]
                        ax.plot(xticks[::len(xticks) - 1], [base] * 2, color=COLOR[name.lower()], lw=0.7, ls="--", alpha=0.7)
                    ax.plot(y.index, y, color=COLOR[name.lower()], lw=1, marker=marker, ms=2.5, mfc="white", mew=0.7, clip_on=False)
            if rank == ranks[0]:
                ax.text(0, 1.04, sem, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_position(("outward", 4))
        row[0].set_ylabel(rank)
    fig.supylabel("Squared error (θ²)" if mse else "Prediction score", fontsize=8, x=0.125 - 0.7 / fig.get_figwidth())  # left of the row labels, whatever the column count
    ax = axes[0, 0]
    ax.set_xlim(xticks[0], xticks[-1]), ax.set_xticks(xticks)
    if mse:
        ax.set_yscale("log"), ax.set_ylim(0.02, 6), ax.set_yticks([0.1, 1]), ax.set_yticklabels(["0.1", "1"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_ylim(0, 100), ax.set_yticks([0, 50, 100])
    fig.supxlabel(xlabel, fontsize=8, y=0.11 - 0.45 / fig.get_figheight())  # just under the tick labels, whatever the row count
    handles = [plt.Line2D([], [], color=COLOR[name.lower()], lw=1, marker=marker, ms=2.5, mfc="white", mew=0.7, label=name.capitalize())
               for name, marker in DOMAIN.items() if name.lower() in set(df.domain)]
    if mse:
        handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.7, ls="--", label="Baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.9 + 0.1 / len(ranks)),
               handletextpad=0.3, columnspacing=0.8, handlelength=1.4)
    out = out.with_name(out.name + tag)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def main() -> int:
    if "--both" in sys.argv:  # the top-1 judge under both protocols, one row each, from the two cached tables
        pair, rate = pd.read_csv(OUT.with_suffix(".csv")), pd.read_csv(OUT.with_name(OUT.name + "_direct").with_suffix(".csv"))
        df = pd.concat([pair[pair.judge_rank == 1].assign(row="Pairwise", mse=lambda d: d.linear),
                        rate.assign(row="Absolute", mse=lambda d: d.direct)])
        plot(df, key="mse", out=OUT.with_name(OUT.name + "_both"), mse=True)
        return 0
    direct = "--direct" in sys.argv  # direct judging (top-1 judge) instead of pairwise (top-3 judges)
    key, out = ("direct", OUT.with_name(OUT.name + "_direct")) if direct else ("linear", OUT)
    csv = out.with_suffix(".csv")
    if "--recompute" in sys.argv or not csv.exists():
        (compute_direct() if direct else compute()).to_csv(csv, index=False, float_format="%.4f")
    df = pd.read_csv(csv)
    score = (100 - 100 * df[key] / df.baseline).groupby([df.judge_rank, df.domain, df.semester, df.n]).median()
    print(score.round(0).unstack("n").to_string())
    tag = ""
    if "--last" in sys.argv:  # only the last K semesters of every domain
        k = int(sys.argv[sys.argv.index("--last") + 1])
        df, tag = df[df.semester.isin(sorted(df.semester.unique())[-k:])], f"_last{k}"
    plot(df, key=key, tag=tag, out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
