"""Judge MSE per thirds condition against the full-transcript baseline, per domain x semester; no judging is run.

    PYTHONPATH=src .venv/bin/python src/challenges/thirds/compare.py [--recompute]

Verdicts of the baseline dir and of every thirds dir present are restricted to the (bench, item, left, right)
rows judged in all of them, then scored as pinned Bradley-Terry (`bt`) and JudgeLinear (`linear`) MSE vs theta*
on all targets and on the frontier (top-3 targets by theta*), next to the mean_ref baseline. Writes
analysis/judge/thirds.{csv,png,svg}: MSE as % of mean_ref, one panel per domain, x = kept thirds. The csv holds
both the RUNS (top reference) and the LOO judges (runs.py); the figure shows the LOO judges.
"""
import functools
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from capability.evaluate import score
from capability.task import Task
from challenges.thirds.runs import LOO, RUNS, harnesses
from configs.domain import DomainConfig
from configs.logs import logger
from protocols import PROTOCOLS
from protocols.pairwise import from_logs
from protocols.pairwise.linear import JudgeLinear
from protocols.pairwise.protocol import PairwiseBT

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/thirds")
KEY = ["bench", "item", "model_left", "model_right"]
ORDER = ["full"] + [f"thirds-{k}" for k in ("12", "23", "13", "1", "2", "3")]  # x axis: most to least shown
COLOR = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#4a3aa7"}
TOP = 3
YMAX = 200


def verdicts(run: Path) -> pd.DataFrame:
    v = from_logs(run).sort_values("imported", kind="stable")
    return v.drop_duplicates(KEY, keep="first")


def forecast(cls, task: Task, v: pd.DataFrame) -> pd.Series:
    p = cls.__new__(cls)
    p.verdicts = lambda train, test: v
    p.fit(task.references)
    return p.predict(task.test)


def run_dirs(domain: str, semester: str) -> list[tuple[str, dict[str, Path]]]:
    """(judge label, condition -> run dir) per RUNS/LOO judge. A `-full` judge also gets `<stem>+40`: its 100-item dirs
    with the missing conditions taken from the 40-item twin (wave 2 of the LOO judges ran on 40 items), so every
    condition is scored on the rows common to all of them, i.e. the 40 items."""
    out = []
    for stem in sorted({RUNS[domain, semester], LOO[domain, semester]}):
        dirs = {c: h.log_dir(semester) for c, h in harnesses(domain, semester, stem)[1].items()}
        out.append((stem, dirs))
        if stem.endswith("-full"):
            twin = {c: h.log_dir(semester) for c, h in harnesses(domain, semester, stem.removesuffix("-full"))[1].items()}
            extra = {c: twin[c] for c, d in dirs.items() if not any(d.glob("*.eval")) and any(twin[c].glob("*.eval"))}
            if extra:
                out.append((f"{stem}+40", dirs | extra))
    return out


def compute() -> pd.DataFrame:
    rows = []
    for (domain, semester), stem, dirs in [(k, s, d) for k in sorted(RUNS) for s, d in run_dirs(*k)]:
        dom = DomainConfig.from_cli(Path(f"configs/domain/{domain}.yml"))
        vs = {c: verdicts(d) for c, d in dirs.items() if any(d.glob("*.eval"))}
        if "full" not in vs or len(vs) == 1:
            print("skip", domain, semester, stem, "conditions on disk:", sorted(vs), file=sys.stderr)
            continue
        task = Task.from_semester(semester, dom)
        common = functools.reduce(lambda a, b: a.merge(b), [v[KEY].drop_duplicates() for v in vs.values()])
        baseline = PROTOCOLS["mean_ref"](task)
        for condition, v in vs.items():
            v = v.merge(common)
            forecasts = {"bt": forecast(PairwiseBT, task, v), "linear": forecast(JudgeLinear, task, v)}
            for targets, truth in (("all", task.truth), ("frontier", task.truth.nlargest(TOP))):
                rows.append(dict(domain=domain, semester=semester, judge=stem, condition=condition, targets=targets,
                                 n_targets=len(truth), n_verdicts=len(v), n_common=len(common),
                                 baseline=score(baseline, truth)["mse"], **{k: score(f, truth)["mse"] for k, f in forecasts.items()}))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, metric: str = "linear") -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    df = df[df.targets == "all"]
    # the LOO judge per domain x semester, preferring its `+40` row (the only one with the single-third conditions)
    df = df[[j.removesuffix("+40") == LOO[d, s] for d, s, j in zip(df.domain, df.semester, df.judge)]]
    df = df.sort_values("judge").groupby(["domain", "semester", "condition"]).last().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(3.6, 1.5), sharey=True, gridspec_kw={"wspace": 0.15})
    for ax, (name, color) in zip(axes, COLOR.items()):
        d = df[df.domain == name]
        for semester, ls, mfc in (("2025-II", "-", color), ("2026-I", "--", "white")):
            s = d[d.semester == semester].set_index("condition").reindex(ORDER)
            if s[metric].isna().all():
                continue
            y = (100 * s[metric] / s.baseline).clip(upper=YMAX)
            ax.plot(range(len(ORDER)), y, ls, color=color, lw=1, marker="o", ms=3.5, mfc=mfc, mec=color, clip_on=False)
        ax.axhline(100, color="#9a9a9a", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(len(ORDER)), [c.removeprefix("thirds-").replace("full", "123") for c in ORDER])
        ax.set_xlim(-0.4, len(ORDER) - 0.6)
        ax.text(0, 1.02, name, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
        ax.tick_params(axis="x", length=0, labelsize=5, labelrotation=90)
    axes[0].set_ylim(0, YMAX)
    axes[0].set_yticks([0, 100, 200])
    axes[0].set_ylabel("MSE (% ref-mean)")
    axes[1].set_xlabel("Thirds shown", x=1.0)
    handles = [plt.Line2D([], [], color="black", lw=1, marker="o", ms=3.5, label="2025-II"),
               plt.Line2D([], [], color="black", lw=1, ls="--", marker="o", ms=3.5, mfc="white", label="2026-I"),
               plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Ref. mean")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98),
               handletextpad=0.3, columnspacing=1.0)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")


def main() -> int:
    csv = OUT.with_suffix(".csv")
    if "--recompute" in sys.argv or not csv.exists():
        df = compute()
        if df.empty:
            print("no thirds runs on disk", file=sys.stderr)
            return 1
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False, float_format="%.4f")
    df = pd.read_csv(csv)
    df["bt_pct"], df["linear_pct"] = 100 * df.bt / df.baseline, 100 * df.linear / df.baseline
    print(df.round(2).to_string(index=False))
    plot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
