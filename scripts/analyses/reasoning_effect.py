"""Effect of showing vs hiding the subjects' reasoning on the strongest judge's MSE, per domain and semester (no judging is run).

    uv run python scripts/analyses/reasoning_effect.py [--recompute]

Judge = the top reference of each semester (data/meta/<domain>/fit.yml `top[0]`). Pairwise JudgeLinear theta-hat
from logs/judge/<domain>/<sem>/<judge>/<cond>/: reasoning shown = 24d6cc32 (or 66c170cd, the 40-item 2026-I
subset) vs hidden = 9c1d3abb (coding, cyber) or the reasoning-off campaign 7ea04f2b (science, math); both
conditions are scored on the (bench, item)s judged in both. MSE is reported as % of the mean_ref baseline
(every reference's theta* averaged), on all targets and on the frontier (top-3 targets by theta*). 2024 semesters have no reasoning models, so they are skipped.
Writes analysis/judge/reasoning_effect.{csv,png,svg}.
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from capability.evaluate import score
from capability.task import Task
from configs.domain import DomainConfig
from configs.logs import logger
from protocols import PROTOCOLS
from protocols.pairwise import from_logs
from protocols.pairwise.linear import JudgeLinear


warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/reasoning_effect")
COLOR = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#4a3aa7"}
SHOWN, HIDDEN = ["24d6cc32", "66c170cd"], ["9c1d3abb", "7ea04f2b"]
TOP = 3  # frontier = the top-3 targets by theta*
YMAX = 200  # % of ref-mean; larger values are drawn as an arrow at the edge


def verdicts(run: Path) -> pd.DataFrame:
    v = from_logs(run).sort_values("imported", kind="stable")
    return v.drop_duplicates(["bench", "item", "model_left", "model_right"], keep="first")


def linear(task: Task, v: pd.DataFrame) -> pd.Series:
    p = JudgeLinear.__new__(JudgeLinear)
    p.verdicts = lambda train, test: v
    p.fit(task.references)
    return p.predict(task.test)


def first_dir(judge_dir: Path, names: list[str]) -> Path | None:
    return next((judge_dir / n for n in names if (judge_dir / n).exists() and not (judge_dir / n).is_symlink()), None)


def compute() -> pd.DataFrame:
    rows = []
    for name in COLOR:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            judge_dir = Path("logs/judge", name, sem, judges[0].lower())
            on, off = first_dir(judge_dir, SHOWN), first_dir(judge_dir, HIDDEN)
            if sem.startswith("2024") or on is None or off is None:
                print("skip", name, sem, judges[0], file=sys.stderr)
                continue
            task = Task.from_semester(sem, domain)
            v_on, v_off = verdicts(on), verdicts(off)
            common = v_on[["bench", "item"]].merge(v_off[["bench", "item"]]).drop_duplicates()
            forecasts = {"baseline": PROTOCOLS["mean_ref"](task), "shown": linear(task, v_on.merge(common)),
                         "hidden": linear(task, v_off.merge(common))}
            for targets, truth in (("all", task.truth), ("frontier", task.truth.nlargest(TOP))):
                rows.append(dict(domain=name, semester=sem, judge=judges[0], targets=targets, n_targets=len(truth),
                                 n_items=len(common), **{k: score(f, truth)["mse"] for k, f in forecasts.items()}))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    df = df[df.targets == "all"]
    fig, axes = plt.subplots(1, 4, figsize=(3.6, 1.7), sharey=True, gridspec_kw={"wspace": 0.15},
                             width_ratios=[df.domain.eq(d).sum() for d in COLOR])
    for ax, (name, color) in zip(axes, COLOR.items()):
        d = df[df.domain == name].sort_values("semester").reset_index(drop=True)
        for x, r in d.iterrows():
            y = 100 * pd.Series({"shown": r.shown, "hidden": r.hidden}) / r.baseline
            clipped = y.clip(upper=YMAX)
            if abs(clipped.shown - clipped.hidden) > 2:  # arrow from shown to hidden: up = hiding hurts
                ax.annotate("", (x, clipped.hidden), (x, clipped.shown), zorder=1,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=color, lw=1, shrinkA=2, shrinkB=2.5))
            for k, mfc in (("shown", color), ("hidden", "white")):
                ax.plot(x, clipped[k], "^" if y[k] > YMAX else "o", mfc=mfc, mec=color, ms=4, zorder=3, clip_on=False)
                if y[k] > YMAX:
                    ax.text(x, YMAX * 1.06, f"{y[k]:.0f}", ha="center", va="bottom", fontsize=4, color=color)
        ax.axhline(100, color="#9a9a9a", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(len(d)), d.semester)
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.text(0, 1.04, name, transform=ax.transAxes, fontsize=7, color=color)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
        ax.tick_params(axis="x", length=0, labelsize=5, labelrotation=90)
    axes[0].set_ylim(0, YMAX)
    axes[0].set_yticks([0, 50, 100, 150, 200])
    axes[0].set_ylabel("MSE (% ref-mean)")
    handles = [plt.Line2D([], [], marker="o", color="black", ls="", ms=4, label="Shown"),
               plt.Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=4, label="Hidden"),
               plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Ref. mean")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98),
               handletextpad=0.3, columnspacing=1.0)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")


def main() -> int:
    csv = OUT.with_suffix(".csv")
    if "--recompute" in sys.argv or not csv.exists():
        df = compute()
        df.to_csv(csv, index=False, float_format="%.4f")
    df = pd.read_csv(csv)
    df["shown_pct"], df["hidden_pct"] = 100 * df.shown / df.baseline, 100 * df.hidden / df.baseline
    print(df.round(1).to_string(index=False))
    plot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
