"""Transcript truncation: full-transcript judge runs against their head / tail / ends cuts.

Direct judging keeps 4,000 characters per transcript, pairwise 2,000 per transcript (the same
~4k judge budget per prompt). Runs are read from logs/direct_truncated-<cut>-4000/ and
logs/judge_truncated-<cut>-2000/ (configs/judges/truncated/), matched to the frozen train
items and scored through calibration.judge_forecast exactly like the full runs, so a cut's
MSE is comparable to the full run's under the same variant. Never launches judging.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import calibration, matching
from analysis.loading import JUDGES, LOGS, JudgeRun, direct_from_logs, fetched, pairwise_from_logs
from analysis.plots import BASELINE, COLORS, DOMAIN_COLORS, DOMAIN_LABELS, LABELS, fix_axes_height
from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import DirectJudgeConfig, JudgeConfig
from protocols.direct_judge import JudgeDirect
from protocols.pairwise import JudgeBT

CUTS = ("head", "tail", "ends")
FULL = "full"
DIRECT_CHARS, PAIRWISE_CHARS = 4000, 2000
PROTOCOLS = ("direct", "pairwise")


def truncated_runs(dom: DomainConfig, task: Task) -> list[JudgeRun]:
    """Every fetched truncated run for the task, condition "truncated-<cut>-<chars>"."""
    runs = []
    for path in sorted((JUDGES / "truncated").glob("*.yml")):
        cfg = JudgeConfig.from_cli(path)
        harness = JudgeBT(cfg, dom)
        run_dir = harness.log_dir(task.semester)
        if run_dir.is_dir() and fetched(run_dir):
            runs.append(JudgeRun("pairwise", cfg.judge.shortname.lower(), f"truncated-{cfg.truncate}-{cfg.max_length}",
                                 harness, pairwise_from_logs(run_dir)))
    for cut in CUTS:
        for run_dir in sorted((LOGS / f"direct_truncated-{cut}-{DIRECT_CHARS}" / dom.name / task.semester).glob("*/*/")):
            judge = run_dir.parts[-2]
            cfg = DirectJudgeConfig.from_cli(JUDGES / f"{judge}.yml")
            cfg.max_length, cfg.truncate = DIRECT_CHARS, cut
            harness = JudgeDirect(cfg, dom, include_reasoning=True)
            assert harness.log_dir(task.semester) == run_dir, (harness.log_dir(task.semester), run_dir)
            if fetched(run_dir):
                runs.append(JudgeRun("direct", judge, f"truncated-{cut}-{DIRECT_CHARS}", harness, direct_from_logs(run_dir)))
    return runs


def forecasts(dom: DomainConfig, task: Task, full: list[JudgeRun], truncated: list[JudgeRun],
              variants: dict[str, str]) -> pd.DataFrame:
    """One row per target x truncated run and, once per (protocol, judge), its full-transcript
    run (reasoning shown): cut is "full" or head / tail / ends; theta_hat under the protocol's
    variant, theta the truth, missing the run's unjudged count."""
    core = {(r.protocol, r.judge): r for r in calibration.core_runs(full)}
    seen, frames = set(), []
    for run in truncated:
        base = core[(run.protocol, run.judge)]
        pairs = [(run, run.condition.split("-")[1])] + ([] if (run.protocol, run.judge) in seen else [(base, FULL)])
        seen.add((run.protocol, run.judge))
        for r, cut in pairs:
            theta_hat, _ = calibration.judge_forecast(matching.match(r, task), task, variants[run.protocol])
            frames.append(pd.DataFrame({"domain": dom.name, "semester": task.semester, "protocol": run.protocol, "judge": run.judge,
                                        "cut": cut, "variant": variants[run.protocol], "model": task.truth.index,
                                        "theta_hat": theta_hat.reindex(task.truth.index).to_numpy(float),
                                        "theta": task.truth.to_numpy(float), "missing": r.missing}))
    return pd.concat(frames, ignore_index=True)


def summary(f: pd.DataFrame) -> pd.DataFrame:
    """One row per task x protocol x cut: MSE, the full run's MSE and their ratio."""
    keys = ["domain", "semester", "protocol", "judge", "cut"]
    mse = f.groupby(keys).apply(lambda g: float(((g.theta_hat - g.theta) ** 2).mean())).rename("mse").reset_index()
    full = mse[mse.cut == FULL].drop(columns="cut").rename(columns={"mse": "full_mse"})
    out = mse[mse.cut != FULL].merge(full)
    return out.assign(relative_mse=out.mse / out.full_mse)


def figure(f: pd.DataFrame, baseline: pd.DataFrame, path: Path, cut: str = "ends") -> None:
    """MSE over target models, per protocol and domain, full transcripts against the `cut`:
    bars in the domain's colour at the MSE over the domain's targets (both semesters pooled,
    the unit is the target model) with a capless percentile-bootstrap 95% interval
    (stats.mse_row); the prediction from saturated items as a dashed grey segment over each
    domain's pair at its MSE over the same targets. `baseline`: the results forecasts rows of
    that prediction (theta_hat, theta per target)."""
    from analysis.stats import mse_row
    protocols = [p for p in PROTOCOLS if p in set(f.protocol)]
    domains = [d for d in DOMAIN_COLORS if d in set(f.domain)]
    fig, ax = plt.subplots(figsize=(2.6, 2.4), layout="constrained")
    x, ticks, centres = 0.0, [], []
    for p in protocols:
        for d in domains:
            for c in (FULL, cut):
                row = mse_row(f[(f.protocol == p) & (f.domain == d) & (f.cut == c)])
                ax.bar(x, row["mse"], color=DOMAIN_COLORS[d], width=0.72, zorder=1)
                ax.plot([x, x], [row["lo"], row["hi"]], color="black", lw=0.8, zorder=4)
                ticks.append((x, "Full" if c == FULL else "Truncated"))
                x += 1
            level = mse_row(baseline[baseline.domain == d])["mse"]
            ax.hlines(level, x - 2.45, x - 0.55, color="grey", lw=0.8, ls="--", zorder=0)
            x += 0.3
        centres.append(x - 0.3 - 2 * len(domains) / 2 - 0.5)
        x += 0.5
    top = ax.get_ylim()[1]
    for p, c in zip(protocols, centres):
        ax.text(c, top, LABELS[p], color="grey", fontsize=7, ha="center", va="bottom")
    ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Prediction Error\n(MSE θ)")
    ax.set_ylim(bottom=0)
    ax.spines["bottom"].set_bounds(ticks[0][0], ticks[-1][0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS[d], label=DOMAIN_LABELS[d]) for d in domains]
    handles.append(plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE]))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
