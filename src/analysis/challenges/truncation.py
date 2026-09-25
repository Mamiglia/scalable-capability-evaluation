"""Transcript truncation: full-transcript judge runs against their head / tail / ends cuts.

Direct judging keeps 4,000 characters per transcript, pairwise 2,000 per transcript (the same
~4k judge budget per prompt). Runs are read from logs/direct_truncated-<cut>-4000/ and
logs/judge_truncated-<cut>-2000/ (configs/judges/truncated/), matched to the frozen train
items and scored through calibration.judge_forecast exactly like the full runs, so a cut's
MSE is comparable to the full run's under the same variant. Never launches judging.
"""


import pandas as pd

from analysis import calibration, matching
from analysis.loading import JUDGES, LOGS, JudgeRun, direct_from_logs, fetched, pairwise_from_logs
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
