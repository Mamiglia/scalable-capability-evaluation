"""Score protocols against the frozen theta* of the semester tasks: the benchmark's scorer and leaderboard.

    uv run evaluate [configs/domain/<domain>.yml ...] [--protocol pairwise mine.py:Mine ...] [--semester 2026-I ...]

Domains default to the paper's four, --protocol to every entry of protocols.PROTOCOLS (the paper's: mean_ref,
glm_regex, embedding, direct, pairwise); `<file.py>:<Class>` loads a Protocol subclass from any file, built with no
arguments. --semester defaults to every task of the domain. --judge / --direct-judge add one
judge's protocols read from its run in logs/ (JudgeBT / JudgeCalibrated / JudgeLinear, JudgeDirect); nothing is ever
judged here (`uv run judge` collects).

The metric is MSE in theta units (every protocol forecasts on the theta* scale). Printed per domain over the main tasks
(MAIN: the paper's 2025-II and 2026-I, 16 targets per domain), pooled over them ("main", 64 targets with all four
domains), and over every evaluated task ("all"; valid because every local fit shares the global item scale). The
per-target forecasts go to results/<domain>/eval_<name>.csv (gitignored).
"""

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import JudgeConfig, DirectJudgeConfig
from protocols import PROTOCOLS, Protocol
from protocols.pairwise import JudgeBT
from protocols.pairwise.calibrated import JudgeCalibrated
from protocols.pairwise.linear import JudgeLinear
from protocols.direct_judge import JudgeDirect
from reference import tables

DOMAINS = [Path(f"configs/domain/{d}.yml") for d in ("science", "math", "coding", "cyber")]
MAIN = ["2025-II", "2026-I"]
RESULTS = Path("results")


def results_path(domain, name: str) -> Path:
    """results/<domain>/eval_<name>.csv: one protocol's per-target forecasts on one domain."""
    return RESULTS / getattr(domain, "name", domain) / f"eval_{name}.csv"


def score(pred: pd.Series, truth: pd.Series) -> dict:
    """MSE of theta-hat against theta* over the models both series index; NaN forecasts are dropped.

    Every protocol forecasts on the task's theta* scale, so the raw
    squared error is the metric: no shift/scale-free variant.
    """
    pred, truth = pred.align(truth, join="inner")
    known = pred.notna()
    pred, truth = pred[known], truth[known]
    if len(pred) == 0:
        return {"n": 0, "mse": np.nan}
    return {"n": len(pred), "mse": float(((pred - truth) ** 2).mean())}


def semester_names(domain: DomainConfig) -> list[str]:
    columns = tables.load(domain)[0].columns
    return [c.removesuffix("_role") for c in columns if c.endswith("_role")]


def forecast(task: Task, protocol: Protocol) -> pd.DataFrame:
    """One row per target: semester, model, theta_hat, theta."""
    theta_hat = protocol(task).reindex(task.truth.index)
    return pd.DataFrame({"semester": task.semester, "model": task.truth.index,
                         "theta_hat": theta_hat.to_numpy(float), "theta": task.truth.to_numpy()})


def evaluate(domain: DomainConfig, protocol: Protocol, name: str = "", semester_list: list[str] | None = None) -> pd.DataFrame:
    """Run the protocol on every semester task; one score row per semester plus "all"."""
    tasks = [Task.from_semester(s, domain) for s in semester_list or semester_names(domain)]
    forecasts = pd.concat([forecast(task, protocol) for task in tasks], ignore_index=True)
    if name:
        results_path(domain, name).parent.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(results_path(domain, name), index=False, float_format="%.4f")

    scores = []
    groups = list(forecasts.groupby("semester", sort=False)) + [("all", forecasts)]
    for semester, group in groups:
        group = group.set_index("model")
        scores.append({"semester": semester, **score(group.theta_hat, group.theta)})
    return pd.DataFrame(scores)


def load_protocol(spec: str) -> Protocol:
    """A registered name, or `<file.py>:<Class>` (a Protocol subclass from that file, built with no arguments)."""
    if spec in PROTOCOLS:
        return PROTOCOLS[spec]
    path, _, cls = spec.partition(":")
    if not (cls and Path(path).is_file()):
        raise SystemExit(f"--protocol {spec}: neither one of {sorted(PROTOCOLS)} nor <file.py>:<Class>")
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(Path(path).stem, path))
    module.__spec__.loader.exec_module(module)
    return getattr(module, cls)()


def leaderboard(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MSE per protocol (rows): each domain's main tasks, "main" pooled over them, "all" over every task; n = targets."""
    f = forecasts.assign(se=(forecasts.theta_hat - forecasts.theta) ** 2)
    main = f[f.semester.isin(MAIN)]
    table = main.pivot_table(index="protocol", columns="domain", values="se", aggfunc="mean", dropna=False)
    table = table.reindex(f.protocol.unique())  # a protocol scored on no main task still gets its row
    table["main"] = main.groupby("protocol").se.mean()
    table["n"] = main.groupby("protocol").se.count().reindex(table.index, fill_value=0)  # NaN forecasts are not counted
    table["all"] = f.groupby("protocol").se.mean()
    return table.sort_values(["main", "all"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", nargs="*", type=Path, default=DOMAINS, help="default: science math coding cyber")
    parser.add_argument("--protocol", nargs="+", default=[], help="registered names or <file.py>:<Class>; default: all registered")
    parser.add_argument("--judge", nargs="+", default=[], type=Path, help="JudgeConfig yml(s) -> judge_<name> protocols")
    parser.add_argument("--direct-judge", nargs="+", default=[], type=Path,
                        help="judge yml(s) -> direct_judge_<name> protocols (1–10 ratings, reasoning shown)")
    parser.add_argument("--semester", nargs="+", default=None, help="default: every semester task")
    parser.add_argument("--truncate", help="override the --judge configs' truncate, e.g. thirds-13 (src/challenges/thirds)")
    args = parser.parse_args()

    protocols = [load_protocol(spec) for spec in args.protocol or PROTOCOLS]
    forecasts = []
    for path in args.domain:
        domain = DomainConfig.from_cli(path)
        # JudgeBT, JudgeCalibrated and JudgeLinear read the same run dir.
        override = {"truncate": args.truncate} if args.truncate else {}  # replace() re-runs the config's validation
        judges = [cls(replace(JudgeConfig.from_cli(p), **override), domain)
                  for p in args.judge for cls in (JudgeBT, JudgeCalibrated, JudgeLinear)]
        judges += [JudgeDirect(DirectJudgeConfig.from_cli(p), domain) for p in args.direct_judge]
        for judge in judges:
            judge.collect = False  # score what logs/ holds; `uv run judge` collects
        semesters = [s for s in semester_names(domain) if args.semester is None or s in args.semester]
        if not semesters:  # evaluate() reads an empty list as every task
            continue
        for protocol in protocols + judges:
            evaluate(domain, protocol, protocol.name, semesters)
            forecasts.append(pd.read_csv(results_path(domain, protocol.name), dtype={"semester": str})
                             .assign(domain=domain.name, protocol=protocol.name))
    print(leaderboard(pd.concat(forecasts)).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
