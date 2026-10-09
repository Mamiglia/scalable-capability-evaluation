"""Effect of the number of judged items on the strongest judge's pairwise MSE, per domain and semester (no judging is run).

    uv run python scripts/analyses/item_count_effect.py [--recompute] [--last K] [--direct | --both]
    # --last K: only the last K semesters (files tagged _lastK); --direct: direct judging by the top-1 judge (files tagged _direct);
    # --both: the top-1 judge, a Pairwise row and an Absolute (direct) row, squared error, from the two cached tables (files tagged _both)

Judges = the top-3 references of each semester (data/meta/<domain>/fit.yml, tables.fit_settings `top`, one figure row per rank), subjects' reasoning shown
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

import numpy as np
import pandas as pd

from capability.evaluate import score
from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import DirectJudgeConfig
from configs.logs import logger
from plots.item_count import plot
from protocols import PROTOCOLS
from protocols.direct_judge import JudgeDirect, from_logs as direct_logs
from protocols.direct_judge.scoring import calibrate
from protocols.pairwise import from_logs
from protocols.pairwise.linear import weighted_theta
from reference import tables

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/item_count_effect")
DOMAINS = ["science", "math", "coding", "cyber"]
CONDITIONS = ["24d6cc32", "66c170cd"]
SIZES = [3, 6, *range(10, 101, 10)]
DRAWS = 30


def forecasts(verdicts: pd.DataFrame, task: Task) -> dict[str, pd.Series]:
    refs, targets = task.references.theta, list(task.test.models.index)
    return {"linear": weighted_theta(verdicts, refs, targets)}


def compute() -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(0)
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = tables.fit_settings(name)["top"]
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
        top = tables.fit_settings(name)["top"]
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
