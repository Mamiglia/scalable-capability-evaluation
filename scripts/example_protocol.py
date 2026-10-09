"""A complete new protocol in ~30 lines, offline: the judges' pairwise win rate against the references, mapped to theta
by a line fit on the references (their win rate among themselves vs their theta*).

    uv run python scripts/example_protocol.py && uv run python scripts/leaderboard.py winrate
"""

import numpy as np
import pandas as pd

from capability.evaluate import evaluate
from capability.task import Task
from configs.domain import DomainConfig
from protocols import Protocol
from reference.tables import META

VERDICTS = pd.read_csv(META / "judgments_pairwise.csv", dtype={"item": str, "semester": str})


def win_rate(verdicts: pd.DataFrame, opponents: pd.Index) -> pd.Series:
    """Per model, the share of its verdicts against `opponents` that it won."""
    left = verdicts[verdicts.model_right.isin(opponents)].assign(model=lambda v: v.model_left, won=lambda v: v.pick_left)
    right = verdicts[verdicts.model_left.isin(opponents)].assign(model=lambda v: v.model_right, won=lambda v: 1 - v.pick_left)
    return pd.concat([left, right]).groupby("model").won.mean()


class WinRate(Protocol):
    name = "winrate"

    def __call__(self, task: Task) -> pd.Series:
        refs = task.references.theta
        # this task's verdicts: its semester, its train items (bench names also pick the domain), every judge
        verdicts = VERDICTS[VERDICTS.semester == task.semester].merge(task.references.items[["bench", "item"]])
        rate = win_rate(verdicts, refs.index)
        known = rate.reindex(refs.index).dropna()
        slope, intercept = np.polyfit(known, refs[known.index], 1)
        return intercept + slope * rate.reindex(task.test.models.index)


if __name__ == "__main__":
    for domain in ["science", "math", "coding", "cyber"]:
        scores = evaluate(DomainConfig.from_cli(f"configs/domain/{domain}.yml"), WinRate(), WinRate.name, ["2025-II", "2026-I"])
        print(domain, scores.round(2).to_dict("records"))
