"""A protocol forecasts a target model's theta* from the frozen semester tables (capability.semesters).

    uv run evaluate configs/domain/science.yml [--protocol chars]

A protocol is a callable Task -> theta-hat. A Task is one (semester, target): the 8 reference
models with their local theta* (anchors), the target, the train items, and correct repeat-0
transcripts of the train items for any of these models. evaluate() scores every task's forecast
against theta* per semester and pooled over all semesters ("all"; valid because every local fit
shares the global item scale).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from capability import semesters
from configs.domain import DomainConfig
from judge.dataset import Transcript, read_transcripts


@dataclass
class Task:
    domain: DomainConfig
    semester: str
    references: list[str]
    theta_ref: pd.Series  # local theta* of the references, indexed by model
    target: str
    items: pd.DataFrame  # train items: bench, item, guess, a, b
    _models: pd.DataFrame  # frozen models.csv (for the log paths)

    def transcripts(self, model: str, include_reasoning: bool = True) -> dict[tuple[str, str], list[Transcript]]:
        """Correct repeat-0 transcripts of the train items: (bench, item) -> [Transcript]."""
        row = self._models.set_index("model").loc[model]
        out = {}
        for bench, ids in self.items.groupby("bench").item:
            path = row.get(f"path_{bench}", "")
            if not path:
                continue
            read = read_transcripts(semesters.DATA / path, require_correct=True,
                                    include_reasoning=include_reasoning, items=set(ids))
            out.update({(bench, i): t for i, t in read.items()})
        return out


Protocol = Callable[[Task], float]


def tasks(domain: DomainConfig) -> Iterator[Task]:
    models, items = semesters.load(domain)
    for sem in [c[: -len("_role")] for c in models.columns if c.endswith("_role")]:
        role, theta = models[f"{sem}_role"], models.set_index("model")[f"{sem}_theta"]
        ref = models.model[role == "reference"].tolist()
        train = items[items[f"{sem}_train"]][["bench", "item", "guess", "a", "b"]].reset_index(drop=True)
        for target in models.model[role == "target"]:
            yield Task(domain, sem, ref, theta.loc[ref], target, train, models)


def evaluate(domain: DomainConfig, protocol: Protocol, name: str = "") -> pd.DataFrame:
    """Per-semester and pooled scores of `protocol`; theta-hat per target -> data/meta/<domain>/eval_<name>.csv."""
    models = semesters.load(domain)[0].set_index("model")
    rows = [dict(semester=t.semester, model=t.target, theta_hat=float(protocol(t)),
                 theta=models.loc[t.target, f"{t.semester}_theta"]) for t in tasks(domain)]
    pred = pd.DataFrame(rows)
    if name:
        pred.to_csv(semesters.frozen_dir(domain) / f"eval_{name}.csv", index=False, float_format="%.4f")
    groups = [(sem, g) for sem, g in pred.groupby("semester", sort=False)] + [("all", pred)]
    return pd.DataFrame([{"semester": sem, **semesters.score(g.set_index("model").theta_hat, g.set_index("model").theta)}
                         for sem, g in groups])


THOUGHT = re.compile(r"<THOUGHT_START>.*?<THOUGHT_END>", re.S)
FEATURES = {
    "chars": lambda t: len(t.text),
    "answer_chars": lambda t: len(THOUGHT.sub("", t.text)),
    "reasoning_chars": lambda t: len(t.text) - len(THOUGHT.sub("", t.text)),
    "tokens": lambda t: t.tokens,
}


def features(task: Task, model: str) -> pd.Series:
    """Mean log1p of each transcript feature over the model's correct train transcripts."""
    rows = [{k: f(t[0]) for k, f in FEATURES.items()} for t in task.transcripts(model).values()]
    return np.log1p(pd.DataFrame(rows, columns=list(FEATURES))).mean()


def _table(task: Task) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({m: features(task, m) for m in task.references + [task.target]}).T
    return x, task.theta_ref


def regression(feature: str) -> Protocol:
    """Least squares of the references' theta* on one log feature, applied to the target."""
    def run(task: Task) -> float:
        x, y = _table(task)
        xr, yr = x.loc[task.references, feature], y.loc[task.references]
        if xr.std() == 0:  # e.g. no reference reasons: the feature says nothing about theta
            return float(yr.mean())
        beta, alpha = np.polyfit(xr, yr, 1)
        return alpha + beta * x.loc[task.target, feature]
    return run


def ridge(task: Task, lam: float = 1.0) -> float:
    """Ridge over every standardised log feature (8 references, so shrinkage is not optional)."""
    x, y = _table(task)
    ref = x.loc[task.references]
    z = (x - ref.mean()) / ref.std().replace(0, 1)
    zr = z.loc[task.references].to_numpy()
    w = np.linalg.solve(zr.T @ zr + lam * np.eye(zr.shape[1]), zr.T @ (y.loc[task.references] - y.loc[task.references].mean()))
    return float(y.loc[task.references].mean() + z.loc[task.target].to_numpy() @ w)


PROTOCOLS: dict[str, Protocol] = {**{f: regression(f) for f in FEATURES}, "ridge": ridge}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--protocol", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    a = parser.parse_args()
    domain = DomainConfig.from_cli(a.domain)
    table = pd.concat([evaluate(domain, PROTOCOLS[n], n).assign(protocol=n) for n in a.protocol])
    print(table.pivot(index="semester", columns="protocol", values=["r", "affine_mse"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
