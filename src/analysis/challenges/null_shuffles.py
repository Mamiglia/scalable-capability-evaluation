"""Null diagnostics: destroy each condition's signal by shuffling and rescore it.

For every semester task, each condition is rescored on shuffled inputs, `draws` times:
    glm_regex, embedding   the per-model feature rows are permuted among the models of a
                           cohort (references among references, targets among targets), so
                           every model gets another model's features
    direct                 the `score` column of the judge's rating table is permuted
    pairwise               the `pick_left` column of the judge's verdict table is permuted
Judge conditions use the semester's best judge on references (the one the results use).
A condition with real signal has an actual MSE below most of its null MSEs; a condition
whose null distribution sits at or below the actual MSE is not reading capability.
"""

import numpy as np
import pandas as pd

from analysis import judges
from analysis.calibration import judge_forecast
from analysis.loading import JudgeRun
from capability.task import Cohort, Task
from protocols.features import TabularRegression


class FrozenFeatures(TabularRegression):
    """A TabularRegression whose feature tables are given (one per cohort id) instead of read.

    tables: {id(cohort): DataFrame indexed by model}. Lets a shuffled table be fitted and
    predicted through the protocol's own estimator and NaN handling."""

    def __init__(self, protocol: TabularRegression, tables: dict[int, pd.DataFrame]):
        super().__init__(protocol.name, protocol.features, protocol.estimator)
        self.tables = tables

    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        """The stored table for this cohort, reindexed like cohort.models."""
        return self.tables[id(cohort)].reindex(cohort.models.index)


def permute_rows(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The same table with its rows assigned to a random permutation of its index."""
    return pd.DataFrame(table.to_numpy()[rng.permutation(len(table))], index=table.index, columns=table.columns)


def mse(theta_hat: pd.Series, truth: pd.Series) -> float:
    """Mean squared error over the targets both series cover (NaN forecasts dropped)."""
    err = (theta_hat.reindex(truth.index) - truth).dropna()
    return float((err ** 2).mean()) if len(err) else float("nan")


def feature_nulls(protocol: TabularRegression, task: Task, draws: int, seed: int) -> tuple[float, list[float]]:
    """(actual MSE, null MSEs) for a feature protocol: features read once, then permuted per draw."""
    train, test = task.references, task.test
    x_train, x_test = protocol.get_features(train), protocol.get_features(test)
    actual = FrozenFeatures(protocol, {id(train): x_train, id(test): x_test})
    actual.fit(train)
    out = [mse(actual.predict(test), task.truth)]
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(draws):
        shuffled = FrozenFeatures(protocol, {id(train): permute_rows(x_train, rng), id(test): permute_rows(x_test, rng)})
        shuffled.fit(train)
        nulls.append(mse(shuffled.predict(test), task.truth))
    return out[0], nulls


def judge_nulls(run: JudgeRun, task: Task, variant: str, draws: int, seed: int) -> tuple[float, list[float]]:
    """(actual MSE, null MSEs) for a judge run: the ratings' `score` (direct) or the verdicts'
    `pick_left` (pairwise) permuted per draw, then converted exactly as the results are."""
    column = "pick_left" if run.protocol == "pairwise" else "score"
    actual = mse(judge_forecast(run, task, variant)[0], task.truth)
    rng = np.random.default_rng(seed)
    nulls, table = [], run.table
    for _ in range(draws):
        run.table = table.assign(**{column: table[column].to_numpy()[rng.permutation(len(table))]})
        nulls.append(mse(judge_forecast(run, task, variant)[0], task.truth))
    run.table = table
    return actual, nulls


def summarise(actual: float, nulls: list[float]) -> dict:
    """actual, null mean / 5% / 95%, and p = share of null draws at or below the actual MSE."""
    n = np.asarray(nulls, float)
    return dict(actual=actual, null_mean=float(np.nanmean(n)), null_p05=float(np.nanpercentile(n, 5)),
                null_p95=float(np.nanpercentile(n, 95)), p_null_le_actual=float(np.mean(n <= actual)), draws=len(n))
