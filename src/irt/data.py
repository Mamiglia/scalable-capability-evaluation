"""Response table -> integer-indexed arrays for the IRT fit.

A response table has one row per (bench, model, item, repeat) with `correct`; items are keyed
by (bench, item) so several benches can share one fit.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd


class IRTData(NamedTuple):
    model_idx: np.ndarray
    item_idx: np.ndarray
    y: np.ndarray  # 0/1 correct
    models: list[str]  # model index -> name
    items: list[tuple]  # item index -> (bench, item)


def encode(responses: pd.DataFrame) -> IRTData:
    """Response table to integer-indexed arrays for the fit."""
    pairs = list(zip(responses["bench"], responses["item"]))
    models = sorted(responses["model"].unique())
    item_ids = sorted(set(pairs))
    m_index = {m: i for i, m in enumerate(models)}
    i_index = {q: i for i, q in enumerate(item_ids)}
    return IRTData(
        model_idx=responses["model"].map(m_index).to_numpy(),
        item_idx=np.array([i_index[p] for p in pairs]),
        y=responses["correct"].astype(int).to_numpy(),
        models=models,
        items=item_ids,
    )
