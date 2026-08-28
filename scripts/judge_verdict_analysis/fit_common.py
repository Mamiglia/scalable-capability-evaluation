"""Artifacts returned by each judge-verdict fitting backend."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class FitResult:
    backend: str
    fold: str
    selected_features: tuple[str, ...]
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    tuning: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: dict[str, object]
