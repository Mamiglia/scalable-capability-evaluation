"""Nested GLM tiers on transcript features: length, length+regex, and (pending RST) structure.

Standardised ridge with the penalty chosen by leave-one-out on the training cohort.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from capability.task import Cohort
from configs.logs import logger
from protocols.features import TabularRegression
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.reader import feature_table

ALPHAS = np.logspace(-3, 3, 13)


class TranscriptRegression(TabularRegression):
    """Fit reference-model features to theta* and predict target-model theta*.

    Args:
        features: Optional column names overriding the variant's defaults.
        variant: A key of FEATURE_SETS ("length", "regex", "structure"); protocol name glm_<variant>.
        reasoning: "visible" (text + raw CoT) or "hidden" (text only); a non-default view is
            appended to the protocol name.
    """
    def __init__(self, features: list[str] | None = None, *, variant: str = "regex", reasoning: str = "visible"):
        if variant not in FEATURE_SETS:
            raise ValueError(f"unknown transcript GLM variant: {variant}")
        self.variant, self.reasoning = variant, reasoning
        name = f"glm_{variant}" + ("" if reasoning == "visible" else f"_reasoning-{reasoning}")
        super().__init__(name, list(FEATURE_SETS[variant] if features is None else features),
                         make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)))

    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        """Selected columns per model; structure columns are NaN for models whose transcripts lack RST labels."""
        return feature_table(cohort, reasoning=self.reasoning)[self.features]

    def fit(self, train: Cohort) -> None:
        super().fit(train)
        if self.fitted:
            logger.info("%s %s: alpha=%g", self.name, train.semester, self.estimator[-1].alpha_)
