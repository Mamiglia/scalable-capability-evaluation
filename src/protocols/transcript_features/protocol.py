"""Three Gaussian identity-link GLMs: length, length+regex, length+regex+RST structure.

All use the same standardised ridge fit on the supplied reference Cohort.
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from capability.task import Cohort
from protocols.features import TabularRegression
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.reader import ClauseTyper, feature_table


class TranscriptRegression(TabularRegression):
    """Fit reference-model features to theta* and predict target-model theta*.

    Args:
        features: Optional list of feature-column names overriding the variant's
            defaults. The variant still controls whether structure is extracted.
        variant: "length", "regex" or "structure", selecting the nested sets in
            FEATURE_SETS. Output protocol names are glm_<variant>.
        clause_typer: Optional callable mapping structured messages to ordered
            clause labels; required when evaluating the structure variant.

    fit(Cohort) trains standardisation and ridge on reference rows. predict(Cohort)
    returns a model-indexed float Series, with NaN for unusable observations.
    Reference selection belongs to the caller; this class does not choose semesters.
    """
    def __init__(self, features: list[str] | None = None, *, variant: str = "regex",
                 clause_typer: ClauseTyper | None = None):
        if variant not in FEATURE_SETS:
            raise ValueError(f"unknown transcript GLM variant: {variant}")
        self.variant = variant
        self.clause_typer = clause_typer
        super().__init__(f"glm_{variant}", list(FEATURE_SETS[variant] if features is None else features),
                         make_pipeline(StandardScaler(), Ridge(alpha=1.0)))

    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        """Return selected float feature columns, indexed like cohort.models.

        Reads the cohort's selected items from its log paths. Raises ValueError
        when the structure variant has no clause labeller; unknown column names
        raise KeyError. Capability labels are not used in feature extraction.
        """
        if self.variant == "structure" and self.clause_typer is None:
            raise ValueError("structure GLM requires an RST clause labeller")
        typer = self.clause_typer if self.variant == "structure" else None
        return feature_table(cohort, clause_typer=typer)[self.features]
