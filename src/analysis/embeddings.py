"""Ridge on the mean transcript embedding of a model's correct train transcripts.

Vectors come from the HF dataset LASR-G5/transcript-embeddings (Qwen3-Embedding-8B), one
.npz per collection log at the log's path with `.eval` replaced by `.<variant>.npz`, under
the directory named by SMALL_JUDGE_EMBEDDINGS. Variant "full" embeds the transcript with its
reasoning (what a judge shown reasoning sees), "answer" the reasoning-stripped text. A
model's feature row is the mean of its train transcripts' L2-normalised vectors; the
estimator is ridge with the penalty chosen by leave-one-out over the references, as for the
surface-feature GLM.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from capability.task import Cohort
from configs.logs import logger
from protocols.features import TabularRegression

EMBEDDINGS = Path(os.environ.get("SMALL_JUDGE_EMBEDDINGS", "transcript-embeddings"))
VARIANTS = ("full", "answer")
ALPHAS = np.logspace(-2, 4, 13)  # unit-norm inputs need stronger shrinkage than standardised features


def npz_path(log_path: str, variant: str) -> Path:
    """The embedding file of a data/ log: same relative path, `.eval` -> `.<variant>.npz`."""
    return EMBEDDINGS / (log_path.rsplit(".eval", 1)[0] + f".{variant}.npz")


def read_vectors(path: Path, ids: set[str]) -> np.ndarray | None:
    """The [k, dim] float32 first-attempt vectors of the samples in `ids`, or None when the
    file is missing (reported) or holds none of them."""
    if not path.exists():
        logger.warning("%s: no embedding file; pull it from LASR-G5/transcript-embeddings", path)
        return None
    with np.load(path) as f:
        keep = np.isin(f["ids"], list(ids)) & (f["epochs"] == 1)
        return f["embedding"][keep].astype(np.float32) if keep.any() else None


class EmbeddingRegression(TabularRegression):
    """Fit reference embeddings to theta*, predict target theta* (protocol name "embedding")."""

    def __init__(self, variant: str = "full"):
        """variant: "full" (reasoning included), "answer" (reasoning stripped) or "channels-<spec>" (the transcript
        rendered with JudgeConfig.channels = <spec>, src/challenges/channels/embed.py)."""
        assert variant in VARIANTS or variant.startswith("channels-"), variant
        self.variant = variant
        super().__init__("embedding", [], RidgeCV(alphas=ALPHAS))

    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        """One row per model of `cohort` (index = cohort.models.index): the mean vector over its
        correct first-attempt train transcripts across benches; a row of NaN for a model with
        no readable vector. Columns are "0".."<dim-1>"."""
        rows: dict[str, np.ndarray | None] = {}
        for model, transcripts in cohort.transcripts.items():
            ids_by_bench: dict[str, set[str]] = {}
            for bench, item in transcripts:
                ids_by_bench.setdefault(bench, set()).add(item)
            vectors = []
            for bench, ids in ids_by_bench.items():
                path = cohort.models.loc[model, f"path_{bench}"]
                if isinstance(path, str) and (v := read_vectors(npz_path(path, self.variant), ids)) is not None:
                    vectors.append(v)
            rows[model] = np.concatenate(vectors).mean(axis=0) if vectors else None
        dim = next((len(v) for v in rows.values() if v is not None), 0)
        if dim == 0:  # no model has a readable vector: a 0-column frame makes sklearn raise
            return pd.DataFrame(np.nan, index=cohort.models.index, columns=["0"])
        table = pd.DataFrame({m: (v if v is not None else np.full(dim, np.nan)) for m, v in rows.items()}).T
        table.columns = [str(i) for i in range(dim)]
        return table.reindex(cohort.models.index)
