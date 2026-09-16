"""Regressions on the mean transcript embedding of a model's correct train transcripts.

Vectors come from the HF dataset LASR-G5/transcript-embeddings (Qwen3-Embedding-8B), one
.npz per collection log at the log's path with `.eval` replaced by `.<variant>.npz`, under
the directory named by SMALL_JUDGE_EMBEDDINGS. Variant "full" embeds the transcript with its
reasoning (what a judge shown reasoning sees), "answer" the reasoning-stripped text. A
model's feature row is the mean of its train transcripts' L2-normalised vectors.

Each .npz holds ids (str), epochs (int, 1 = first attempt), correct (bool), n_tokens, n_chunks,
embedding (float16 [n, 4096], L2-normalised, token-weighted mean over 8192-token chunks) and a
json `meta`; scripts/embeddings/embed.py writes them.

Protocols:
    embedding        ridge with the penalty chosen by leave-one-out over the references, as for
                     the surface-feature GLM (registered in protocols.features)
    emb_kr           kernel ridge with a cosine kernel, alpha by leave-one-out on the references
    emb_knn          distance-weighted cosine 3-nearest-neighbour regression
    emb_ens          PCA-6 of the mean embedding, then the mean of cosine 3-NN and cosine kernel
                     ridge: the family scripts/embeddings/hillclimb2_cross.py selects on the other
                     domains for every held-out domain
    *_answer         the same on the answer-only ("answer") variant
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from capability.task import Cohort
from configs.logs import logger
from protocols.features import TabularRegression

EMBEDDINGS = Path(os.environ.get("SMALL_JUDGE_EMBEDDINGS", "transcript-embeddings"))
VARIANTS = ("full", "answer")
ALPHAS = np.logspace(-2, 4, 13)  # unit-norm inputs need stronger shrinkage than standardised features


def npz_path(log_path: str, variant: str) -> Path:
    """The embedding file of a data/ log: same relative path, `.eval` -> `.<variant>.npz`."""
    return EMBEDDINGS / (log_path.rsplit(".eval", 1)[0] + f".{variant}.npz")


def embedding_path(log: str | Path, variant: str) -> Path:
    """npz_path for the scripts (scripts/embeddings/*) that hold a Path."""
    return npz_path(str(log), variant)


def read_vectors(path: Path, ids: set[str]) -> np.ndarray | None:
    """The [k, dim] float32 first-attempt vectors of the samples in `ids`, or None when the
    file is missing (reported) or holds none of them."""
    if not path.exists():
        logger.warning("%s: no embedding file; pull it from LASR-G5/transcript-embeddings", path)
        return None
    with np.load(path) as f:
        keep = np.isin(f["ids"], list(ids)) & (f["epochs"] == 1)
        return f["embedding"][keep].astype(np.float32) if keep.any() else None


@lru_cache(maxsize=1024)
def read_embeddings(path: Path) -> pd.DataFrame:
    """Correct first-attempt rows of one embedding file as float32, indexed by item (item-level
    access for the sweeps in scripts/embeddings); empty when the file is missing."""
    if not path.exists():
        logger.warning("%s: no embedding file; pull it from LASR-G5/transcript-embeddings", path)
        return pd.DataFrame()
    with np.load(path) as store:
        keep = store["correct"] & (store["epochs"] == 1)
        index = pd.Index(store["ids"][keep].astype(str), name="item")
        return pd.DataFrame(store["embedding"][keep].astype(np.float32), index=index)


def kernel_ridge_cosine(alphas=np.logspace(-3, 1, 9)):
    """Kernel ridge on unit vectors (a linear kernel there is the cosine kernel); alpha by leave-one-out CV."""
    search = GridSearchCV(KernelRidge(kernel="linear"), {"alpha": alphas},
                          cv=LeaveOneOut(), scoring="neg_mean_squared_error")
    return make_pipeline(Normalizer(), search)


class PcaKnnKernelEnsemble(BaseEstimator, RegressorMixin):
    """PCA fit on the training rows, L2-normalise, then the mean of cosine k-NN and cosine kernel ridge forecasts."""

    def __init__(self, n_components: int = 6, n_neighbors: int = 3):
        self.n_components, self.n_neighbors = n_components, n_neighbors

    def _project(self, X):
        return Normalizer().fit_transform(self.pca_.transform(X))

    def fit(self, X, y):
        self.n_samples_fit_ = len(X)  # EmbeddingRegression.fit reads this: k-NN needs n_neighbors rows
        self.pca_ = PCA(n_components=min(self.n_components, len(X) - 1), random_state=0).fit(X)
        Z = self._project(X)
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors, metric="cosine", weights="distance").fit(Z, y)
        self.kr_ = GridSearchCV(KernelRidge(kernel="linear"), {"alpha": np.logspace(-4, 2, 13)},
                                cv=LeaveOneOut(), scoring="neg_mean_squared_error").fit(Z, y)
        return self

    def predict(self, X):
        Z = self._project(X)
        return (self.knn_.predict(Z) + self.kr_.predict(Z)) / 2


class EmbeddingRegression(TabularRegression):
    """Fit reference embeddings to theta*, predict target theta* (protocol name "embedding")."""

    def __init__(self, variant: str = "full", name: str = "embedding", estimator=None):
        """variant: "full" (reasoning included) or "answer" (reasoning stripped); estimator defaults
        to RidgeCV over ALPHAS."""
        assert variant in VARIANTS, variant
        self.variant = variant
        super().__init__(name, [], estimator if estimator is not None else RidgeCV(alphas=ALPHAS))

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

    def fit(self, train: Cohort) -> None:
        super().fit(train)
        # A nearest-neighbour estimator raises at predict time with fewer training rows than neighbours.
        needed = getattr(self.estimator, "n_neighbors", 0)
        if self.fitted and needed and getattr(self.estimator, "n_samples_fit_", 0) < needed:
            self.fitted = False


def _knn():
    return KNeighborsRegressor(n_neighbors=3, metric="cosine", weights="distance")


PROTOCOLS: dict[str, EmbeddingRegression] = {
    "emb_kr": EmbeddingRegression("full", "emb_kr", kernel_ridge_cosine()),
    "emb_knn": EmbeddingRegression("full", "emb_knn", _knn()),
    "emb_ens": EmbeddingRegression("full", "emb_ens", PcaKnnKernelEnsemble()),
    "emb_kr_answer": EmbeddingRegression("answer", "emb_kr_answer", kernel_ridge_cosine()),
    "emb_knn_answer": EmbeddingRegression("answer", "emb_knn_answer", _knn()),
}
