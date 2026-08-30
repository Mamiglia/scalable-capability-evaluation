"""Artifacts returned by each judge-verdict fitting backend."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .config import AnalysisConfig


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


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dictionary: pd.DataFrame,
    feature_space: str,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return raw features unchanged or a fold-local PC projection."""
    if feature_space not in {"raw", "pc"}:
        raise ValueError("feature_space must be 'raw' or 'pc'")
    if {"feature", "block"} - set(dictionary):
        raise ValueError("feature dictionary must contain feature and block")
    features = dictionary.feature.astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be non-empty and unique")
    if missing := sorted(set(features) - set(train) | set(features) - set(test)):
        raise ValueError(f"feature rows missing columns: {missing}")
    if feature_space == "raw":
        return train.copy(), test.copy(), dictionary.copy(), {
            "feature_space": "raw",
            "features": features,
            "loadings": pd.DataFrame(columns=["component", "feature", "loading"]),
            "explained_variance": pd.DataFrame(columns=[
                "component", "explained_variance", "explained_variance_ratio",
                "cumulative_explained_variance_ratio", "singular_value",
            ]),
        }

    train_values = train[features].to_numpy(dtype=float)
    test_values = test[features].to_numpy(dtype=float)
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("PC features must be finite")
    count = min(config.fit_pca_components, len(features))
    model = TruncatedSVD(
        n_components=count,
        algorithm=config.fit_pca_algorithm,
        random_state=config.random_seed,
    ).fit(train_values)
    names = [f"pc_{index:02d}" for index in range(1, count + 1)]

    def project(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        metadata = frame.drop(columns=features).reset_index(drop=True)
        scores = pd.DataFrame(model.transform(values), columns=names)
        return pd.concat([metadata, scores], axis=1)

    ratios = model.explained_variance_ratio_
    loadings = pd.DataFrame([
        {"component": component, "feature": feature, "loading": float(loading)}
        for component, row in zip(names, model.components_)
        for feature, loading in zip(features, row)
    ])
    variance = pd.DataFrame({
        "component": names,
        "explained_variance": model.explained_variance_,
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance_ratio": np.cumsum(ratios),
        "singular_value": model.singular_values_,
    })
    prepared_dictionary = pd.DataFrame({"feature": names, "block": "pc"})
    return project(train, train_values), project(test, test_values), prepared_dictionary, {
        "feature_space": "pc",
        "features": names,
        "source_features": features,
        "loadings": loadings,
        "explained_variance": variance,
    }
