"""Artifacts returned by each judge-verdict fitting backend."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    """Return raw features unchanged or project through the frozen global basis."""
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

    if config.pc_basis_root is None:
        raise ValueError("pc_basis_root is required for PC features")
    loadings_path = config.pc_basis_root / "loadings.parquet"
    variance_path = config.pc_basis_root / "explained-variance.csv"
    if not loadings_path.is_file() or not variance_path.is_file():
        raise ValueError(f"incomplete global PC basis: {config.pc_basis_root}")
    loadings = pd.read_parquet(loadings_path)
    variance = pd.read_csv(variance_path)
    variance_columns = [
        "component", "explained_variance", "explained_variance_ratio",
        "cumulative_explained_variance_ratio", "singular_value",
    ]
    if {"component", "feature", "loading"} - set(loadings) or set(variance_columns) - set(variance):
        raise ValueError("invalid global PC basis schema")
    loadings = loadings.copy()
    variance = variance.copy()
    loadings[["component", "feature"]] = loadings[["component", "feature"]].astype(str)
    variance["component"] = variance.component.astype(str)
    if variance.component.duplicated().any():
        raise ValueError("global PC basis has duplicate component rows")
    try:
        variance[variance_columns[1:]] = variance[variance_columns[1:]].apply(
            pd.to_numeric, errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError("global PC explained variance must have finite numeric values") from error
    if not np.isfinite(variance[variance_columns[1:]].to_numpy(dtype=float)).all():
        raise ValueError("global PC explained variance must have finite numeric values")
    components = variance.component.astype(str).tolist()[:config.fit_pca_components]
    if len(components) != config.fit_pca_components or len(components) != len(set(components)):
        raise ValueError("global PC basis has too few or duplicate components")
    if loadings.duplicated(["component", "feature"]).any():
        raise ValueError("global PC basis has duplicate loadings")
    selected = loadings[loadings.component.isin(components)]
    if set(selected.component) != set(components) or any(
        set(group.feature) != set(features)
        for _, group in selected.groupby("component", sort=False)
    ):
        raise ValueError("global PC basis selected components do not match the design matrix")
    matrix = loadings.pivot(index="component", columns="feature", values="loading").reindex(
        index=components, columns=features,
    ).to_numpy(dtype=float)
    train_values = train[features].to_numpy(dtype=float)
    test_values = test[features].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("global PC loadings must be finite and complete")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("PC features must be finite")

    def project(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        metadata = frame.drop(columns=features).reset_index(drop=True)
        scores = pd.DataFrame(values @ matrix.T, columns=components)
        return pd.concat([metadata, scores], axis=1)

    selected_loadings = pd.DataFrame({
        "component": np.repeat(components, len(features)),
        "feature": features * len(components),
        "loading": matrix.ravel(),
    })
    selected_variance = variance.set_index("component").loc[components].reset_index()
    prepared_dictionary = pd.DataFrame({"feature": components, "block": "pc"})
    return project(train, train_values), project(test, test_values), prepared_dictionary, {
        "feature_space": "pc",
        "features": components,
        "source_features": features,
        "pc_basis_root": str(config.pc_basis_root),
        "loadings": selected_loadings,
        "explained_variance": selected_variance,
    }
