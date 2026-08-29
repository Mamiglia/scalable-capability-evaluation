"""Benchmark-specific PCA summaries for selected readable transcript features."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_tables(
    transcripts: pd.DataFrame,
    dictionary: pd.DataFrame,
    n_components: int = 20,
    common_support: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit benchmark-specific standardised PCAs and return variance/loadings.

    When ``common_support`` is true, every benchmark uses the same feature basis:
    only columns with non-zero variance in every benchmark are retained. This is
    required for a cross-benchmark loading plot; otherwise domain-only plan/tool
    features can appear important only where they exist.
    """
    columns = dictionary.column.tolist()
    required = {"bench"} | set(columns)
    if missing := required - set(transcripts):
        raise ValueError(f"transcript table missing columns: {sorted(missing)}")
    names = dictionary.set_index("column")[["feature", "block"]]
    if common_support:
        # Intersect support before fitting any benchmark so every PCA sees the
        # same columns. Per-benchmark constant dropping remains a safety check.
        variable = [
            set(column for column in columns
                if np.nanstd(group[column].to_numpy(float)) > 0)
            for _, group in transcripts.groupby("bench", sort=False)
        ]
        columns = [column for column in columns if all(column in keep for keep in variable)]
    variance_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    for bench, group in transcripts.groupby("bench", sort=False):
        values = group[columns].to_numpy(float)
        variable = np.nanstd(values, axis=0) > 0
        used = [column for column, keep in zip(columns, variable) if keep]
        if not used:
            continue
        values = np.nan_to_num(group[used].to_numpy(float))
        values = StandardScaler().fit_transform(values)
        count = min(n_components, values.shape[0], values.shape[1])
        model = PCA(n_components=count).fit(values)
        cumulative = np.cumsum(model.explained_variance_ratio_)
        for index, (ratio, total) in enumerate(zip(model.explained_variance_ratio_, cumulative), 1):
            variance_rows.append({"bench": bench, "pc": index,
                                  "explained_variance_ratio": ratio, "cumulative": total,
                                  "n_rows": len(group), "n_features": len(used)})
        for index, component in enumerate(model.components_, 1):
            for column, value in zip(used, component):
                loading_rows.append({"bench": bench, "pc": index,
                                     "feature": names.loc[column, "feature"],
                                     "block": names.loc[column, "block"],
                                     "loading": value, "abs_loading": abs(value)})
    return pd.DataFrame(variance_rows), pd.DataFrame(loading_rows)


def align_pc_signs(
    loadings: pd.DataFrame,
    associations: pd.DataFrame,
    association_column: str,
    alignment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orient each PC toward stronger or judge-preferred transcripts.

    PCA signs are arbitrary. For each benchmark/component, this function takes
    the dot product of its loading vector and the feature-wise target-association
    vector. Negative projections flip the component. Repeated association rows
    (one per judge) are averaged before alignment.

    Returns:
        Aligned loading rows and one direction/audit row per benchmark/component.
    """
    required_loadings = {"bench", "pc", "feature", "loading"}
    required_associations = {"bench", "feature", association_column}
    if missing := required_loadings - set(loadings):
        raise ValueError(f"loading table missing columns: {sorted(missing)}")
    if missing := required_associations - set(associations):
        raise ValueError(f"association table missing columns: {sorted(missing)}")
    target = (associations.groupby(["bench", "feature"], as_index=False)[association_column]
              .mean())
    merged = loadings.merge(target, on=["bench", "feature"], how="left")
    rows: list[dict[str, object]] = []
    for (bench, pc), group in merged.groupby(["bench", "pc"], sort=False):
        valid = group.dropna(subset=[association_column])
        projection = float((valid.loading * valid[association_column]).sum())
        sign = -1 if projection < 0 else 1
        rows.append({"bench": bench, "pc": pc, "alignment": alignment,
                     "method": "loading_dot_feature_association",
                     "sign": sign, "projection_before": projection,
                     "projection_after": abs(projection), "n_features": len(valid)})
    directions = pd.DataFrame(rows)
    aligned = loadings.merge(directions[["bench", "pc", "alignment", "sign"]],
                             on=["bench", "pc"], how="left", validate="many_to_one")
    aligned["loading"] *= aligned.sign
    aligned["abs_loading"] = aligned.loading.abs()
    return aligned, directions
