"""Feature processing and transformation for judge-verdict models."""
from collections.abc import Mapping
from typing import Literal

import pandas as pd

from .config import CONFIG


CorrelationMethod = Literal["pearson", "spearman"]


def common_support(
    table: pd.DataFrame,
    dictionary: pd.DataFrame,
    benchmarks: tuple[str, ...],
    splits: tuple[str, ...],
) -> pd.DataFrame:
    """Keep dictionary rows variable in every benchmark's selected splits."""
    selected = table[table.bench.isin(benchmarks) & table.split.isin(splits)]
    columns = dictionary.column.tolist()
    retained = [
        column for column in columns
        if all(
            selected.loc[selected.bench == bench, column].nunique(dropna=True) > 1
            for bench in benchmarks
        )
    ]
    return dictionary[dictionary.column.isin(retained)].copy()


def feature_correlations(
    table: pd.DataFrame,
    dictionary: pd.DataFrame,
    method: CorrelationMethod,
) -> pd.DataFrame:
    """Return all defined within-benchmark feature correlations in long form."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    names = dictionary.set_index("column")[["feature", "block"]]
    columns = dictionary.column.tolist()
    rows: list[dict[str, object]] = []
    for bench, group in table.groupby("bench", sort=False):
        correlation = group[columns].corr(method=method)
        for index, left in enumerate(columns):
            for right in columns[index + 1:]:
                value = correlation.loc[left, right]
                if pd.notna(value):
                    rows.append({
                        "bench": bench,
                        "feature_a": names.loc[left, "feature"], "block_a": names.loc[left, "block"],
                        "feature_b": names.loc[right, "feature"], "block_b": names.loc[right, "block"],
                        "correlation": float(value), "abs_correlation": abs(float(value)),
                        "method": method,
                        "n_complete": int(group[[left, right]].notna().all(axis=1).sum()),
                    })
    return pd.DataFrame(rows, columns=[
        "bench", "feature_a", "block_a", "feature_b", "block_b",
        "correlation", "abs_correlation", "method", "n_complete"])


def select_representatives(
    groups: pd.DataFrame,
    rule: str,
    overrides: Mapping[str, str],
    aliases: Mapping[str, str],
) -> pd.DataFrame:
    """Mark one readable original feature per correlation group for retention."""
    if rule != "shortest_name":
        raise ValueError("rule must be 'shortest_name'")
    result = groups.copy()
    representatives = {
        group: overrides.get(group, min(frame.feature, key=lambda feature: (len(feature), feature)))
        for group, frame in result.groupby("correlation_group", sort=False)
    }
    if invalid := {
        group: feature for group, feature in representatives.items()
        if feature not in set(result.loc[result.correlation_group == group, "feature"])
    }:
        raise ValueError(f"representative overrides are not group members: {invalid}")
    result["representative"] = result.correlation_group.map(representatives)
    result["retained"] = result.feature == result.representative
    result["processed_name"] = result.representative.map(aliases).fillna(result.representative)
    return result


def correlation_groups(
    pairs: pd.DataFrame,
    dictionary: pd.DataFrame,
    threshold: float = CONFIG.correlation_threshold,
    transitive: bool = True,
) -> pd.DataFrame:
    """Create shared groups from mean absolute correlations.

    Correlations are averaged across available benchmarks for each feature pair.
    Pairs at or above threshold are joined. With ``transitive=False``, groups
    merge only when every cross-group pair passes.

    Args:
        pairs: Pairwise feature correlations by benchmark.
        dictionary: Feature names and their original blocks.
        threshold: Minimum mean absolute correlation for joining features.
        transitive: Whether correlation chains can join an entire group.

    Returns:
        result: pd.DataFrame
            One row per feature with its correlation group, group size, and
            original block.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    required = {"feature_a", "feature_b", "abs_correlation"}
    if missing := required - set(pairs):
        raise ValueError(f"correlation table missing columns: {sorted(missing)}")

    metadata = dictionary[["feature", "block"]].drop_duplicates("feature")
    # Union-find tracks group roots; members supports the stricter all-pairs check.
    parent = {feature: feature for feature in metadata.feature}
    members = {feature: {feature} for feature in metadata.feature}

    def root(feature: str) -> str:
        # Compress paths because the same feature can appear in many candidate edges.
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    # Each benchmark contributes equally to one shared correlation per feature pair.
    means = (pairs.groupby(["feature_a", "feature_b"], as_index=False)
             .abs_correlation.mean())
    similarities = {
        frozenset((edge.feature_a, edge.feature_b)): edge.abs_correlation
        for edge in means.itertuples()
    }
    # Strongest-first ordering makes the non-transitive partition deterministic.
    edges = means[means.abs_correlation >= threshold].sort_values(
        ["abs_correlation", "feature_a", "feature_b"],
        ascending=[False, True, True],
    )
    for edge in edges.itertuples():
        a, b = root(edge.feature_a), root(edge.feature_b)
        if a != b:
            # Reject a strict merge if even one cross-group pair is weak or missing.
            if not transitive and any(
                similarities.get(frozenset((left, right)), -1) < threshold
                for left in members[a] for right in members[b]
            ):
                continue
            parent[b] = a
            members[a].update(members.pop(b))

    # Stable anonymous IDs keep output independent of correlation-table row order.
    group_ids = {
        feature: f"group-{index:03d}"
        for index, group in enumerate(sorted(map(sorted, members.values())), start=1)
        for feature in group
    }
    result = metadata.rename(columns={"block": "original_block"}).copy()
    result["correlation_group"] = result.feature.map(group_ids)
    result["group_size"] = result.groupby("correlation_group").feature.transform("size")
    return result
