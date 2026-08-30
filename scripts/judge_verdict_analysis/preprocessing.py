"""Feature processing and transformation for judge-verdict models."""
from collections.abc import Mapping
import json
from typing import Literal

import pandas as pd

from .config import CONFIG


CorrelationMethod = Literal["pearson", "spearman"]


def _combined_weights(
    left: Mapping[str, float],
    right: Mapping[str, float],
    right_sign: int,
    average: bool,
) -> dict[str, float]:
    scale = 0.5 if average else 1.0
    weights = {feature: scale * weight for feature, weight in left.items()}
    for feature, weight in right.items():
        weights[feature] = weights.get(feature, 0.0) + scale * right_sign * weight
    return {feature: float(weights[feature]) for feature in sorted(weights)}


def _merge_values(
    values: Mapping[str, pd.Series],
    left: str,
    right: str,
    right_sign: int,
    support: Mapping[str, Mapping[str, bool]],
    benchmarks: list[str],
    bench: pd.Series,
) -> pd.Series:
    output = pd.Series(float("nan"), index=bench.index, dtype=float)
    for benchmark in benchmarks:
        mask = bench == benchmark
        left_active = support[left][benchmark]
        right_active = support[right][benchmark]
        if left_active and right_active:
            output.loc[mask] = pd.concat(
                [values[left].loc[mask], right_sign * values[right].loc[mask]], axis=1,
            ).mean(axis=1, skipna=True)
        elif left_active:
            output.loc[mask] = values[left].loc[mask]
        elif right_active:
            output.loc[mask] = right_sign * values[right].loc[mask]
    return output


def apply_composites(
    table: pd.DataFrame,
    dictionary: pd.DataFrame,
    provenance: Mapping[str, object],
    *,
    neutralize: bool = True,
) -> pd.DataFrame:
    """Replay a fitted composite tree, optionally neutralizing unavailable rows."""
    if "bench" not in table:
        raise ValueError("composite table must contain bench")
    required = {"feature", "column"}
    if missing := required - set(dictionary):
        raise ValueError(f"composite dictionary missing columns: {sorted(missing)}")
    columns = dictionary.set_index("feature").column.astype(str).to_dict()
    source_features = [str(feature) for feature in provenance["source_features"]]
    if missing := [columns[feature] for feature in source_features if columns.get(feature) not in table]:
        raise ValueError(f"composite table missing source columns: {missing}")
    benchmarks = [str(benchmark) for benchmark in provenance["benchmarks"]]
    bench = table.bench.astype(str)
    if unknown := sorted(set(bench) - set(benchmarks)):
        raise ValueError(f"composite table has unknown benchmarks: {unknown}")

    leaf_support = provenance["leaf_support"]
    values: dict[str, pd.Series] = {}
    support: dict[str, dict[str, bool]] = {}
    for feature in source_features:
        values[feature] = table[columns[feature]].astype(float).copy()
        support[feature] = {
            benchmark: bool(leaf_support[feature][benchmark])
            for benchmark in benchmarks
        }
        for benchmark, active in support[feature].items():
            if not active:
                values[feature].loc[bench == benchmark] = float("nan")

    for merge in provenance["merges"]:
        left = str(merge["left_parent_id"])
        right = str(merge["right_parent_id"])
        output = str(merge["output_id"])
        right_sign = -1 if merge["branch_sign_flip"] else 1
        support[output] = {
            benchmark: bool(merge["active_support"][benchmark]["output"])
            for benchmark in benchmarks
        }
        values[output] = _merge_values(
            values, left, right, right_sign, support, benchmarks, bench,
        )

    final_features = [str(feature) for feature in provenance["final_features"]]
    output = pd.DataFrame(
        {feature: values[feature] for feature in final_features},
        index=table.index,
    )
    return output.fillna(0.0) if neutralize else output


def fit_composites(
    table: pd.DataFrame,
    dictionary: pd.DataFrame,
    threshold: float = CONFIG.correlation_threshold,
    method: CorrelationMethod = "pearson",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Greedily merge the strongest sign-consistent correlation and record its tree."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    required = {"feature", "column", "block"}
    if missing := required - set(dictionary):
        raise ValueError(f"composite dictionary missing columns: {sorted(missing)}")
    if dictionary.feature.duplicated().any():
        raise ValueError("composite dictionary contains duplicate features")
    if "bench" not in table:
        raise ValueError("composite table must contain bench")

    metadata = dictionary.set_index("feature")
    source_features = sorted(metadata.index.astype(str))
    columns = metadata.column.astype(str).to_dict()
    if missing := [columns[feature] for feature in source_features if columns[feature] not in table]:
        raise ValueError(f"composite table missing source columns: {missing}")
    benchmarks = sorted(table.bench.astype(str).unique())
    bench = table.bench.astype(str)
    source_blocks = {feature: str(metadata.loc[feature, "block"]) for feature in source_features}
    values = {feature: table[columns[feature]].astype(float).copy() for feature in source_features}
    support = {
        feature: {
            benchmark: bool(values[feature].loc[bench == benchmark].nunique(dropna=True) > 1)
            for benchmark in benchmarks
        }
        for feature in source_features
    }
    for feature in source_features:
        for benchmark, active in support[feature].items():
            if not active:
                values[feature].loc[bench == benchmark] = float("nan")
    nodes = {
        feature: {
            "leaf_members": [feature],
            "source_blocks": [source_blocks[feature]],
            "leaf_weights": {feature: 1.0},
            "leaf_weights_by_benchmark": {
                benchmark: ({feature: 1.0} if support[feature][benchmark] else {})
                for benchmark in benchmarks
            },
        }
        for feature in source_features
    }
    current = set(source_features)
    merges: list[dict[str, object]] = []

    pair_correlations: dict[tuple[str, str], dict[str, float]] = {}
    for benchmark in benchmarks:
        active = [feature for feature in source_features if support[feature][benchmark]]
        mask = bench == benchmark
        correlation = pd.DataFrame({
            feature: values[feature].loc[mask].to_numpy() for feature in active
        }).corr(method=method)
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                value = correlation.loc[left, right]
                if pd.notna(value):
                    pair_correlations.setdefault((left, right), {})[benchmark] = round(
                        float(value), 15,
                    )

    def candidate(
        left: str, right: str, correlations: dict[str, float],
    ) -> tuple[float, str, str, dict[str, float], int] | None:
        has_positive = any(value > 0 for value in correlations.values())
        has_negative = any(value < 0 for value in correlations.values())
        if has_positive and has_negative:
            return None
        score = round(sum(abs(value) for value in correlations.values()) / len(correlations), 15)
        if score < threshold:
            return None
        return score, left, right, correlations, -1 if has_negative else 1

    candidates = {
        pair: result
        for pair, correlations in pair_correlations.items()
        if (result := candidate(*pair, correlations)) is not None
    }
    while True:
        if not candidates:
            break
        score, left, right, correlations, right_sign = min(
            candidates.values(), key=lambda value: (-value[0], value[1], value[2]),
        )
        output = f"composite-{len(merges) + 1:03d}"
        if output in nodes:
            raise ValueError(f"generated composite name collides with source feature: {output}")
        output_values = _merge_values(
            values, left, right, right_sign, support, benchmarks, bench,
        )
        output_support = {
            benchmark: bool(output_values.loc[bench == benchmark].nunique(dropna=True) > 1)
            for benchmark in benchmarks
        }
        support[output] = output_support
        values[output] = output_values
        leaf_weights_by_benchmark = {}
        for benchmark in benchmarks:
            left_active = support[left][benchmark]
            right_active = support[right][benchmark]
            leaf_weights_by_benchmark[benchmark] = (
                _combined_weights(
                    nodes[left]["leaf_weights_by_benchmark"][benchmark],
                    nodes[right]["leaf_weights_by_benchmark"][benchmark],
                    right_sign,
                    left_active and right_active,
                )
                if left_active or right_active else {}
            )
        nodes[output] = {
            "leaf_members": sorted(nodes[left]["leaf_members"] + nodes[right]["leaf_members"]),
            "source_blocks": sorted(set(nodes[left]["source_blocks"] + nodes[right]["source_blocks"])),
            "leaf_weights": _combined_weights(
                nodes[left]["leaf_weights"], nodes[right]["leaf_weights"], right_sign, True,
            ),
            "leaf_weights_by_benchmark": leaf_weights_by_benchmark,
        }
        active_support = {
            benchmark: {
                "left": bool(support[left][benchmark]),
                "right": bool(support[right][benchmark]),
                "output": bool(output_support[benchmark]),
            }
            for benchmark in benchmarks
        }
        merges.append({
            "step": len(merges) + 1,
            "left_parent_id": left,
            "right_parent_id": right,
            "output_id": output,
            "output_name": output,
            "threshold_score": score,
            "signed_correlations_by_benchmark": correlations,
            "branch_sign_flip": right_sign == -1,
            "active_support": active_support,
            **nodes[output],
        })
        current.difference_update((left, right))
        current.add(output)
        candidates = {
            pair: value for pair, value in candidates.items()
            if left not in pair and right not in pair
        }
        for other in sorted(current - {output}):
            pair = tuple(sorted((other, output)))
            correlations = {}
            for benchmark in benchmarks:
                if not (support[other][benchmark] and support[output][benchmark]):
                    continue
                mask = bench == benchmark
                value = values[other].loc[mask].corr(
                    values[output].loc[mask], method=method,
                )
                if pd.notna(value):
                    correlations[benchmark] = round(float(value), 15)
            if correlations and (result := candidate(*pair, correlations)) is not None:
                candidates[pair] = result

    final_features = sorted(current)
    final_composites = [
        {
            "output_id": feature,
            "output_name": feature,
            "block": (
                nodes[feature]["source_blocks"][0]
                if len(nodes[feature]["source_blocks"]) == 1 else "mixed"
            ),
            **nodes[feature],
        }
        for feature in final_features
    ]
    provenance: dict[str, object] = {
        "algorithm": "greedy-maximum-correlation-hierarchical-mean",
        "correlation_method": method,
        "threshold": float(threshold),
        "threshold_inclusive": True,
        "benchmarks": benchmarks,
        "source_features": source_features,
        "source_blocks": source_blocks,
        "leaf_support": {
            feature: {benchmark: bool(active) for benchmark, active in supports.items()}
            for feature, supports in sorted(support.items()) if feature in source_features
        },
        "merges": merges,
        "final_features": final_features,
        "final_composites": final_composites,
    }
    rows = []
    for composite in final_composites:
        feature = composite["output_id"]
        if feature in metadata.index:
            row = metadata.loc[feature].to_dict()
            row["feature"] = feature
        else:
            row = {column: pd.NA for column in dictionary.columns}
            row.update({
                "feature": feature,
                "column": feature,
                "block": composite["block"],
                "source": "correlation-composite",
                "transform": "hierarchical-signed-mean",
                "n_missing": 0,
            })
        row.update({
            "group_size": len(composite["leaf_members"]),
            "leaf_members": json.dumps(composite["leaf_members"]),
            "source_blocks": json.dumps(composite["source_blocks"]),
        })
        rows.append(row)
    output_dictionary = pd.DataFrame(rows)
    output_dictionary = output_dictionary[[
        "feature", *[column for column in output_dictionary if column != "feature"]
    ]]
    return apply_composites(table, dictionary, provenance), output_dictionary, provenance


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
    transitive: bool = CONFIG.correlation_transitive,
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
