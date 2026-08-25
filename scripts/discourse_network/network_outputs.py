"""Transcript-level topology grids and exposure-adjusted circuit outputs."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .circuits import topology_features
from .attachments import topology_sensitivity
from .edge_views import aggregate_activations, aggregate_dense, weighted_kind_summary
from .nodes import stable_id


def _accepted_edges(candidates: pd.DataFrame, structural: list[dict[str, Any]], *,
                    threshold: float, max_parents: int, budget: int,
                    config_hash: str) -> list[dict[str, Any]]:
    accepted = list(structural)
    eligible = candidates[candidates["candidate_rank"].le(budget) &
                          candidates["probability"].ge(threshold)]
    for (transcript_key, target), group in eligible.groupby(["transcript_key", "target_node_id"]):
        for row in group.sort_values(["probability", "source_node_id"], ascending=[False, True]).head(
                max_parents).to_dict("records"):
            accepted.append({"edge_id": stable_id(transcript_key, row["source_node_id"], target,
                                                    "inferred", config_hash),
                             "transcript_key": transcript_key,
                             "source_node_id": row["source_node_id"],
                             "target_node_id": target, "edge_source": "inferred"})
    return accepted


def _topology_table(nodes: pd.DataFrame, candidates: pd.DataFrame,
                    structural: list[dict[str, Any]], configs: Iterable[tuple[str, float, int, int]],
                    config_hash: str) -> pd.DataFrame:
    rows = []
    for config_id, threshold, parents, budget in configs:
        accepted = _accepted_edges(candidates, structural, threshold=threshold,
                                   max_parents=parents, budget=budget, config_hash=config_hash)
        for transcript_key, node_group in nodes.groupby("transcript_key", sort=True):
            local_nodes = node_group.sort_values("node_id").to_dict("records")
            for node in local_nodes:
                node["is_final_sink"] = bool(node.get("is_final_sink")) if pd.notna(
                    node.get("is_final_sink")) else False
            local_edges = [edge for edge in accepted if edge["transcript_key"] == transcript_key]
            local_candidates = candidates[candidates["transcript_key"].eq(transcript_key)]
            features = topology_features(local_nodes, local_edges)
            rows.append({"transcript_key": transcript_key, "config_id": config_id,
                         "item_key": node_group.iloc[0].get("item_key"),
                         "split": node_group.iloc[0].get("split"),
                         "threshold": threshold, "maximum_parents": parents,
                         "candidate_budget": budget,
                         "candidate_population": int(local_candidates["candidate_rank"].le(budget).sum()),
                         "realized_candidate_size": min(
                             budget, int(local_candidates["candidate_rank"].max()))
                         if len(local_candidates) else 0,
                         **features})
    return pd.DataFrame(rows)


def build_topology_outputs(nodes: pd.DataFrame, candidates: pd.DataFrame,
                           structural_edges: list[dict[str, Any]], *, threshold: float,
                           primary_parents: int, primary_budget: int,
                           parent_grid: Iterable[int], threshold_offsets: Iterable[float],
                           budget_grid: Iterable[int], config_hash: str
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_config = [("primary", threshold, primary_parents, primary_budget)]
    grid_config = [(f"p{parents}-t{offset:+.2f}-b{budget}",
                    min(.95, max(.05, threshold + offset)), parents, budget)
                   for parents, offset, budget in product(parent_grid, threshold_offsets, budget_grid)]
    return (_topology_table(nodes, candidates, structural_edges, primary_config, config_hash),
            _topology_table(nodes, candidates, structural_edges, grid_config, config_hash))


def topology_robustness(primary: pd.DataFrame, grid: pd.DataFrame,
                        headline: Iterable[str]) -> dict[str, Any]:
    headline = [name for name in headline if name in primary]
    primary_values = {name: primary.sort_values("transcript_key")[name].to_numpy()
                      for name in headline}
    alternatives = [{name: group.sort_values("transcript_key")[name].to_numpy() for name in headline}
                    for _, group in grid.groupby("config_id", sort=True)]
    correlations = topology_sensitivity(primary_values, alternatives)
    populations = (grid.groupby(["candidate_budget", "config_id"])["candidate_population"].sum()
                   .groupby(level=0).first())
    return {"topology_grid_cells": grid["config_id"].nunique(),
            "nonconstant_headline_statistics": sorted(correlations),
            "topology_spearman": correlations,
            "candidate_budget_populations": {str(int(key)): int(value)
                                               for key, value in populations.items()},
            "max_realized_candidate_size": int(grid["realized_candidate_size"].max())}


def _final_ancestry_flags(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, bool]:
    flags = {}
    for transcript_key, node_group in nodes.groupby("transcript_key", sort=True):
        sinks = node_group[node_group["is_final_sink"].fillna(False)]["node_id"].astype(int).tolist()
        incoming = {}
        local_edges = edges[edges["transcript_key"].eq(transcript_key)]
        for row in local_edges.itertuples():
            incoming.setdefault(int(row.target_node_id), []).append(int(row.source_node_id))
        ancestry, stack = (set(sinks), list(sinks)) if len(sinks) == 1 else (set(), [])
        while stack:
            for source in incoming.get(stack.pop(), ()):
                if source not in ancestry:
                    ancestry.add(source)
                    stack.append(source)
        for row in local_edges.itertuples():
            flags[str(row.edge_id)] = (int(row.source_node_id) in ancestry and
                                       int(row.target_node_id) in ancestry)
    return flags


def aggregate_edge_transcripts(edges: pd.DataFrame, dense: np.ndarray,
                               activations: pd.DataFrame, nodes: pd.DataFrame, *,
                               feature_ids: Iterable[int] | None = None,
                               cluster_ids: Iterable[int] | None = None
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist joint and per-source transcript estimands with sparse zeros included."""
    ancestry = _final_ancestry_flags(nodes, edges)
    feature_ids = (set(activations.get("feature_id", pd.Series(dtype=int)).astype(int))
                   if feature_ids is None else {int(feature) for feature in feature_ids})
    cluster_ids = (None if cluster_ids is None else
                   sorted({int(cluster) for cluster in cluster_ids}))
    summaries, feature_rows = [], []
    for transcript_key, transcript in edges.groupby("transcript_key", sort=True):
        scopes = [("combined", transcript)] + [
            (str(source), group) for source, group in transcript.groupby("edge_source", sort=True)]
        for scope, group in scopes:
            positions = group["dense_row"].astype(int).to_numpy()
            flags = np.array([ancestry[str(edge_id)] for edge_id in group["edge_id"]])
            row = {"transcript_key": transcript_key, "edge_scope": scope,
                   "edge_count": len(group), "item_key": group.iloc[0].get("item_key"),
                   "split": group.iloc[0].get("split")}
            for name, values in aggregate_dense(np.asarray(dense)[positions], flags).items():
                row.update({f"dense_{name}_{index}": float(value)
                            for index, value in enumerate(values)})
            clusters = cluster_ids
            if clusters is None and group["cluster_id"].notna().any():
                clusters = range(int(edges["cluster_id"].dropna().max()) + 1)
            if clusters is not None:
                counts = group["cluster_id"].dropna().astype(int).value_counts()
                row.update({f"cluster_{cluster}_proportion": float(counts.get(cluster, 0) / len(group))
                            for cluster in clusters})
            summaries.append(row)
            inventory = [{"unit_id": str(edge_id), "in_final_ancestry": bool(flag)}
                         for edge_id, flag in zip(group["edge_id"], flags)]
            selected = activations[activations["edge_id"].isin(group["edge_id"])]
            metrics = aggregate_activations(
                selected.rename(columns={"edge_id": "unit_id"}).to_dict("records"),
                inventory=inventory, feature_ids=feature_ids)
            feature_rows.extend({"transcript_key": transcript_key, "edge_scope": scope,
                                 "item_key": group.iloc[0].get("item_key"),
                                 "split": group.iloc[0].get("split"),
                                 "feature_id": feature, **values}
                                for feature, values in metrics.items())
    return pd.DataFrame(summaries), pd.DataFrame(feature_rows)


def aggregate_circuit_transcripts(circuits: pd.DataFrame, dense: np.ndarray,
                                  activations: pd.DataFrame, *,
                                  feature_ids: Iterable[int] | None = None
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, feature_rows = [], []
    feature_ids = (set(activations.get("feature_id", pd.Series(dtype=int)).astype(int))
                   if feature_ids is None else {int(feature) for feature in feature_ids})
    for transcript_key, group in circuits.groupby("transcript_key", sort=True):
        positions = (group["dense_row"].astype(int).to_numpy() if "dense_row" in group and
                     group["dense_row"].notna().all() else group.index.to_numpy())
        vectors = dense[positions]
        flags = group["final_ancestry"].astype(bool).to_numpy()
        kinds = sorted(set(group["extractor_kind"]))
        kind_summaries = {kind: aggregate_dense(
            vectors[group["extractor_kind"].eq(kind).to_numpy()],
            flags[group["extractor_kind"].eq(kind).to_numpy()]) for kind in kinds}
        precap = {kind: int(group[group["extractor_kind"].eq(kind)]["precap_count"].iloc[0])
                  for kind in kinds}
        precap_total = sum(precap.values())
        proportions = {kind: count / precap_total for kind, count in precap.items()}
        recorded = {kind: float(group[group["extractor_kind"].eq(kind)][
            "true_kind_proportion"].iloc[0]) for kind in kinds}
        if any(not np.isclose(recorded[kind], proportions[kind]) for kind in kinds):
            raise ValueError("circuit kind proportions disagree with exact pre-cap counts")
        cluster_by_kind = {}
        if group["cluster_id"].notna().any():
            width = int(group["cluster_id"].dropna().max()) + 1
            for kind in kinds:
                subset = group[group["extractor_kind"].eq(kind)]["cluster_id"]
                vector = np.zeros(width)
                for cluster, count in subset.dropna().astype(int).value_counts().items():
                    vector[cluster] = count / len(subset)
                cluster_by_kind[kind] = vector
        for kind in kinds:
            kind_group = group[group["extractor_kind"].eq(kind)]
            row = {"transcript_key": transcript_key, "summary_scope": "extractor_kind",
                   "extractor_kind": kind, "estimand_basis": "retained_sample_within_kind",
                   "retained_count": len(kind_group),
                   "precap_count": precap[kind],
                   "true_kind_proportion": proportions[kind],
                   "item_key": group.iloc[0].get("item_key"), "split": group.iloc[0].get("split")}
            for name, values in kind_summaries[kind].items():
                row.update({f"dense_{name}_{index}": float(value)
                            for index, value in enumerate(values)})
            if kind in cluster_by_kind:
                row.update({f"cluster_{index}_proportion": float(value)
                            for index, value in enumerate(cluster_by_kind[kind])})
            summaries.append(row)
        weighted = {"mean": weighted_kind_summary(
            {kind: kind_summaries[kind]["mean"] for kind in kinds}, proportions)}
        row = {"transcript_key": transcript_key, "summary_scope": "combined_weighted",
               "extractor_kind": None, "estimand_basis": "exact_precap_kind_weighted_means",
               "retained_count": len(group), "precap_count": precap_total,
               "true_kind_proportion": 1.0,
               "item_key": group.iloc[0].get("item_key"), "split": group.iloc[0].get("split")}
        for name, values in weighted.items():
            row.update({f"dense_{name}_{index}": float(value) for index, value in enumerate(values)})
        for name in ("median", "q10", "q90", "ancestry_contrast"):
            row.update({f"dense_{name}_{index}": np.nan for index in range(vectors.shape[1])})
        if cluster_by_kind:
            weighted_clusters = weighted_kind_summary(cluster_by_kind, proportions)
            row.update({f"cluster_{index}_proportion": float(value)
                        for index, value in enumerate(weighted_clusters)})
        summaries.append(row)
        selected = activations[activations["circuit_id"].isin(group["circuit_id"])]
        by_kind = {}
        for kind in kinds:
            kind_group = group[group["extractor_kind"].eq(kind)]
            kind_inventory = [{"unit_id": circuit_id, "in_final_ancestry": bool(ancestry)}
                              for circuit_id, ancestry in zip(kind_group["circuit_id"],
                                                              kind_group["final_ancestry"])]
            kind_rows = selected[selected["circuit_id"].isin(kind_group["circuit_id"])].rename(
                columns={"circuit_id": "unit_id"}).to_dict("records")
            by_kind[kind] = aggregate_activations(kind_rows, inventory=kind_inventory,
                                                   feature_ids=feature_ids)
        for kind in kinds:
            feature_rows.extend({"transcript_key": transcript_key,
                                 "summary_scope": "extractor_kind", "extractor_kind": kind,
                                 "estimand_basis": "retained_sample_within_kind",
                                 "item_key": group.iloc[0].get("item_key"),
                                 "split": group.iloc[0].get("split"), "feature_id": feature,
                                 **metrics} for feature, metrics in by_kind[kind].items())
        for feature in sorted(feature_ids):
            metrics = {field: sum(proportions[kind] * by_kind[kind][feature][field]
                                  for kind in kinds)
                       for field in ("mean", "proportion_active")}
            metrics.update(median=np.nan, q90=np.nan, ancestry_contrast=np.nan)
            feature_rows.append({"transcript_key": transcript_key,
                                 "summary_scope": "combined_weighted", "extractor_kind": None,
                                 "estimand_basis": "exact_precap_kind_weighted_means",
                                 "item_key": group.iloc[0].get("item_key"),
                                 "split": group.iloc[0].get("split"),
                                 "feature_id": feature, **metrics})
    return pd.DataFrame(summaries), pd.DataFrame(feature_rows)
